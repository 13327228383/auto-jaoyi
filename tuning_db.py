# -*- coding: utf-8 -*-
"""
tuning_db.py —— 参数落库 + 自动调参（数据库层）
================================================================================
目标：把策略参数从"写死代码"改为"以数据库为单一事实源 + 按实盘表现自动调参"。

数据库：本地 MySQL（服务 MySQL97，host=127.0.0.1:3306）。库名 TONGHUASHUN。
连接凭据从 config.ini [db] 读取（config.ini 已在 .gitignore，不提交敏感信息）。

表结构：
  param_sets   候选/上线的参数组合 + 回测画像（SLOPE_WINDOW/SR_TRAIL_PCT/.../backtest_json/source/status）
  trade_log    实盘成交（由 auto_run.journal 双写，param_set_id 标记当时生效参数），含已往回填的历史流水
  active_param 当前生效参数（单行，记录在 "selected_cohort"）—— auto_run 读取后覆盖 CFG 可调键
  tuning_audit 每次自动评估/切换的审计日志

安全模型：
  - DB 连不上时所有函数返回 None / 空，auto_run 保持用本地回测最优参数，绝不停摆、不误下单。
  - 自动切换由 auto_tuner.run() 触发，配有护栏（最小实盘样本/最小提升阈值/最小切换间隔），
    未达条件则维持现役参数。现役参数永远有一个"已上线"的研究最优守卫（出厂备份）。
"""
import os, json, time
import configparser
import pymysql

HERE = os.path.dirname(os.path.abspath(__file__))
DB_NAME = "TONGHUASHUN"

# auto_run.CFG 中可被自动调参覆盖的键（其余如风控框架/标的池/宏观层保持代码常量，不在调参范围）。
# 值类型用于建表/强制转换。允许自动切换的必须是"研究已验证、独立影响收益/磨损"的旋钮。
# 注意 MIN_HOLD_DAYS 属"交易节奏"，曾在此列→盘中热更可能悄悄放开换仓锁触发当日换仓。
# 已移出（R3 堵漏）：固定为 auto_run.CFG 代码常量，不再被 DB 下发/盘中热改。
# 表结构(active_param/param_sets)保留该列以兼容历史数据，但读取/下发/版本匹配均忽略它。
TUNABLE_KEYS = {
    "SLOPE_WINDOW": int,
    "SR_TRAIL_PCT": float,
    "DEF_PEAK_STOP": float,
    "DEF_MOM_DAYS": int,
    "DEF_MOM_ENTER": float,
    "DEF_MOM_EXIT": float,
    "HOLD_N": int,
}

# 出厂/研究最优守卫（当前稳健最优，grid_final 抗摩擦选点：0.2%滑点抗性最高、回撤最小、SW不变）
FALLBACK_ACTIVE = {
    "SLOPE_WINDOW": 30, "SR_TRAIL_PCT": 0.02, "DEF_PEAK_STOP": 0.02,
    "MIN_HOLD_DAYS": 5, "DEF_MOM_DAYS": 10, "DEF_MOM_ENTER": 0.005,
    "DEF_MOM_EXIT": -0.008, "HOLD_N": 1,
    "note": "研究稳健最优(grid_final)：trail2%/SW30/moh5，0.1%年化25.6%、0.2%抗性10.7%、回撤-23.4%、换仓310",
    "source": "research_seed",
}


def _read_db_conf():
    """从 config.ini [db] 读连接；缺失则用本机默认(root/无密码)。
    安全：DB 密码**只**从 config.ini[db].password 读取（config.ini 已在 .gitignore，严禁提交），
    代码内不再含任何明文密码默认值。config.ini 无密码时以空串尝试连接（一般会鉴权失败，
    tuning_db 全部函数都优雅返回 None，auto_run 保持用研发现役参数，不因 DB 缺失停摆）。"""
    conf = {"host": "127.0.0.1", "port": 3306, "user": "root", "password": "", "charset": "utf8mb4"}
    try:
        cp = configparser.ConfigParser()
        cp.read(os.path.join(HERE, "config.ini"), encoding="utf-8")
        if "db" in cp:
            for k in ("host", "port", "user", "password", "charset"):
                if cp["db"].get(k):
                    v = cp["db"][k]
                    conf[k] = int(v) if k == "port" else v
    except Exception:
        pass
    return conf


def connect(database=None, autocommit=True):
    conf = _read_db_conf()
    kw = dict(host=conf["host"], port=int(conf["port"]), user=conf["user"],
              password=conf["password"], charset=conf.get("charset", "utf8mb4"))
    if database:
        kw["database"] = database
    return pymysql.connect(connect_timeout=5, autocommit=autocommit, **kw)


def ensure_db():
    """建库 + 建表（幂等）。返回 True 表示可用，False 表示连不上（调用方需优雅降级）。"""
    try:
        c = connect()
    except Exception as e:
        print(f"[tuning_db] MySQL 连接失败，走本地兜底：{e}")
        return False
    try:
        with c.cursor() as cur:
            cur.execute(f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}` "
                        "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        c.select_db(DB_NAME)
        with c.cursor() as cur:
            cur.execute("""CREATE TABLE IF NOT EXISTS param_sets(
                id INT PRIMARY KEY AUTO_INCREMENT,
                name VARCHAR(120) NOT NULL,
                SLOPE_WINDOW INT, SR_TRAIL_PCT DOUBLE, DEF_PEAK_STOP DOUBLE,
                MIN_HOLD_DAYS INT, DEF_MOM_DAYS INT, DEF_MOM_ENTER DOUBLE,
                DEF_MOM_EXIT DOUBLE, HOLD_N INT,
                status VARCHAR(32) NOT NULL DEFAULT 'candidate',
                backtest_json JSON NULL,
                source VARCHAR(64) NULL,
                note VARCHAR(500) NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY uq_cfg (SLOPE_WINDOW,SR_TRAIL_PCT,DEF_PEAK_STOP,MIN_HOLD_DAYS,DEF_MOM_DAYS,DEF_MOM_ENTER,DEF_MOM_EXIT,HOLD_N)
            ) ENGINE=InnoDB""")
            cur.execute("""CREATE TABLE IF NOT EXISTS trade_log(
                id INT PRIMARY KEY AUTO_INCREMENT,
                ts DATETIME NOT NULL,
                side VARCHAR(16) NOT NULL,
                code VARCHAR(16) NOT NULL,
                price DOUBLE NULL, qty BIGINT NULL, amount DOUBLE NULL,
                reason VARCHAR(255) NULL,
                param_set_id INT NULL,
                is_sim TINYINT NOT NULL DEFAULT 0,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                KEY idx_ts (ts), KEY idx_param (param_set_id), KEY idx_sim (is_sim)
            ) ENGINE=InnoDB""")
            cur.execute("""CREATE TABLE IF NOT EXISTS active_param(
                id INT PRIMARY KEY AUTO_INCREMENT,
                param_set_id INT NULL,
                SLOPE_WINDOW INT, SR_TRAIL_PCT DOUBLE, DEF_PEAK_STOP DOUBLE,
                MIN_HOLD_DAYS INT, DEF_MOM_DAYS INT, DEF_MOM_ENTER DOUBLE,
                DEF_MOM_EXIT DOUBLE, HOLD_N INT,
                note VARCHAR(500) NULL,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            ) ENGINE=InnoDB""")
            cur.execute("""CREATE TABLE IF NOT EXISTS tuning_audit(
                id INT PRIMARY KEY AUTO_INCREMENT,
                run_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                prev_param_set_id INT NULL, new_param_set_id INT NULL,
                changed TINYINT NOT NULL DEFAULT 0,
                reason VARCHAR(500) NULL,
                detail JSON NULL
            ) ENGINE=InnoDB""")
            cur.execute("""CREATE TABLE IF NOT EXISTS sim_panel(
                id INT PRIMARY KEY AUTO_INCREMENT,
                param_set_id INT NOT NULL,
                window_start DATE NULL, window_end DATE NULL,
                ann DOUBLE NULL, maxdd DOUBLE NULL, switches INT NULL,
                note VARCHAR(255) NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY uq_ps_win (param_set_id, window_end)
            ) ENGINE=InnoDB""")
        return True
    except Exception as e:
        print(f"[tuning_db] 初始化失败：{e}")
        return False


def seed_from_research(cfg_rows, meta_note=""):
    """把研究候选（dict 列表，含 TUNABLE_KEYS + backtest 字段）写入 param_sets。
    status: 首个=active（现役），其余=candidate。返回主键列表。"""
    if not ensure_db():
        return []
    sql = ("INSERT IGNORE INTO param_sets "
           "(name,SLOPE_WINDOW,SR_TRAIL_PCT,DEF_PEAK_STOP,MIN_HOLD_DAYS,DEF_MOM_DAYS,DEF_MOM_ENTER,DEF_MOM_EXIT,HOLD_N,"
           "status,backtest_json,source,note) "
           "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)")
    ids = []
    c = connect(DB_NAME)
    try:
        for i, row in enumerate(cfg_rows):
            bj = json.dumps(row.get("backtest", {}), ensure_ascii=False)
            name = row.get("name") or f"research_{row.get('SLOPE_WINDOW')}_{int(row.get('SR_TRAIL_PCT',0)*10000)}"
            with c.cursor() as cur:
                cur.execute(sql, (
                    name, row["SLOPE_WINDOW"], row["SR_TRAIL_PCT"], row.get("DEF_PEAK_STOP", row["SR_TRAIL_PCT"]),
                    row.get("MIN_HOLD_DAYS", 5), row.get("DEF_MOM_DAYS", 10),
                    row.get("DEF_MOM_ENTER", 0.005), row.get("DEF_MOM_EXIT", -0.008),
                    row.get("HOLD_N", 1), "candidate", bj, row.get("source", "research"), row.get("note", "")))
                cur.execute("SELECT id FROM param_sets WHERE "
                            "SLOPE_WINDOW=%s AND SR_TRAIL_PCT=%s AND DEF_PEAK_STOP=%s AND MIN_HOLD_DAYS=%s "
                            "AND DEF_MOM_DAYS=%s AND DEF_MOM_ENTER=%s AND DEF_MOM_EXIT=%s AND HOLD_N=%s",
                            (row["SLOPE_WINDOW"], row["SR_TRAIL_PCT"], row.get("DEF_PEAK_STOP", row["SR_TRAIL_PCT"]),
                             row.get("MIN_HOLD_DAYS", 5), row.get("DEF_MOM_DAYS", 10),
                             row.get("DEF_MOM_ENTER", 0.005), row.get("DEF_MOM_EXIT", -0.008), row.get("HOLD_N", 1)))
                rid = cur.fetchone()[0]
                ids.append(rid)
                if i == 0:
                    set_active(rid, row.get("note", meta_note))
    finally:
        c.close()
    return ids


def get_param_set_id(row):
    """by cfg dict → id（不存在则返回 None）。
    R5 堵漏：须匹配含 DEF_PEAK_STOP 的完整参数才视为同版本，减少成交版本号错位。
    注意：param_sets 允许存在"7 参相同、仅 MIN_HOLD_DAYS 不同"的多行，此函数会取 id 升序第一行。
    为精确定位真实版本，优先用 get_active_param_set_id()（读 active_param.param_set_id）。"""
    if not row:
        return None
    try:
        c = connect(DB_NAME)
        with c.cursor() as cur:
            cur.execute("SELECT id FROM param_sets WHERE SLOPE_WINDOW=%s AND SR_TRAIL_PCT=%s AND "
                        "DEF_PEAK_STOP=%s AND DEF_MOM_DAYS=%s AND DEF_MOM_ENTER=%s AND DEF_MOM_EXIT=%s AND HOLD_N=%s",
                        (row.get("SLOPE_WINDOW"), row.get("SR_TRAIL_PCT"),
                         row.get("DEF_PEAK_STOP", row.get("SR_TRAIL_PCT")),
                         row.get("DEF_MOM_DAYS", 10), row.get("DEF_MOM_ENTER", 0.005),
                         row.get("DEF_MOM_EXIT", -0.008), row.get("HOLD_N", 1)))
            r = cur.fetchone()
            return r[0] if r else None
    except Exception:
        return None
    finally:
        try: c.close()
        except Exception: pass


def get_active_params():
    """读取现役参数 dict（只含 TUNABLE_KEYS）。无入库/连不上 → 返回 FALLBACK_ACTIVE（研究最优守卫）。"""
    if not ensure_db():
        return {k: v for k, v in FALLBACK_ACTIVE.items() if k in TUNABLE_KEYS} | {"_fallback": True}
    try:
        c = connect(DB_NAME)
        with c.cursor() as cur:
            cur.execute("SELECT SLOPE_WINDOW,SR_TRAIL_PCT,DEF_PEAK_STOP,DEF_MOM_DAYS,"
                        "DEF_MOM_ENTER,DEF_MOM_EXIT,HOLD_N FROM active_param ORDER BY id DESC LIMIT 1")
            r = cur.fetchone()
        cols = list(TUNABLE_KEYS)
        if r and r[0] is not None:
            return {k: TUNABLE_KEYS[k](v) for k, v in zip(cols, r) if v is not None}
        return {k: v for k, v in FALLBACK_ACTIVE.items() if k in TUNABLE_KEYS}
    except Exception:
        return {k: v for k, v in FALLBACK_ACTIVE.items() if k in TUNABLE_KEYS}
    finally:
        try: c.close()
        except Exception: pass


def get_active_param_set_id():
    """现役**真实参数版本**(active_param.param_set_id，由 set_active 写入时记录)。
    P1 彻底堵漏：版本标识直接用此值，避免按参数反查时在"仅 MIN_HOLD_DAYS 不同"的重复行间歧义，
    从而保证 insert_trade/auto_tuner 的成交分组都落到正确的参数版本。无现役→None。"""
    if not ensure_db():
        return None
    try:
        c = connect(DB_NAME)
        with c.cursor() as cur:
            cur.execute("SELECT param_set_id FROM active_param ORDER BY id DESC LIMIT 1")
            r = cur.fetchone()
        c.close()
        return r[0] if (r and r[0] is not None) else None
    except Exception:
        return None
    finally:
        try: c.close()
        except Exception: pass


def get_param(row_id):
    """按 id 读某个参数组的完整 8 键值（含 MIN_HOLD_DAYS）。不存在/DB不可用→None。"""
    if not ensure_db():
        return None
    try:
        c = connect(DB_NAME)
        with c.cursor() as cur:
            cur.execute("SELECT SLOPE_WINDOW,SR_TRAIL_PCT,DEF_PEAK_STOP,MIN_HOLD_DAYS,DEF_MOM_DAYS,"
                        "DEF_MOM_ENTER,DEF_MOM_EXIT,HOLD_N FROM param_sets WHERE id=%s", (row_id,))
            r = cur.fetchone()
        c.close()
        if not r:
            return None
        cols = ["SLOPE_WINDOW", "SR_TRAIL_PCT", "DEF_PEAK_STOP", "MIN_HOLD_DAYS",
                "DEF_MOM_DAYS", "DEF_MOM_ENTER", "DEF_MOM_EXIT", "HOLD_N"]
        return dict(zip(cols, r))
    except Exception:
        return None
    finally:
        try: c.close()
        except Exception: pass


def upsert_param_set(params7, note="auto_tuner staging"):
    """按 7 个可调键查找；已存在则返回现有 id，不存在则插入一条 candidate 并返回其 id。
    用于 auto_tuner 的"逐步逼近"：把中间参数作为【真实 param_sets 行】落库，从而
    auto_run 应用时永远 applied==DB、版号精确（避免折中域外孤儿态）。"""
    if not ensure_db():
        return None
    p = dict(params7)
    try:
        c = connect(DB_NAME)
        with c.cursor() as cur:
            cur.execute("SELECT id FROM param_sets WHERE SLOPE_WINDOW=%s AND SR_TRAIL_PCT=%s AND "
                        "DEF_PEAK_STOP=%s AND MIN_HOLD_DAYS=%s AND DEF_MOM_DAYS=%s AND DEF_MOM_ENTER=%s "
                        "AND DEF_MOM_EXIT=%s AND HOLD_N=%s",
                        (p.get("SLOPE_WINDOW"), p.get("SR_TRAIL_PCT"), p.get("DEF_PEAK_STOP"),
                         p.get("MIN_HOLD_DAYS", 5), p.get("DEF_MOM_DAYS", 10),
                         p.get("DEF_MOM_ENTER", 0.005), p.get("DEF_MOM_EXIT", -0.008), p.get("HOLD_N", 1)))
            r = cur.fetchone()
            if r:
                return r[0]
            cur.execute("INSERT INTO param_sets"
                        "(name,SLOPE_WINDOW,SR_TRAIL_PCT,DEF_PEAK_STOP,MIN_HOLD_DAYS,DEF_MOM_DAYS,DEF_MOM_ENTER,DEF_MOM_EXIT,HOLD_N,status,source,note) "
                        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                        (f"staging_sw{p.get('SLOPE_WINDOW')}", p.get("SLOPE_WINDOW"), p.get("SR_TRAIL_PCT"),
                         p.get("DEF_PEAK_STOP"), p.get("MIN_HOLD_DAYS", 5), p.get("DEF_MOM_DAYS", 10),
                         p.get("DEF_MOM_ENTER", 0.005), p.get("DEF_MOM_EXIT", -0.008), p.get("HOLD_N", 1),
                         "candidate", "auto_tuner_staging", note[:490]))
            return cur.lastrowid
    except Exception:
        return None
    finally:
        try: c.close()
        except Exception: pass


def set_active(param_set_id, note=""):
    """切换现役参数（写入 active_param）。"""
    if not ensure_db():
        return False
    try:
        c = connect(DB_NAME)
        with c.cursor() as cur:
            cur.execute("SELECT SLOPE_WINDOW,SR_TRAIL_PCT,DEF_PEAK_STOP,MIN_HOLD_DAYS,DEF_MOM_DAYS,"
                        "DEF_MOM_ENTER,DEF_MOM_EXIT,HOLD_N FROM param_sets WHERE id=%s", (param_set_id,))
            r = cur.fetchone()
            if not r:
                return False
            cur.execute("DELETE FROM active_param")
            cur.execute("INSERT INTO active_param"
                        "(param_set_id,SLOPE_WINDOW,SR_TRAIL_PCT,DEF_PEAK_STOP,MIN_HOLD_DAYS,DEF_MOM_DAYS,DEF_MOM_ENTER,DEF_MOM_EXIT,HOLD_N,note) "
                        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                        (param_set_id, *r, note[:490] if note else None))
        return True
    except Exception:
        return False
    finally:
        try: c.close()
        except Exception: pass


def insert_trade(rec, param_set_id=None, is_sim=0):
    """写入一条实盘成交到 trade_log（由 auto_run.journal 双写）。
    is_sim=1 表示模拟/回放成交（tune_sim），自动调参器默认排除，防止假数据触发真钱换参。"""
    if not ensure_db():
        return False
    try:
        c = connect(DB_NAME)
        with c.cursor() as cur:
            cur.execute("INSERT INTO trade_log(ts,side,code,price,qty,amount,reason,param_set_id,is_sim) "
                        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                        (rec["ts"], rec["side"], rec["code"], rec.get("price"), rec.get("qty"),
                         rec.get("amount"), rec.get("reason"), param_set_id, is_sim))
        return True
    except Exception:
        return False
    finally:
        try: c.close()
        except Exception: pass


def real_trades(param_set_id=None, include_sim=False):
    """读取 trade_log（可按参数组过滤）。默认排除模拟成交(is_sim=1)，保证真钱调参只用真实数据。"""
    if not ensure_db():
        return []
    try:
        c = connect(DB_NAME)
        with c.cursor() as cur:
            if param_set_id is not None:
                cur.execute("SELECT ts,side,code,price,qty,amount,reason,param_set_id,is_sim FROM trade_log "
                            "WHERE param_set_id=%s AND is_sim=%s ORDER BY ts", (param_set_id, 1 if include_sim else 0))
            else:
                cur.execute("SELECT ts,side,code,price,qty,amount,reason,param_set_id,is_sim FROM trade_log "
                            "WHERE is_sim=%s ORDER BY ts", (1 if include_sim else 0,))
            cols = ["ts", "side", "code", "price", "qty", "amount", "reason", "param_set_id", "is_sim"]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    except Exception:
        return []
    finally:
        try: c.close()
        except Exception: pass


def list_param_sets(include_candidates=True):
    """列出 param_sets 主键与可调参数。返回 [(id, {SW,trail,min_hold,...})]。"""
    if not ensure_db():
        return []
    try:
        c = connect(DB_NAME)
        with c.cursor() as cur:
            cur.execute("SELECT id,SLOPE_WINDOW,SR_TRAIL_PCT,MIN_HOLD_DAYS,DEF_MOM_DAYS,"
                        "DEF_MOM_ENTER,DEF_MOM_EXIT,HOLD_N FROM param_sets ORDER BY id")
            cols = ["SW", "trail", "MOH", "DEF_MOM_DAYS", "DEF_MOM_ENTER", "DEF_MOM_EXIT", "HOLD_N"]
            return [(r[0], dict(zip(cols, r[1:]))) for r in cur.fetchall()]
    except Exception:
        return []
    finally:
        try: c.close()
        except Exception: pass


def save_sim_panel(param_set_id, win_start, win_end, ann, maxdd, switches, note=""):
    """写入（幂等，按 param+window_end 去重）某参数组的近一年模拟表现。"""
    if not ensure_db():
        return
    try:
        c = connect(DB_NAME)
        with c.cursor() as cur:
            cur.execute("INSERT INTO sim_panel(param_set_id,window_start,window_end,ann,maxdd,switches,note) "
                        "VALUES (%s,%s,%s,%s,%s,%s,%s) "
                        "ON DUPLICATE KEY UPDATE ann=VALUES(ann),maxdd=VALUES(maxdd),"
                        "switches=VALUES(switches),note=VALUES(note),window_start=VALUES(window_start)",
                        (param_set_id, win_start, win_end, ann, maxdd, switches, note))
    except Exception:
        pass
    finally:
        try: c.close()
        except Exception: pass


def get_sim_panel():
    """近一年模拟面板。返回 [(param_set_id, ann, maxdd, switches)]（按 ann 降序）。"""
    if not ensure_db():
        return []
    try:
        c = connect(DB_NAME)
        with c.cursor() as cur:
            cur.execute("SELECT param_set_id,ann,maxdd,switches FROM sim_panel "
                        "ORDER BY ann DESC")
            return [tuple(r) for r in cur.fetchall()]
    except Exception:
        return []
    finally:
        try: c.close()
        except Exception: pass


def audit(prev_id, new_id, changed, reason, detail=None):
    """写调参审计。"""
    if not ensure_db():
        return
    try:
        c = connect(DB_NAME)
        with c.cursor() as cur:
            cur.execute("INSERT INTO tuning_audit(prev_param_set_id,new_param_set_id,changed,reason,detail) "
                        "VALUES (%s,%s,%s,%s,%s)",
                        (prev_id, new_id, 1 if changed else 0, reason or "",
                         json.dumps(detail or {}, ensure_ascii=False)))
    except Exception:
        pass
    finally:
        try: c.close()
        except Exception: pass


if __name__ == "__main__":
    ok = ensure_db()
    print(f"[tuning_db] ensure_db = {ok}")
    act = get_active_params()
    print(f"[tuning_db] active = {act}")