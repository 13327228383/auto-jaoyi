# -*- coding: utf-8 -*-
"""exec32_trade.py —— 用 pywinauto 填表 + 物理鼠标点击 提交同花顺买入/卖出委托。

用法:
  python exec32_trade.py sell sh511010 141.187 100
  python exec32_trade.py buy  sz511010 141.187 100
说明:
  - 填入代码/价格/数量 → 点(买入/卖出)按钮 → 处理委托确认弹窗(物理点"是")。
  - 实测: pywinauto 的 .click() 在此客户端关不掉确认弹窗，必须用
    SetForegroundWindow + SetCursorPos + mouse_event 物理点击。
"""
import re
import sys
import time
import win32api
import win32con
import win32gui
import win32process
from pywinauto import Application

# 统一 stdout/stderr 为 UTF-8：父进程按 utf-8 解码子进程输出，避免 GBK/UTF-8 错位导致
# "UnicodeDecodeError: 'utf-8' codec can't decode byte 0xcf" 崩溃（tasklist 含中文所致）。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

TITLE = "网上股票交易系统5.0"
MAIN_HANDLE = 0  # 运行时按 pid+标题自动定位，不再写死（客户端重启后句柄会变）
SHORTCUT = {"buy": "{F1}", "sell": "{F2}"}
XIAIDAN_IMG = "xiadan.exe"


def find_main_handle(pid):
    """在 xiadan pid 的顶层窗口里，按标题定位主交易窗口句柄。"""
    import win32gui
    want = TITLE
    h = [0]

    def cb(hwnd, _):
        if h[0] != 0:
            return
        # 注意：此处不能按 IsWindowVisible 过滤！主窗口可能被【隐藏】(SW_HIDE, vis=0)，
        # 但仍在屏幕上、坐标正常。按可见性过滤会导致"未找到主窗口"，进而连累整链失败。
        try:
            _, p = win32process.GetWindowThreadProcessId(hwnd)
        except Exception:
            return
        if p == pid and (win32gui.GetWindowText(hwnd) or "").startswith(want):
            h[0] = hwnd

    win32gui.EnumWindows(cb, None)
    return h[0]


def ths_pid():
    import subprocess
    out = subprocess.check_output(
        'tasklist /FI "IMAGENAME eq xiadan.exe"', shell=True,
        text=True, encoding="gbk", errors="replace",
    )
    for line in out.splitlines():
        if line.lower().startswith("xiadan.exe"):
            return int(line.split()[1])
    return None


def norm_btn(t):
    """把 '是(&Y)' / '确定(O)' 等按钮文本归一化为 '是'/'确定'。"""
    t = (t or "").replace("&", "")
    t = re.sub(r"\([A-Za-z]\)\s*$", "", t).strip()
    return t


def phys_click(hwnd):
    """物理鼠标点击窗口中心。"""
    win32gui.SetForegroundWindow(hwnd)
    time.sleep(0.2)
    r = win32gui.GetWindowRect(hwnd)
    cx = (r[0] + r[2]) // 2
    cy = (r[1] + r[3]) // 2
    win32api.SetCursorPos((cx, cy))
    time.sleep(0.1)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    time.sleep(0.06)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
    time.sleep(0.2)


def get_main(pid):
    h = find_main_handle(pid)
    if h == 0:
        raise SystemExit(f"ERROR: 未找到 {TITLE} 主窗口(xiadan pid={pid})")
    app = Application(backend="win32").connect(handle=h)
    w = app.window(handle=h)
    return app, w


def find_visible_edit(w, control_id):
    """在同 id 的多个 Edit 中，挑选可见、坐标合理、最靠左上方的那个。"""
    cand = []
    for ctrl in w.descendants(class_name="Edit"):
        try:
            if ctrl.control_id() != control_id:
                continue
            r = ctrl.rectangle()
            wdt = r.right - r.left
            hgt = r.bottom - r.top
            if r.left >= 0 and r.top >= 0 and wdt > 20 and hgt > 5 and ctrl.is_visible():
                cand.append((r.top, r.left, ctrl))
        except Exception:
            continue
    if cand:
        cand.sort(key=lambda x: (x[0], x[1]))
        return cand[0][2]
    return None


def fill_edit(w, control_id, text):
    """键盘输入：Ctrl+End 光标到末尾后逐字 Backspace 清空，再输入文本。
    注意：此客户端自绘输入框不接受 Ctrl+A/Delete 清空（会变成追加），
    必须用 Ctrl+End+Backspace（见 project_memory 关于清空输入字段的约定）。"""
    e = find_visible_edit(w, control_id)
    if e is None:
        return "NONE"
    e.set_focus()
    time.sleep(0.1)
    e.type_keys("^{END}")            # 光标到行尾
    time.sleep(0.05)
    for _ in range(20):              # 逐个删除，清空残留
        e.type_keys("{BACKSPACE}")
    time.sleep(0.05)
    e.type_keys(str(text))
    time.sleep(0.15)
    if control_id == 1032:  # 代码框填完回车触发名称联动
        e.type_keys("{ENTER}")
        time.sleep(0.3)
    return e.window_text()


def _btn_shape_ok(ctrl):
    """仅用尺寸+屏幕内坐标判断按钮可点（is_visible/is_enabled 在同花顺上不可靠，忽略）。"""
    try:
        r = ctrl.rectangle()
        if r.width() <= 5 or r.height() <= 5:
            return False
        if r.right <= r.left or r.bottom <= r.top:
            return False
    except Exception:
        return False
    return True


def click_board_button(w, text):
    """物理点击主窗口内 买入/卖出 按钮。

    实测(2026-09): 买卖提交按钮共用 control_id=1006（买入页文本"买入"，卖出页
    文本"卖出"）；control_id=1007 是"重填"按钮，极易误点，绝不可当卖出用。
    因此统一按 1006 定位，并校验文本属于 买入/卖出，再判断可点性。"""
    btn = None
    for c in w.descendants(class_name="Button"):
        try:
            if c.control_id() != 1006:
                continue
            if not c.is_visible():
                continue
            if not _btn_shape_ok(c):
                continue
            if norm_btn(c.window_text()) not in ("买入", "卖出"):
                continue
            btn = c
            break
        except Exception:
            continue
    if btn is None:
        # 兜底：按归一化文本匹配任意可见按钮
        for ctrl in w.descendants(class_name="Button"):
            try:
                if norm_btn(ctrl.window_text()) != norm_btn(text):
                    continue
                if not ctrl.is_visible():
                    continue
                if not _btn_shape_ok(ctrl):
                    continue
                btn = ctrl
                break
            except Exception:
                continue
    if btn is None:
        print("ERROR: 未找到交易按钮: " + text)
        return False
    print("命中按钮: id=%s text=%r" % (btn.control_id(), btn.window_text()))
    phys_click(btn.handle)
    time.sleep(0.4)
    return True


def find_confirm_dialog(pid):
    """找 xiadan PID 下可见的 #32770 确认弹窗。返回句柄或 None。"""
    dlg = [None]

    def cb(hwnd, _):
        if dlg[0] is not None:
            return
        if not win32gui.IsWindowVisible(hwnd):
            return
        try:
            _, p = win32process.GetWindowThreadProcessId(hwnd)
        except Exception:
            return
        if p == pid and win32gui.GetClassName(hwnd) == "#32770":
            dlg[0] = hwnd

    win32gui.EnumWindows(cb, None)
    return dlg[0]


def handle_dialogs(pid, max_rounds=8):
    """物理点击确认弹窗的 是/确定。返回 (msgs, collected)：msgs 为操作记录、
    collected 为每一轮弹窗收集的全部 Static 文本（用于校验委托价格）。
    若某轮是「委托确认」，其文本中含 买入价格/买入数量 等行。"""
    msgs = []
    collected = []
    for _ in range(max_rounds):
        dlg = find_confirm_dialog(pid)
        if dlg is None:
            break
        # 收集弹窗内 static 文本
        texts = []
        def cb(child, _p):
            try:
                if win32gui.GetClassName(child) == "Static":
                    t = win32gui.GetWindowText(child).strip()
                    if t:
                        texts.append(t)
            except Exception:
                pass
        win32gui.EnumChildWindows(dlg, cb, None)
        collected.append(texts)
        msgs.append("弹窗: " + " | ".join(texts[:6]))
        # 找 是/确定 按钮
        btn = [None]
        def cb2(child, _p):
            if btn[0] is not None:
                return
            try:
                if win32gui.GetClassName(child) == "Button" and \
                        norm_btn(win32gui.GetWindowText(child)) in ("是", "确定"):
                    btn[0] = child
            except Exception:
                pass
        win32gui.EnumChildWindows(dlg, cb2, None)
        if btn[0] is not None:
            phys_click(btn[0])
            msgs.append("  → 物理点击 是/确定")
        else:
            msgs.append("  → 未找到确认按钮，按 Esc")
            win32api.keybd_event(win32con.VK_ESCAPE, 0, 0, 0)
            win32api.keybd_event(win32con.VK_ESCAPE, 0, win32con.KEYEVENTF_KEYUP, 0)
        time.sleep(0.8)
    return msgs, collected


def _extract_dlg_price(collected):
    """从委托确认弹窗文本里解析 买入价格。返回字符串或 None。"""
    import re
    for texts in collected:
        for t in texts:
            if ("买入价格" not in t) and ("卖出价格" not in t):
                continue
            # 该 Static 行含 HTML(如 color=0x323232 带数字)，取第一个带小数点的数字，即真实价格
            m = re.search(r"\d+\.\d+(?:\.\d+)*", t)
            if m:
                return m.group(0)
    return None


def ensure_main_visible(pid):
    """还原(若最小化)并前置主窗口, 把窗口拉回可视区给足尺寸。
    关键：主窗口可能被【隐藏】(vis=0, 前置会话 SW_HIDE 所致)，须先 ShowWindow(SW_SHOW)，
    仅 IsIconic 判断不够。用于抗"用户抢鼠标/把窗口拖走/最小化/隐藏"等干扰。"""
    mh = find_main_handle(pid)
    if mh == 0:
        raise SystemExit("ERROR: 未找到主窗口")
    if not win32gui.IsWindowVisible(mh):
        win32gui.ShowWindow(mh, win32con.SW_SHOW)
        time.sleep(0.4)
    if win32gui.IsIconic(mh):
        win32gui.ShowWindow(mh, win32con.SW_RESTORE)
        time.sleep(0.4)
    r = win32gui.GetWindowRect(mh)
    wdt, hgt = r[2] - r[0], r[3] - r[1]
    if wdt < 100 or hgt < 100 or r[0] < -100 or r[1] < -100:
        win32gui.MoveWindow(mh, 60, 60, 1600, 1000, True)
        time.sleep(0.5)
    win32gui.SetForegroundWindow(mh)
    time.sleep(0.3)
    return mh


def wait_dialog_present(pid, timeout=2.5, interval=0.3):
    """轮询直到出现 #32770 弹窗。出现即代表"点中买卖按钮、委托流程被触发"。
    超时返回 None, 调用方据此判定本次点击失败并重试(抗抢鼠标/没点中)。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        dlg = find_confirm_dialog(pid)
        if dlg is not None:
            return dlg
        time.sleep(interval)
    return None


def do_trade(side, security, price, amount, fill_price=True, retries=3):
    """下单, 带自动重试。成功判据: 点买卖按钮后出现委托确认弹窗。
    用户抢鼠标/焦点丢失导致没点中时, 弹窗不出现 → 记录并整轮重试。"""
    code = security[-6:]
    price = str(price)  # 保留原始小数精度
    expect = float(price) if fill_price else None
    board_text = "买入" if side == "buy" else "卖出"
    pid = ths_pid()
    print("xiadan PID:", pid)
    app, w = get_main(pid)
    print("主窗口:", repr(w.window_text()))

    for attempt in range(1, retries + 1):
        print(f"\n--- 尝试 {attempt}/{retries} ({board_text}) ---")
        ensure_main_visible(pid)                     # 还原+前置窗口, 抗抢/被移走

        # 切入买入/卖出页
        w.type_keys(SHORTCUT[side], set_foreground=False)
        time.sleep(1.0)

        # 填表顺序关键: 先填代码回车触发名称/最新价联动, 等联动稳定后再填价格与数量,
        # 避免联动异步刷新把价格覆盖回"最新价"。
        vals = {}
        vals["代码"] = fill_edit(w, 1032, code)      # 内部会回车触发联动
        time.sleep(1.2)                              # 等待联动取价完成
        if fill_price:
            vals["价格"] = fill_edit(w, 1033, price)
            vals["数量"] = fill_edit(w, 1034, str(amount))
        else:
            vals["数量"] = fill_edit(w, 1034, str(amount))
        print("[填写结果]", vals)

        # 点击 买入/卖出 按钮 (统一 id=1006)
        if not click_board_button(w, board_text):
            print("  FAIL: 未点到交易按钮, 本尝试作废")
            time.sleep(0.6)
            continue

        # 等待委托确认框出现(点中按钮的标志)
        dlg = wait_dialog_present(pid)
        if dlg is None:
            print("  未出现委托确认弹窗(可能被抢焦点/没点中), 重试")
            time.sleep(0.6)
            continue

        # 处理连续弹窗(委托确认 + 成交预警等), 并从弹窗校验委托价格
        msgs, collected = handle_dialogs(pid)
        for m in msgs:
            print("  →", m)

        if expect is not None:
            got = _extract_dlg_price(collected)
            if got is not None:
                g = float(got)
                tol = max(0.001, expect * 0.003)      # 0.3% 容差
                if abs(g - expect) <= tol:
                    print(f"价格校验 OK  期望={expect}  实际={got}")
                else:
                    print(f"价格校验 FAIL 期望={expect}  实际={got} (价差>0.3%, 委托价被联动最新价覆盖)")
            else:
                print("价格校验 SKIP  未能从弹窗解析委托价格")
        print("TRADE_DONE")
        return True

    print(f"FAIL: 已重试 {retries} 次仍未触发委托确认")
    return False


if __name__ == "__main__":
    side = sys.argv[1]
    fill_price = True
    if len(sys.argv) > 5:
        fill_price = sys.argv[5].lower() in ("1", "true", "yes")
    do_trade(side, sys.argv[2], sys.argv[3], sys.argv[4], fill_price=fill_price)