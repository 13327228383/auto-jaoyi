# -*- coding: utf-8 -*-
"""keep_awake.py —— 防止系统进入睡眠(休眠)/关闭显示器。

用 SetThreadExecutionState(ES_CONTINUOUS|ES_SYSTEM_REQUIRED|ES_DISPLAY_REQUIRED)
周期刷新, 让机器在交易期间保持唤醒。由 start_auto.bat 后台启动,
auto_run.py 退出后由 bat 结束。无需管理员权限, 不影响用户后续改回电源设置。

用法:
  python keep_awake.py        # (通常由 bat 以 /min 后台启动)
"""
import ctypes
import sys
import time

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x1
ES_DISPLAY_REQUIRED = 0x2


def main():
    flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
    try:
        kernel32 = ctypes.windll.kernel32
    except Exception as ex:
        print("keep_awake: 无法加载 kernel32:", ex, file=sys.stderr)
        sys.exit(1)

    try:
        res = kernel32.SetThreadExecutionState(flags)
        if res == 0:
            print("keep_awake: SetThreadExecutionState 被拒绝, 系统可能仍会休眠", file=sys.stderr)
    except Exception as ex:
        print("keep_awake: 初始化失败:", ex, file=sys.stderr)

    print("keep_awake: 已锁定 不休眠+不熄屏 (20s 周期刷新), Ctrl+C 停止", flush=True)
    while True:
        time.sleep(20)
        kernel32.SetThreadExecutionState(flags)  # 定期刷新, 保持状态持续


if __name__ == "__main__":
    main()