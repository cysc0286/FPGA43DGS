#!/usr/bin/env python3
"""串口控制台：自动找 CP210x，读启动日志，或发命令收输出。

用途：板子上电后第一时间拿到 PS 端启动日志，判断系统/登录/报错。

示例
----
    python serial_console.py --list
    python serial_console.py --log 20 --save boot.log
    python serial_console.py --send "uname -a" --wait 3
    python serial_console.py --send "ls /" --send "free -m" --wait 2

设计说明
--------
- 只依赖 pyserial，不引入项目内任何模块（PC 侧独立可用）。
- 默认 115200 8N1，与本板 CP2103 的 PS UART 一致。
- 打开串口后先不发任何字节，只监听，避免打扰正在启动的 bootloader。
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import serial
    from serial.tools import list_ports
except ImportError:  # pragma: no cover
    sys.exit("缺少 pyserial：python -m pip install pyserial")


SILABS_VID = 0x10C4
CP210X_PIDS = {0xEA60, 0xEA61, 0xEA63, 0xEA70, 0xEA71, 0xEA7A, 0xEA7B}


def ports():
    """列出所有串口，标注哪些是 CP210x。"""
    found = []
    for p in list_ports.comports():
        is_cp = (p.vid == SILABS_VID and p.pid in CP210X_PIDS) or "CP210" in (p.description or "")
        found.append((p.device, is_cp, p.description or "", p.hwid or ""))
    return found


def cmd_list() -> int:
    items = ports()
    if not items:
        print("没有找到任何串口。检查 USB 线 / 板子是否上电。")
        return 1
    print(f"{'端口':<8} {'CP210x':<7} 描述")
    print("-" * 78)
    for dev, is_cp, desc, hwid in items:
        print(f"{dev:<8} {'  *   ' if is_cp else '      '} {desc}")
        if is_cp:
            print(f"{'':<17} {hwid}")
    cps = [d for d, cp, _, _ in items if cp]
    print()
    if cps:
        print(f"可用板卡串口: {', '.join(cps)}")
        return 0
    print("没有 CP210x 设备。若设备管理器里是带感叹号的未知设备，说明驱动没装好。")
    return 2


def pick_port(explicit: str | None) -> str:
    if explicit:
        return explicit
    cps = [d for d, cp, _, _ in ports() if cp]
    if not cps:
        sys.exit("没有找到 CP210x 串口，请先装驱动或用 --port 指定。")
    if len(cps) > 1:
        print(f"发现多个 CP210x 串口 {cps}，默认用第一个；可用 --port 指定。", file=sys.stderr)
    return cps[0]


def wait_for_port(port: str | None, timeout: float) -> str:
    """等板子上电后 USB 重新枚举。CP2103 掉电时整个设备会从系统消失，
    此时直接 open 会报找不到端口，所以要轮询。

    port 为 None 时轮询"任意 CP210x 出现"，返回最终使用的端口名。
    """
    deadline = time.time() + timeout
    announced = False
    while time.time() < deadline:
        target = port
        if target is None:
            cps = [d for d, cp, _, _ in ports() if cp]
            target = cps[0] if cps else None
        if target:
            try:
                with serial.Serial(target, 115200, timeout=0.1):
                    print(f"[serial] {target} 已上线", file=sys.stderr)
                    return target
            except (serial.SerialException, OSError):
                pass
        if not announced:
            what = port or "CP210x 串口"
            print(f"[serial] 等待 {what} 上线（请给板子上电，最多 {timeout:.0f}s）...",
                  file=sys.stderr)
            announced = True
        time.sleep(0.25)
    sys.exit(f"{timeout:.0f}s 内没有等到串口。检查板子是否上电（SW8 拨到 ON）。")


def drain(ser: serial.Serial, deadline: float, out: list[str], echo: bool) -> None:
    """读到 deadline 为止，边读边打印。"""
    buf = b""
    while time.time() < deadline:
        chunk = ser.read(ser.in_waiting or 1)
        if not chunk:
            continue
        buf += chunk
        # 按行切，保留不完整的尾巴
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            text = line.decode("utf-8", "replace").rstrip("\r")
            out.append(text)
            if echo:
                print(text, flush=True)
    if buf:
        text = buf.decode("utf-8", "replace").rstrip("\r")
        out.append(text)
        if echo:
            print(text, flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CP210x 串口控制台（读日志 / 发命令）")
    ap.add_argument("--list", action="store_true", help="只列出串口，不打开")
    ap.add_argument("--port", help="串口名，如 COM7；默认自动找 CP210x")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--log", type=float, default=0.0, help="打开后纯监听多少秒")
    ap.add_argument("--send", action="append", default=[], help="要发送的命令，可重复")
    ap.add_argument("--wait", type=float, default=3.0, help="每条命令后等待多少秒")
    ap.add_argument("--save", type=Path, help="把收到的文本存到文件")
    ap.add_argument("--quiet", action="store_true", help="不回显到屏幕，只写文件")
    ap.add_argument("--wait-port", type=float, default=0.0,
                    help="先等端口出现多少秒（板子未上电时用）")
    ap.add_argument("--script", type=Path,
                    help="按行发送文件里的命令（每行一条，用于批量上传）")
    args = ap.parse_args(argv)

    if args.list:
        return cmd_list()

    port = pick_port(args.port) if (args.port or args.wait_port <= 0) else None
    if args.wait_port > 0:
        port = wait_for_port(port, args.wait_port)
    if port is None:
        sys.exit("没有可用串口。")
    print(f"[serial] {port} @ {args.baud} 8N1", file=sys.stderr)

    lines: list[str] = []
    started = datetime.now()
    with serial.Serial(port, args.baud, bytesize=8, parity="N", stopbits=1,
                       timeout=0.2, write_timeout=2) as ser:
        ser.reset_input_buffer()
        # 板端 PS UART 未接硬件流控，显式关掉，避免意外挂起
        ser.dtr = False
        ser.rts = False

        if args.log > 0:
            drain(ser, time.time() + args.log, lines, not args.quiet)

        for command in args.send:
            ser.write((command + "\r\n").encode())
            ser.flush()
            drain(ser, time.time() + args.wait, lines, not args.quiet)

        if args.script:
            script_lines = [ln.rstrip("\r\n") for ln in
                            args.script.read_text(encoding="utf-8").splitlines()]
            script_lines = [ln for ln in script_lines if ln.strip()]
            for index, command in enumerate(script_lines, 1):
                ser.write((command + "\r\n").encode())
                ser.flush()
                # 每行只等很短时间；最后一行多等一会，让远端把文件落盘
                pause = args.wait if index == len(script_lines) else 0.15
                drain(ser, time.time() + pause, lines, not args.quiet)
                if index % 10 == 0:
                    print(f"[serial] 已发送 {index}/{len(script_lines)} 行", file=sys.stderr)

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        header = f"# {port} {args.baud}8N1  {started:%Y-%m-%d %H:%M:%S}\n"
        args.save.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
        print(f"[serial] 已保存 {len(lines)} 行 -> {args.save}", file=sys.stderr)

    if not lines:
        print("[serial] 一个字节都没收到。三种可能：板子没上电 / 波特率不对 / TX 没接上。",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
