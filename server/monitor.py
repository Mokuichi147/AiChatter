#!/usr/bin/env python3
"""シリアルモニター (DTR/RTS無効、ポート出現待ち、自動再接続、リトライあり)"""
import os
import sys
import time
import serial

PORT = "/dev/cu.usbmodem1101"
BAUD = 115200
WAIT_SEC = 120  # ポート出現待ちタイムアウト (秒)

def try_open():
    """ポートへの接続を試みる。失敗したらNoneを返す"""
    if not os.path.exists(PORT):
        return None
    try:
        ser = serial.Serial()
        ser.port = PORT
        ser.baudrate = BAUD
        ser.dtr = False
        ser.rts = False
        ser.timeout = 1
        ser.open()
        return ser
    except Exception:
        return None

def connect_wait():
    """ポートが出現して安定して接続できるまで待機する"""
    deadline = time.time() + WAIT_SEC
    port_seen = os.path.exists(PORT)

    if not port_seen:
        print(f"[monitor] {PORT} 待機中...", flush=True)

    while time.time() < deadline:
        ser = try_open()
        if ser is not None:
            print(f"[monitor] 接続: {PORT} @ {BAUD}bps", flush=True)
            return ser
        time.sleep(0.2)

    print("[monitor] タイムアウト", flush=True)
    return None

def main():
    print("[monitor] 開始 (Ctrl+C で終了)", flush=True)
    try:
        while True:
            ser = connect_wait()
            if ser is None:
                break
            try:
                while True:
                    line = ser.readline()
                    if line:
                        print(line.decode("utf-8", errors="replace"),
                              end="", flush=True)
            except (serial.SerialException, OSError) as e:
                print(f"\n[monitor] 切断 ({e})", flush=True)
                try:
                    ser.close()
                except Exception:
                    pass
                # ポートが消えるまで待つ
                for _ in range(50):
                    if not os.path.exists(PORT):
                        break
                    time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    print("[monitor] 終了", flush=True)

if __name__ == "__main__":
    main()
