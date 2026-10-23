"""通过 GitHub 设备授权（Device Flow）把本地提交推上去，全程不落盘明文 token。

为什么需要它
------------
这台机器上没有持久化的 GitHub 凭据，Git Credential Manager 在非交互环境下
既不会弹窗也不会自动授权。设备授权流程是最省事的方式：

    1. 脚本申请一个 user_code（形如 05D2-3CEA）
    2. 你在浏览器打开 https://github.com/login/device 输入这个码并点同意
    3. 脚本轮询拿到 access_token，用它推送一次，然后丢弃

token 只在内存和这一次 git 子进程的命令行里出现，不写入任何配置文件。

前置：网络需要走系统代理（脚本自动读取 HTTPS_PROXY），
本项目环境里 git 的 schannel 吊销检查会失败，因此推送时临时用 `sslVerify=false`。

用法：
    python scripts/github_device_push.py                # 申请码 + 等授权 + 推送
    python scripts/github_device_push.py --resume       # 复用上次申请的 device_code 继续等
"""

import argparse
import json
import os
import ssl
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEVICE_FILE = PROJECT_ROOT / ".cache" / "device_code.json"

#: GitHub CLI 的公开 OAuth client_id（开源工具通用，非本项目私有）
CLIENT_ID = "178c6fc778ccc68e1d6a"
SCOPE = "repo"
REMOTE = "https://github.com/mztkj02/lab-safety-agent-v0.1.git"


def _opener():
    """带代理、且跳过证书链校验的 opener。

    跳过校验是因为本机的 HTTPS 走本地透明代理，代理自签证书不在 git/openssl
    的信任链里，schannel 的吊销检查也会失败。传输本身仍在代理隧道内。
    """
    proxy = (os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
             or os.environ.get("HTTP_PROXY") or "")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    handlers = [urllib.request.HTTPSHandler(context=ctx)]
    if proxy:
        handlers.insert(0, urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    return urllib.request.build_opener(*handlers)


def request_device_code() -> dict:
    data = urllib.parse.urlencode({"client_id": CLIENT_ID, "scope": SCOPE}).encode()
    req = urllib.request.Request(
        "https://github.com/login/device/code", data=data,
        headers={"Accept": "application/json", "User-Agent": "lab-safety-agent/1.0"},
    )
    payload = json.loads(_opener().open(req, timeout=25).read().decode())
    DEVICE_FILE.parent.mkdir(parents=True, exist_ok=True)
    DEVICE_FILE.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def poll_token(device: dict) -> str:
    interval = max(5, int(device.get("interval", 5)))
    deadline = time.time() + int(device.get("expires_in", 900))
    print("等待你在浏览器里完成授权……（每 5 秒检查一次）", flush=True)

    while time.time() < deadline:
        time.sleep(interval)
        data = urllib.parse.urlencode({
            "client_id": CLIENT_ID,
            "device_code": device["device_code"],
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
        }).encode()
        req = urllib.request.Request(
            "https://github.com/login/oauth/access_token", data=data,
            headers={"Accept": "application/json", "User-Agent": "lab-safety-agent/1.0"},
        )
        try:
            resp = json.loads(_opener().open(req, timeout=25).read().decode())
        except Exception as exc:  # noqa: BLE001
            print(f"  轮询出错（继续等）：{type(exc).__name__}", flush=True)
            continue

        if resp.get("access_token"):
            return resp["access_token"]
        err = resp.get("error")
        if err == "authorization_pending":
            continue
        if err == "slow_down":
            interval += 3
            continue
        print(f"  授权失败：{err} {resp.get('error_description', '')}", flush=True)
        raise SystemExit(1)

    print("设备码已过期，请重新运行本脚本申请新的码。")
    raise SystemExit(1)


def push_with_token(token: str) -> bool:
    auth_url = f"https://x-access-token:{token}@{REMOTE.split('https://')[1]}"
    cmd = [
        "git",
        "-c", f"http.proxy={os.environ.get('HTTPS_PROXY', '')}",
        "-c", "http.sslVerify=false",
        "push", auth_url, "main",
    ]
    print("开始推送……", flush=True)
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    # 输出里可能回显带 token 的 URL，替换掉再打印
    safe = out.replace(token, "***")
    print(safe.strip()[-1500:] or "(无输出)")
    return proc.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="GitHub 设备授权后推送")
    parser.add_argument("--resume", action="store_true", help="复用上次的 device_code")
    args = parser.parse_args()

    if args.resume and DEVICE_FILE.exists():
        device = json.loads(DEVICE_FILE.read_text(encoding="utf-8"))
        print(f"复用设备码：{device.get('user_code')}")
    else:
        device = request_device_code()
        print("=" * 52)
        print(f"  浏览器打开：{device.get('verification_uri')}")
        print(f"  输入代码：  {device.get('user_code')}")
        print(f"  有效期：    {device.get('expires_in')} 秒")
        print("=" * 52)

    token = poll_token(device)
    print("授权成功，已获得临时 token（不会写入任何文件）")
    ok = push_with_token(token)
    if ok:
        print("推送完成。token 已丢弃。")
        try:
            DEVICE_FILE.unlink()
        except OSError:
            pass
    else:
        print("推送失败，可重试：python scripts/github_device_push.py --resume")
        sys.exit(1)


if __name__ == "__main__":
    main()
