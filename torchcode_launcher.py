#!/usr/bin/env python3
import os
import sys
import time
import signal
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"

children = []
stopping = False

def log(msg):
    print(f"[TorchCode] {msg}", flush=True)

def find_python():
    candidates = []
    if os.name == "nt":
        candidates += [
            ROOT / ".venv" / "Scripts" / "python.exe",
            ROOT / "venv" / "Scripts" / "python.exe",
        ]
    else:
        candidates += [
            ROOT / ".venv" / "bin" / "python",
            ROOT / "venv" / "bin" / "python",
        ]
    for p in candidates:
        if p.exists():
            return str(p)
    return sys.executable

def find_npm():
    npm = shutil.which("npm")
    if not npm:
        raise RuntimeError("未找到 npm。请先安装 Node.js（建议 Node.js 20+）。")
    return npm

def check_project():
    required = [
        ROOT / "api" / "main.py",
        ROOT / "api" / "requirements.txt",
        WEB / "package.json",
        ROOT / "torch_judge",
    ]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if missing:
        raise RuntimeError(
            "脚本必须放在 TorchCode 项目根目录。\n缺少: " + ", ".join(missing)
        )

def terminate_process_tree(proc):
    if proc is None or proc.poll() is not None:
        return

    try:
        if os.name == "nt":
            # /T kills child processes (e.g. npm -> next, uvicorn reload worker)
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                return

            deadline = time.time() + 3
            while time.time() < deadline:
                if proc.poll() is not None:
                    return
                time.sleep(0.1)

            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
    except Exception:
        pass

def cleanup(*_):
    global stopping
    if stopping:
        return
    stopping = True
    print()
    log("正在关闭前端和后端……")
    for proc in reversed(children):
        terminate_process_tree(proc)
    log("已关闭。")
    raise SystemExit(0)

def spawn(cmd, cwd):
    kwargs = {
        "cwd": str(cwd),
        "stdin": None,
    }

    if os.name == "nt":
        # CREATE_NEW_PROCESS_GROUP makes signal/process-tree handling cleaner.
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True

    return subprocess.Popen(cmd, **kwargs)

def main():
    check_project()

    python = find_python()
    npm = find_npm()

    signal.signal(signal.SIGINT, cleanup)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, cleanup)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, cleanup)

    log(f"项目目录: {ROOT}")
    log(f"Python: {python}")

    # Helpful dependency checks with clear messages.
    try:
        subprocess.run(
            [python, "-c", "import fastapi, uvicorn, torch, pydantic"],
            cwd=str(ROOT),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        raise RuntimeError(
            "Python 依赖未安装完整。\n"
            "请先在项目根目录执行：\n"
            "  pip install -r api/requirements.txt\n"
            "  pip install -e ."
        )

    if not (WEB / "node_modules").exists():
        log("首次运行：正在执行 npm install……")
        subprocess.run([npm, "install"], cwd=str(WEB), check=True)

    log("启动 FastAPI 后端：http://localhost:8000")
    backend = spawn(
        [python, "-m", "uvicorn", "api.main:app", "--port", "8000", "--reload"],
        ROOT,
    )
    children.append(backend)

    time.sleep(1.0)
    if backend.poll() is not None:
        raise RuntimeError("后端启动失败，请查看上方错误信息。")

    log("启动 Next.js 前端：http://localhost:3000")
    frontend = spawn([npm, "run", "dev"], WEB)
    children.append(frontend)

    time.sleep(1.0)
    if frontend.poll() is not None:
        raise RuntimeError("前端启动失败，请查看上方错误信息。")

    print()
    log("TorchCode 已启动。")
    log("浏览器打开：http://localhost:3000")
    log("关闭这个窗口，或按 Ctrl+C，即可同时关闭前后端。")
    print()

    # Keep the launcher alive; if either service exits, close the other.
    while True:
        if backend.poll() is not None:
            log("后端已退出，正在关闭前端。")
            cleanup()
        if frontend.poll() is not None:
            log("前端已退出，正在关闭后端。")
            cleanup()
        time.sleep(0.5)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cleanup()
    except Exception as e:
        print()
        print(f"[TorchCode] 启动失败：{e}", flush=True)
        for proc in reversed(children):
            terminate_process_tree(proc)

        # Keep error visible for double-click launches.
        try:
            input("\n按 Enter 关闭窗口……")
        except EOFError:
            pass
        raise SystemExit(1)
