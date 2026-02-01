import os, sys, time, subprocess, requests

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_URL  = os.getenv("AUTOSCALER_API", f"http://{API_HOST}:{API_PORT}")

def wait_api(timeout_sec: int = 25) -> bool:
    t0 = time.time()
    health_url = API_URL.rstrip("/") + "/health"
    while time.time() - t0 < timeout_sec:
        try:
            r = requests.get(health_url, timeout=1.0)
            if r.ok and r.json().get("ok"):
                return True
        except Exception:
            pass
        time.sleep(0.3)
    return False

def main():
    env = os.environ.copy()
    env["PYTHONPATH"] = SRC_DIR           # like: PYTHONPATH=./src
    env["AUTOSCALER_API"] = API_URL       # streamlit reads this

    # start API (cwd=ROOT_DIR so results/... paths match your manual run)
    api_cmd = [
        sys.executable, "-m", "uvicorn",
        "optimize.api_scaler:app",
        "--host", API_HOST,
        "--port", str(API_PORT),
    ]
    api_proc = subprocess.Popen(api_cmd, cwd=ROOT_DIR, env=env)

    st_proc = None
    try:
        if not wait_api():
            print("❌ API not ready. Check uvicorn logs above.")
            api_proc.terminate()
            return 1

        st_cmd = [
            sys.executable, "-m", "streamlit",
            "run", os.path.join(SRC_DIR, "app", "app_preview_api.py"),
        ]
        st_proc = subprocess.Popen(st_cmd, cwd=ROOT_DIR, env=env)
        return st_proc.wait()

    finally:
        if st_proc is not None:
            try: st_proc.terminate()
            except Exception: pass

        try:
            api_proc.terminate()
            api_proc.wait(timeout=5)
        except Exception:
            try: api_proc.kill()
            except Exception: pass

if __name__ == "__main__":
    raise SystemExit(main())