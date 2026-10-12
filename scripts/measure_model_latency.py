"""Measure generation latency and the loaded model's GPU residency.

Answers one question that a specification cannot: does the configured chat model
fit in this machine's VRAM, or is part of it running on the CPU? Ollama reports
both the model's total size and how much of it is resident on the GPU. Residency
is useful diagnostic evidence; time to first visible content is measured separately.

Run it against an inference API that is already up. No settings are written
and no models are pulled. Fixed public prompts measure LLM generation only;
they do not exercise retrieval or prove response quality.

    .\\.venv\\Scripts\\python.exe scripts\\measure_model_latency.py

Compare configurations by restarting the server with different environment
variables, never by editing .env, so the machine is left as it was found:

    $env:OLLAMA_KEEP_ALIVE='24h'

Standard library only, so it runs on a fresh clone.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import time
import urllib.error
import urllib.request

QUESTIONS = [
    "Suối Giàng có gì đặc biệt?",
    "Trạm Tấu có suối khoáng nóng không?",
    "Mù Cang Chải ruộng bậc thang mùa nào đẹp?",
    "Hồ Thác Bà đi mấy ngày là đủ?",
]


def get_json(url: str, timeout: float = 30.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def gpu_report() -> str:
    """Name and memory of the GPU, straight from the driver."""
    if platform.system() != "Windows" and platform.system() != "Linux":
        return "khong doc duoc"
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "khong co nvidia-smi"
    line = out.stdout.strip().splitlines()
    return line[0].strip() if line else "khong co GPU NVIDIA"


def residency(ollama_url: str) -> list[dict]:
    """Which models Ollama holds, and how much of each is on the GPU."""
    try:
        models = get_json(f"{ollama_url}/api/ps").get("models", [])
    except (urllib.error.URLError, TimeoutError, OSError):
        return []
    rows = []
    for model in models:
        total = float(model.get("size") or 0)
        on_gpu = float(model.get("size_vram") or 0)
        rows.append({
            "name": model.get("name", "?"),
            "total_gb": total / 1e9,
            "gpu_gb": on_gpu / 1e9,
            "gpu_percent": (on_gpu / total * 100.0) if total else 0.0,
        })
    return rows


def print_residency(ollama_url: str, label: str) -> None:
    rows = residency(ollama_url)
    if not rows:
        print(f"  [{label}] Ollama khong giu model nao")
        return
    for row in rows:
        verdict = "toan bo tren GPU" if row["gpu_percent"] >= 99.5 else "CO PHAN CHAY BANG CPU"
        print(
            f"  [{label}] {row['name']:<26} {row['total_gb']:5.2f} GB, "
            f"tren GPU {row['gpu_gb']:5.2f} GB ({row['gpu_percent']:5.1f}%) — {verdict}"
        )


def ask(app_url: str, question: str, timeout: float) -> tuple[float, float, int]:
    """Measure visible tokens only when the SSE stream completes successfully."""
    request = urllib.request.Request(
        f"{app_url}/api/chat/stream",
        data=json.dumps({"message": question}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first: float | None = None
    pieces = 0
    event = ""
    completed = False
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            try:
                line = raw.decode("utf-8").strip()
            except UnicodeError:
                raise RuntimeError("Inference returned an invalid stream.") from None
            if line.startswith("event:"):
                event = line[6:].strip()
                continue
            if not line.startswith("data:"):
                continue
            if event == "error":
                # Provider payload may contain sensitive internals; never echo it.
                raise RuntimeError("Inference stream failed.")
            try:
                payload = json.loads(line[5:])
            except json.JSONDecodeError:
                raise RuntimeError("Inference returned an invalid stream.") from None
            if not isinstance(payload, dict):
                raise RuntimeError("Inference returned an invalid stream.")
            if event == "done":
                completed = True
                break
            if event != "token" or not isinstance(payload.get("text"), str):
                raise RuntimeError("Inference returned an invalid stream.")
            if payload["text"]:
                pieces += 1
                if first is None:
                    first = time.perf_counter() - started
    if not completed:
        raise RuntimeError("Inference stream ended before completion.")
    if first is None:
        raise RuntimeError("Inference completed without visible tokens.")
    total = time.perf_counter() - started
    return first, total, pieces


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-url", default="http://127.0.0.1:8000")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--label", default="", help="Ten cau hinh dang do, in ra cho de doi chieu")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--warmup", action="store_true",
                        help="Hoi mot cau bo di truoc khi do, de bo qua lan nap dau tien")
    args = parser.parse_args()

    app_url = args.app_url.rstrip("/")
    ollama_url = args.ollama_url.rstrip("/")

    try:
        live = get_json(f"{app_url}/api/health/live", timeout=10)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"Khong ket noi duoc toi {app_url}: {exc}")
        print("Chay truoc: python -B -m uvicorn app.main:app --host 127.0.0.1 --port 8000")
        return 1
    if live.get("status") != "ok":
        print(f"API khong bao trang thai tien trinh ok tai {app_url}: {live}")
        return 1

    try:
        info = get_json(f"{app_url}/api/info", timeout=10)
    except (urllib.error.URLError, TimeoutError, OSError):
        info = {}

    print("=" * 74)
    if args.label:
        print(f"CAU HINH: {args.label}")
    print(f"GPU     : {gpu_report()}")
    print(f"May     : {platform.node()}")
    print("API     : tien trinh ok; khong khang dinh model da san sang")
    if info:
        print(f"Provider: {info.get('provider', '?')} | model: {info.get('model', '?')}")
        print(f"History : {info.get('history', '?')} | retrieval noi bo: {info.get('rag_enabled', '?')}")
    else:
        print("Metadata: khong doc duoc /api/info")
    print("=" * 74)
    print_residency(ollama_url, "truoc khi do")

    firsts: list[float] = []
    totals: list[float] = []
    try:
        if args.warmup:
            print("  (hoi mot cau bo di de bo qua lan nap dau tien)")
            ask(app_url, QUESTIONS[0], args.timeout)
        print()
        for question in QUESTIONS:
            first, total, pieces = ask(app_url, question, args.timeout)
            firsts.append(first)
            totals.append(total)
            print(f"  chu dau {first:6.2f}s | tong {total:7.2f}s | {pieces:4d} manh | {question}")
    except (RuntimeError, urllib.error.URLError, TimeoutError, OSError):
        print("Khong hoan tat chat; khong tong hop latency nhu mot lan sinh thanh cong.")
        return 1

    print()
    print(f"  CHU DAU TIEN : trung binh {statistics.mean(firsts):6.2f}s  "
          f"(thap nhat {min(firsts):.2f}s, cao nhat {max(firsts):.2f}s)")
    print(f"  TONG THOI GIAN: trung binh {statistics.mean(totals):6.2f}s  "
          f"(thap nhat {min(totals):.2f}s, cao nhat {max(totals):.2f}s)")
    print()
    print_residency(ollama_url, "sau khi do ")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
