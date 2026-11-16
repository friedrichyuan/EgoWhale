"""Live stage reporting shared by experiment workers and their subprocesses."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import threading

_LOCK = threading.Lock()


def report_progress(stage, current=None, total=None, message=""):
    payload = {"stage": stage, "current": current, "total": total, "message": message,
               "updated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}
    label = os.environ.get("EGOHAND_VIDEO_LABEL", "")
    fraction = f" {current}/{total}" if total else ""
    with _LOCK:
        print(f"[进度] {label} | {stage}{fraction} {message}".strip(), flush=True)
        destination = os.environ.get("EGOHAND_PROGRESS_FILE")
        if destination:
            path = Path(destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
            temporary.replace(path)


def run_streamed_command(args, *, cwd=None, stdout_path, stderr_path, timeout, stage_markers=(), progress_only=False):
    """Tee both pipes immediately, preserving separate logs and a bounded error tail."""
    stdout_path, stderr_path = Path(stdout_path), Path(stderr_path)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    tails = {"stdout": [], "stderr": []}
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with stdout_path.open("w") as stdout_file, stderr_path.open("w") as stderr_file:
        process = subprocess.Popen(args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, encoding="utf-8", errors="replace", bufsize=1, env=env)

        def drain(pipe, log, name):
            try:
                for line in pipe:
                    log.write(line)
                    log.flush()
                    # Optimizer stdout contains large debug arrays. Retain it
                    # in its log while showing progress and all stderr errors.
                    if not progress_only or name == "stderr" or line.startswith("[进度]"):
                        print(line, end="", flush=True)
                    for marker, stage in stage_markers:
                        if marker in line.lower():
                            report_progress(stage)
                    tails[name].append(line)
                    if len(tails[name]) > 80:
                        del tails[name][0]
            finally:
                pipe.close()

        threads = [threading.Thread(target=drain, args=(process.stdout, stdout_file, "stdout"), daemon=True),
                   threading.Thread(target=drain, args=(process.stderr, stderr_file, "stderr"), daemon=True)]
        for thread in threads:
            thread.start()
        try:
            returncode = process.wait(timeout=timeout)
        except BaseException:
            if process.poll() is None:
                process.kill()
                process.wait()
            raise
        finally:
            for thread in threads:
                thread.join()
    return subprocess.CompletedProcess(args, returncode, "".join(tails["stdout"]), "".join(tails["stderr"]))
