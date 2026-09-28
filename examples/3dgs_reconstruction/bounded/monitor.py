"""Sampled resource watchdog. It is NOT a kernel-enforced memory reservation."""
import os
import signal
import subprocess
import time
from pathlib import Path
import psutil
try:
    from .common import save
except ImportError:
    from common import save


def stop_tree(proc, parent, known):
    try:
        known.update({p.pid: p for p in parent.children(recursive=True)})
    except psutil.Error:
        pass
    if os.name != "nt":
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    for p in list(known.values()) + [parent]:
        try:
            p.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(list(known.values()) + [parent], timeout=2)
    for p in alive:
        try:
            p.kill()
        except psutil.Error:
            pass
    if os.name != "nt":
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    proc.wait(timeout=5)


def run(command, folder, env, rss_mib=0, timeout_s=0, reserve_mib=0, sample_s=.05):
    """Write logs even on failure/cancel. Folder must be new to avoid erasing evidence."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    peak = 0
    min_available = psutil.virtual_memory().available / 2**20
    state, returncode, samples, known = "starting", None, 0, {}
    parent = proc = None
    error = None
    cpu_peak_threads = 0
    try:
        if min_available < reserve_mib:
            raise RuntimeError("Insufficient available RAM before launch")
        with (folder/"output.log").open("w", encoding="utf-8") as log:
            proc = subprocess.Popen([str(c) for c in command], stdout=log, stderr=subprocess.STDOUT,
                                    env=env, start_new_session=(os.name != "nt"))
            parent = psutil.Process(proc.pid)
            state = "running"
            while proc.poll() is None:
                rss, threads = 0, 0
                try:
                    known.update({p.pid: p for p in parent.children(recursive=True)})
                except psutil.Error:
                    pass
                for p in [parent]+list(known.values()):
                    try:
                        rss += p.memory_info().rss
                        threads += p.num_threads()
                    except psutil.Error:
                        pass
                samples += 1
                peak = max(peak, rss/2**20)
                cpu_peak_threads = max(cpu_peak_threads, threads)
                available = psutil.virtual_memory().available/2**20
                min_available = min(min_available, available)
                if rss_mib and rss/2**20 > rss_mib:
                    state = "rss_limit"
                elif timeout_s and time.perf_counter()-start > timeout_s:
                    state = "timeout"
                elif reserve_mib and available < reserve_mib:
                    state = "host_reserve_limit"
                if state != "running":
                    stop_tree(proc, parent, known)
                    break
                time.sleep(sample_s)
            returncode = proc.wait()
            if state == "running":
                state = "passed" if returncode == 0 else "process_failed"
            # Stages do not own persistent services. Clean up observed descendants
            # even if the parent exited before its workers did.
            if any(p.is_running() for p in known.values()):
                stop_tree(proc, parent, known)
    except BaseException as exc:
        state = "interrupted" if isinstance(exc, KeyboardInterrupt) else "launch_failed"
        error = str(exc) or type(exc).__name__
        if proc is not None and parent is not None:
            stop_tree(proc, parent, known)
            returncode = proc.returncode
        raise
    finally:
        result = {"command": [str(c) for c in command], "status": state, "returncode": returncode,
                  "wall_s": time.perf_counter()-start, "peak_process_tree_rss_mib_sampled": peak,
                  "min_host_available_mib_sampled": min_available, "peak_tree_threads_sampled": cpu_peak_threads,
                  "sample_interval_s": sample_s, "samples": samples, "error": error,
                  "limits": {"rss_mib": rss_mib, "timeout_s": timeout_s, "reserve_mib": reserve_mib},
                  "memory_boundary": "sampled process-tree RSS; shared pages may be counted twice; spikes can exceed limits before termination"}
        save(folder/"measurement.json", result)
    if state != "passed":
        raise RuntimeError("Stage stopped: %s; evidence: %s" % (state, folder))
    return result
