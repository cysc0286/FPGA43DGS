"""Linux stdlib-only process-tree timing/RSS watchdog for board commands."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def tree_rss(pid, proc_root=Path('/proc')):
    # This BSP omits /proc/PID/task/TID/children (CONFIG_PROC_CHILDREN).
    # PPid in status also covers children created by secondary threads.
    parents, resident = {}, {}
    for proc in proc_root.iterdir():
        if not proc.name.isdecimal():
            continue
        try:
            fields = dict(line.split(':', 1) for line in
                          (proc/'status').read_text().splitlines() if ':' in line)
            parent = int(fields['PPid'].strip())
            current = int(proc.name)
            resident[current] = int(fields.get('VmRSS', '0 kB').split()[0])*1024
            parents.setdefault(parent, []).append(current)
        except (OSError, ValueError, KeyError):
            continue  # A process can exit during a /proc sample.
    pending, seen, rss = [pid], set(), 0
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        rss += resident.get(current, 0)
        pending.extend(parents.get(current, ()))
    return rss


def available_mib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])/1024
    raise RuntimeError("Cannot read available system memory")


def main():
    import resource  # Linux-only CLI; pure /proc parsing can be tested elsewhere.
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rss-mib", type=float, default=650)
    p.add_argument("--reserve-mib", type=float, default=128)
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ["--"] else a.command
    if a.out.exists() or not command:
        p.error("Use a fresh measurement path and a command")
    start = time.perf_counter()
    peak, samples, minimum, proc = 0., 0, available_mib(), None
    state = "launch_failed"
    try:
        if minimum < a.reserve_mib:
            raise RuntimeError("Insufficient free memory before launch")
        proc = subprocess.Popen(command, start_new_session=True)
        state = "running"
        while proc.poll() is None:
            peak = max(peak, tree_rss(proc.pid)/1024**2)
            minimum = min(minimum, available_mib())
            samples += 1
            if a.rss_mib and peak > a.rss_mib:
                state = "rss_limit"
            elif minimum < a.reserve_mib:
                state = "host_reserve_limit"
            elif time.perf_counter()-start > a.timeout:
                state = "timeout"
            if state != "running":
                break
            time.sleep(.05)
        if state == "running":
            state = "passed" if proc.returncode == 0 else "process_failed"
    except KeyboardInterrupt:
        state = 'interrupted'
    finally:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=5)
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        result = dict(status=state, returncode=None if proc is None else proc.returncode,
                      command=command, wall_seconds=time.perf_counter()-start,
                      child_cpu_seconds=usage.ru_utime+usage.ru_stime,
                      peak_process_tree_rss_mib_sampled=peak, samples=samples,
                      child_maxrss_mib_kernel=usage.ru_maxrss/1024,
                      min_available_mib_sampled=minimum, sample_interval_seconds=.05,
                      limits=dict(rss_mib=a.rss_mib, reserve_mib=a.reserve_mib, timeout=a.timeout),
                      memory_scope="Sampled tree RSS sums shared pages; kernel maxrss is max process, not simultaneous total")
        a.out.write_text(json.dumps(result, indent=2))
    if state != "passed":
        raise SystemExit("Board stage failed: " + state)


if __name__ == "__main__":
    main()
