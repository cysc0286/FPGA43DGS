"""Linux stdlib-only process-tree timing/RSS/PSS watchdog for board commands."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def tree_memory(pid, proc_root=Path('/proc'), include_pss=False):
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
    pss = 0
    if include_pss:
        for current in seen:
            proc = proc_root/str(current)
            try:
                fields = dict(line.split(':',1) for line in
                    (proc/'smaps_rollup').read_text().splitlines() if ':' in line)
                pss += int(fields['Pss'].split()[0])*1024
            except (OSError, ValueError, KeyError):
                if proc.exists():
                    raise RuntimeError('Cannot account process PSS: '+str(current))
                # A child may exit between status and rollup sampling.
    return dict(rss_bytes=rss, pss_bytes=pss if include_pss else None)


def tree_rss(pid, proc_root=Path('/proc')):
    return tree_memory(pid, proc_root)['rss_bytes']


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
    p.add_argument("--pss-mib", type=float, default=0,
                   help="Optional proportional process-tree budget; --rss-mib 0 avoids double-counting shared pages")
    p.add_argument("--reserve-mib", type=float, default=128)
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ["--"] else a.command
    if a.out.exists() or not command:
        p.error("Use a fresh measurement path and a command")
    start = time.perf_counter()
    peak, peak_pss, samples, minimum, proc = 0., 0., 0, available_mib(), None
    state, error = "launch_failed", None
    try:
        if minimum < a.reserve_mib:
            raise RuntimeError("Insufficient free memory before launch")
        proc = subprocess.Popen(command, start_new_session=True)
        state = "running"
        while proc.poll() is None:
            memory = tree_memory(proc.pid, include_pss=bool(a.pss_mib))
            peak = max(peak, memory['rss_bytes']/1024**2)
            if a.pss_mib:
                peak_pss = max(peak_pss, memory['pss_bytes']/1024**2)
            minimum = min(minimum, available_mib())
            samples += 1
            if a.rss_mib and peak > a.rss_mib:
                state = "rss_limit"
            elif a.pss_mib and peak_pss > a.pss_mib:
                state = "pss_limit"
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
    except Exception as exc:
        state, error = 'measurement_failed', type(exc).__name__+': '+str(exc)
    finally:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=5)
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        result = dict(status=state, error=error, returncode=None if proc is None else proc.returncode,
                      command=command, wall_seconds=time.perf_counter()-start,
                      child_cpu_seconds=usage.ru_utime+usage.ru_stime,
                      peak_process_tree_rss_mib_sampled=peak, samples=samples,
                      peak_process_tree_pss_mib_sampled=peak_pss if a.pss_mib else None,
                      child_maxrss_mib_kernel=usage.ru_maxrss/1024,
                      min_available_mib_sampled=minimum, sample_interval_seconds=.05,
                      limits=dict(rss_mib=a.rss_mib, pss_mib=a.pss_mib, reserve_mib=a.reserve_mib, timeout=a.timeout),
                      memory_scope="RSS sums shared pages; optional PSS apportions them among mappings. "
                                   "MemAvailable protects total host headroom including tmpfs/kernel. "
                                   "Kernel maxrss is max process, not simultaneous total")
        a.out.write_text(json.dumps(result, indent=2))
    if state != "passed":
        raise SystemExit("Board stage failed: " + state)


if __name__ == "__main__":
    main()
