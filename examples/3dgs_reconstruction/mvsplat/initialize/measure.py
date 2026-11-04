"""Linux stdlib-only process-tree timing/RSS/PSS watchdog for board commands."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time


def _read_error(source, error):
    return dict(source=source, error_type=type(error).__name__,
                errno=getattr(error, 'errno', None), message=str(error)[:256])


class PssAccountingError(RuntimeError):
    def __init__(self, current, attempts):
        self.details = dict(pid=current, attempts=attempts)
        last = '; '.join(item['source']+'='+item['error_type']+
                         '(errno='+str(item['errno'])+'): '+item['message']
                         for item in attempts[-1]['reads'])
        super().__init__('Cannot account process PSS: '+str(current)+'; '+last)


def _process_has_exited(proc, current, diagnostics=None):
    # Only a fresh exit-state sample or a reaped PID proves its mm is gone.
    # Neither missing smaps nor an unreadable /proc file proves zero memory.
    try:
        status = dict(line.split(':', 1) for line in
                      (proc/'status').read_text().splitlines() if ':' in line)
        state = status.get('State', '').strip().split()[0:1]
        if diagnostics is not None:
            diagnostics.append(dict(source='status', state=state))
        if state in (['Z'], ['X']):
            return True
    except (OSError, ValueError) as error:
        if diagnostics is not None:
            diagnostics.append(_read_error('status', error))
    try:
        value = (proc/'stat').read_text()
        _, separator, tail = value.rpartition(')')
        state = tail.strip().split()[0:1] if separator else []
        valid = value.startswith(str(current)+' (') and bool(separator)
        if diagnostics is not None:
            diagnostics.append(dict(source='stat', valid=valid, state=state))
        if valid and state in (['Z'], ['X']):
            return True
    except (OSError, ValueError) as error:
        if diagnostics is not None:
            diagnostics.append(_read_error('stat', error))
    try:
        proc.stat()
    except FileNotFoundError as error:
        if diagnostics is not None:
            diagnostics.append(_read_error('proc_directory', error))
        return True
    except OSError as error:
        if diagnostics is not None:
            diagnostics.append(_read_error('proc_directory', error))
    else:
        if diagnostics is not None:
            diagnostics.append(dict(source='proc_directory', exists=True))
    return False


def _pss_bytes(line):
    fields = line.split()
    if (len(fields) != 3 or fields[0] != 'Pss:' or fields[2] != 'kB'
            or not fields[1].isdecimal()):
        raise ValueError('Invalid nonnegative Pss kB field: '+line[:96])
    return int(fields[1])*1024


def _rollup_pss(text):
    values = [line.strip() for line in text.splitlines() if line.strip().startswith('Pss:')]
    if len(values) != 1:
        raise ValueError('smaps_rollup needs exactly one Pss field')
    return _pss_bytes(values[0])


_VMA_HEADER = re.compile(r'^([0-9a-fA-F]+)-([0-9a-fA-F]+)\s+[rwxps-]{4}\s+'
                         r'[0-9a-fA-F]+\s+[0-9a-fA-F]+:[0-9a-fA-F]+\s+[0-9]+(?:\s+.*)?$')
_VMA_PREFIX = re.compile(r'^[0-9a-fA-F]+-[0-9a-fA-F]+\s')
_VMA_FIRST = frozenset('0123456789abcdefABCDEF')


def _smaps_pss(text):
    # Sum the actual Pss of *every* VMA. An empty/truncated/partly parsed live
    # address space is unknown, not zero. Pss_Dirty/Pss_Anon must not be added.
    mappings, value, total = 0, None, 0
    for line in text.splitlines():
        # A genuine or malformed VMA prefix must begin with an ASCII hex
        # digit and contain '-'. Most smaps lines are named counters/flags;
        # do not run two regular expressions on each of those at 20 Hz.
        # The full range/permissions/offset/device/inode check stays intact.
        if line and line[0] in _VMA_FIRST and '-' in line:
            header = _VMA_HEADER.fullmatch(line)
            if header:
                if mappings and value is None:
                    raise ValueError('Full smaps mapping is missing Pss')
                if int(header[2], 16) <= int(header[1], 16):
                    raise ValueError('Invalid full smaps mapping range')
                if value is not None:
                    total += value
                mappings, value = mappings+1, None
                continue
            if _VMA_PREFIX.match(line):
                raise ValueError('Malformed full smaps VMA header')
        # Normal kernel output is unindented. Preserve the old acceptance of
        # surrounding whitespace, including non-ASCII whitespace, for Pss.
        if line.startswith('Pss:') or (line and line[0].isspace() and line.lstrip().startswith('Pss:')):
            if not mappings or value is not None:
                raise ValueError('Full smaps Pss is unscoped or duplicated')
            value = _pss_bytes(line.strip())
    if not mappings or value is None:
        raise ValueError('Full smaps is empty or final mapping is missing Pss')
    return total+value


def _read_pss(proc, current):
    # exec can temporarily replace an mm while /proc/PID remains live. Retry
    # briefly, but every accepted sample still needs actual PSS or exit proof.
    attempts = []
    # A compiler exec was observed to outlast three immediate /proc reads.
    # Bound recovery to ten samples / nine 20 ms sleeps; never reuse a previous
    # PSS sample or interpret an unreadable live address space as zero.
    for attempt in range(10):
        details = dict(attempt=attempt+1, reads=[], exit_probe=[])
        errors = []
        for source, parse in (('smaps_rollup', _rollup_pss), ('smaps', _smaps_pss)):
            try:
                text = (proc/source).read_text(encoding='utf-8', errors='surrogateescape')
                return parse(text)
            except (OSError, ValueError) as error:
                errors.append(error)
                details['reads'].append(_read_error(source, error))
        attempts.append(details)
        if _process_has_exited(proc, current, details['exit_probe']):
            return 0
        if any(isinstance(error, PermissionError) for error in errors) or attempt == 9:
            raise PssAccountingError(current, attempts) from errors[-1]
        time.sleep(.02)


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
            pss += _read_pss(proc, current)
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
    state, error, error_details = "launch_failed", None, None
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
        error_details = getattr(exc, 'details', None)
    finally:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=5)
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        result = dict(status=state, error=error, error_details=error_details,
                      returncode=None if proc is None else proc.returncode,
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
