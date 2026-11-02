"""Keep the fastest qualified experiment; prune only explicitly owned results.

Each direct trial directory needs ``memory_trial.json`` with schema
``memory_trial_v1``, protocol, and state (complete/failed/running). A winner must
pass correctness, finite, image, and memory gates and have a positive finite
scene_preparation_seconds. Producers must be stopped or marked running; pass
currently displayed/rollback directory names as ``protected``. Dry-run is the
default. This tool never cleans recordings or unmarked directories.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import tempfile


MARKER = 'memory_trial.json'
SCHEMA = 'memory_trial_v1'
SUMMARY = 'retention_summary.json'
GATES = ('correctness_passed', 'finite_passed', 'image_passed', 'memory_passed')


def _link(info):
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, 'st_file_attributes', 0) & 0x400)


def _root(path):
    path = Path(path).absolute()
    for ancestor in reversed((path, *path.parents)):
        info = ancestor.lstat()
        if _link(info) or not stat.S_ISDIR(info.st_mode):
            raise ValueError('Root has a link or non-directory component: '+str(ancestor))
    return path.resolve(strict=True)


def _json_file(path, limit):
    info = path.lstat()
    if _link(info) or not stat.S_ISREG(info.st_mode) or info.st_size > limit:
        raise ValueError('Unsafe or oversized metadata: '+str(path))
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _fingerprint(path, device):
    """Reject links, special files and nested mounts, without following them."""
    rows = []
    pending = [path]
    while pending:
        current = pending.pop()
        info = current.lstat()
        if (_link(info) or info.st_dev != device or os.path.ismount(current) or
                not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode))):
            raise ValueError('Link, mount or special file in trial: '+str(current))
        rows.append((str(current.relative_to(path)), info.st_dev, info.st_ino,
                     info.st_mode, info.st_size, info.st_mtime_ns))
        if stat.S_ISDIR(info.st_mode):
            pending.extend(current.iterdir())
    return sorted(rows)


def _compact(value, depth=0):
    if depth > 4:
        return '[omitted]'
    if value is None or type(value) in (bool, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value[:1000]
    if isinstance(value, dict):
        return {str(k)[:100]:_compact(v, depth+1) for k,v in list(value.items())[:32]}
    if isinstance(value, list):
        return [_compact(v, depth+1) for v in value[:16]]
    return str(value)[:1000]


def _record(name, data, digest):
    keys = ('protocol', 'state', 'scene_preparation_seconds', *GATES,
            'memory', 'hashes', 'config', 'error')
    return dict(name=name, marker_sha256=digest,
                **{key:_compact(data[key]) for key in keys if key in data})


def _atomic_summary(root, result, records):
    target = root/SUMMARY
    if target.exists() or target.is_symlink():
        info = target.lstat()
        if _link(info) or not stat.S_ISREG(info.st_mode):
            raise ValueError('Unsafe summary path')
    payload = dict(schema='memory_retention_summary_v1', root=str(root),
                   records=records, last_action=result)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=root,
                                         prefix='.retention.', delete=False) as output:
            temporary = Path(output.name)
            json.dump(payload, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write('\n')
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def retain_best(root, protocol, protected=(), apply=False):
    """Plan/apply retention for one protocol, preserving compact failure evidence.

    No qualified winner means no deletion. ``protected`` contains only direct
    child names, not paths. Returns actual removed names separately from plans.
    """
    root = _root(root)
    if not isinstance(protocol, str) or not protocol or len(protocol) > 256:
        raise ValueError('Expected a bounded explicit protocol identifier')
    protected = set(protected)
    if any(not isinstance(n, str) or n in ('', '.', '..') or
           '/' in n or '\\' in n or Path(n).name != n for n in protected):
        raise ValueError('Protected entries must be direct child names')
    history = {}
    summary_path = root/SUMMARY
    if summary_path.exists() or summary_path.is_symlink():
        previous, _ = _json_file(summary_path, 4*1024**2)
        if previous.get('schema') != 'memory_retention_summary_v1' or previous.get('root') != str(root):
            raise ValueError('Existing summary belongs to another root/schema')
        history = {(r['protocol'],r['name']):r for r in previous['records']}
    device = root.stat().st_dev
    trials, skipped, kept = {}, {}, {}
    for child in sorted(root.iterdir()):
        if child.name == SUMMARY or child.name.startswith('.retention.'):
            continue
        info = child.lstat()
        if _link(info):
            skipped[child.name] = 'link_or_reparse_point'
            continue
        if not stat.S_ISDIR(info.st_mode):
            continue
        if child.name in protected:
            kept[child.name] = 'protected'
        marker = child/MARKER
        if not marker.exists() and not marker.is_symlink():
            skipped[child.name] = 'unmarked'
            continue
        try:
            data, digest = _json_file(marker, 64*1024)
            if data.get('schema') != SCHEMA or not isinstance(data.get('protocol'), str):
                raise ValueError('Unknown trial marker')
            if data['protocol'] != protocol:
                kept.setdefault(child.name, 'other_protocol')
                continue
            history[(protocol,child.name)] = _record(child.name, data, digest)
            if data.get('state') not in ('complete', 'failed') or data.get('active'):
                kept.setdefault(child.name, 'active_or_nonterminal')
                continue
            fingerprint = _fingerprint(child, device)
            trials[child.name] = dict(data=data, marker_sha256=digest, fingerprint=fingerprint)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            skipped[child.name] = str(exc)
    def eligible(item):
        data = item[1]['data']
        seconds = data.get('scene_preparation_seconds')
        return (data.get('state') == 'complete' and all(data.get(g) is True for g in GATES)
                and type(seconds) in (int,float) and math.isfinite(seconds) and seconds > 0)
    qualified = [item for item in trials.items() if eligible(item)]
    best = min(qualified, key=lambda item:(item[1]['data']['scene_preparation_seconds'],item[0]))[0] if qualified else None
    if best is not None:
        kept[best] = 'fastest_qualified'
    planned = sorted(name for name in trials if name != best and name not in protected) if best is not None else []
    if best is None:
        kept.update({name:'no_qualified_winner' for name in trials})
    result = dict(protocol=protocol, apply=bool(apply), best=best, kept=kept,
                  planned_removals=planned, removed=[], skipped=skipped)
    records = sorted(history.values(), key=lambda r:(r['protocol'],r['name']))
    if not apply:
        return dict(result, records=records)
    # Persist evidence and the deletion plan before changing any result directory.
    _atomic_summary(root, result, records)
    for name in planned:
        child = root/name
        try:
            _, digest = _json_file(child/MARKER, 64*1024)
            if digest != trials[name]['marker_sha256'] or _fingerprint(child, device) != trials[name]['fingerprint']:
                raise ValueError('Trial changed after planning')
            shutil.rmtree(child)
            result['removed'].append(name)
        except (OSError, ValueError) as exc:
            result['skipped'][name] = str(exc)
            _atomic_summary(root, result, records)
            raise RuntimeError('Retention stopped; summaries preserved: '+name) from exc
    _atomic_summary(root, result, records)
    return dict(result, records=records)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--protocol', required=True)
    p.add_argument('--protected', nargs='*', default=[])
    p.add_argument('--apply', action='store_true', help='Default is a read-only deletion preview')
    a = p.parse_args(argv)
    print(json.dumps(retain_best(a.root,a.protocol,a.protected,a.apply), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
