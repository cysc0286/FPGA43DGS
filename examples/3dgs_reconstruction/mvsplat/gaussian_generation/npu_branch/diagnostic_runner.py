"""Run one unregistered diagnostic ONNX candidate through the resident NPU bridge.

This tool intentionally does not use PartitionRuntime/catalog and cannot qualify
or attach a production partition. It consumes a single compiled diagnostic
manifest entry, verifies its source/capture identities and explicit ABI, then
runs the real A*3/B*3/A*3 sequence in one SDK session.
"""
import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np

from common import save, sha
from gaussian_generation.npu_branch.artifacts import from_wire, relative_file, to_wire
from gaussian_generation.npu_branch.catalog import PARTITIONS
from gaussian_generation.npu_branch.compile_graphs import audit
from gaussian_generation.npu_branch.protocol import check_bridge, check_direct_sfb


RTOL = 2e-3
ATOL = 2e-4


def array_sha(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(value.tobytes()).hexdigest()


def compare(actual, expected, rtol=RTOL, atol=ATOL):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        return dict(passed=False, shape_matches=False, finite=False,
                    actual_shape=list(actual.shape), expected_shape=list(expected.shape))
    finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
    if not finite:
        return dict(passed=False, shape_matches=True, finite=False)
    delta = np.abs(actual.astype(np.float64)-expected.astype(np.float64))
    failed = delta > atol+rtol*np.abs(expected)
    return dict(passed=not bool(failed.any()), shape_matches=True, finite=True,
                exact=bool(np.array_equal(actual, expected)),
                max_abs=float(delta.max(initial=0)), mean_abs=float(delta.mean()),
                rmse=float(np.sqrt(np.mean(delta*delta))),
                outside_tolerance_fraction=float(failed.mean()),
                tolerance=dict(rtol=rtol, atol=atol))


def _read_fixture(path, expected_shape, label):
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {'input', 'expected'}:
            raise ValueError(label+' fixture must contain only input and expected')
        value, expected = data['input'].copy(), data['expected'].copy()
    for key, array, shape in (('input', value, expected_shape[0]),
                              ('expected', expected, expected_shape[1])):
        if (array.dtype != np.float32 or list(array.shape) != shape
                or not np.isfinite(array).all()):
            raise ValueError(f'{label} {key} violates manifest FP32 shape/finite contract')
    return value, expected


def _capture_binding(capture_root, candidate, *, allow_synthetic_b=False):
    """Bind A/B diagnostics to the exact source capture, rejecting zero-B by default."""
    capture_root = Path(capture_root).resolve()
    capture_manifest_path = capture_root/'capture_manifest.json'
    if not capture_manifest_path.is_file():
        raise ValueError('Missing capture_manifest.json')
    if sha(capture_manifest_path) != candidate.get('capture_manifest_sha256'):
        raise ValueError('Capture manifest identity differs from exported candidate')
    capture = json.loads(capture_manifest_path.read_text(encoding='utf-8'))
    module = candidate.get('module')
    boundary = candidate.get('boundary') or {}
    node_name = boundary.get('name')
    kind = candidate.get('kind')
    try:
        traces = capture['modules'][module]['captures']
        trace_a, trace_b = traces['A'], traces['B']
        a_boundary = trace_a['boundaries'][node_name]
        b_boundary = trace_b['boundaries'][node_name]
    except (KeyError, TypeError):
        raise ValueError('Capture manifest lacks A/B for requested boundary') from None
    for label, trace in (('A', trace_a), ('B', trace_b)):
        if not trace.get('complete'):
            raise ValueError('Incomplete source capture '+label)
    origin_b = str(trace_b.get('source_origin', ''))
    if ('synthetic' in origin_b.lower() or 'zero' in origin_b.lower()) and not allow_synthetic_b:
        raise ValueError('B is synthetic; supply two actual contexts or explicitly allow --allow-synthetic-b')
    if kind not in ('block', 'prefix'):
        raise ValueError('Candidate must be a diagnostic block or prefix')
    return dict(module=module, boundary=node_name, kind=kind,
                source_a=trace_a.get('source_origin'), source_b=origin_b,
                a_boundary=a_boundary, b_boundary=b_boundary,
                capture_manifest_sha256=sha(capture_manifest_path))


def _loaded_sdk_libraries():
    maps = Path('/proc/self/maps')
    if not maps.is_file():
        return {}
    result = {}
    for line in maps.read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith('/'):
            path = Path(fields[5])
            if 'icraft' in path.name.lower() and path.is_file():
                result[str(path.resolve())] = sha(path)
    return result


def validate_candidate(graphs, bundle, name, capture_root, *, allow_synthetic_b=False):
    graphs, bundle = Path(graphs).resolve(), Path(bundle).resolve()
    if Path(name).name != name or name in ('', '.', '..'):
        raise ValueError('Candidate name must be one manifest key')
    if name in PARTITIONS:
        raise ValueError('Diagnostic runner refuses registered production catalog entries')
    source_manifest_path, compiled_manifest_path = graphs/'manifest.json', bundle/'manifest.json'
    source = json.loads(source_manifest_path.read_text(encoding='utf-8'))
    compiled = json.loads(compiled_manifest_path.read_text(encoding='utf-8'))
    if source.get('schema') != 'mvsplat_partitions_v1' or compiled.get('schema') != 'mvsplat_compiled_v1':
        raise ValueError('Unexpected export/compiled manifest schema')
    if compiled.get('source_manifest_sha256') != sha(source_manifest_path):
        raise ValueError('Compiled graph is not bound to the supplied export manifest')
    if compiled.get('weight_sha256') != source.get('weight_sha256'):
        raise ValueError('Checkpoint identity mismatch')
    item = compiled.get('partitions', {}).get(name)
    if not item or not item.get('complete') or item.get('status') != 'compiled_not_executed':
        raise ValueError('Expected one completed, not-yet-executed compiled candidate')
    exported = item.get('export') or {}
    if (not exported.get('diagnostic_only') or exported.get('production_qualified') is not False
            or exported.get('npu_executed') is not False or not exported.get('host_onnx_passed')):
        raise ValueError('Candidate is not a host-verified diagnostic-only export')
    if exported.get('module') not in ('corr_refine_net', 'refine_unet'):
        raise ValueError('Only corr/refine U-Net diagnostic candidates are accepted')
    source_item = source.get('partitions', {}).get(name)
    if not source_item or not source_item.get('complete'):
        raise ValueError('Source candidate is missing or incomplete')
    graph = relative_file(bundle, item['graph'])
    raw = relative_file(bundle, item['raw'])
    actual_abi = audit(graph, raw, exported)
    if actual_abi != item.get('audit'):
        raise ValueError('Compiled ABI/placement/hash differs from manifest')
    if actual_abi.get('npu_executed') is not False or actual_abi.get('host_compute_ops') != 0:
        raise ValueError('Compiled audit does not prove an NPU-only diagnostic graph')
    if exported.get('dtype') != 'float32' or exported.get('logical_layout') != 'NCHW':
        raise ValueError('Expected logical FP32 NCHW oracle')
    input_spec, output_spec = actual_abi['input'], actual_abi['output']
    in_shape, out_shape = list(exported['input_shape']), list(exported['output_shape'])
    input_wire_shape = list(input_spec['shape'])
    output_wire_shape = list(output_spec['shape'])
    if input_spec['logical_shape'] != in_shape or output_spec['logical_shape'] != out_shape:
        raise ValueError('Compiled wire ABI does not map to exported logical shapes')
    folder = (graphs/name).resolve()
    if graphs not in folder.parents or not folder.is_dir():
        raise ValueError('Candidate export folder escapes source manifest root')
    oracle_a, oracle_b = folder/'oracle.npz', folder/'oracle_b.npz'
    if sha(oracle_a) != exported.get('oracle_sha256'):
        raise ValueError('A oracle hash mismatch')
    if not oracle_b.is_file() or sha(oracle_b) != exported.get('oracle_b_sha256'):
        raise ValueError('A real B oracle is required and must match the export manifest')
    a, b = _read_fixture(oracle_a, (in_shape, out_shape), 'A'), _read_fixture(oracle_b, (in_shape, out_shape), 'B')
    if np.array_equal(a[0], b[0]):
        raise ValueError('A and B inputs are identical')
    if compare(a[1], b[1])['passed']:
        raise ValueError('CPU reference outputs are not distinguishable at the configured tolerance')
    capture_identity = _capture_binding(capture_root, source_item,
                                        allow_synthetic_b=allow_synthetic_b)
    if capture_identity['module'] != exported.get('module') or capture_identity['boundary'] != (exported.get('boundary') or {}).get('name'):
        raise ValueError('Capture topology differs from candidate export')
    # Validate the candidate fixtures against the exact source hook outputs.
    for label, sample, trace, node in (('A', a, capture_identity['a_boundary'], 'A'),
                                       ('B', b, capture_identity['b_boundary'], 'B')):
        capture_path = relative_file(capture_root, Path(capture_identity['module'])/trace['oracle'])
        if sha(capture_path) != trace.get('oracle_sha256'):
            raise ValueError('Source boundary capture hash mismatch: '+label)
        with np.load(capture_path, allow_pickle=False) as captured:
            if source_item.get('kind') == 'block':
                expected_in, expected_out = captured['input'].copy(), captured['expected'].copy()
            else:
                trace_obj = json.loads((capture_root/'capture_manifest.json').read_text(encoding='utf-8'))
                mod_trace = trace_obj['modules'][capture_identity['module']]['captures'][node]
                source_fixture = relative_file(capture_root,
                    Path(capture_identity['module'])/mod_trace['source_fixture'])
                if sha(source_fixture) != mod_trace.get('source_fixture_sha256'):
                    raise ValueError('Original source fixture hash mismatch: '+label)
                with np.load(source_fixture, allow_pickle=False) as original:
                    expected_in = original['input'].copy()
                expected_out = captured['expected'].copy()
        if not np.array_equal(sample[0], expected_in) or not np.array_equal(sample[1], expected_out):
            raise ValueError('Candidate oracle differs from captured source tensors: '+label)
    return dict(name=name, graphs=graphs, bundle=bundle, graph=graph, raw=raw,
                input_spec=dict(layout=input_spec['layout'], shape=input_wire_shape),
                output_spec=dict(layout=output_spec['layout'], shape=output_wire_shape),
                input_shape=in_shape, output_shape=out_shape, samples={'A':a, 'B':b},
                export=exported, compiled=item, capture_identity=capture_identity)


def _execute_wire(forward_into, value, input_spec, output_spec):
    input_wire = to_wire(value, input_spec)
    output_wire = np.full(output_spec['shape'], np.nan, dtype=np.float32)
    timings = forward_into(input_wire, output_wire)
    if not np.isfinite(output_wire).all():
        raise ValueError('Bridge left nonfinite or unwritten output elements')
    return from_wire(output_wire, output_spec), timings


def run_sequence(session, samples, input_spec, output_spec, *, session_factory=None,
                 session_records=None):
    """Execute A*3/B*3/A*3, optionally resetting SDK state at each input change."""
    a, b = samples['A'], samples['B']
    sequence = [('A_initial', a)]*3 + [('B', b)]*3 + [('A_return', a)]*3
    rows, outputs = [], []
    error = None
    attempted = 0
    active_session, manager, active_segment = session, None, None
    context_count = 0 if session_factory else 1
    try:
        for index, (segment, (value, expected)) in enumerate(sequence):
            started = time.perf_counter()
            attempted += 1
            try:
                if session_factory and segment != active_segment:
                    if manager is not None:
                        manager.__exit__(None, None, None)
                        manager = None
                    manager = session_factory()
                    active_session = manager.__enter__()
                    active_segment = segment
                    context_count += 1
                    if session_records is not None:
                        session_records.append(dict(bindings=active_session.bindings,
                            memory_layout=active_session.memory_layout,
                            sdk_libraries=_loaded_sdk_libraries()))
                actual, timing = _execute_wire(active_session.forward_into, value,
                                               input_spec, output_spec)
            except Exception as exc:
                error = type(exc).__name__+': '+str(exc)
                break
            elapsed = time.perf_counter()-started
            outputs.append(actual)
            rows.append(dict(index=index, segment=segment, input_sha256=array_sha(value),
                output_sha256=array_sha(actual), finite=True,
                output_min=float(actual.min()), output_max=float(actual.max()),
                cpu_reference=compare(actual, expected), seconds=elapsed,
                bridge_timing_seconds=timing))
    finally:
        if manager is not None:
            manager.__exit__(None, None, None)
    complete = len(rows) == 9
    has_a_repeat = len(outputs) >= 3
    has_b_repeat = len(outputs) >= 6
    has_a_return = len(outputs) == 9
    summary = dict(complete=complete, diagnostic_only=True, acceptance=False,
        sequence='A*3/B*3/A*3', sdk_forward_calls=len(rows), attempted_calls=attempted,
        a_initial_repeat_exact=has_a_repeat and all(np.array_equal(outputs[0], v) for v in outputs[1:3]),
        b_repeat_exact=has_b_repeat and all(np.array_equal(outputs[3], v) for v in outputs[4:6]),
        a_return_exact=has_a_return and all(np.array_equal(outputs[0], v) for v in outputs[6:9]),
        expected_a_b_distinguishable=not compare(a[1], b[1])['passed'],
        actual_a_b_distinguishable=(len(outputs) >= 4 and
                                    not compare(outputs[0], outputs[3])['passed']),
        strict_reference_pass=bool(rows) and all(row['cpu_reference']['passed'] for row in rows),
        session_policy='one_context_per_input_segment' if session_factory else 'one_context',
        sdk_context_count=context_count, tolerance=dict(rtol=RTOL, atol=ATOL))
    summary['diagnostic_pass'] = complete and all((summary['a_initial_repeat_exact'],
        summary['b_repeat_exact'], summary['a_return_exact'],
        summary['expected_a_b_distinguishable'], summary['actual_a_b_distinguishable'],
        summary['strict_reference_pass']))
    result = dict(complete=complete, summary=summary, calls=rows)
    if error:
        result['error'] = error
    return result


class BridgeSession:
    """One SDK context using the production bridge ABI and SFB dump conversion."""
    def __init__(self, spec, library_path):
        if platform.machine().lower() not in ('aarch64', 'arm64'):
            raise RuntimeError('Real NPU execution requires the ARM board')
        import fcntl
        self._lock_file = open('/run/lock/fpga43dgs-npu.lock', 'a')
        try:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._lock_file.close()
            raise RuntimeError('Another process owns the NPU diagnostic lock') from None
        self.lib = None
        self.ctx = None
        try:
            self.lib = ct.CDLL(str(Path(library_path).resolve()))
            if os.environ.get('MGS_NPU_DIRECT_SFB') == '1':
                check_direct_sfb(self.lib)
            check_bridge(self.lib)
            self.lib.mgs_error.restype = ct.c_char_p
            self.lib.mgs_create.argtypes = [ct.c_char_p, ct.c_char_p, ct.c_size_t, ct.c_size_t]
            self.lib.mgs_create.restype = ct.c_void_p
            self.lib.mgs_destroy.argtypes = [ct.c_void_p]
            self.lib.mgs_destroy.restype = None
            self.lib.mgs_bindings.argtypes = [ct.c_void_p]
            self.lib.mgs_bindings.restype = ct.c_int
            self.lib.mgs_forward.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t,
                ct.c_void_p, ct.c_size_t, ct.POINTER(ct.c_double)]
            self.lib.mgs_forward.restype = ct.c_int
            self.forward_detailed = getattr(self.lib, 'mgs_forward_detailed', None)
            if self.forward_detailed is not None:
                self.forward_detailed.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t,
                    ct.c_void_p, ct.c_size_t, ct.POINTER(ct.c_double)]
                self.forward_detailed.restype = ct.c_int
            guard_mib = int(os.environ.get('MGS_NPU_DDR_GUARD_MIB', '0'))
            if guard_mib:
                try:
                    memory_layout = self.lib.mgs_memory_layout
                except AttributeError:
                    raise RuntimeError('Bridge lacks required PL DDR reservation support') from None
                memory_layout.argtypes = [ct.c_void_p]
                memory_layout.restype = ct.c_char_p
            self.ctx = self.lib.mgs_create(str(spec['graph']).encode(), str(spec['raw']).encode(),
                int(np.prod(spec['input_spec']['shape']))*4,
                int(np.prod(spec['output_spec']['shape']))*4)
            if not self.ctx:
                raise RuntimeError(self._error())
            bindings = self.lib.mgs_bindings(self.ctx)
            if bindings <= 0:
                raise RuntimeError('No ZG330 compute binding in diagnostic session')
            self.bindings = bindings
            if guard_mib:
                raw = memory_layout(self.ctx)
                regions = json.loads(raw) if raw else []
                guard_bytes = guard_mib*1024*1024
                if not regions or min(regions, key=lambda x: x['begin'])['bytes'] != guard_bytes:
                    raise RuntimeError('PL DDR reservation could not be verified')
                self.memory_layout = regions
            else:
                self.memory_layout = None
        except BaseException:
            self.close()
            raise

    def _error(self):
        raw = self.lib.mgs_error()
        return raw.decode(errors='replace') if raw else 'Unknown bridge error'

    def forward_into(self, input_wire, output_wire):
        width = 5 if self.forward_detailed is not None else 3
        measured = (ct.c_double*width)()
        forward = self.forward_detailed or self.lib.mgs_forward
        rc = forward(self.ctx, input_wire.ctypes.data, input_wire.nbytes,
                     output_wire.ctypes.data, output_wire.nbytes, measured)
        if rc:
            raise RuntimeError(self._error())
        names = (['input_write','forward_submit','wait','output_convert','sdk_total']
                 if width == 5 else ['input_write','execute_wait','output_convert'])
        return dict(zip(names, map(float, measured)))

    def close(self):
        if getattr(self, 'ctx', None):
            self.lib.mgs_destroy(self.ctx)
            self.ctx = None
        lock_file = getattr(self, '_lock_file', None)
        if lock_file is not None:
            lock_file.close()
            self._lock_file = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def execute_diagnostic(graphs, bundle, partition, capture_root, library, out,
                       *, allow_synthetic_b=False, fresh_session_per_segment=False):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    result_path = out/'diagnostic_result.json'
    record = dict(complete=False, diagnostic_only=True, acceptance=False,
                  production_runtime_modified=False, partition=partition,
                  scope='single diagnostic graph; isolated SDK state diagnostic; no production qualification')
    try:
        spec = validate_candidate(graphs, bundle, partition, capture_root,
                                  allow_synthetic_b=allow_synthetic_b)
        fresh_input = os.environ.get('MGS_NPU_FRESH_INPUT')
        if fresh_input not in (None, '1'):
            raise ValueError('U-Net diagnostic requires MGS_NPU_FRESH_INPUT=1')
        os.environ['MGS_NPU_FRESH_INPUT'] = '1'
        record.update(candidate=dict(name=partition, graph_sha256=sha(spec['graph']),
            raw_sha256=sha(spec['raw']), export_manifest_sha256=sha(Path(graphs)/'manifest.json'),
            compiled_manifest_sha256=sha(Path(bundle)/'manifest.json'),
            capture_manifest_sha256=spec['capture_identity']['capture_manifest_sha256'],
            input_shape=spec['input_shape'], output_shape=spec['output_shape'],
            input_layout=spec['input_spec']['layout'], output_layout=spec['output_spec']['layout'],
            source_a=spec['capture_identity']['source_a'], source_b=spec['capture_identity']['source_b']),
            input_hashes={label:array_sha(pair[0]) for label,pair in spec['samples'].items()},
            expected_hashes={label:array_sha(pair[1]) for label,pair in spec['samples'].items()},
            bridge_sha256=sha(library), rtol=RTOL, atol=ATOL,
            execution_environment=dict(fresh_input=True,
                force_output_wait=os.environ.get('MGS_NPU_FORCE_WAIT') == '1',
                direct_sfb=os.environ.get('MGS_NPU_DIRECT_SFB') == '1',
                ddr_guard_mib=int(os.environ.get('MGS_NPU_DDR_GUARD_MIB', '0'))))
        if fresh_session_per_segment:
            session_records = []
            run = run_sequence(None, spec['samples'], spec['input_spec'], spec['output_spec'],
                session_factory=lambda: BridgeSession(spec, library),
                session_records=session_records)
            record['session'] = dict(one_session=False, context_policy='one fresh SDK context per A/B segment',
                                     contexts=session_records, sdk_calls=9, sdk_passes_per_request=1)
        else:
            with BridgeSession(spec, library) as session:
                record['session'] = dict(bindings=session.bindings, memory_layout=session.memory_layout,
                                         sdk_libraries=_loaded_sdk_libraries(),
                                         one_session=True, sdk_calls=9, sdk_passes_per_request=1)
                run = run_sequence(session, spec['samples'], spec['input_spec'], spec['output_spec'])
        record.update(run)
        record['complete'] = bool(run['complete'])
        save(result_path, record)
        if not record['complete']:
            raise RuntimeError(run.get('error', 'Diagnostic sequence stopped early'))
        return record
    except BaseException as exc:
        record['error'] = type(exc).__name__+': '+str(exc)
        save(result_path, record)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graphs', required=True, type=Path, help='unet_probe_export output root')
    parser.add_argument('--bundle', required=True, type=Path, help='compile_graphs output root')
    parser.add_argument('--partition', required=True, help='one unregistered candidate from both manifests')
    parser.add_argument('--capture-root', required=True, type=Path, help='source capture manifest bound by export')
    parser.add_argument('--library', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path, help='new evidence directory; must not exist')
    parser.add_argument('--allow-synthetic-b', action='store_true',
                        help='diagnostic only: explicitly permit synthetic zero B')
    parser.add_argument('--fresh-session-per-segment', action='store_true',
                        help='diagnostic only: create a new SDK context at each A/B segment boundary')
    args = parser.parse_args(argv)
    result = execute_diagnostic(args.graphs, args.bundle, args.partition,
        args.capture_root, args.library, args.out, allow_synthetic_b=args.allow_synthetic_b,
        fresh_session_per_segment=args.fresh_session_per_segment)
    print(json.dumps(dict(complete=result['complete'], diagnostic_pass=result['summary']['diagnostic_pass'],
                          output=str(Path(args.out).resolve()/'diagnostic_result.json'))))


if __name__ == '__main__':
    main()
