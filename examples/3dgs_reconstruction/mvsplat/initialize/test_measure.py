"""Memory limits must include native descendants on kernels without children files."""
from pathlib import Path
import errno
import json
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from initialize.measure import tree_rss, tree_memory


def legacy_smaps_pss(text):
    """Pre-optimization parser retained only as an equivalence/benchmark oracle."""
    from initialize.measure import _VMA_HEADER, _pss_bytes
    mappings, value, total = 0, None, 0
    for line in text.splitlines():
        header = _VMA_HEADER.fullmatch(line)
        if header:
            if mappings and value is None:
                raise ValueError('Full smaps mapping is missing Pss')
            if int(header[2], 16) <= int(header[1], 16):
                raise ValueError('Invalid full smaps mapping range')
            if value is not None:
                total += value
            mappings, value = mappings+1, None
        elif line.strip().startswith('Pss:'):
            if not mappings or value is not None:
                raise ValueError('Full smaps Pss is unscoped or duplicated')
            value = _pss_bytes(line.strip())
        elif re.match(r'^[0-9a-fA-F]+-[0-9a-fA-F]+\s', line):
            raise ValueError('Malformed full smaps VMA header')
    if not mappings or value is None:
        raise ValueError('Full smaps is empty or final mapping is missing Pss')
    return total+value


def realistic_smaps(mappings=512):
    """Kernel-shaped text: 26 accounting/flag lines per complete VMA."""
    counters = ('Size', 'KernelPageSize', 'MMUPageSize', 'Rss', 'Pss_Dirty',
                'Shared_Clean', 'Shared_Dirty', 'Private_Clean', 'Private_Dirty',
                'Referenced', 'Anonymous', 'KSM', 'LazyFree', 'AnonHugePages',
                'ShmemPmdMapped', 'FilePmdMapped', 'Shared_Hugetlb', 'Private_Hugetlb',
                'Swap', 'SwapPss', 'Locked')
    result = []
    for index in range(mappings):
        first = 0x7f0000000000+index*0x10000
        result.append('%x-%x r-xp 00001000 08:01 %d /usr/lib/a-library-%d.so\n' %
                      (first, first+0x10000, 200000+index, index))
        result.extend(name+': 4 kB\n' for name in counters)
        result.append('Pss: %d kB\n' % (index % 97))
        result.extend(('THPeligible: 0\n', 'ProtectionKey: 0\n',
                       'VmFlags: rd ex mr mw me sd\n', '\n'))
    return ''.join(result)


class SmapsParserEquivalenceTests(unittest.TestCase):
    def assert_same_parse(self, text):
        from initialize.measure import _smaps_pss
        try:
            expected = legacy_smaps_pss(text)
        except ValueError as expected_error:
            with self.assertRaises(ValueError) as actual:
                _smaps_pss(text)
            self.assertEqual(str(actual.exception), str(expected_error))
        else:
            self.assertEqual(_smaps_pss(text), expected)

    def test_kernel_shaped_large_smaps_has_identical_complete_vma_sum(self):
        from initialize.measure import _smaps_pss
        text = realistic_smaps()
        self.assertGreater(len(text), 200000)
        self.assert_same_parse(text)
        self.assertEqual(_smaps_pss(text), sum(i % 97 for i in range(512))*1024)

    def test_ordinary_counter_lines_do_not_invoke_vma_regexes(self):
        from initialize import measure
        text = realistic_smaps(4)
        with patch.object(measure, '_VMA_HEADER', wraps=measure._VMA_HEADER) as header, \
                patch.object(measure, '_VMA_PREFIX', wraps=measure._VMA_PREFIX) as prefix:
            self.assertEqual(measure._smaps_pss(text), 6*1024)
        self.assertEqual(header.fullmatch.call_count, 4)
        prefix.match.assert_not_called()

    def test_invalid_headers_ranges_and_pss_preserve_exact_errors(self):
        header = '1000-2000 rw-p 00000000 00:00 0 [one]\n'
        valid = header+'Pss: 4 kB\n'
        bad_headers = (
            '2000-3000 rw-z 00000000 00:00 0\n',
            '2000-3000 rwp 00000000 00:00 0\n',
            '2000-3000 rw-p 0000000g 00:00 0\n',
            '2000-3000 rw-p 00000000 zz:00 0\n',
            '2000-3000 rw-p 00000000 00:00 -1\n',
            '2000-3000 \n',
            '2000-3000\trw-z 0 0:0 0\n',
            '2000-2000 rw-p 0 0:0 0\n',
            '3000-2000 rw-p 0 0:0 0\n',
        )
        cases = [valid+bad+'Pss: 2 kB\n' for bad in bad_headers]
        cases += ['', 'Pss: 4 kB\n', header, header+header+'Pss: 4 kB\n',
                  valid+'Pss: 1 kB\n', valid+'\u2003Pss: 2 kB\u2003\n',
                  valid.replace('Pss: 4 kB', 'Pss: -4 kB'),
                  valid.replace('Pss: 4 kB', 'Pss: 4 MB'),
                  valid.replace('Pss: 4 kB', 'Pss: 4 kB extra')]
        for text in cases:
            with self.subTest(text=text):
                self.assert_same_parse(text)

    def test_whitespace_unknown_counters_and_uppercase_ranges_remain_equivalent(self):
        text = ('A000-B000\trw-p\t0\t0:0\t0 [path-with-hyphen]\n'
                'Anonymous: 1024 kB\nAnonHugePages: 0 kB\n'
                'Pss_Dirty: 999 kB\nPss_Anon: 999 kB\n'
                'Dead-counter-with-hyphen: 999 kB\n'
                '  2000-3000 rw-z 0 0:0 0\n'
                '\u2003 Pss: 4 kB\t\n'
                'b000-c000 rw-p 0 0:0 0\n\tPss: 0 kB\r\n')
        self.assert_same_parse(text)


class TreeRssTests(unittest.TestCase):
    def process(self, root, pid, parent, rss=None):
        proc=root/str(pid)
        proc.mkdir()
        (proc/'status').write_text('Name:\ttest\nPPid:\t%d\n'%parent +
            ('' if rss is None else 'VmRSS:\t%d kB\n'%rss))

    def test_embedded_kernel_counts_native_child_and_grandchild(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.process(root,100,1,1024)
            self.process(root,101,100,2048)
            self.process(root,102,101,4096)
            self.process(root,200,1,8192)
            (root/'self').mkdir()
            self.assertEqual(tree_rss(100,root),7*1024**2)

    def test_disappeared_process_does_not_discard_other_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.process(root,10,1,256)
            (root/'11').mkdir()
            self.process(root,12,10,512)
            self.assertEqual(tree_rss(10,root),768*1024)
            self.assertEqual(tree_rss(99,root),0)

    def test_nonresident_parent_keeps_descendant_and_graph_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.process(root,10,11)
            self.process(root,11,10,128)
            self.assertEqual(tree_rss(10,root),128*1024)

    def test_pss_counts_shared_mappings_proportionally(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for pid, parent in ((100,1),(101,100),(200,1)):
                self.process(root,pid,parent,1024)
                (root/str(pid)/'smaps_rollup').write_text('Rss: 1024 kB\nPss: 512 kB\n')
            self.assertEqual(tree_memory(100,root,True),
                             dict(rss_bytes=2*1024**2,pss_bytes=1024**2))

    def test_missing_live_pss_is_an_error_instead_of_zero_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.process(root,100,1,1024)
            with self.assertRaisesRegex(RuntimeError,'PSS'):
                tree_memory(100,root,True)

    def test_reaped_address_space_keeps_live_descendant_accounting(self):
        for state in ('Z (zombie)', 'X (dead)'):
            with self.subTest(state=state), tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                self.process(root,100,1)
                with (root/'100/status').open('a') as stream:
                    stream.write('State:\t'+state+'\n')
                self.process(root,101,100,1024)
                (root/'101/smaps_rollup').write_text('Pss: 512 kB\n')
                self.assertEqual(tree_memory(100,root,True)['pss_bytes'],512*1024)

    def test_live_zero_rss_without_pss_is_still_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.process(root,100,1,0)
            with (root/'100/status').open('a') as stream:
                stream.write('State:\tS (sleeping)\n')
            with self.assertRaisesRegex(RuntimeError,'PSS'):
                tree_memory(100,root,True)

    def test_pss_exit_race_status_still_live_but_stat_confirms_zombie(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.process(root,100,1,1024)
            (root/'100/status').write_text('Name: test\nPPid: 1\nVmRSS: 1024 kB\nState: S (sleeping)\n')
            (root/'100/stat').write_text('100 (test) S 1 0 0 0\n')
            self.process(root,101,100,512)
            (root/'101/smaps_rollup').write_text('Pss: 256 kB\n')
            original=Path.read_text
            calls=0
            def race(path,*args,**kwargs):
                nonlocal calls
                value=original(path,*args,**kwargs)
                if path == root/'100/status':
                    calls+=1
                    if calls == 2:
                        # status was fetched before the last mm was released;
                        # stat is sampled afterwards and observes exit.
                        (root/'100/stat').write_text('100 (test) Z 1 0 0 0\n')
                return value
            with patch.object(Path,'read_text',race):
                self.assertEqual(tree_memory(100,root,True)['pss_bytes'],256*1024)
            self.assertGreaterEqual(calls,2)

    def test_pss_missing_with_both_status_and_stat_live_still_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.process(root,100,1,0)
            with (root/'100/status').open('a') as stream:
                stream.write('State: S (sleeping)\n')
            (root/'100/stat').write_text('100 (test with ) parentheses) S 1 0 0\n')
            with self.assertRaisesRegex(RuntimeError,'Cannot account process PSS'):
                tree_memory(100,root,True)

    def test_pss_exit_race_reaped_between_status_and_stat_keeps_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.process(root,100,1,0)
            with (root/'100/status').open('a') as stream:
                stream.write('State: R (running)\n')
            self.process(root,101,100,512)
            (root/'101/smaps_rollup').write_text('Pss: 256 kB\n')
            original=Path.read_text
            calls=0
            def race(path,*args,**kwargs):
                nonlocal calls
                value=original(path,*args,**kwargs)
                if path == root/'100/status':
                    calls+=1
                    if calls == 2:
                        (root/'100/status').unlink()
                        (root/'100').rmdir()
                return value
            with patch.object(Path,'read_text',race):
                self.assertEqual(tree_memory(100,root,True)['pss_bytes'],256*1024)

    def test_unreadable_live_status_and_stat_do_not_become_zero_pss(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.process(root,100,1,0)
            original=Path.read_text
            calls=0
            def denied(path,*args,**kwargs):
                nonlocal calls
                if path == root/'100/status':
                    calls+=1
                    if calls > 1:
                        raise PermissionError('live status denied')
                if path == root/'100/stat':
                    raise PermissionError('live stat denied')
                return original(path,*args,**kwargs)
            with patch.object(Path,'read_text',denied):
                with self.assertRaisesRegex(RuntimeError,'Cannot account process PSS'):
                    tree_memory(100,root,True)

    def test_exec_transient_pss_failure_retries_and_counts_actual_pss(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.process(root,100,1,1024)
            (root/'100/stat').write_text('100 (compiler) R 1 0 0\n')
            (root/'100/smaps_rollup').write_text('Pss: 768 kB\n')
            original=Path.read_text
            calls=0
            def transient(path,*args,**kwargs):
                nonlocal calls
                if path == root/'100/smaps_rollup':
                    calls+=1
                    if calls < 3:
                        raise FileNotFoundError('mm replaced during exec')
                return original(path,*args,**kwargs)
            with patch.object(Path,'read_text',transient), patch('initialize.measure.time.sleep') as sleep:
                self.assertEqual(tree_memory(100,root,True)['pss_bytes'],768*1024)
            self.assertEqual(calls,3)
            self.assertEqual(sleep.call_args_list,[unittest.mock.call(.02)]*2)

    def test_exec_mm_replacement_recovers_actual_pss_after_three_failed_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.process(root,100,1,1024)
            with (root/'100/status').open('a') as stream:
                stream.write('State: R (running)\n')
            (root/'100/stat').write_text('100 (compiler) R 1 0 0\n')
            (root/'100/smaps_rollup').write_text('Pss: 896 kB\n')
            (root/'100/smaps').write_text('')
            original=Path.read_text
            calls=0
            def replacing_mm(path,*args,**kwargs):
                nonlocal calls
                if path == root/'100/smaps_rollup':
                    calls+=1
                    if calls <= 3:
                        raise ProcessLookupError(errno.ESRCH,'No such process')
                return original(path,*args,**kwargs)
            with patch.object(Path,'read_text',replacing_mm), patch('initialize.measure.time.sleep') as sleep:
                self.assertEqual(tree_memory(100,root,True)['pss_bytes'],896*1024)
            self.assertEqual(calls,4)
            self.assertEqual(sleep.call_args_list,[unittest.mock.call(.02)]*3)

    def test_persistent_live_pss_failure_still_fails_after_bounded_retries(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.process(root,100,1,1024)
            (root/'100/stat').write_text('100 (compiler) R 1 0 0\n')
            original=Path.read_text
            calls=0
            def unavailable(path,*args,**kwargs):
                nonlocal calls
                if path == root/'100/smaps_rollup':
                    calls+=1
                    raise OSError('temporarily unreadable mm')
                return original(path,*args,**kwargs)
            with patch.object(Path,'read_text',unavailable), patch('initialize.measure.time.sleep') as sleep:
                with self.assertRaisesRegex(RuntimeError,'Cannot account process PSS: 100'):
                    tree_memory(100,root,True)
            self.assertEqual(calls,10)
            self.assertEqual(sleep.call_args_list,[unittest.mock.call(.02)]*9)

    def test_persistent_esrch_and_empty_smaps_never_accept_live_unknown_pss(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.process(root,100,1,1024)
            with (root/'100/status').open('a') as stream:
                stream.write('State: R (running)\n')
            (root/'100/stat').write_text('100 (compiler) R 1 0 0\n')
            (root/'100/smaps').write_text('')
            original=Path.read_text
            calls={'smaps_rollup':0,'smaps':0}
            def unavailable(path,*args,**kwargs):
                if path.parent == root/'100' and path.name in calls:
                    calls[path.name]+=1
                    if path.name=='smaps_rollup':
                        raise ProcessLookupError(errno.ESRCH,'No such process')
                return original(path,*args,**kwargs)
            with patch.object(Path,'read_text',unavailable), patch('initialize.measure.time.sleep') as sleep:
                with self.assertRaisesRegex(RuntimeError,'Cannot account process PSS: 100') as caught:
                    tree_memory(100,root,True)
            self.assertEqual(calls,{'smaps_rollup':10,'smaps':10})
            self.assertEqual(sleep.call_args_list,[unittest.mock.call(.02)]*9)
            attempts=caught.exception.details['attempts']
            self.assertEqual([item['attempt'] for item in attempts],list(range(1,11)))
            for attempt in attempts:
                self.assertEqual(attempt['reads'][0]['errno'],errno.ESRCH)
                self.assertEqual(attempt['reads'][1]['error_type'],'ValueError')
                states=[item.get('state') for item in attempt['exit_probe'] if 'state' in item]
                self.assertEqual(states,[['R'],['R']])

    def test_exit_after_extended_pss_retry_keeps_actual_live_child_pss(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.process(root,100,1,1024)
            with (root/'100/status').open('a') as stream:
                stream.write('State: R (running)\n')
            (root/'100/stat').write_text('100 (compiler) R 1 0 0\n')
            (root/'100/smaps').write_text('')
            self.process(root,101,100,512)
            (root/'101/smaps_rollup').write_text('Pss: 256 kB\n')
            original=Path.read_text
            calls=0
            def exiting(path,*args,**kwargs):
                nonlocal calls
                if path == root/'100/smaps_rollup':
                    calls+=1
                    raise ProcessLookupError(errno.ESRCH,'No such process')
                value=original(path,*args,**kwargs)
                if path == root/'100/status' and calls==4:
                    return value.replace('State: R (running)','State: Z (zombie)')
                return value
            with patch.object(Path,'read_text',exiting), patch('initialize.measure.time.sleep') as sleep:
                self.assertEqual(tree_memory(100,root,True)['pss_bytes'],256*1024)
            self.assertEqual(calls,4)
            self.assertEqual(sleep.call_args_list,[unittest.mock.call(.02)]*3)

    def test_pss_permission_denied_fails_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.process(root,100,1,1024)
            (root/'100/stat').write_text('100 (compiler) R 1 0 0\n')
            original=Path.read_text
            calls=0
            def denied(path,*args,**kwargs):
                nonlocal calls
                if path == root/'100/smaps_rollup':
                    calls+=1
                    raise PermissionError('PSS access denied')
                return original(path,*args,**kwargs)
            with patch.object(Path,'read_text',denied), patch('initialize.measure.time.sleep') as sleep:
                with self.assertRaisesRegex(RuntimeError,'Cannot account process PSS: 100'):
                    tree_memory(100,root,True)
            self.assertEqual(calls,1)
            sleep.assert_not_called()

    def smaps(self, root, pid, pss=(512, 256)):
        text = ''
        for i, value in enumerate(pss):
            text += ('%x-%x rw-p 00000000 00:00 0 [mapping %d]\n'
                     'Size: 1024 kB\nRss: 1024 kB\nPss: %d kB\n'
                     'Pss_Dirty: 0 kB\nVmFlags: rd wr mr mw me ac\n' %
                     (0x1000+i*0x1000, 0x2000+i*0x1000, i, value))
        (root/str(pid)/'smaps').write_text(text)

    def test_missing_rollup_uses_all_complete_smaps_mappings(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.process(root,100,1,2048)
            self.smaps(root,100)
            with patch('initialize.measure.time.sleep') as sleep:
                self.assertEqual(tree_memory(100,root,True)['pss_bytes'],768*1024)
            sleep.assert_not_called()

    def test_rollup_permission_error_can_use_independently_readable_full_smaps(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.process(root,100,1,2048)
            self.smaps(root,100)
            original=Path.read_text
            def read(path,*args,**kwargs):
                if path.name=='smaps_rollup':raise PermissionError('rollup unavailable')
                return original(path,*args,**kwargs)
            with patch.object(Path,'read_text',read),patch('initialize.measure.time.sleep') as sleep:
                self.assertEqual(tree_memory(100,root,True)['pss_bytes'],768*1024)
            sleep.assert_not_called()

    def test_malformed_or_negative_rollup_falls_back_instead_of_undercounting(self):
        for rollup in ('Pss: -2 kB\n','Pss: 2 MB\n','Pss: 5 kB\nPss: 1 kB\n'):
            with self.subTest(rollup=rollup),tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                self.process(root,100,1,2048)
                (root/'100/smaps_rollup').write_text(rollup)
                self.smaps(root,100)
                self.assertEqual(tree_memory(100,root,True)['pss_bytes'],768*1024)

    def test_full_smaps_rejects_partial_duplicate_negative_and_unscoped_pss(self):
        valid='1000-2000 rw-p 00000000 00:00 0\nPss: 4 kB\n'
        invalid=(valid+'2000-3000 rw-p 00000000 00:00 0\nRss: 8 kB\n',
                 valid+'Pss: 5 kB\n',valid.replace('4 kB','-1 kB'),
                 valid.replace('4 kB','4 MB'),'Pss: 4 kB\n','')
        for smaps in invalid:
            with self.subTest(smaps=smaps),tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                self.process(root,100,1,2048)
                (root/'100/smaps').write_text(smaps)
                with patch('initialize.measure.time.sleep'):
                    with self.assertRaisesRegex(RuntimeError,'Cannot account process PSS: 100'):
                        tree_memory(100,root,True)

    def test_failure_retains_both_pss_sources_and_exit_probe_in_bounded_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.process(root,100,1,2048)
            (root/'100/stat').write_text('100 (compiler) R 1 0 0\n')
            (root/'100/smaps').write_text('')
            with patch('initialize.measure.time.sleep') as sleep:
                with self.assertRaisesRegex(RuntimeError,'smaps_rollup.*smaps') as caught:
                    tree_memory(100,root,True)
            details=caught.exception.details
            self.assertEqual(details['pid'],100)
            self.assertEqual(len(details['attempts']),10)
            self.assertEqual(sleep.call_args_list,[unittest.mock.call(.02)]*9)
            for attempt in details['attempts']:
                self.assertEqual([item['source'] for item in attempt['reads']],['smaps_rollup','smaps'])
                self.assertTrue(attempt['exit_probe'])
                self.assertIn('FileNotFoundError',attempt['reads'][0]['error_type'])

    def test_real_read_zero_pss_is_distinct_from_missing_live_pss(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.process(root,100,1,0)
            self.smaps(root,100,(0,0))
            self.assertEqual(tree_memory(100,root,True)['pss_bytes'],0)

    def test_cli_preserves_strict_budgets_and_serializes_failure_details(self):
        from initialize.measure import main, PssAccountingError
        attempts=[dict(attempt=1,reads=[dict(source='smaps_rollup',error_type='OSError',
            errno=5,message='simulated read failure'),dict(source='smaps',error_type='ValueError',
            errno=None,message='simulated missing Pss')],exit_probe=[dict(source='stat',state=['R'])])]
        error=PssAccountingError(123,attempts)
        process=SimpleNamespace(pid=123,returncode=None,poll=lambda:None)
        process.wait=lambda **kwargs:setattr(process,'returncode',-15)
        resource=SimpleNamespace(RUSAGE_CHILDREN=0,getrusage=lambda _:SimpleNamespace(
            ru_utime=0.,ru_stime=0.,ru_maxrss=1024.))
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'measurement.json'
            argv=['measure.py','--out',str(output),'--rss-mib','0','--pss-mib','700',
                  '--reserve-mib','96','--','fake-compiler']
            with patch('sys.argv',argv),patch.dict('sys.modules',{'resource':resource}), \
                    patch('initialize.measure.available_mib',return_value=200.), \
                    patch('initialize.measure.subprocess.Popen',return_value=process), \
                    patch('initialize.measure.tree_memory',side_effect=error), \
                    patch('initialize.measure.os.killpg',create=True):
                with self.assertRaisesRegex(SystemExit,'measurement_failed'):
                    main()
            record=json.loads(output.read_text())
            self.assertEqual(record['status'],'measurement_failed')
            self.assertEqual(record['error_details'],error.details)
            self.assertEqual(record['limits']['pss_mib'],700.)
            self.assertEqual(record['limits']['reserve_mib'],96.)
            self.assertIn('smaps_rollup',record['error'])
            self.assertIn('smaps',record['error'])


if __name__=='__main__':
    unittest.main()
