"""Only owned inactive experiments can be pruned, after preserving metrics."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from initialize.result_retention import MARKER, SUMMARY, retain_best


class RetentionTests(unittest.TestCase):
    def trial(self, root, name, seconds=10., **changes):
        path = root/name
        path.mkdir()
        record = dict(schema='memory_trial_v1',protocol='fixed-v1',state='complete',
            correctness_passed=True,finite_passed=True,image_passed=True,memory_passed=True,
            scene_preparation_seconds=seconds,memory=dict(peak_pss_mib=500),
            hashes=dict(image_sha256='abc'))
        record.update(changes)
        (path/MARKER).write_text(json.dumps(record))
        (path/'gaussians.npz').write_bytes(b'large output')
        return path

    def test_keeps_fastest_not_latest_and_preserves_failure_summary(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            self.trial(root,'best',9.)
            self.trial(root,'later',12.)
            self.trial(root,'bad',1.,state='failed',image_passed=False,error='nonfinite output')
            result=retain_best(root,'fixed-v1',apply=True)
            self.assertEqual(result['best'],'best')
            self.assertEqual(result['removed'],['bad','later'])
            self.assertTrue((root/'best/gaussians.npz').exists())
            summary=json.loads((root/SUMMARY).read_text())
            failure=next(r for r in summary['records'] if r['name']=='bad')
            self.assertEqual(failure['error'],'nonfinite output')
            self.assertEqual(failure['memory']['peak_pss_mib'],500)
            self.assertEqual(failure['hashes']['image_sha256'],'abc')
            again=retain_best(root,'fixed-v1',apply=True)
            self.assertEqual(len(again['records']),3)

    def test_dry_run_writes_and_deletes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.trial(root,'a',1.);self.trial(root,'b',2.)
            result=retain_best(root,'fixed-v1')
            self.assertEqual(result['planned_removals'],['b'])
            self.assertTrue((root/'b/gaussians.npz').exists())
            self.assertFalse((root/SUMMARY).exists())

    def test_no_valid_winner_means_no_deletion(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for n,changes in [('a',dict(finite_passed=False)),('b',dict(memory_passed=False)),
                              ('c',dict(scene_preparation_seconds=float('nan'))),('d',dict(state='failed'))]:
                self.trial(root,n,**changes)
            result=retain_best(root,'fixed-v1',apply=True)
            self.assertIsNone(result['best']);self.assertEqual(result['removed'],[])

    def test_protected_qualified_scene_still_participates_in_ranking(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.trial(root,'displayed',1.);self.trial(root,'slower',2.)
            result=retain_best(root,'fixed-v1',protected=['displayed'],apply=True)
            self.assertEqual(result['best'],'displayed')
            self.assertEqual(result['removed'],['slower'])

    def test_metadata_write_failure_does_not_delete_anything(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.trial(root,'best',1.);self.trial(root,'slow',2.)
            with patch('initialize.result_retention._atomic_summary',side_effect=OSError('disk full')):
                with self.assertRaises(OSError):retain_best(root,'fixed-v1',apply=True)
            self.assertTrue((root/'slow').exists())

    def test_protocol_protected_running_and_unknown_are_retained(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            self.trial(root,'best',3.)
            self.trial(root,'foreign',1.,protocol='other-v2')
            self.trial(root,'fallback',8.)
            self.trial(root,'active',1.,state='running')
            self.trial(root,'active_flag',1.,active=True)
            (root/'recording').mkdir()
            result=retain_best(root,'fixed-v1',protected=['fallback'],apply=True)
            self.assertEqual(result['best'],'best')
            self.assertEqual(result['removed'],[])
            self.assertEqual(result['kept']['foreign'],'other_protocol')
            self.assertEqual(result['kept']['fallback'],'protected')
            self.assertEqual(result['skipped']['recording'],'unmarked')

    def test_nested_mount_is_rejected_and_not_walked(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.trial(root,'best',1.);other=self.trial(root,'slow',2.)
            native=os.path.ismount
            with patch('initialize.result_retention.os.path.ismount',side_effect=lambda p:Path(p)==other or native(p)):
                result=retain_best(root,'fixed-v1',apply=True)
            self.assertIn('slow',result['skipped']);self.assertTrue(other.exists())

    def test_summary_written_before_removal(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.trial(root,'best',1.);self.trial(root,'slow',2.)
            import shutil
            native=shutil.rmtree
            def remove(path):
                summary=json.loads((root/SUMMARY).read_text())
                self.assertEqual(summary['last_action']['planned_removals'],['slow'])
                self.assertEqual(len(summary['records']),2)
                native(path)
            with patch('initialize.result_retention.shutil.rmtree',side_effect=remove):
                retain_best(root,'fixed-v1',apply=True)

    def test_protected_paths_cannot_escape_root(self):
        with tempfile.TemporaryDirectory() as d:
            for value in ('../outside','nested/child','nested\\child','..'):
                with self.assertRaises(ValueError):retain_best(d,'v1',protected=[value],apply=True)

    def test_marker_symlink_or_payload_symlink_never_deleted(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.trial(root,'best',1.);linked=self.trial(root,'linked',2.)
            target=root/'outside.txt';target.write_text('keep')
            try:(linked/'outside').symlink_to(target)
            except OSError:self.skipTest('Symlink creation unavailable')
            result=retain_best(root,'fixed-v1',apply=True)
            self.assertIn('linked',result['skipped']);self.assertTrue(linked.exists())
            self.assertEqual(target.read_text(),'keep')

    def test_symlink_root_and_summary_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);actual=root/'actual';actual.mkdir()
            try:(root/'linked').symlink_to(actual,target_is_directory=True)
            except OSError:self.skipTest('Symlink creation unavailable')
            with self.assertRaises(ValueError):retain_best(root/'linked','v1',apply=True)
            (actual/SUMMARY).symlink_to(root/'external_summary.json')
            with self.assertRaises(ValueError):retain_best(actual,'v1',apply=True)

    def test_changed_trial_after_summary_is_not_removed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.trial(root,'best',1.);self.trial(root,'slow',2.)
            from initialize import result_retention as module
            native=module._atomic_summary
            def change(*args):
                native(*args)
                (root/'slow/gaussians.npz').write_bytes(b'changed')
            with patch.object(module,'_atomic_summary',side_effect=change):
                with self.assertRaisesRegex(RuntimeError,'stopped'):
                    retain_best(root,'fixed-v1',apply=True)
            self.assertTrue((root/'slow').exists())


if __name__=='__main__':
    unittest.main()
