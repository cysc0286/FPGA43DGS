"""Memory limits must include native descendants on kernels without children files."""
from pathlib import Path
import tempfile
import unittest
from measure import tree_rss, tree_memory


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


if __name__=='__main__':
    unittest.main()
