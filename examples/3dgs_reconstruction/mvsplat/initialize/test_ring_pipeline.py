"""Ring coordination must render validated arrays before diagnostic archival."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from common import gaussian_content_sha256, sha
from initialize.ring import prepare_ring_scene


class RingPipelineTests(unittest.TestCase):
    def scenario(self, directory, bad_frame=False):
        arrays = dict(means=np.array([[0., 0., 2.]], np.float32),
                      covariances=np.eye(3, dtype=np.float32)[None]*.01,
                      harmonics=np.zeros((1, 3, 25), np.float32),
                      opacities=np.full(1, .5, np.float32))
        prepared = dict(pairs=[{}, {}, {}], timing={'decode_seconds':1.}, coverage={},
                        quality_scope='fixed-centre test fixture', views=[dict(role='target', name='target',
                        width=128, height=128, c2w=np.eye(4).tolist(),
                        intrinsics_normalized=[[1.,0.,.5],[0.,1.,.5],[0.,0.,1.]])])
        inference = dict(gaussian_sha256=gaussian_content_sha256(arrays),
                         gaussian_identity_kind='array_content_v1')
        order = []
        class Events:
            def emit(self, event, **fields):
                order.append(event)
        class Renderer:
            def load_rows(self, rows):
                self.rows = rows.copy()
                order.append('load_rows')
                return dict(gaussians=len(rows))
            def render_camera(self, camera, out):
                self.camera = camera
                Path(out).mkdir(parents=True)
                (Path(out)/'frame.bin').write_bytes(b'actual renderer frame fixture')
                order.append('render_camera')
                return dict(complete=True, frame_sha256='wrong' if bad_frame else sha(Path(out)/'frame.bin'))
        renderer = Renderer()
        def archive(record, gaussians, out):
            self.assertIn('FRAME_COMPLETE', order)
            self.assertEqual(record['gaussian_sha256'], gaussian_content_sha256(gaussians))
            order.append('archive')
        def export(*args, **kwargs):
            self.assertEqual(order[-1], 'archive')
            self.assertIs(kwargs['gaussians'], arrays)
            order.append('export')
        return arrays, prepared, inference, order, Events(), renderer, archive, export

    def test_auto_pair_count_and_first_frame_precede_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            g, prepared, inference, order, events, renderer, archive, export = self.scenario(directory)
            with patch('video_input.ring.prepare_ring', return_value=prepared) as prepare, \
                 patch('gaussian_generation.ring.generate_ring', return_value=(inference,g)) as generate, \
                 patch('gaussian_generation.ring.persist_ring_archive', side_effect=archive), \
                 patch('gaussian_generation.adapter.main', side_effect=export):
                result=prepare_ring_scene('video', directory, object(), renderer, events,
                                          groups='auto', max_pairs=8, min_pairs=8,
                                          coverage_target=.95, feature_contrast=.01)
            self.assertTrue(generate.call_args.kwargs['defer_archive'])
            self.assertEqual(prepare.call_args.kwargs['groups'], 'auto')
            self.assertEqual(prepare.call_args.kwargs['max_pairs'], 8)
            self.assertEqual(prepare.call_args.kwargs['min_pairs'], 8)
            self.assertEqual(prepare.call_args.kwargs['feature_contrast'], .01)
            self.assertEqual(order, ['load_rows','SCENE_READY','render_camera','FRAME_COMPLETE','archive','export'])
            self.assertTrue(result['frame']['complete'])
            self.assertEqual(renderer.rows.shape,(1,62))

    def test_bad_first_frame_cannot_publish_completion_or_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            g, prepared, inference, order, events, renderer, archive, export = self.scenario(directory,True)
            with patch('video_input.ring.prepare_ring', return_value=prepared), \
                 patch('gaussian_generation.ring.generate_ring', return_value=(inference,g)), \
                 patch('gaussian_generation.ring.persist_ring_archive', side_effect=archive) as persist, \
                 patch('gaussian_generation.adapter.main', side_effect=export) as adapter:
                with self.assertRaisesRegex(ValueError,'Incomplete'):
                    prepare_ring_scene('video', directory, object(), renderer, events)
            self.assertNotIn('FRAME_COMPLETE', order)
            persist.assert_not_called()
            adapter.assert_not_called()


if __name__=='__main__':
    unittest.main()
