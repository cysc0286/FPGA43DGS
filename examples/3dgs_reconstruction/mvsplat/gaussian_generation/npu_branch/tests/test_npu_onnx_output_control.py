"""Tiny ONNX tail controls: no full MVSplat inference or hardware access."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

from common import save, sha
from gaussian_generation.npu_branch.onnx_output_control import change_output_tail, prepare, verify


def toy():
    x = helper.make_tensor_value_info('features',TensorProto.FLOAT,[1,2,3,5])
    y = helper.make_tensor_value_info('result',TensorProto.FLOAT,[1,3,5,2])
    graph = helper.make_graph([
        helper.make_node('Identity',['features'],['base']),
        helper.make_node('Transpose',['base'],['result'],perm=[0,2,3,1])],
        'small_layout_example',[x],[y])
    model = helper.make_model(graph,opset_imports=[helper.make_opsetid('',17)])
    model.ir_version = 9
    return model


def run(model, x):
    session=ort.InferenceSession(model.SerializeToString(),providers=['CPUExecutionProvider'])
    return session.run(None,{'features':x})[0]


class OutputControlTests(unittest.TestCase):
    def test_remove_only_final_transpose_preserves_model(self):
        model=toy();before=model.SerializeToString()
        candidate,change=change_output_tail(model,[1,2,3,5])
        x=np.arange(30,dtype=np.float32).reshape(1,2,3,5)
        self.assertTrue(np.array_equal(run(candidate,x),x))
        self.assertEqual(model.SerializeToString(),before)
        self.assertEqual(len(candidate.graph.node),1)
        self.assertEqual(change['onnx_output_shape'],[1,2,3,5])
        self.assertFalse(change['compiled_layout_modified'])

    def test_identity_conv_nchw_is_exact_and_has_actual_weights(self):
        model,change=change_output_tail(toy(),[1,2,3,5],identity_conv=True)
        x=np.arange(30,dtype=np.float32).reshape(1,2,3,5)/17
        self.assertTrue(np.array_equal(run(model,x),x))
        self.assertEqual(model.graph.node[-1].op_type,'Conv')
        self.assertEqual(len(model.graph.initializer),2)
        self.assertEqual(change['identity_conv_multiplications'],60)

    def test_identity_conv_nhwc_is_exact(self):
        model,change=change_output_tail(toy(),[1,2,3,5],identity_conv=True,output_layout='NHWC')
        x=np.arange(30,dtype=np.float32).reshape(1,2,3,5)/19
        self.assertTrue(np.array_equal(run(model,x),x.transpose(0,2,3,1)))
        self.assertEqual(change['onnx_output_shape'],[1,3,5,2])
        self.assertEqual([n.op_type for n in model.graph.node],['Identity','Conv','Transpose'])

    def test_reject_wrong_tail_permutation(self):
        model=toy()
        model.graph.node[-1].attribute[0].ints[:] = [0,3,1,2]
        with self.assertRaisesRegex(ValueError,'exact terminal'):
            change_output_tail(model,[1,2,3,5])

    def test_reject_shape_and_nonstatic_output(self):
        with self.assertRaisesRegex(ValueError,'shape/dtype'):
            change_output_tail(toy(),[1,4,3,5])
        with self.assertRaisesRegex(ValueError,'Static positive'):
            change_output_tail(toy(),[1,2,0,5])

    def test_prepare_stays_incomplete_until_independent_ort_verification(self):
        with tempfile.TemporaryDirectory() as t:
            source=Path(t)/'source';folder=source/'refine_unet';folder.mkdir(parents=True)
            onnx.save(toy(),str(folder/'model.onnx'))
            x=np.arange(30,dtype=np.float32).reshape(1,2,3,5)/13
            np.savez(folder/'oracle.npz',input=x,expected=x)
            save(source/'manifest.json',dict(schema='mvsplat_partitions_v1',weight_sha256='fixture-only',
                input_shape=[1,2,3,5],partitions={'refine_unet':dict(complete=True,host_onnx_passed=True,
                onnx_output_layout='NHWC',output_shape=[1,2,3,5],input_shape=[1,2,3,5],parameters=0,
                graph_sha256=sha(folder/'model.onnx'),oracle_sha256=sha(folder/'oracle.npz'))}))
            target=Path(t)/'candidate'
            prepared=prepare(source,'refine_unet',target,identity_conv=True,output_layout='NHWC')
            item=prepared['partitions']['refine_unet']
            self.assertFalse(item['complete'])
            self.assertFalse(item['host_onnx_passed'])
            self.assertEqual(item['parameters'],6)
            self.assertEqual(item['oracle_sha256'],sha(folder/'oracle.npz'))
            checked=verify(target,'refine_unet')['partitions']['refine_unet']
            self.assertTrue(checked['complete'])
            self.assertTrue(checked['host_onnx_check']['exact'])
            self.assertFalse(checked['npu_executed'])
            self.assertEqual(json.loads((target/'manifest.json').read_text())['partitions']['refine_unet'],checked)


if __name__=='__main__':
    unittest.main()
