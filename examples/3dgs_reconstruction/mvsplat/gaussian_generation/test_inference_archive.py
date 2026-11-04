"""Array-content/NPZ-file identity contracts using tiny persisted Gaussian scenes."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from common import gaussian_content_sha256, save, sha
from gaussian_generation.adapter import main as export, validate_inference_archive
from web.viewer import verify_saved_scene


class InferenceArchiveIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.inference=self.root/'inference';self.inference.mkdir()
        self.g=dict(means=np.array([[0.,0.,2.]],np.float32),
            covariances=np.eye(3,dtype=np.float32)[None]*.01,
            harmonics=np.zeros((1,3,25),np.float32),opacities=np.array([.5],np.float32))
        np.savez(self.inference/'gaussians.npz',**self.g)
        self.content=gaussian_content_sha256(self.g)
        self.record=dict(complete=True,gaussian_identity_kind='array_content_v1',
            gaussian_sha256=self.content,gaussian_content_sha256=self.content,
            gaussian_archive_sha256=sha(self.inference/'gaussians.npz'),archive_verified=True)

    def test_legacy_and_explicit_npz_identity_remain_compatible(self):
        for kind in (None,'npz_file_v1'):
            record=dict(complete=True,gaussian_sha256=sha(self.inference/'gaussians.npz'))
            if kind is not None:record['gaussian_identity_kind']=kind
            self.assertIsNone(validate_inference_archive(self.inference,record))

    def test_array_identity_verifies_file_and_original_array_contents(self):
        result=validate_inference_archive(self.inference,self.record)
        for key in self.g:np.testing.assert_array_equal(result[key],self.g[key])
        self.assertIsNone(validate_inference_archive(self.inference,self.record,gaussians=self.g))
        self.assertNotEqual(self.record['gaussian_sha256'],self.record['gaussian_archive_sha256'])

    def test_unknown_kind_missing_fields_and_unverified_archive_are_rejected(self):
        for kind in ('','unknown',None,123):
            with self.subTest(kind=kind),self.assertRaisesRegex(ValueError,'identity kind'):
                validate_inference_archive(self.inference,dict(self.record,gaussian_identity_kind=kind))
        for field in ('gaussian_sha256','gaussian_archive_sha256','archive_verified'):
            record=dict(self.record);record.pop(field)
            with self.subTest(field=field),self.assertRaises(ValueError):
                validate_inference_archive(self.inference,record)
        for verified in (False,1,'true',None):
            with self.subTest(verified=verified),self.assertRaisesRegex(ValueError,'not verified'):
                validate_inference_archive(self.inference,dict(self.record,archive_verified=verified))

    def test_archive_byte_tampering_and_rehashing_changed_arrays_are_both_rejected(self):
        archive=self.inference/'gaussians.npz'
        archive.write_bytes(archive.read_bytes()+b'changed file bytes')
        with self.assertRaisesRegex(ValueError,'file identity'):
            validate_inference_archive(self.inference,self.record)
        changed={key:value.copy() for key,value in self.g.items()};changed['means'][0,0]=.25
        np.savez(archive,**changed)
        record=dict(self.record,gaussian_archive_sha256=sha(archive))
        with self.assertRaisesRegex(ValueError,'content identity'):
            validate_inference_archive(self.inference,record)

    def test_content_alias_and_in_memory_array_tampering_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'fields disagree'):
            validate_inference_archive(self.inference,dict(self.record,gaussian_content_sha256='0'*64))
        changed={key:value.copy() for key,value in self.g.items()};changed['means'][0,0]=.25
        with self.assertRaisesRegex(ValueError,'In-memory Gaussian content'):
            validate_inference_archive(self.inference,self.record,gaussians=changed)

    def export_scene(self,kind='array_content_v1',gaussians=None):
        prepared=self.root/'input';prepared.mkdir()
        np.savez(prepared/'context.npz',image=np.zeros((1,),np.float32))
        context=sha(prepared/'context.npz')
        view=dict(name='frame_000001.png',index=1,role='target',c2w=np.eye(4).tolist(),
            intrinsics_normalized=[[1,0,.5],[0,1,.5],[0,0,1]],width=128,height=128)
        save(prepared/'input.json',dict(context_sha256=context,views=[view]))
        record=dict(self.record,input_sha256=context)
        if kind=='npz_file_v1':
            record['gaussian_identity_kind']=kind;record['gaussian_sha256']=record['gaussian_archive_sha256']
        save(self.inference/'inference.json',record)
        out=self.root/'renderer_input'
        with redirect_stdout(io.StringIO()):
            export(['--input',str(prepared),'--inference',str(self.inference),'--out',str(out)],gaussians=gaussians)
        return json.loads((out/'manifest.json').read_text(encoding='utf-8'))

    def test_array_export_and_saved_viewer_bind_both_explicit_identities(self):
        manifest=self.export_scene(gaussians=self.g)
        self.assertEqual(manifest['gaussian_identity_kind'],'array_content_v1')
        self.assertEqual(manifest['gaussian_sha256'],self.content)
        self.assertEqual(manifest['gaussian_archive_sha256'],self.record['gaussian_archive_sha256'])
        self.assertEqual(verify_saved_scene(self.root),self.root.resolve())
        for field in ('gaussian_identity_kind','gaussian_sha256','gaussian_archive_sha256'):
            changed=dict(manifest);changed.pop(field)
            save(self.root/'renderer_input/manifest.json',changed)
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'export identity'):
                verify_saved_scene(self.root)
        save(self.root/'renderer_input/manifest.json',manifest)
        self.record['input_sha256']=manifest['input_sha256']
        save(self.inference/'inference.json',dict(self.record,gaussian_sha256='0'*64))
        with self.assertRaisesRegex(ValueError,'validation failed'):verify_saved_scene(self.root)

    def test_npz_kind_export_and_saved_viewer_keep_old_file_hash_contract(self):
        manifest=self.export_scene(kind='npz_file_v1')
        self.assertEqual(manifest['gaussian_sha256'],sha(self.inference/'gaussians.npz'))
        self.assertEqual(verify_saved_scene(self.root),self.root.resolve())
        for kind in ('unknown',None,'array_content_v1'):
            save(self.root/'renderer_input/manifest.json',dict(manifest,gaussian_identity_kind=kind))
            with self.subTest(kind=kind),self.assertRaisesRegex(ValueError,'export identity'):
                verify_saved_scene(self.root)

    def test_export_rejects_different_memory_before_creating_model(self):
        changed={key:value.copy() for key,value in self.g.items()};changed['means'][0,0]=.5
        with self.assertRaisesRegex(ValueError,'In-memory Gaussian content'):
            self.export_scene(gaussians=changed)
        self.assertFalse((self.root/'renderer_input').exists())


if __name__=='__main__':unittest.main()
