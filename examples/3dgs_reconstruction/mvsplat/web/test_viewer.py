"""Loopback HTTP and camera ABI checks; fake renderer never accesses FPGA."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np

from rendering.runtime import LiveFrame
from web.viewer import camera_view, make_server, orbit_view, verify_saved_scene, viewer_metadata


BASE = dict(c2w=np.eye(4).tolist(), intrinsics_normalized=[[1, 0, .5], [0, 1, .5], [0, 0, 1]],
            width=128, height=128)


class ViewerTests(unittest.TestCase):
    def test_actual_viewer_finishes_pixel_handoff_without_a_visible_animation_frame(self):
        node=shutil.which('node')
        if not node:self.skipTest('Node unavailable for actual viewer-script execution')
        program=r'''
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
let drawn=0,closed=0,animationRequests=0,failRender=false;const elements=new Map();
const graphics=new Proxy({drawImage(){drawn++;}},{get:(o,k)=>o[k]||(()=>{})});
const element=id=>{if(!elements.has(id))elements.set(id,{width:480,height:480,options:[],
  classList:{toggle(){}},getContext:()=>graphics,addEventListener(){},setPointerCapture(){}});return elements.get(id);};
const win={addEventListener(){}};win.parent=win;
const context=vm.createContext({document:{getElementById:element},window:win,
  location:{hash:'#token=test'},URLSearchParams,performance:{now:()=>100},
  requestAnimationFrame(){animationRequests++;},setTimeout(){},
  createImageBitmap:async()=>({width:128,height:128,close(){closed++;}}),
  fetch:async path=>path==='/render'?(failRender?{ok:false,text:async()=>'test request failed'}:
    {ok:true,headers:{get:k=>({'X-Request-Id':'0','X-Render-Ms':'52.0','X-Scene-Version':'1'})[k]},
      blob:async()=>({})}):new Promise(()=>{})});
vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),context,{timeout:1000});
vm.runInContext('scene={base:{},focus_distance:1}',context);
(async()=>{let complete=false;const operation=context.draw().then(()=>complete=true);
  for(let i=0;i<20;i++)await Promise.resolve();
  assert.equal(drawn,1);assert.equal(closed,1);
  assert.equal(complete,true,'offscreen rAF must not block an already copied canvas frame');
  assert.equal(element('confirm').disabled,false);
  assert.equal(vm.runInContext('busy',context),false);
  assert.equal(vm.runInContext('hasFrame',context),true);
  assert.match(element('status').textContent,/画面已就绪/);
  assert.equal(animationRequests,0);await operation;
  failRender=true;let failedComplete=false;
  const failed=context.draw().then(()=>failedComplete=true);
  for(let i=0;i<20;i++)await Promise.resolve();
  assert.equal(failedComplete,true);assert.equal(element('confirm').disabled,false);
  assert.equal(vm.runInContext('busy',context),false);
  assert.equal(vm.runInContext('hasFrame',context),true);
  assert.equal(drawn,1);assert.equal(closed,1);
  assert.match(element('status').textContent,/test request failed/);
  assert.equal(animationRequests,0);await failed;
})().catch(e=>{console.error(e);process.exitCode=1;});
'''
        result=subprocess.run([node,'-e',program,str(Path(__file__).with_name('viewer.js'))],
                              capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_saved_scene_requires_model_gaussian_and_context_hashes(self):
        from common import sha
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('renderer_input','inference','input'):(root/name).mkdir()
            model=root/'renderer_input/model.ply';model.write_bytes(b'model')
            gaussians=root/'inference/gaussians.npz';gaussians.write_bytes(b'gaussians')
            context=root/'input/context.npz';context.write_bytes(b'context')
            (root/'renderer_input/manifest.json').write_text(json.dumps(dict(model_sha256=sha(model))))
            (root/'inference/inference.json').write_text(json.dumps(dict(complete=True,gaussian_sha256=sha(gaussians),input_sha256=sha(context))))
            (root/'input/input.json').write_text(json.dumps(dict(context_sha256=sha(context))))
            self.assertEqual(verify_saved_scene(root),root.resolve())
            for path in (model,gaussians,context):
                original=path.read_bytes();path.write_bytes(b'tampered')
                with self.assertRaisesRegex(ValueError,'validation'):verify_saved_scene(root)
                path.write_bytes(original)

    def test_orbit_identity_and_proper_camera_axes(self):
        np.testing.assert_allclose(orbit_view(BASE, {}, 10)["c2w"], np.eye(4), atol=1e-12)
        for yaw, pitch in ((30, -20), (-30, 20), (180, 85), (-180, -85), (90, 0)):
            view = np.array(orbit_view(BASE, dict(yaw=yaw, pitch=pitch), 10)["c2w"])
            rotation = view[:3, :3]
            np.testing.assert_allclose(rotation.T@rotation, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(rotation), 1)
            np.testing.assert_allclose(view[:3, 3]+10*rotation[:, 2], [0, 0, 10], atol=1e-12)
        for controls in ({"distance": -1}, {"yaw": float("nan")}, {"request_id": True}, {"bad": 1}):
            with self.assertRaises(ValueError):
                orbit_view(BASE, controls, 10)

    def test_look_turns_a_full_circle_without_moving_the_declared_centre(self):
        center=[4.,-3.,2.]
        for yaw in (-180,-90,0,90,180):
            for pitch in (-85,0,85):
                view=np.array(camera_view(BASE,dict(mode='look',yaw=yaw,pitch=pitch),10,
                                          capture_center=center)['c2w'])
                np.testing.assert_allclose(view[:3,3],center,atol=1e-12)
                np.testing.assert_allclose(view[:3,:3].T@view[:3,:3],np.eye(3),atol=1e-12)
                self.assertAlmostEqual(np.linalg.det(view[:3,:3]),1)
        turned=np.array(camera_view(BASE,dict(mode='look',yaw=90),10,capture_center=center)['c2w'])
        np.testing.assert_allclose(turned[:3,2],[1,0,0],atol=1e-12)
        reference=np.eye(4)
        reference[:3,:3]=[[0,0,1],[0,1,0],[-1,0,0]]
        reference[:3,3]=[500,600,700]
        relative=np.array(camera_view(dict(BASE,c2w=reference.tolist()),dict(mode='look',yaw=90),
                                      10,capture_center=center)['c2w'])
        np.testing.assert_allclose(relative[:3,2],reference[:3,0],atol=1e-12)
        np.testing.assert_allclose(relative[:3,3],center,atol=1e-12)
        for controls in ({'mode':'look'},{'mode':'orbit','yaw':181},{'mode':'bad'}):
            with self.assertRaises(ValueError):camera_view(BASE,controls,10)
        for controls in ({'distance':1},{'distance':1.1},{'pan_x':0},{'pan_x':.1},{'pan_y':-.1}):
            with self.assertRaisesRegex(ValueError,'translation'):
                camera_view(BASE,dict(mode='look',**controls),10,capture_center=center)

    def test_fixed_centre_is_explicit_and_coverage_metadata_is_bounded(self):
        self.assertEqual(viewer_metadata()['camera_controls']['modes'],['orbit'])
        selected=viewer_metadata(dict(capture_center=[5,6,7],default_view_mode='look',
            depth_observable=False,coverage=dict(directions=[[0,0,1],[1,0,0]],
            crop_horizontal_fov_degrees=55,observed_yaw_span_degrees=280,pair_groups=2)))
        self.assertEqual(selected['capture_center'],[5.,6.,7.])
        self.assertEqual(selected['camera_controls']['default_mode'],'look')
        self.assertEqual(selected['camera_controls']['modes'],['look'])
        self.assertEqual(viewer_metadata(dict(capture_center=[0,0,0],default_view_mode='look',
                                             depth_observable=True))['camera_controls']['modes'],['look','orbit'])
        self.assertFalse(selected['depth_observable'])
        for metadata in (dict(default_view_mode='look'),dict(capture_center=[0,0,float('nan')]),
                         dict(coverage=dict(directions=[[0,0,0]])),
                         dict(coverage=dict(directions=[[0,0,1]],pair_groups=True))):
            with self.assertRaises(ValueError):viewer_metadata(metadata)

    def test_horizontal_coverage_metrics_are_bounded_and_passed_without_3d_claims(self):
        coverage=dict(sample_covered_fraction=.75,coverage_target=.95,target_met=False,
                      achieved_fraction=.75,max_continuous_gap_degrees=90,
                      covered_yaw_sample_degrees=list(range(270)))
        exposed=viewer_metadata(dict(coverage=coverage))['coverage']
        self.assertEqual(exposed['sample_covered_fraction'],.75)
        self.assertEqual(exposed['coverage_target'],.95);self.assertFalse(exposed['target_met'])
        self.assertEqual(exposed['max_continuous_gap_degrees'],90)
        self.assertEqual(exposed['covered_yaw_sample_degrees'],list(range(270)))
        for field,value in (('sample_covered_fraction',1.1),('coverage_target',float('nan')),
                            ('target_met','true'),('max_continuous_gap_degrees',361),
                            ('covered_yaw_sample_degrees',[1,1]),('covered_yaw_sample_degrees',[360])):
            with self.subTest(field=field),self.assertRaises(ValueError):
                viewer_metadata(dict(coverage=dict(coverage,**{field:value})))

    def test_live_http_pixels_authentication_errors_and_quit(self):
        class FakeRenderer:
            calls = []
            def render_camera(self, camera):
                self.calls.append(camera)
                image = np.zeros((128, 128, 3), np.uint8)
                image[:, :, 0] = 230
                return LiveFrame(image, dict(wall_ms=1.25))
        renderer = FakeRenderer()
        server = make_server(renderer, BASE, 10, port=0)
        def run():
            while not server.stopping:
                server.handle_request()
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}"
        def post(path, payload, token=None):
            request = Request(url+path, data=json.dumps(payload).encode(), headers={
                "Authorization": "Bearer "+(token or server.session_token), "Content-Type": "application/json"})
            return urlopen(request, timeout=5)
        try:
            with urlopen(url, timeout=5) as response:
                self.assertIn('id="confirm"', response.read().decode())
            with urlopen(url+'/viewer.js', timeout=5) as response:
                self.assertIn("createImageBitmap", response.read().decode())
            with self.assertRaises(HTTPError) as denied:
                urlopen(url+'/scene',timeout=5)
            self.assertEqual(denied.exception.code,403)
            with urlopen(Request(url+'/scene',headers={'Authorization':'Bearer '+server.session_token}),timeout=5) as response:
                selected=json.load(response)
                self.assertEqual(selected['base'],BASE)
                self.assertEqual(selected['focus_distance'],10)
            with self.assertRaises(HTTPError) as failed:
                post("/render", {}, "bad")
            self.assertEqual(failed.exception.code, 403)
            with self.assertRaises(HTTPError) as failed:
                post("/render", {"yaw": 181})
            self.assertEqual(failed.exception.code, 400)
            with post("/render", {"request_id": 7, "yaw": 10}) as response:
                pixels = cv2.imdecode(np.frombuffer(response.read(), np.uint8), cv2.IMREAD_COLOR)
                self.assertEqual(response.headers["X-Request-Id"], "7")
                self.assertEqual(pixels[0, 0].tolist(), [0, 0, 230])
            self.assertEqual(len(renderer.calls), 1)
            self.assertEqual((len(renderer.calls[0]), renderer.calls[0][:8]), (136, b"FLCAM001"))
            with post("/quit", {}) as response:
                self.assertEqual(response.status, 200)
            thread.join(5)
            self.assertFalse(thread.is_alive())
        finally:
            server.stopping = True
            thread.join(2)
            server.server_close()

    def test_http_look_camera_abi_keeps_manifest_anchor_at_large_yaw(self):
        import struct
        class FakeRenderer:
            camera=None
            def render_camera(self,camera):
                self.camera=camera
                return LiveFrame(np.zeros((128,128,3),np.uint8),dict(wall_ms=1.))
        renderer=FakeRenderer()
        server=make_server(renderer,BASE,10,port=0,points=[[100,200,300]],
            view_metadata=dict(capture_center=[2,3,4],default_view_mode='look',depth_observable=False))
        thread=threading.Thread(target=lambda:[server.handle_request() for _ in range(2)],daemon=True)
        thread.start()
        try:
            url=f'http://127.0.0.1:{server.server_port}'
            headers={'Authorization':'Bearer '+server.session_token}
            with urlopen(Request(url+'/scene',headers=headers),timeout=5) as response:
                scene=json.load(response)
            self.assertEqual(scene['default_view_mode'],'look')
            self.assertEqual(scene['capture_center'],[2,3,4])
            request=Request(url+'/render',headers=headers,data=b'{"yaw":180,"pitch":45,"request_id":1}')
            with urlopen(request,timeout=5) as response:self.assertEqual(response.status,200)
            values=struct.unpack('<4I14d',renderer.camera[8:])
            np.testing.assert_allclose(values[13:16],[2,3,4])
            np.testing.assert_allclose(np.array(values[4:13]).reshape(3,3)[:,2],[0,-2**-.5,-2**-.5],atol=1e-12)
            thread.join(5)
        finally:
            server.server_close()

    def test_renderer_timeout_stops_session_instead_of_reusing_a_late_reply(self):
        class FailingRenderer:
            def render_camera(self, camera):
                raise TimeoutError("test delayed native reply")
        server = make_server(FailingRenderer(), BASE, 10, port=0)
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()
        try:
            request = Request(f"http://127.0.0.1:{server.server_port}/render", data=b"{}",
                              headers={"Authorization": "Bearer "+server.session_token})
            with self.assertRaises(HTTPError) as error:
                urlopen(request, timeout=5)
            self.assertEqual(error.exception.code, 500)
            thread.join(5)
            self.assertTrue(server.stopping)
            self.assertIn("TimeoutError", server.stats["fatal_error"])
        finally:
            server.server_close()

    def test_rotation_only_http_rejects_orbit_and_translation_before_fpga_access(self):
        class FakeRenderer:
            calls=0
            def render_camera(self,camera):
                self.calls+=1
                return LiveFrame(np.zeros((128,128,3),np.uint8),dict(wall_ms=1.))
        renderer=FakeRenderer()
        server=make_server(renderer,BASE,10,port=0,
            view_metadata=dict(capture_center=[0,0,0],default_view_mode='look',depth_observable=False))
        def run():
            while not server.stopping:server.handle_request()
        thread=threading.Thread(target=run,daemon=True);thread.start()
        url=f'http://127.0.0.1:{server.server_port}'
        def post(payload):
            return urlopen(Request(url+'/render',data=json.dumps(payload).encode(),
                headers={'Authorization':'Bearer '+server.session_token}),timeout=5)
        try:
            for controls in ({'mode':'orbit'},{'mode':'orbit','distance':1},
                             {'distance':1},{'pan_x':0},{'pan_y':0}):
                with self.assertRaises(HTTPError) as error:post(controls)
                self.assertEqual(error.exception.code,400)
            self.assertEqual(renderer.calls,0)
            self.assertFalse(server.stopping)
            with post(dict(yaw=180,pitch=85,request_id=4)) as response:
                self.assertEqual(response.status,200)
            self.assertEqual(renderer.calls,1)
            self.assertNotIn('fatal_error',server.stats)
        finally:
            server.stopping=True;thread.join(2);server.server_close()


if __name__ == "__main__":
    unittest.main()
