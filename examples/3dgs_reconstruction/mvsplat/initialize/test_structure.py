"""The six owners must form a deployable pipeline without old stage aliases."""
from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


MVS = Path(__file__).resolve().parents[1]


class StructureTests(unittest.TestCase):
    def test_source_package_keeps_build_scripts_but_skips_build_directories(self):
        tool = MVS.parents[2]/'tools/package_core.py'
        spec = importlib.util.spec_from_file_location('core_package_test', tool)
        package = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(package)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = ['examples/rendering/build_resident.sh',
                     'examples/rendering/build_hls.ps1',
                     'examples/rendering/build_cache/generated.cpp']
            for relative in files:
                path = root/relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('source fixture\n', encoding='utf-8')
            (root/'tools').mkdir()
            (root/'tools/core_sources.json').write_text(json.dumps(dict(
                files=[], trees=['examples'], skip_directories=[],
                extensions=['.sh', '.ps1', '.cpp'])), encoding='utf-8')
            with patch.object(package, 'ROOT', root), patch.object(package.subprocess,
                    'check_output', return_value=('\0'.join(files)+'\0').encode()):
                packed = {p.relative_to(root).as_posix() for p in package.source_files()}
            self.assertEqual(packed, set(files[:2]))

    def test_public_help_and_dispatch_do_not_load_weights(self):
        sys.path.insert(0, str(MVS.parent))
        import pipeline
        for command, target in (('initialize', 'initialize.run.main'),
                                ('web', 'web.camera_ui.main'),
                                ('view', 'web.viewer.main')):
            with self.subTest(command=command), patch(target) as entry:
                pipeline.main([command, '--out', 'example'])
                entry.assert_called_once_with(['--out', 'example'])
        result = subprocess.run([sys.executable, str(MVS.parent/'pipeline.py'), '--help'],
                                capture_output=True, text=True, check=True)
        self.assertIn('{initialize,web,view}', result.stdout)
        for command in ('reconstruct', 'gaussian', 'export', 'pose'):
            result = subprocess.run([sys.executable, str(MVS.parent/'pipeline.py'), command],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, command)

    def test_deployment_contains_both_web_entries_and_no_missing_files(self):
        from initialize.board_sync import SOURCE_FILES
        for relative in SOURCE_FILES:
            with self.subTest(relative=relative):
                self.assertTrue((MVS/relative).is_file())
        for relative in ('web/camera_ui.py', 'web/recording_server.py', 'web/viewer.py',
                         'web/viewer.html', 'web/viewer.css', 'web/viewer.js',
                         'web/recording_page.html', 'web/upload_page.html',
                         'initialize/measure.py', 'gaussian_generation/memory.py',
                         'gaussian_generation/upstream.py', 'rendering/contracts.py'):
            self.assertIn(relative, SOURCE_FILES)
        self.assertFalse(any('/tests/' in name or name.startswith('test_') for name in SOURCE_FILES))
        self.assertEqual({name for name in SOURCE_FILES if name.endswith('.json')},
                         {'rendering/mainline.json', 'rendering/acceptance.json'})

    def test_video_input_does_not_need_html_or_http_to_import(self):
        script = ("import sys; import video_input.recording, video_input.upload; "
                  "assert 'web.recording_server' not in sys.modules; "
                  "assert 'web.viewer' not in sys.modules")
        subprocess.run([sys.executable, '-c', script], cwd=MVS, check=True)

    def test_render_camera_math_does_not_import_http(self):
        script = ("import sys; from rendering.camera import camera_view; "
                  "assert 'web.viewer' not in sys.modules; "
                  "assert 'http.server' not in sys.modules")
        subprocess.run([sys.executable, '-c', script], cwd=MVS, check=True)


if __name__ == '__main__':
    unittest.main()
