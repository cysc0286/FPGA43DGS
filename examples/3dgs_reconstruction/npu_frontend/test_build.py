import tempfile
import unittest
from pathlib import Path
from build_runner import sdk_paths


class SDKLayout(unittest.TestCase):
    def test_extracted_debian_sdk(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            header = root/"usr/include/icraft-xrt/core/session.h"
            header.parent.mkdir(parents=True)
            header.touch()
            lib = root/"usr/lib/aarch64-linux-gnu/libicraft_xrt.so"
            lib.parent.mkdir(parents=True)
            lib.touch()
            self.assertEqual(sdk_paths(root), (root/"usr/include", lib.parent))
            self.assertEqual(sdk_paths(root/"usr"), (root/"usr/include", lib.parent))

    def test_headers_only_require_syntax_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            header = root/"include/icraft-xrt/core/session.h"
            header.parent.mkdir(parents=True)
            header.touch()
            self.assertEqual(sdk_paths(root, False)[0], root/"include")
            with self.assertRaises(ValueError):
                sdk_paths(root)


if __name__ == "__main__":
    unittest.main(verbosity=2)
