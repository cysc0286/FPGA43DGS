"""Public entry for the resident video-to-Gaussian-to-FPGA pipeline and web UI."""
import argparse
from pathlib import Path
import sys

MODULE_ROOT = Path(__file__).resolve().parent / 'mvsplat'

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ('initialize', 'web', 'view'):
        sys.path.insert(0, str(MODULE_ROOT))
        if argv[0] == 'initialize':
            from initialize.run import main as entry
        elif argv[0] == 'web':
            from web.camera_ui import main as entry
        else:
            from web.viewer import main as entry
        return entry(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('initialize', help='Prewarm, receive video and render the first frame')
    commands.add_parser('web', help='Capture/upload web UI over the same resident pipeline')
    commands.add_parser('view', help='Reopen an existing validated scene in the web viewer')
    parser.parse_args(argv)

if __name__ == '__main__':
    main()
