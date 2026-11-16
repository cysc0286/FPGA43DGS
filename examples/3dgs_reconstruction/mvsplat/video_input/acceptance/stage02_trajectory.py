"""Stage02: generate and independently execute the production bow trajectory."""
from pathlib import Path
from video_input.camera_scan import write_plan
from .common import (DEFAULT_PROFILE, base_report, finish_cli, new_output, parser,
                     require_board, require_motion, run_logged, scan_command, write_json)


def main(argv=None):
    p = parser(__doc__, motor=True)
    p.add_argument('--profile', type=Path, default=DEFAULT_PROFILE)
    a = p.parse_args(argv)
    if a.execute:
        require_motion(a)
    out = new_output(a.out)
    report = base_report(2, a)
    try:
        plan = out / 'trajectory.csv'
        summary = write_plan(a.profile, plan)
        command = scan_command(a.native, plan, out / 'motion.csv')
        report.update(plan=summary, command=command,
                      note='capture events in the CSV are position markers; stage02 never starts recording.')
        if a.execute:
            require_board()
            run_logged(command, out / 'native.log', timeout=summary['nominal_duration_seconds'] * 1.3 + 10)
            report['complete'] = True
        print('Stage02: %s; planned %.2f seconds' %
              ('motion finished' if a.execute else 'DRY RUN', summary['nominal_duration_seconds']))
    except BaseException as error:
        report['error'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        write_json(out / 'report.json', report)


if __name__ == '__main__':
    finish_cli(main)
