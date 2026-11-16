"""Stage01: native read-only status and independent small-angle motor tests."""
from .common import (base_report, finish_cli, new_output, parser, require_board,
                     require_motion, run_logged, write_json)


def main(argv=None):
    p = parser(__doc__, motor=True)
    p.add_argument('--action', choices=('status', 'pan', 'tilt', 'both'), default='status')
    a = p.parse_args(argv)
    if a.execute and a.action != 'status':
        require_motion(a)
    out = new_output(a.out)
    report = base_report(1, a)
    actions = ['pan', 'tilt'] if a.action == 'both' else [a.action]
    commands = [[str(a.native.resolve()), 'status' if item == 'status' else 'smoke'] +
                ([] if item == 'status' else [item, '--execute', '--wiring-j17-23-21'])
                for item in actions]
    report.update(commands=commands, action=a.action,
                  pulse_sequence_us=[1500, 1450, 1500, 1550, 1500],
                  note='Dry-run commands are descriptions only. Register readback is not a measured mechanical angle.')
    try:
        if a.execute:
            require_board()
            for item, command in zip(actions, commands):
                run_logged(command, out / (item + '.log'), timeout=20)
            report['complete'] = True
        print('Stage01: ' + ('command sequence finished; observe physical movement' if a.execute else 'DRY RUN; no device opened'))
    except BaseException as error:
        report['error'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        write_json(out / 'report.json', report)


if __name__ == '__main__':
    finish_cli(main)
