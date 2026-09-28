"""Upload, compile and test over SSH. Uses known_hosts; never stores credentials."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import time

import paramiko

EXAMPLE = Path(__file__).resolve().parents[1]
REMOTE = '/root/fpga43dgs_basic_alu'


def command(client, text, log, timeout):
    channel = client.get_transport().open_session(timeout=10)
    channel.set_combine_stderr(True)
    channel.exec_command(text)
    deadline = time.monotonic() + timeout
    payload = bytearray()
    try:
        while True:
            while channel.recv_ready():
                chunk = channel.recv(65536)
                payload.extend(chunk)
                print(chunk.decode('utf-8', 'replace'), end='', flush=True)
            if channel.exit_status_ready() and not channel.recv_ready():
                break
            if time.monotonic() > deadline:
                raise TimeoutError('SSH operation timed out; inspect board state before retrying')
            time.sleep(.05)
        status = channel.recv_exit_status()
    finally:
        channel.close()
        log.write_bytes(payload)
    if status != 0:
        raise RuntimeError(f'Board command exited {status}; see {log}')
    return payload.decode('utf-8', 'replace')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', type=int, default=22)
    parser.add_argument('--known-hosts', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((EXAMPLE / 'data/manifest.json').read_text(encoding='utf-8'))
    data = EXAMPLE / 'data/vectors.txt'
    if hashlib.sha256(data.read_bytes()).hexdigest() != manifest['sha256']:
        raise RuntimeError('Local vector SHA256 differs from manifest')
    logs = EXAMPLE / 'build' / time.strftime('hardware_%Y%m%d_%H%M%S')
    logs.mkdir(parents=True)
    client = paramiko.SSHClient()
    client.load_host_keys(str(args.known_hosts))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    password = os.environ.get('FPGA_BOARD_PASSWORD')
    try:
        client.connect(args.host, port=args.port, username='root', password=password,
                       look_for_keys=password is None, allow_agent=password is None,
                       timeout=10, banner_timeout=10, auth_timeout=10)
        client.get_transport().set_keepalive(10)
        command(client, 'uname -a; cat /proc/sys/kernel/random/boot_id; '
                'mkdir -p ' + shlex.quote(REMOTE), logs / 'board_identity.log', 20)
        with client.open_sftp() as sftp:
            for local in (EXAMPLE/'board/test_basic_alu.cpp', EXAMPLE/'board/build_native.sh', data):
                sftp.put(str(local), REMOTE + '/' + local.name)
        prefix = 'cd ' + shlex.quote(REMOTE) + ' && '
        sha = command(client, prefix + 'sha256sum vectors.txt', logs/'vectors_sha256.log', 20)
        if sha.split()[0] != manifest['sha256']:
            raise RuntimeError('Board vector SHA256 mismatch')
        command(client, prefix+'timeout 180 sh build_native.sh', logs/'native_build.log', 200)
        output = command(client, prefix+'timeout 120 ./test_basic_alu vectors.txt', logs/'hardware_test.log', 140)
        if 'PASS: FPGA basic_alu 2800/2800 exact matches' not in output:
            raise RuntimeError('Hardware test did not report the expected PASS')
        record = {'host': args.host, 'port': args.port, 'vectors': 2800,
                  'vector_sha256': manifest['sha256'], 'result': 'PASS',
                  'cpp_sha256': hashlib.sha256((EXAMPLE/'board/test_basic_alu.cpp').read_bytes()).hexdigest(),
                  'log_directory': str(logs)}
        (logs/'result.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
        print('HARDWARE_TEST_PASSED='+str(logs))
    finally:
        client.close()


if __name__ == '__main__':
    main()
