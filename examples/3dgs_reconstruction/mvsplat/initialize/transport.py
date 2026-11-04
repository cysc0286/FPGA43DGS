"""Bounded SSH to the lab board using its previously verified public host key."""
import base64
import os
import time
from pathlib import Path
import paramiko

HOST='192.168.126.49'
HOST_KEY='AAAAC3NzaC1lZDI1NTE5AAAAIJEJjNFzYnjluPa+kwz4imgxJHRfVt462buoRP0JRtu6'

def connect():
    password=os.environ.get('FPGA_BOARD_PASSWORD')
    if password is None:raise RuntimeError('Supply FPGA_BOARD_PASSWORD privately in the environment')
    client=paramiko.SSHClient()
    client.get_host_keys().add(HOST,'ssh-ed25519',paramiko.Ed25519Key(data=base64.b64decode(HOST_KEY)))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect(HOST,username='root',password=password,look_for_keys=False,allow_agent=False,
                   timeout=8,banner_timeout=8,auth_timeout=8)
    client.get_transport().set_keepalive(10)
    return client

def run(client,command,timeout=60,log=None,check=True):
    channel=client.get_transport().open_session(timeout=8)
    channel.set_combine_stderr(True);channel.exec_command(command)
    chunks=[];deadline=time.monotonic()+timeout
    try:
        while True:
            while channel.recv_ready():
                chunk=channel.recv(65536)
                if not chunk:break
                chunks.append(chunk)
            if channel.exit_status_ready() and not channel.recv_ready():break
            if time.monotonic()>deadline:raise TimeoutError('Inspect board process/state before retrying timed-out operation')
            time.sleep(.03)
        status=channel.recv_exit_status()
    finally:
        channel.close();content=b''.join(chunks).decode('utf-8','replace')
        if log:Path(log).write_text(content,encoding='utf-8')
    if check and status:raise RuntimeError(f'Board exit {status}:\n{content[-6000:]}')
    return status,content
