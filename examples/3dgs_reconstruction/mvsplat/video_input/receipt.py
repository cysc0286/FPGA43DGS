"""Accept a closed local video file after READY (live capture is a future adapter)."""
from pathlib import Path
import time
from common import sha


def receive_file(path, events):
    path = Path(path).resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError("Expected a completed, nonempty video file")
    before = path.stat()
    # This event marks handoff of an already completed local file. Hash checking
    # belongs to scene preparation, not to an invented camera/upload interval.
    events.emit("VIDEO_COMPLETE", path=str(path), bytes=before.st_size,
                receipt_scope="closed local file handed to pipeline; acquisition not measured")
    started = time.monotonic()
    digest = sha(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Video changed during receipt verification")
    return dict(path=str(path), sha256=digest, bytes=before.st_size,
                verify_seconds=time.monotonic()-started,
                video_input_seconds_record_only=None)
