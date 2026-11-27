"""Single-instance hand-off (I18, I35).

The secondary must send absolute paths (resolved against *its* cwd), and the
primary must receive long file lists completely without blocking the GUI.
"""
from __future__ import annotations

import os
import subprocess
import sys
import uuid
from pathlib import Path

from PyQt6.QtCore import QTimer
from PyQt6.QtNetwork import QLocalSocket

import single_instance as si_mod
from single_instance import SingleInstance

ROOT = Path(__file__).resolve().parents[1]


def _primary():
    inst = SingleInstance(f"ImageViewerTest-{uuid.uuid4().hex[:12]}")
    assert inst.check_and_connect() is True
    return inst


def _run_secondary(app_id, cwd, args):
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]);"
        "from PyQt6.QtCore import QCoreApplication;"
        "app = QCoreApplication([]);"
        "from single_instance import SingleInstance;"
        "inst = SingleInstance(sys.argv[2]);"
        "assert inst.check_and_connect() is False, 'expected secondary';"
        "sys.exit(0 if inst.send_files(sys.argv[3:]) else 3)"
    )
    return subprocess.Popen(
        [sys.executable, "-c", code, str(ROOT), app_id, *args],
        cwd=str(cwd),
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )


def test_absolute_paths_resolves_relative_against_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    absolute = os.path.abspath(os.sep + "abs" + os.sep + "c.raw")
    out = si_mod.absolute_paths(["a.png", "-flag", "sub/b.tif", absolute])
    assert out == [
        os.path.abspath(str(tmp_path / "a.png")),
        os.path.abspath(str(tmp_path / "sub" / "b.tif")),
        absolute,
    ]


def test_message_roundtrip_handles_newlines_and_unicode():
    files = ["/tmp/a b.png", "/tmp/日本\nweird.tif"]
    msg = si_mod.encode_message(files)
    assert si_mod.decode_payload(msg[si_mod._HEADER_SIZE:]) == files
    assert si_mod.decode_payload(b"not json") == []


def test_large_relative_handoff_is_absolute_complete_and_nonblocking(qtbot, tmp_path):
    primary = _primary()
    try:
        # >64KB of relative paths, sent from a different cwd than ours.
        sender_cwd = tmp_path / "launch_dir"
        sender_cwd.mkdir()
        rel = [f"sub/{'x' * 200}_{i:05d}.png" for i in range(600)]
        assert sum(len(p) for p in rel) > 64 * 1024

        ticks = []
        timer = QTimer()
        timer.setInterval(5)
        timer.timeout.connect(lambda: ticks.append(1))
        timer.start()

        with qtbot.waitSignal(primary.files_received, timeout=15000) as blocker:
            proc = _run_secondary(primary.app_id, sender_cwd, rel)
        timer.stop()
        assert proc.wait(timeout=15) == 0

        received = blocker.args[0]
        expected = [os.path.abspath(os.path.join(str(sender_cwd), p)) for p in rel]
        assert received == expected
        # The event loop kept running while the hand-off was in progress.
        assert len(ticks) > 0
        assert primary._buffers == {}
    finally:
        primary.server.close()


def test_incomplete_message_is_dropped(qtbot):
    primary = _primary()
    try:
        got = []
        primary.files_received.connect(got.append)
        client = QLocalSocket()
        client.connectToServer(primary.app_id)
        assert client.waitForConnected(1000)
        full = si_mod.encode_message(["/a.png", "/b.png"])
        client.write(full[:-3])
        client.flush()
        qtbot.waitUntil(lambda: len(primary._buffers) == 1, timeout=2000)
        client.disconnectFromServer()
        qtbot.waitUntil(lambda: primary._buffers == {}, timeout=2000)
        assert got == []
    finally:
        primary.server.close()
