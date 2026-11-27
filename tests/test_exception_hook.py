"""Regression tests (I48): an uncaught exception in a Qt slot must be
reported (stderr + log file + non-modal notice) instead of aborting the app.
The hook is only installed by main.py's __main__ block, never under pytest;
these tests install it explicitly."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_slot_exception_does_not_abort_process(tmp_path):
    log = tmp_path / "logs" / "iv.log"
    script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(ROOT)!r})
        from PyQt6.QtCore import QTimer
        from PyQt6.QtWidgets import QApplication
        import main
        app = QApplication(sys.argv)
        main.install_exception_hook(app, log_path={str(log)!r})
        def boom():
            raise RuntimeError("boom-from-slot")
        def alive():
            print("STILL-ALIVE", flush=True)
            app.quit()
        QTimer.singleShot(0, boom)
        QTimer.singleShot(0, boom)
        QTimer.singleShot(300, alive)
        sys.exit(app.exec())
    """)
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    proc = subprocess.run([sys.executable, "-c", script], cwd=str(tmp_path),
                          env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert "STILL-ALIVE" in proc.stdout
    assert "Traceback" in proc.stderr
    assert "RuntimeError: boom-from-slot" in proc.stderr
    text = log.read_text(encoding="utf-8")
    assert text.count("RuntimeError: boom-from-slot") == 2


def test_hook_reports_and_rate_limits_dialog(qtbot, monkeypatch, tmp_path, capsys):
    from PyQt6.QtWidgets import QApplication, QMessageBox
    import main

    shown = []
    monkeypatch.setattr(QMessageBox, "show", lambda self: shown.append(self.text()))
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)  # restored after test

    hook = main.install_exception_hook(QApplication.instance(),
                                       log_path=str(tmp_path / "iv.log"))
    assert sys.excepthook is hook

    def raise_and_report(msg):
        try:
            raise ValueError(msg)
        except ValueError:
            hook(*sys.exc_info())

    raise_and_report("same")
    raise_and_report("same")
    qtbot.wait(50)
    assert len(shown) == 1 and "ValueError: same" in shown[0]
    assert capsys.readouterr().err.count("ValueError: same") == 2
