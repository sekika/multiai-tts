"""Test setup shared by the suite.

The package imports :mod:`sounddevice` at module load, which requires the
native PortAudio library. Playback is never exercised in the (mocked) tests,
so when PortAudio is unavailable we substitute a lightweight stub. This keeps
the unit tests runnable on machines/CI without the audio backend installed.
"""
import sys
import types

try:  # pragma: no cover - depends on the host environment
    import sounddevice  # noqa: F401
except OSError:
    stub = types.ModuleType("sounddevice")
    stub.play = lambda *args, **kwargs: None
    stub.wait = lambda *args, **kwargs: None
    sys.modules["sounddevice"] = stub
