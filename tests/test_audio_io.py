import numpy as np
import pytest
import soundfile as sf

from irodori_speaker.audio_io import concat_mp3, decode, ffmpeg_available, write_mp3

pytestmark = pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg が無い")


def _tone(n, sr=48000):
    return (0.1 * np.sin(np.arange(n) / sr * 2 * np.pi * 440)).astype("float32")


def test_write_mp3_then_decode_roundtrip(tmp_path):
    x = _tone(48000)
    stereo = np.stack([x, x], axis=1)
    p = tmp_path / "a.mp3"
    write_mp3(stereo, 48000, p, "128k")
    y = decode(p, 24000, 2)
    assert y.dtype == np.float32 and y.shape[1] == 2 and abs(len(y) - 24000) < 2000
    assert not (tmp_path / "a.tmp.mp3").exists()


def test_decode_mono_shape_and_window(tmp_path):
    p = tmp_path / "a.wav"
    sf.write(p, _tone(48000 * 3), 48000)
    y = decode(p, 16000, 1, start=1.0, seconds=1.0)
    assert y.ndim == 1 and abs(len(y) - 16000) < 200


def test_concat_mp3(tmp_path):
    a, b, out = tmp_path / "a.mp3", tmp_path / "b.mp3", tmp_path / "ab.mp3"
    write_mp3(_tone(48000), 48000, a, "128k")
    write_mp3(_tone(48000), 48000, b, "128k")
    concat_mp3([a, b], out)
    assert abs(len(decode(out, 24000, 1)) - 48000) < 4000


def test_decode_missing_file_raises(tmp_path):
    with pytest.raises(RuntimeError):
        decode(tmp_path / "none.mp3", 16000, 1)
