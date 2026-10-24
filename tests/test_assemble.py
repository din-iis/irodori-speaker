import numpy as np
import pytest
import soundfile as sf

from irodori_speaker.assemble import Clip, concat_wavs, write_chapter
from irodori_speaker.config import SilenceConfig


def _write(path, sr, n=100):
    sf.write(path, np.zeros(n, dtype="float32"), sr)
    return path


def test_concat_wavs_rejects_mixed_sample_rates(tmp_path):
    a = _write(tmp_path / "a.wav", 48000)
    b = _write(tmp_path / "b.wav", 24000)
    with pytest.raises(ValueError):
        concat_wavs([a, b], tmp_path / "out.wav")


def test_concat_wavs_sums_samples(tmp_path):
    a = _write(tmp_path / "a.wav", 48000, 100)
    b = _write(tmp_path / "b.wav", 48000, 250)
    out = tmp_path / "out.wav"
    concat_wavs([a, b], out)
    assert sf.info(out).frames == 350


def test_concat_wavs_failure_keeps_existing_output_and_leaves_no_tmp(tmp_path):
    a = _write(tmp_path / "a.wav", 48000)
    b = _write(tmp_path / "b.wav", 24000)
    out = tmp_path / "out.wav"
    out.write_bytes(b"old")
    with pytest.raises(ValueError):
        concat_wavs([a, b], out)
    assert out.read_bytes() == b"old"
    assert not (tmp_path / "out.tmp.wav").exists()


def test_write_chapter_failure_keeps_existing_output_and_leaves_no_tmp(tmp_path):
    a = _write(tmp_path / "a.wav", 48000)
    b = _write(tmp_path / "b.wav", 24000)
    clips = [Clip(a, 0, 0, 0), Clip(b, 0, 0, 1)]
    out = tmp_path / "ch.wav"
    out.write_bytes(b"old")
    with pytest.raises(ValueError):
        write_chapter(out, clips, SilenceConfig())
    assert out.read_bytes() == b"old"
    assert not (tmp_path / "ch.tmp.wav").exists()
