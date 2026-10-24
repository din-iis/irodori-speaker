import numpy as np
import pytest
import soundfile as sf

from irodori_speaker.assemble import concat_wavs


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
