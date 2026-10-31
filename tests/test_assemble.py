import json

import numpy as np
import pytest
import soundfile as sf

from irodori_speaker.assemble import Clip, concat_wavs, times_path, write_chapter
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


def test_write_chapter_writes_paragraph_times(tmp_path):
    a = _write(tmp_path / "a.wav", 1000, 100)
    b = _write(tmp_path / "b.wav", 1000, 200)
    c = _write(tmp_path / "c.wav", 1000, 50)
    out = tmp_path / "ch.wav"
    write_chapter(out, [Clip(a, 0, 0, 0), Clip(b, 0, 0, 1), Clip(c, 0, 1, 0)],
                  SilenceConfig(sentence=0.01, paragraph=0.1, chapter=1.0))
    t = json.loads(times_path(out).read_text(encoding="utf-8"))
    # 章頭 1000 + a100 + 文間 10 + b200 = 1310、段落間 100、c50
    assert t == {"sr": 1000, "frames": 1460, "paragraphs": [[0, 1000, 1310], [1, 1410, 1460]]}
    assert sf.info(out).frames == 1460


def test_write_chapter_failure_removes_old_times(tmp_path):
    a = _write(tmp_path / "a.wav", 1000, 100)
    bad = _write(tmp_path / "bad.wav", 2000, 100)
    out = tmp_path / "ch.wav"
    times_path(out).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        write_chapter(out, [Clip(a, 0, 0, 0), Clip(bad, 0, 0, 1)], SilenceConfig())
    assert not times_path(out).exists()
