import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from irodori_speaker.assemble import (
    Clip,
    concat_wavs,
    ends_with_ellipsis,
    gap_seconds,
    times_path,
    write_chapter,
)
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
    assert t == {
        "sr": 1000,
        "frames": 1460,
        "paragraphs": [[0, 1000, 1310], [1, 1410, 1460]],
        "segments": [[0, 0, 1000, 1100], [0, 1, 1110, 1310], [1, 0, 1410, 1460]],
    }
    assert sf.info(out).frames == 1460


def test_write_chapter_failure_removes_old_times(tmp_path):
    a = _write(tmp_path / "a.wav", 1000, 100)
    bad = _write(tmp_path / "bad.wav", 2000, 100)
    out = tmp_path / "ch.wav"
    times_path(out).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        write_chapter(out, [Clip(a, 0, 0, 0), Clip(bad, 0, 0, 1)], SilenceConfig())
    assert not times_path(out).exists()


# ---------- 区切り・章題・間 ----------

SIL = dict(sentence=0.01, paragraph=0.1, chapter=0.0, title=0.2, scene=0.5, turn=0.05, ellipsis=0.07)


def _times(out):
    return json.loads(times_path(out).read_text(encoding="utf-8"))


def test_clip_kind_and_text_are_optional():
    c = Clip(None, 0, 0, 0)
    assert c.kind == "narration" and c.text == ""


def test_break_clip_uses_scene_silence_and_zero_width_paragraph(tmp_path):
    a = _write(tmp_path / "a.wav", 1000, 100)
    c = _write(tmp_path / "c.wav", 1000, 50)
    out = tmp_path / "ch.wav"
    write_chapter(out, [Clip(a, 0, 0, 0), Clip(None, 0, 1, 0, kind="break"), Clip(c, 0, 2, 0)],
                  SilenceConfig(**{**SIL, "chapter": 0.0}))
    t = _times(out)
    # a100, 間 = max(paragraph 0.1, scene 0.5) = 500 → c は 600 から
    assert t["paragraphs"] == [[0, 0, 100], [1, 100, 100], [2, 600, 650]]
    assert t["segments"] == [[0, 0, 0, 100], [2, 0, 600, 650]]
    assert t["frames"] == 650
    assert sf.info(out).frames == 650


def test_scene_smaller_than_paragraph_uses_paragraph():
    s = SilenceConfig(**{**SIL, "scene": 0.05})
    a, b = Clip(Path("a"), 0, 0, 0), Clip(Path("b"), 0, 2, 0)
    assert gap_seconds(a, b, s, after_break=True) == pytest.approx(0.1)


def test_title_silence_after_title(tmp_path):
    t = _write(tmp_path / "t.wav", 1000, 100)
    b = _write(tmp_path / "b.wav", 1000, 50)
    out = tmp_path / "ch.wav"
    write_chapter(out, [Clip(t, 0, 0, 0, kind="title", text="題"), Clip(b, 0, 1, 0)],
                  SilenceConfig(**SIL))
    assert _times(out)["paragraphs"] == [[0, 0, 100], [1, 300, 350]]  # 題の後 0.2 s


def test_turn_silence_inside_paragraph_only():
    s = SilenceConfig(**SIL)
    d = Clip(Path("d"), 0, 0, 0, kind="dialogue", text="「はい」")
    n = Clip(Path("n"), 0, 0, 1, kind="narration", text="と言った。")
    assert gap_seconds(d, n, s, after_break=False) == pytest.approx(0.05)
    n2 = Clip(Path("n"), 0, 1, 0, kind="narration", text="次。")
    assert gap_seconds(d, n2, s, after_break=False) == pytest.approx(0.1)  # 段落をまたぐ → paragraph


@pytest.mark.parametrize("text", ["そう……", "そう‥", "そう―", "そう—", "そう……」", "「そう……」"])
def test_ends_with_ellipsis(text):
    assert ends_with_ellipsis(text)


def test_not_ellipsis():
    assert not ends_with_ellipsis("そう。")
    assert not ends_with_ellipsis("……そう")


def test_ellipsis_silence_is_larger_of():
    s = SilenceConfig(**SIL)
    a = Clip(Path("a"), 0, 0, 0, text="そう……")
    b = Clip(Path("b"), 0, 0, 1, text="か。")
    assert gap_seconds(a, b, s, after_break=False) == pytest.approx(0.07)


def test_break_at_chapter_end_adds_no_trailing_silence(tmp_path):
    a = _write(tmp_path / "a.wav", 1000, 100)
    out = tmp_path / "ch.wav"
    write_chapter(out, [Clip(a, 0, 0, 0), Clip(None, 0, 1, 0, kind="break")], SilenceConfig(**SIL))
    assert _times(out)["frames"] == 100
    assert _times(out)["paragraphs"] == [[0, 0, 100], [1, 100, 100]]


def test_chapter_without_audio_raises(tmp_path):
    with pytest.raises(ValueError):
        write_chapter(tmp_path / "ch.wav", [Clip(None, 0, 0, 0, kind="break")], SilenceConfig())
