import os

import numpy as np
import pytest
import soundfile as sf

from irodori_speaker.voices import describe_voice, list_voices, resolve_voice, voice_key


def _wav(path, seconds=0.5, sr=16000):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros(int(seconds * sr), dtype="float32"), sr)
    return path


@pytest.fixture
def voices(tmp_path):
    _wav(tmp_path / "narrator.wav")
    _wav(tmp_path / "alice" / "01.wav")
    _wav(tmp_path / "alice" / "02.wav")
    _wav(tmp_path / ".hidden.wav")
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "empty").mkdir()
    return tmp_path


def test_list_voices_files_and_dirs_sorted(voices):
    assert list_voices(voices) == ["alice", "narrator"]


def test_list_voices_missing_dir(tmp_path):
    assert list_voices(tmp_path / "nope") == []


def test_resolve_voice_by_id_file(voices):
    assert resolve_voice(voices, "narrator") == [voices / "narrator.wav"]


def test_resolve_voice_by_id_dir_sorted(voices):
    assert resolve_voice(voices, "alice") == [voices / "alice" / "01.wav", voices / "alice" / "02.wav"]


def test_resolve_voice_by_path(voices):
    p = voices / "narrator.wav"
    assert resolve_voice(voices / "other", str(p)) == [p]
    assert resolve_voice(voices / "other", str(voices / "alice")) == [
        voices / "alice" / "01.wav",
        voices / "alice" / "02.wav",
    ]


def test_resolve_voice_ambiguous(voices):
    _wav(voices / "alice.wav")
    with pytest.raises(ValueError, match="曖昧"):
        resolve_voice(voices, "alice")


def test_resolve_voice_not_found_lists_candidates(voices):
    with pytest.raises(ValueError) as e:
        resolve_voice(voices, "bob")
    assert "alice" in str(e.value) and "narrator" in str(e.value)


def test_resolve_voice_dir_without_audio(voices):
    with pytest.raises(ValueError):
        resolve_voice(voices, "empty")


def test_voice_key_ignores_mtime(voices):
    p = voices / "narrator.wav"
    k = voice_key([p])
    os.utime(p, (1, 1))
    assert voice_key([p]) == k


def test_voice_key_changes_with_content_and_order(voices):
    a, b = voices / "alice" / "01.wav", voices / "alice" / "02.wav"
    k = voice_key([a, b])
    assert voice_key([b, a]) != k
    with open(a, "ab") as f:
        f.write(b"\0")
    assert voice_key([a, b]) != k


def test_describe_voice_with_durations(voices):
    s = describe_voice(resolve_voice(voices, "alice"))
    assert s == "alice（2 クリップ、合計 1.0 秒）"


def test_describe_voice_unreadable_clip(voices):
    bad = voices / "x" / "bad.mp3"
    bad.parent.mkdir()
    bad.write_bytes(b"not audio")
    s = describe_voice([voices / "x" / "bad.mp3", voices / "alice" / "01.wav"])
    assert "秒数不明のクリップあり" in s
    assert "合計 0.5 秒" in s


def test_uppercase_extension_listed_and_resolved(tmp_path):
    _wav(tmp_path / "narrator.WAV")
    assert list_voices(tmp_path) == ["narrator"]
    assert resolve_voice(tmp_path, "narrator") == [tmp_path / "narrator.WAV"]


@pytest.mark.parametrize("value, expected", [
    ("none", True), (" none ", True), ("None", False), ("narrator", False), ("", False), (None, False),
])
def test_is_no_ref(value, expected):
    from irodori_speaker.voices import is_no_ref

    assert is_no_ref(value) is expected
