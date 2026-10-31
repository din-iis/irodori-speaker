from __future__ import annotations

import csv
import json
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from irodori_speaker import check
from irodori_speaker.config import Config
from irodori_speaker.synth import cache_key

SR = 16000


def _sine(seconds: float, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(SR * seconds), dtype=np.float32)


def _stats(**kw) -> check.AudioStats:
    base = dict(seconds=2.0, leading_silence=0.1, trailing_silence=0.1, tail_noise=0.0, clipping=0)
    base.update(kw)
    return check.AudioStats(**base)


def _row(stats="default", z=None, cer=None, key="k", chars=10) -> check.CheckRow:
    return check.CheckRow(
        chapter=0, para=0, seg=0, key=key, text="あ", chars=chars,
        stats=_stats() if stats == "default" else stats, z=z, offset=None, flags=[], cer=cer,
    )


# ---- 文字数 ----


def test_count_chars_は絵文字と句点と空白を除く():
    assert check.count_chars("😊こんにちは。") == 5
    assert check.count_chars("😠😠あ い") == 2
    assert check.count_chars("「😭ひどい。」") == 5  # 閉じ括弧の前の句点も外す。括弧は数える


def test_count_chars_は絵文字だけなら_0():
    assert check.count_chars("😊😊") == 0


# ---- robust_z ----


def test_robust_z_は中央値と_MAD_で求める():
    z = check.robust_z([1.0, 2.0, 3.0, 4.0, 100.0])
    assert z[2] == 0.0
    assert z[4] == pytest.approx(97 / 1.4826, rel=1e-3)
    assert z[0] < 0


def test_robust_z_は_MAD_が_0_なら全部_0():
    assert check.robust_z([1.0, 1.0, 1.0, 9.0]) == [0.0, 0.0, 0.0, 0.0]


def test_robust_z_は_None_を飛ばす():
    z = check.robust_z([1.0, None, 2.0, 3.0, 4.0, 100.0])
    assert z[1] is None
    assert z[5] == pytest.approx(check.robust_z([1.0, 2.0, 3.0, 4.0, 100.0])[4])


def test_robust_z_は全部_None_でも落ちない():
    assert check.robust_z([None, None]) == [None, None]


# ---- 音声の解析 ----


def test_analyze_audio_は前後の無音と秒数を測る():
    data = np.concatenate([_silence(0.5), _sine(1.0), _silence(0.2)])
    st = check.analyze_audio(data, SR)
    assert st.seconds == pytest.approx(1.7, abs=0.01)
    assert st.leading_silence == pytest.approx(0.5, abs=0.03)
    assert st.trailing_silence == pytest.approx(0.2, abs=0.03)
    assert st.tail_noise == 0.0
    assert st.clipping == 0


def test_analyze_audio_は末尾の孤立した音を_tail_とする():
    data = np.concatenate([_sine(1.0), _silence(0.5), _sine(0.2)])
    st = check.analyze_audio(data, SR)
    assert st.tail_noise == pytest.approx(0.2, abs=0.03)


def test_analyze_audio_は末尾まで続く発話を_tail_にしない():
    data = np.concatenate([_sine(1.0), _silence(0.5), _sine(1.0)])
    assert check.analyze_audio(data, SR).tail_noise == 0.0


def test_analyze_audio_は直前の無音が短ければ_tail_にしない():
    data = np.concatenate([_sine(1.0), _silence(0.1), _sine(0.2)])
    assert check.analyze_audio(data, SR).tail_noise == 0.0


def test_analyze_audio_はクリッピングのサンプル数を数える():
    data = _sine(1.0)
    data[[10, 20, 30]] = 1.0
    data[40] = -1.0
    assert check.analyze_audio(data, SR).clipping == 4


def test_analyze_file_は無ければ_None(tmp_path):
    assert check.analyze_file(tmp_path / "none.wav") is None


def test_analyze_file_はステレオを平均する(tmp_path):
    p = tmp_path / "s.wav"
    mono = _sine(1.0)
    sf.write(p, np.stack([mono, mono], axis=1), SR)
    st = check.analyze_file(p)
    assert st.seconds == pytest.approx(1.0, abs=0.01)
    assert st.leading_silence == pytest.approx(0.0, abs=0.03)


# ---- 印 ----


def test_flags_for_は_long_と_short():
    assert check.flags_for(_row(z=3.0), 3.0, 0.3, set()) == ["long"]
    assert check.flags_for(_row(z=-3.0), 3.0, 0.3, set()) == ["short"]
    assert check.flags_for(_row(z=2.9), 3.0, 0.3, set()) == []
    assert check.flags_for(_row(z=None), 3.0, 0.3, set()) == []


def test_flags_for_は_clip_と_tail():
    assert check.flags_for(_row(stats=_stats(clipping=2)), 3.0, 0.3, set()) == ["clip"]
    assert check.flags_for(_row(stats=_stats(tail_noise=0.2)), 3.0, 0.3, set()) == ["tail"]


def test_flags_for_は音声が無ければ_missing():
    assert check.flags_for(_row(stats=None), 3.0, 0.3, set()) == ["missing"]


def test_flags_for_は_warnings_log_にあれば_warned():
    assert check.flags_for(_row(key="abc"), 3.0, 0.3, {"abc"}) == ["warned"]
    assert check.flags_for(_row(key="abc"), 3.0, 0.3, {"zzz"}) == []


def test_flags_for_は_cer_が閾値以上で_cer():
    assert check.flags_for(_row(cer=0.3), 3.0, 0.3, set()) == ["cer"]
    assert check.flags_for(_row(cer=0.29), 3.0, 0.3, set()) == []
    assert check.flags_for(_row(cer=None), 3.0, 0.3, set()) == []


def test_flags_for_は複数の印を決まった順に返す():
    row = _row(stats=_stats(clipping=1, tail_noise=0.1), z=5.0, cer=0.9, key="w")
    assert check.flags_for(row, 3.0, 0.3, {"w"}) == ["long", "clip", "tail", "warned", "cer"]


# ---- キーと warnings.log ----


def test_segment_key_は_cache_key_と一致し_seed_列を使う():
    synth = types.SimpleNamespace(cfg=Config(device="cpu"), latent_key="L")
    row = {"text": "😊あ", "seed": ""}
    assert check.segment_key(synth, row) == cache_key(synth.cfg, "😊あ", "L")
    assert check.segment_key(synth, {**row, "seed": "7"}) == cache_key(synth.cfg, "😊あ", "L", 7)
    assert check.segment_key(synth, {"text": "😊あ"}) == cache_key(synth.cfg, "😊あ", "L")


def test_read_warned_keys_は各行の最初のタブまで(tmp_path):
    p = tmp_path / "warnings.log"
    p.write_text("aaa\t本文\t警告\nbbb\tx\ty\n\n", encoding="utf-8")
    assert check.read_warned_keys(p) == {"aaa", "bbb"}
    assert check.read_warned_keys(tmp_path / "none.log") == set()


# ---- 時刻 ----


def test_offset_for_は_segments_があれば文の開始を使う():
    times = {"sr": 100, "frames": 1000, "paragraphs": [[0, 50, 400]],
             "segments": [[0, 0, 50, 200], [0, 1, 250, 400]]}
    assert check.offset_for(times, 0, 1) == pytest.approx(2.5)


def test_offset_for_は_segments_が無ければ段落の開始を使う():
    times = {"sr": 100, "frames": 1000, "paragraphs": [[0, 50, 400], [1, 500, 900]]}
    assert check.offset_for(times, 1, 3) == pytest.approx(5.0)


def test_offset_for_は見つからなければ_None():
    assert check.offset_for(None, 0, 0) is None
    assert check.offset_for({"sr": 100, "paragraphs": []}, 0, 0) is None


def test_load_times_json_は無ければ_None(tmp_path):
    wav = tmp_path / "01_a.wav"
    assert check.load_times_json(wav) is None
    (tmp_path / "01_a.times.json").write_text(json.dumps({"sr": 1, "frames": 1, "paragraphs": []}))
    assert check.load_times_json(wav)["sr"] == 1


# ---- CER ----


def test_normalize_for_cer_は記号と空白と絵文字を除く():
    assert check.normalize_for_cer("😊 こんにちは、世界！") == "こんにちは世界"
    assert check.normalize_for_cer("ＡＢＣ…") == "ABC"


def test_cer_は編集距離を参照の長さで割る():
    assert check.cer("こんにちは", "こんにちは") == 0.0
    assert check.cer("あいう", "かきく") == 1.0
    assert check.cer("あいうえ", "あいうお") == pytest.approx(0.25)
    assert check.cer("", "なんでも") == 0.0


# ---- check.tsv ----


def _full_row(**kw) -> check.CheckRow:
    base = dict(chapter=1, para=2, seg=3, key="kk", text="本文", chars=4, stats=_stats(seconds=2.0),
                z=1.234, offset=12.345, flags=["long", "tail"])
    base.update(kw)
    return check.CheckRow(**base)


def _read_tsv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
        return list(reader.fieldnames), rows


def test_write_check_tsv_は列と書式を守る(tmp_path):
    p = tmp_path / "check.tsv"
    check.write_check_tsv(p, [_full_row()], asr=False)
    header, rows = _read_tsv(p)
    assert header == check.CHECK_FIELDS
    r = rows[0]
    assert (r["chapter"], r["para"], r["seg"]) == ("1", "2", "3")
    assert r["seconds"] == "2.00" and r["sec_per_char"] == "0.500" and r["z"] == "1.23"
    assert r["offset"] == "12.35" and r["flags"] == "long|tail" and r["chars"] == "4"


def test_write_check_tsv_は_None_を空欄にする(tmp_path):
    p = tmp_path / "check.tsv"
    check.write_check_tsv(p, [_full_row(stats=None, z=None, offset=None, flags=["missing"], chars=0)], asr=False)
    r = _read_tsv(p)[1][0]
    assert r["seconds"] == "" and r["sec_per_char"] == "" and r["z"] == "" and r["offset"] == ""


def test_write_check_tsv_は_asr_のときだけ_cer_と_transcript_列を足す(tmp_path):
    p = tmp_path / "check.tsv"
    check.write_check_tsv(p, [_full_row(cer=0.5, transcript="ほんぶん")], asr=True)
    header, rows = _read_tsv(p)
    assert header == check.CHECK_FIELDS + check.ASR_FIELDS
    assert rows[0]["cer"] == "0.500" and rows[0]["transcript"] == "ほんぶん"
