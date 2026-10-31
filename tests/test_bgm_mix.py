import json
from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf

from irodori_speaker.bgm_mix import (
    Layout,
    Track,
    TimesMismatch,
    build_layout,
    load_times,
    loop_to,
    mix_signature,
    render,
    rms_db,
    trim_silence,
    voice_level_db,
    write_credits,
)
from irodori_speaker.bgm_types import Choice, Scene, Sound
from irodori_speaker.config import BgmConfig

# sr=10 の小さい値で数を確かめる
BGM = BgmConfig(
    intro_seconds=20.0,
    bridge_seconds=8.0,
    music_fade_in=2.0,
    music_fade_out=3.0,
    ambient_crossfade=4.0,
)
SR = 10


def _sound(i: int = 1, name: str = "rain", user: str = "bob") -> Sound:
    return Sound(
        id=i,
        name=name,
        tags=[],
        description="",
        duration=60.0,
        license="https://creativecommons.org/licenses/by/4.0/",
        username=user,
        url=f"https://freesound.org/people/{user}/sounds/{i}/",
        preview="https://example.invalid/p.mp3",
    )


def _choice(kind: str, scene: int, sid: int = 1, chapter: int = 0, ok: bool = True) -> Choice:
    return Choice(
        chapter=chapter,
        scene=scene,
        kind=kind,
        query="q",
        sound=_sound(sid) if ok else None,
        score=0.5 if ok else None,
        reason=None if ok else "no_result",
    )


def _scene(index: int, start: int, end: int, ambient: str | None = "rain", chapter: int = 0) -> Scene:
    return Scene(
        chapter=chapter,
        index=index,
        start_para=start,
        end_para=end,
        description="d",
        ambient_query=ambient,
        mood="calm",
        music_query="piano",
    )


def _choices(*cs: Choice) -> dict[str, Choice]:
    return {f"{c.chapter}:{c.scene}:{c.kind}": c for c in cs}


PARAS = {0: (10, 400), 1: (410, 1000)}


def test_layout_intro_two_scenes_with_bridge():
    scenes = [_scene(0, 0, 0), _scene(1, 1, 1)]
    choices = _choices(
        _choice("intro", 0), _choice("bridge", 1), _choice("ambient", 0), _choice("ambient", 1)
    )
    L = build_layout(SR, 1000, PARAS, scenes, choices, BGM)
    assert L.music == [Track("0:0:intro", 0, 200, 20, 30), Track("0:1:bridge", 600, 680, 20, 30)]
    assert L.voice == [(0, 400, 200), (400, 1000, 680)]
    assert L.ambient == [Track("0:0:ambient", 170, 640, 30, 40), Track("0:1:ambient", 650, 1310, 30, 30)]
    assert L.total == 1310


def test_layout_without_music_crossfades_ambient_at_boundary():
    scenes = [_scene(0, 0, 0), _scene(1, 1, 1)]
    choices = _choices(_choice("ambient", 0), _choice("ambient", 1))
    L = build_layout(SR, 1000, PARAS, scenes, choices, BGM)
    assert L.music == []
    assert L.voice == [(0, 400, 0), (400, 1000, 400)]
    # 境界 b=400。前は b+X/2 で終わり、後は b-X/2 から始まる
    assert L.ambient == [Track("0:0:ambient", 0, 420, 40, 40), Track("0:1:ambient", 380, 1030, 40, 30)]
    assert L.total == 1030


def test_layout_single_scene_has_no_bridge():
    scenes = [_scene(0, 0, 1)]
    choices = _choices(_choice("intro", 0), _choice("ambient", 0))
    L = build_layout(SR, 1000, PARAS, scenes, choices, BGM)
    assert [t.key for t in L.music] == ["0:0:intro"]
    assert L.voice == [(0, 1000, 200)]
    assert L.ambient == [Track("0:0:ambient", 170, 1230, 30, 30)]
    assert L.total == 1230


def test_layout_without_intro_starts_ambient_at_zero():
    scenes = [_scene(0, 0, 1)]
    L = build_layout(SR, 1000, PARAS, scenes, _choices(_choice("ambient", 0)), BGM)
    assert L.music == []
    assert L.ambient == [Track("0:0:ambient", 0, 1030, 40, 30)]
    assert L.total == 1030


def test_layout_silent_last_scene_has_no_tail():
    scenes = [_scene(0, 0, 0), _scene(1, 1, 1, ambient=None)]
    L = build_layout(SR, 1000, PARAS, scenes, _choices(_choice("ambient", 0)), BGM)
    assert L.voice == [(0, 400, 0), (400, 1000, 400)]
    assert L.total == 1000


def test_layout_silent_scene_next_to_bridge():
    # 場面 1 が無音 → Track は場面 0 と 2 だけ、bridge は 2 つ
    paras = {0: (0, 100), 1: (100, 200), 2: (200, 300)}
    scenes = [_scene(0, 0, 0), _scene(1, 1, 1, ambient=None), _scene(2, 2, 2)]
    choices = _choices(
        _choice("ambient", 0),
        _choice("bridge", 1),
        _choice("bridge", 2),
        _choice("ambient", 2),
    )
    L = build_layout(SR, 300, paras, scenes, choices, BGM)
    assert [t.key for t in L.music] == ["0:1:bridge", "0:2:bridge"]
    assert [t.key for t in L.ambient] == ["0:0:ambient", "0:2:ambient"]
    # 境界 100 で bridge [100,180)、場面 1 の朗読は 180 から 280、境界 280 で bridge [280,360)
    assert L.music == [Track("0:1:bridge", 100, 180, 20, 30), Track("0:2:bridge", 280, 360, 20, 30)]
    assert L.voice == [(0, 100, 0), (100, 200, 180), (200, 300, 360)]
    assert L.ambient[0] == Track("0:0:ambient", 0, 140, 40, 40)
    assert L.ambient[1] == Track("0:2:ambient", 330, 490, 30, 30)
    assert L.total == 490


def test_layout_unselected_choice_is_ignored():
    scenes = [_scene(0, 0, 0), _scene(1, 1, 1)]
    choices = _choices(
        _choice("intro", 0, ok=False), _choice("bridge", 1, ok=False), _choice("ambient", 0, ok=False)
    )
    L = build_layout(SR, 1000, PARAS, scenes, choices, BGM)
    assert L.music == [] and L.ambient == []
    assert L.voice == [(0, 400, 0), (400, 1000, 400)]
    assert L.total == 1000


def test_loop_to_short_source():
    x = np.ones((30, 2), dtype="float32")  # sr=10 で 3 秒
    y = loop_to(x, 500, 10)
    assert y.shape == (500, 2)
    assert np.allclose(y, 1.0, atol=1e-5)  # 一定値はクロスフェードしても一定


def test_loop_to_long_source_is_cut():
    x = np.arange(100, dtype="float32")
    assert np.array_equal(loop_to(x, 40, 10), x[:40])


def test_trim_silence_strips_leading_and_trailing():
    x = np.zeros((100, 2), dtype="float32")
    x[30:60] = 0.5
    y = trim_silence(x, 10)
    assert y.shape == (30, 2)
    assert np.all(y == 0.5)


def test_trim_silence_all_silent_returns_empty():
    assert trim_silence(np.zeros((50, 2), dtype="float32"), 10).shape[0] == 0


def _sine(n: int, sr: int, db: float) -> np.ndarray:
    t = np.arange(n) / sr
    return (np.sin(2 * np.pi * 100 * t) * 10 ** (db / 20) * np.sqrt(2)).astype("float32")


def _noise(n: int, db: float, channels: int = 2) -> np.ndarray:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n, channels)).astype("float32")
    x *= 10 ** (db / 20) / np.sqrt(np.mean(x**2))
    return x


def test_voice_level_db_ignores_silent_frames():
    sr = 1000
    x = np.concatenate([_sine(sr, sr, -20.0), np.zeros(sr, dtype="float32")])
    assert voice_level_db(x, sr) == pytest.approx(-20.0, abs=0.5)
    assert rms_db(_sine(sr, sr, -20.0)) == pytest.approx(-20.0, abs=0.1)
    assert rms_db(np.zeros(10, dtype="float32")) == -np.inf


def test_render_levels():
    sr = 1000
    n = 20 * sr
    narration = np.concatenate([_sine(10 * sr, sr, -20.0), np.zeros(10 * sr, dtype="float32")])
    layout = Layout(total=n, voice=[(0, n, 0)], ambient=[Track("0:0:ambient", 0, n, 0, 0)], music=[])
    out = render(layout, narration, {"0:0:ambient": _noise(5 * sr, -10.0)}, sr, BgmConfig())
    assert out.shape == (n, 2)
    # 朗読の無い区間の環境音は 朗読 -20 + ambient_db(-20) = -40 dBFS
    assert rms_db(out[12 * sr : 19 * sr]) == pytest.approx(-40.0, abs=1.5)


def test_render_music_level():
    sr = 1000
    n = 10 * sr
    narration = _sine(n, sr, -20.0)
    layout = Layout(total=n, voice=[], ambient=[], music=[Track("0:0:intro", 0, n, 0, 0)])
    out = render(layout, narration, {"0:0:intro": _noise(8 * sr, -5.0)}, sr, BgmConfig())
    assert rms_db(out[sr : 9 * sr]) == pytest.approx(-23.0, abs=1.5)


def test_render_applies_fades():
    sr = 1000
    n = 10 * sr
    layout = Layout(total=n, voice=[], ambient=[], music=[Track("m", 0, n, sr, sr)])
    out = render(layout, _sine(n, sr, -20.0), {"m": np.ones((n, 2), dtype="float32")}, sr, BgmConfig())
    assert abs(out[0]).max() < 0.01 * abs(out[n // 2]).max()
    assert abs(out[-1]).max() < 0.01 * abs(out[n // 2]).max()


def test_render_peak_limited_to_minus_1_dbfs():
    sr = 1000
    n = 5 * sr
    narration = np.ones(n, dtype="float32") * 0.9
    layout = Layout(total=n, voice=[(0, n, 0)], ambient=[], music=[Track("m", 0, n, 0, 0)])
    out = render(layout, narration, {"m": _noise(n, -1.0)}, sr, BgmConfig(music_db=6.0))
    assert np.abs(out).max() <= 10 ** (-1 / 20) + 1e-4


def test_render_voice_centered_both_channels():
    sr = 1000
    narration = _sine(sr, sr, -20.0)
    layout = Layout(total=sr + 100, voice=[(0, sr, 100)], ambient=[], music=[])
    out = render(layout, narration, {}, sr, BgmConfig())
    assert np.array_equal(out[:, 0], out[:, 1])
    assert np.array_equal(out[100 : 100 + sr, 0], narration)
    assert np.all(out[:100] == 0)


def test_render_skips_silent_or_missing_sound():
    sr = 1000
    narration = _sine(sr, sr, -20.0)
    layout = Layout(
        total=sr,
        voice=[(0, sr, 0)],
        ambient=[Track("a", 0, sr, 0, 0), Track("b", 0, sr, 0, 0)],
        music=[],
    )
    out = render(layout, narration, {"a": np.zeros((sr, 2), dtype="float32")}, sr, BgmConfig())
    assert np.array_equal(out[:, 0], narration)


def test_mix_signature_changes_with_choice_and_config():
    scenes = [_scene(0, 0, 0), _scene(1, 1, 1)]
    base = _choices(_choice("ambient", 0, sid=1))
    sig = mix_signature("abc", scenes, base, BGM)
    assert sig == mix_signature("abc", scenes, base, BGM)
    assert sig != mix_signature("abd", scenes, base, BGM)
    assert sig != mix_signature("abc", scenes, _choices(_choice("ambient", 0, sid=2)), BGM)
    assert sig != mix_signature("abc", scenes[:1], base, BGM)
    assert sig != mix_signature("abc", scenes, base, replace(BGM, ambient_db=-18.0))
    assert sig != mix_signature("abc", scenes, base, replace(BGM, mp3_bitrate="128k"))
    # 重ね合わせに使わない設定では変わらない
    assert sig == mix_signature("abc", scenes, base, replace(BGM, candidates=3))


def test_load_times_reads_paragraph_positions(tmp_path):
    wav = tmp_path / "01_a.wav"
    sf.write(wav, np.zeros(100, dtype="float32"), 10, subtype="PCM_16")
    wav.with_suffix(".times.json").write_text(
        json.dumps({"sr": 10, "frames": 100, "paragraphs": [[0, 0, 40], [1, 50, 100]]}),
        encoding="utf-8",
    )
    assert load_times(wav) == (10, 100, {0: (0, 40), 1: (50, 100)})


def test_load_times_mismatch_raises(tmp_path):
    wav = tmp_path / "01_a.wav"
    sf.write(wav, np.zeros(100, dtype="float32"), 10, subtype="PCM_16")
    with pytest.raises(TimesMismatch):  # .times.json が無い
        load_times(wav)
    wav.with_suffix(".times.json").write_text(
        json.dumps({"sr": 10, "frames": 99, "paragraphs": [[0, 0, 40]]}), encoding="utf-8"
    )
    with pytest.raises(TimesMismatch):  # frames が違う
        load_times(wav)


def test_write_credits_lists_each_sound_once(tmp_path):
    a = _choice("ambient", 0, sid=1)
    b = _choice("ambient", 1, sid=1)  # 同じ音が 2 場面
    c = Choice(0, 0, "intro", "q", _sound(2, "piano", "amy"), 0.4, None)
    d = _choice("bridge", 1, ok=False)
    path = tmp_path / "CREDITS.txt"
    write_credits(path, [a, b, c, d])
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines == [
        "rain by bob — https://freesound.org/people/bob/sounds/1/ — CC BY 4.0",
        "piano by amy — https://freesound.org/people/amy/sounds/2/ — CC BY 4.0",
    ]
