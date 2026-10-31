from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from .assemble import times_path
from .bgm_types import Choice, Scene, license_label
from .config import BgmConfig

MIX_RULE = 1
OUTRO_SECONDS = 3.0  # 章末の環境音の余韻
PEAK_LIMIT = 10 ** (-1 / 20)
VOICE_FRAME_SECONDS = 0.05
VOICE_FLOOR_DB = -50.0
FALLBACK_VOICE_DB = -20.0  # 朗読に音のある部分が無いときの基準


class TimesMismatch(Exception):
    pass


@dataclass
class Track:
    key: str
    start: int
    end: int
    fade_in: int
    fade_out: int


@dataclass
class Layout:
    total: int
    voice: list[tuple[int, int, int]]  # (元の開始, 元の終了, 出力の開始)
    ambient: list[Track] = field(default_factory=list)
    music: list[Track] = field(default_factory=list)


def load_times(wav: Path) -> tuple[int, int, dict[int, tuple[int, int]]]:
    """章 WAV の段落ごとの (開始, 終了) サンプル位置を読む。"""
    path = times_path(wav)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        sr, frames = int(data["sr"]), int(data["frames"])
        paras = {int(p): (int(s), int(e)) for p, s, e in data["paragraphs"]}
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise TimesMismatch(f"{path.name} を読めません: {e}") from e
    actual = sf.info(str(wav)).frames
    if frames != actual:
        raise TimesMismatch(f"{path.name} のサンプル数 {frames} が WAV の {actual} と違います")
    return sr, frames, paras


def _chosen(choices: dict[str, Choice], key: str) -> bool:
    c = choices.get(key)
    return c is not None and c.sound is not None


def build_layout(
    sr: int,
    frames: int,
    paras: dict[int, tuple[int, int]],
    scenes: list[Scene],
    choices: dict[str, Choice],
    bgm: BgmConfig,
) -> Layout:
    def sec(x: float) -> int:
        return round(x * sr)

    intro_len, bridge_len = sec(bgm.intro_seconds), sec(bgm.bridge_seconds)
    fi, fo = sec(bgm.music_fade_in), sec(bgm.music_fade_out)
    xf, tail = sec(bgm.ambient_crossfade), sec(OUTRO_SECONDS)

    layout = Layout(total=0, voice=[], ambient=[], music=[])
    # 場面ごとの環境音の (開始, fade_in, 終了, fade_out)。Track にするかは最後に決める
    spans: list[list[int]] = [[0, 0, 0, 0] for _ in scenes]

    t = 0
    if scenes and _chosen(choices, f"{scenes[0].chapter}:0:intro"):
        layout.music.append(Track(f"{scenes[0].chapter}:0:intro", 0, intro_len, fi, fo))
        t = intro_len
        spans[0][0], spans[0][1] = max(0, t - fo), fo
    elif scenes:
        spans[0][0], spans[0][1] = 0, xf

    prev_end = 0
    for k, sc in enumerate(scenes):
        try:
            end = frames if k == len(scenes) - 1 else paras[sc.end_para][1]
        except KeyError as e:
            raise TimesMismatch(f"段落 {sc.end_para} の時刻がありません") from e
        layout.voice.append((prev_end, end, t))
        t += end - prev_end
        prev_end = end
        if k == len(scenes) - 1:
            break
        b = t
        nxt = scenes[k + 1]
        bridge_key = f"{nxt.chapter}:{nxt.index}:bridge"
        if _chosen(choices, bridge_key):
            layout.music.append(Track(bridge_key, b, b + bridge_len, fi, fo))
            spans[k][2], spans[k][3] = b + xf, xf
            spans[k + 1][0], spans[k + 1][1] = b + bridge_len - fo, fo
            t = b + bridge_len
        else:
            spans[k][2], spans[k][3] = b + xf // 2, xf
            spans[k + 1][0], spans[k + 1][1] = b - xf // 2, xf

    last = len(scenes) - 1
    for k, sc in enumerate(scenes):
        if k == last:
            spans[k][2], spans[k][3] = t + tail, tail
        key = f"{sc.chapter}:{sc.index}:ambient"
        if sc.ambient_query is not None and _chosen(choices, key):
            layout.ambient.append(Track(key, *_span_args(spans[k])))
    last_has_ambient = bool(layout.ambient) and layout.ambient[-1].key == (
        f"{scenes[last].chapter}:{scenes[last].index}:ambient" if scenes else ""
    )
    layout.total = t + tail if last_has_ambient else t
    return layout


def _span_args(s: list[int]) -> tuple[int, int, int, int]:
    start, fade_in, end, fade_out = s
    return start, end, fade_in, fade_out


def rms_db(x: np.ndarray) -> float:
    if x.size == 0:
        return -np.inf
    ms = float(np.mean(np.square(x.astype("float64"))))
    return 10 * np.log10(ms) if ms > 0 else -np.inf


def voice_level_db(narration: np.ndarray, sr: int) -> float:
    """朗読の音のある部分（50 ms のフレームの RMS が -50 dBFS を超える部分）の RMS。"""
    n = max(1, round(sr * VOICE_FRAME_SECONDS))
    count = len(narration) // n
    if count == 0:
        return rms_db(narration)
    frames = narration[: count * n].reshape(count, n)
    loud = frames[np.mean(np.square(frames.astype("float64")), axis=1) > 10 ** (VOICE_FLOOR_DB / 10)]
    return rms_db(loud)


def trim_silence(x: np.ndarray, sr: int, threshold_db: float = -50.0) -> np.ndarray:
    """冒頭と末尾の無音（振幅が threshold_db 以下）を削る。"""
    amp = np.abs(x) if x.ndim == 1 else np.abs(x).max(axis=1)
    idx = np.flatnonzero(amp > 10 ** (threshold_db / 20))
    if idx.size == 0:
        return x[:0]
    return x[idx[0] : idx[-1] + 1]


def _ramp(k: int) -> np.ndarray:
    """0 と 1 を含まない k 点の上り坂。"""
    return np.linspace(0.0, 1.0, k + 2, dtype="float32")[1:-1]


def loop_to(x: np.ndarray, n: int, sr: int, xfade: float = 2.0) -> np.ndarray:
    """x を長さ n にする。足りなければ、つなぎ目をクロスフェードにしてループする。"""
    length = len(x)
    if length >= n:
        return x[:n]
    if length == 0:
        return np.zeros((n,) + x.shape[1:], dtype=x.dtype)
    xf = min(round(xfade * sr), length // 2)
    up = _ramp(xf)
    down = 1.0 - up
    if x.ndim > 1:
        up, down = up[:, None], down[:, None]
    out = np.zeros((n + length,) + x.shape[1:], dtype="float32")
    out[:length] = x
    pos = length
    while pos < n:
        out[pos - xf : pos] = out[pos - xf : pos] * down + x[:xf] * up
        out[pos : pos + length - xf] = x[xf:]
        pos += length - xf
    return out[:n]


def _track_audio(
    track: Track, material: np.ndarray, sr: int, level_db: float, target_db: float
) -> np.ndarray | None:
    n = track.end - track.start
    if n <= 0:
        return None
    trimmed = trim_silence(material, sr)
    material_db = rms_db(trimmed)
    if not np.isfinite(material_db):
        return None
    audio = loop_to(trimmed, n, sr)
    audio = audio * np.float32(10 ** ((level_db + target_db - material_db) / 20))
    audio = audio.copy()
    fin, fout = min(track.fade_in, n), min(track.fade_out, n)
    if fin > 0:
        audio[:fin] *= _ramp(fin)[:, None]
    if fout > 0:
        audio[n - fout :] *= _ramp(fout)[::-1][:, None]
    return audio


def render(
    layout: Layout,
    narration: np.ndarray,
    sounds: dict[str, np.ndarray],
    sr: int,
    bgm: BgmConfig,
) -> np.ndarray:
    out = np.zeros((layout.total, 2), dtype="float32")
    level = voice_level_db(narration, sr)
    if not np.isfinite(level):
        level = FALLBACK_VOICE_DB

    for tracks, target in ((layout.ambient, bgm.ambient_db), (layout.music, bgm.music_db)):
        for tr in tracks:
            material = sounds.get(tr.key)
            if material is None:
                continue
            if material.ndim == 1:
                material = np.stack([material, material], axis=1)
            audio = _track_audio(tr, material, sr, level, target)
            if audio is None:
                continue
            start, stop = max(tr.start, 0), min(tr.end, layout.total)
            if stop > start:
                out[start:stop] += audio[start - tr.start : stop - tr.start]

    for src_start, src_end, dst in layout.voice:
        seg = narration[src_start:src_end]
        seg = seg[: max(0, layout.total - dst)]
        out[dst : dst + len(seg)] += seg[:, None]

    peak = float(np.abs(out).max()) if out.size else 0.0
    if peak > PEAK_LIMIT:
        out *= np.float32(PEAK_LIMIT / peak)
    return out


_MIX_KEYS = (
    "ambient_db",
    "music_db",
    "intro_seconds",
    "bridge_seconds",
    "music_fade_in",
    "music_fade_out",
    "ambient_crossfade",
    "mp3_bitrate",
)


def mix_signature(
    narration_sig: str,
    scenes: list[Scene],
    choices: dict[str, Choice],
    bgm: BgmConfig,
) -> str:
    payload = {
        "rule": MIX_RULE,
        "narration": narration_sig,
        "scenes": [[s.chapter, s.index, s.start_para, s.end_para] for s in scenes],
        "choices": {
            k: [c.kind, c.sound.id if c.sound is not None else None]
            for k, c in sorted(choices.items())
        },
        "bgm": {k: getattr(bgm, k) for k in _MIX_KEYS},
    }
    return hashlib.sha1(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def write_credits(path: Path, choices: list[Choice]) -> None:
    """使った音ごとに 1 行（同じ音は 1 回）。一時ファイル経由で書く。"""
    seen: set[int] = set()
    lines: list[str] = []
    for c in choices:
        s = c.sound
        if s is None or s.id in seen:
            continue
        seen.add(s.id)
        lines.append(f"{s.name} by {s.username} — {s.url} — {license_label(s.license)}\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text("".join(lines), encoding="utf-8")
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
