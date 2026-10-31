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

MIX_RULE = 2  # 2: 音量の物差しを RMS から K 特性の音量（BS.1770）に替えた
OUTRO_SECONDS = 3.0  # 章末の環境音の余韻
PEAK_LIMIT = 10 ** (-1 / 20)
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


LOUDNESS_BLOCK_SECONDS = 0.4
LOUDNESS_OFFSET_DB = -0.691  # BS.1770 の定数。声と素材の両方に同じくかかるので差には効かない
ABSOLUTE_GATE_DB = -70.0
RELATIVE_GATE_DB = -10.0
DUAL_MONO_DB = 10 * np.log10(2)  # モノラルの声を「両チャンネルに同じ信号」として測ったときの差
_BLOCKS_PER_CHUNK = 256  # rfft を一度にかけるブロック数（長い章でもメモリを食わない）


def _k_weight_power(freqs: np.ndarray, sr: int) -> np.ndarray:
    """BS.1770 の K 特性（高域シェルフ + 低域 RLB ハイパス）の |H(f)|²。

    BS.1770 は 48 kHz の係数だけを定めるので、pyloudnorm と同じ式（双一次変換）で
    任意の sr 向けの係数を求め、その周波数応答を直接評価する。
    """
    w = 2 * np.pi * freqs / sr
    z1 = np.exp(-1j * w)
    z2 = z1 * z1

    # 高域シェルフ
    gain, q, fc = 3.999843853973347, 0.7071752369554196, 1681.9744509555319
    k = np.tan(np.pi * fc / sr)
    vh = 10 ** (gain / 20)
    vb = vh**0.4996667741545416
    a0 = 1 + k / q + k * k
    b = ((vh + vb * k / q + k * k) / a0, 2 * (k * k - vh) / a0, (vh - vb * k / q + k * k) / a0)
    a = (1.0, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0)
    shelf = (b[0] + b[1] * z1 + b[2] * z2) / (a[0] + a[1] * z1 + a[2] * z2)

    # 低域のハイパス
    q, fc = 0.5003270373238773, 38.13547087602444
    k = np.tan(np.pi * fc / sr)
    a0 = 1 + k / q + k * k
    a = (1.0, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0)
    high = (1 - 2 * z1 + z2) / (a[0] + a[1] * z1 + a[2] * z2)

    return np.abs(shelf * high) ** 2


def _block_mean_squares(x: np.ndarray, sr: int) -> np.ndarray:
    """400 ms のブロックごとの K 重み付き平均二乗（チャンネルの和）。

    フィルタを時間領域でかけず、ブロックの rfft のパワーに |H|² をかけて Parseval で
    平均二乗を出す。ブロックは重ならない（BS.1770 は 75% 重ねるが、全サンプルを一度ずつ
    使うほうが速く、定常な環境音や朗読では結果がほぼ変わらない）。末尾の端数は捨てる。
    ただし全体が 1 ブロックに満たなければ、全体を 1 ブロックとして測る。
    """
    if x.ndim == 1:
        x = x[:, None]
    block = max(1, round(sr * LOUDNESS_BLOCK_SECONDS))
    count = len(x) // block
    if count == 0:
        block, count = len(x), 1
    freqs = np.fft.rfftfreq(block, 1 / sr)
    weight = _k_weight_power(freqs, sr)
    # rfft の片側スペクトルを両側のパワーに戻す重み（直流とナイキストは 1、他は 2）
    fold = np.full(len(freqs), 2.0)
    fold[0] = 1.0
    if block % 2 == 0:
        fold[-1] = 1.0
    weight = weight * fold / (block * block)

    out = np.zeros(count)
    for ch in range(x.shape[1]):
        for i in range(0, count, _BLOCKS_PER_CHUNK):
            n = min(_BLOCKS_PER_CHUNK, count - i)
            seg = x[i * block : (i + n) * block, ch].astype("float64").reshape(n, block)
            out[i : i + n] += (np.abs(np.fft.rfft(seg, axis=1)) ** 2) @ weight
    return out


def loudness_db(x: np.ndarray, sr: int) -> float:
    """BS.1770 の考え方の簡略版の音量（LUFS 相当の dB）。音が無ければ -inf。

    K 特性で重み付けし、絶対ゲート（-70）と相対ゲート（-10）を通ったブロックの平均。
    ステレオはチャンネルごとの平均二乗の和（BS.1770 と同じ、L/R の重みは 1）で、モノラル
    はそのまま測る。同じ信号を両チャンネルに入れた音は、モノラルより 3.01 dB 大きくなる。
    """
    if x.size == 0:
        return -np.inf
    ms = _block_mean_squares(x, sr)
    with np.errstate(divide="ignore"):
        lufs = LOUDNESS_OFFSET_DB + 10 * np.log10(ms)
    kept = ms[lufs > ABSOLUTE_GATE_DB]
    if kept.size == 0:
        return -np.inf
    relative = LOUDNESS_OFFSET_DB + 10 * np.log10(np.mean(kept)) + RELATIVE_GATE_DB
    kept = ms[(lufs > ABSOLUTE_GATE_DB) & (lufs > relative)]
    return float(LOUDNESS_OFFSET_DB + 10 * np.log10(np.mean(kept)))


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
    # 素材はステレオで測る。声は dual mono として測るので（render）、物差しが揃う
    material_db = loudness_db(trimmed, sr)
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
    # 声は両チャンネルに同じ信号で入る。ステレオの素材と同じ物差しにするため dual mono として測る
    level = loudness_db(narration, sr) + DUAL_MONO_DB
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
