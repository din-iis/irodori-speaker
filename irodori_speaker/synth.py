from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import Config
from .voices import voice_key as _voice_key

_DURATION_RE = re.compile(
    r"info: predicted duration frames=([\d.]+), scale=([\d.]+), using_frames=(\d+)(?: \(([\d.]+)s\))?"
)


def _sha1(payload) -> str:
    return hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def check_vram(cfg: Config) -> None:
    if not cfg.device.startswith("cuda") or cfg.min_free_vram_gb <= 0:
        return
    import torch  # GPU マシンでのみ必要なので遅延 import

    if not torch.cuda.is_available():
        return
    free, _total = torch.cuda.mem_get_info(cfg.device)
    free_gb = free / 2**30
    if free_gb < cfg.min_free_vram_gb:
        raise RuntimeError(
            f"空き VRAM が {free_gb:.1f} GB しかありません（必要 {cfg.min_free_vram_gb} GB）。"
            "LLM サーバー（Ollama なら `ollama stop <model>`）を停止してから再実行してください"
        )


def latent_key(
    voice_key: str, codec_repo: str, normalize_db: float | None, max_ref_seconds: float | None
) -> str:
    return _sha1([voice_key, codec_repo, normalize_db, max_ref_seconds])


def latent_key_for(cfg: Config, voice_key: str) -> str:
    # max_ref_seconds が None のときの実効上限はチェックポイント依存なので、キーに checkpoint を含める
    max_ref = cfg.max_ref_seconds if cfg.max_ref_seconds is not None else f"checkpoint:{cfg.checkpoint}"
    return latent_key(voice_key, cfg.codec_repo, cfg.ref_normalize_db, max_ref)


# モデルに渡す文字列の加工規則。変えると文のキャッシュと章 WAV の署名が作り直しになる
TEXT_RULE = "strip-final-punct-v2"

_CLOSERS = "」』）)】"
_FINAL_PUNCT = "。．、，,"


def tts_text(text: str) -> str:
    """文末の句点・読点（閉じ括弧の直前を含む）を外す。これらで終わると文末に余計な音が入るため。"""
    body = text.rstrip()
    tail = ""
    while body and body[-1] in _CLOSERS:
        tail = body[-1] + tail
        body = body[:-1]
    stripped = body.rstrip(_FINAL_PUNCT)
    if not stripped.strip():
        return text
    return stripped + tail


def cache_key(cfg: Config, text: str, latent_key: str) -> str:
    s = cfg.sampling
    return _sha1(
        {
            "text": text,
            "text_rule": TEXT_RULE,
            "latent": latent_key,
            "checkpoint": cfg.checkpoint,
            "num_steps": s.num_steps,
            "t_schedule_mode": s.t_schedule_mode,
            "sway_coeff": s.sway_coeff,
            "cfg_scale_text": s.cfg_scale_text,
            "cfg_scale_speaker": s.cfg_scale_speaker,
            "duration_scale": s.duration_scale,
            "max_seconds": s.max_seconds,
            "seed": s.seed,
        }
    )


def resolve_checkpoint(checkpoint: str) -> str:
    local = Path(checkpoint).expanduser()
    if local.exists():
        return str(local)
    from irodori_tts.inference_runtime import download_hf_checkpoint

    return download_hf_checkpoint(checkpoint)


_LOADED_KEY = None  # 直近に get_cached_runtime へ渡した RuntimeKey（上流の runtime が載っている目印）


def release_runtime() -> None:
    """上流のキャッシュから runtime を解放する。LLM など別の GPU 処理の前に呼ぶ。"""
    global _LOADED_KEY
    try:
        from irodori_tts.inference_runtime import clear_cached_runtime
    except ImportError:
        return
    clear_cached_runtime()
    _LOADED_KEY = None


def _runtime_key(cfg: Config, checkpoint: str | None = None):
    from irodori_tts.inference_runtime import RuntimeKey

    return RuntimeKey(
        checkpoint=checkpoint if checkpoint is not None else resolve_checkpoint(cfg.checkpoint),
        model_device=cfg.device,
        codec_device=cfg.device,
        model_precision=cfg.precision,
        codec_precision=cfg.precision,
        codec_repo=cfg.codec_repo,
    )


def load_runtime(cfg: Config, checkpoint: str | None = None):
    """上流のキャッシュから runtime を得る。キーが変わるときだけ VRAM を確認する。"""
    global _LOADED_KEY
    from irodori_tts.inference_runtime import clear_cached_runtime, get_cached_runtime

    key = _runtime_key(cfg, checkpoint)
    if key != _LOADED_KEY:
        if _LOADED_KEY is not None:
            # 上流は新 runtime を作ってから旧を unload するので、先に解放して二重ロードと VRAM 不足の誤検知を避ける
            clear_cached_runtime()
            _LOADED_KEY = None
        check_vram(cfg)
    try:
        runtime, _loaded_now = get_cached_runtime(key)
    except BaseException:
        _LOADED_KEY = None
        raise
    _LOADED_KEY = key
    if not runtime.model_cfg.use_speaker_condition_resolved:
        raise ValueError("このチェックポイントは参照音声に対応していません")
    return runtime


def _load_clip(path: Path):
    """(C, T) の tensor とサンプリングレートを返す。torchaudio が失敗したら soundfile で読む。"""
    import torch
    import torchaudio

    try:
        return torchaudio.load(str(path))
    except RuntimeError:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)  # (T, C)
        return torch.from_numpy(data.T.copy()), sr


def encode_voice(runtime, paths: list[Path], cache_dir: Path, cfg: Config) -> Path:
    max_ref = (
        cfg.max_ref_seconds if cfg.max_ref_seconds is not None else float(runtime.default_max_ref_seconds)
    )
    key = latent_key_for(cfg, _voice_key(paths))
    cache_dir = Path(cache_dir)
    path = cache_dir / f"{key}.pt"
    if path.exists():
        return path
    import torch

    print(f"[synth] 参照音声 {len(paths)} クリップを潜在表現に変換します")
    pieces = []
    for p in paths:
        wav, sr = _load_clip(Path(p))
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if len(paths) == 1 and max_ref > 0 and wav.shape[1] > max_ref * sr:
            wav = wav[:, : int(max_ref * sr)]
        pieces.append(
            runtime.codec.encode_waveform(
                wav, sample_rate=sr, normalize_db=cfg.ref_normalize_db, ensure_max=True
            ).cpu()
        )
    latent = torch.cat(pieces, dim=1)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(latent[0], tmp)
    tmp.replace(path)
    return path


def collect_warnings(messages: list[str]) -> list[str]:
    warnings = []
    for m in messages:
        if not m.startswith("info:"):
            warnings.append(m)
            continue
        hit = _DURATION_RE.search(m)
        if hit:
            frames, scale, using = float(hit[1]), float(hit[2]), int(hit[3])
            if frames * scale > using + 0.5:
                secs = f"（約 {hit[4]}s）" if hit[4] else ""
                warnings.append(
                    f"max_seconds で切り詰め: 予測 {hit[1]} フレーム × {hit[2]} > {using}{secs}"
                )
    return warnings


def _sampling_request(**kwargs):
    from irodori_tts.inference_runtime import SamplingRequest

    return SamplingRequest(**kwargs)


def _save_wav(path, audio, sample_rate) -> None:
    from irodori_tts.inference_runtime import save_wav

    save_wav(path, audio, sample_rate)


_SAMPLING_FIELDS = (
    "num_steps", "t_schedule_mode", "sway_coeff", "cfg_scale_text",
    "cfg_scale_speaker", "duration_scale", "max_seconds", "seed",
)


class Synthesizer:
    def __init__(
        self,
        cfg: Config,
        segments_dir: Path,
        voice_paths: list[Path],
        latent_cache_dir: Path,
        warnings_path: Path | None = None,
    ) -> None:
        self.cfg = cfg
        self.segments_dir = Path(segments_dir)
        self.segments_dir.mkdir(parents=True, exist_ok=True)
        self.voice_paths = [Path(p) for p in voice_paths]
        self.latent_cache_dir = Path(latent_cache_dir)
        self.warnings_path = Path(warnings_path) if warnings_path else None
        self.last_warnings: list[str] = []
        self.latent_key = latent_key_for(cfg, _voice_key(self.voice_paths))
        self._checkpoint: str | None = None
        self._latent_path: Path | None = None
        self._count = 0

    @property
    def runtime(self):
        # 上流は別キーで呼ばれると旧 runtime を unload するので、インスタンスには保持しない
        if self._checkpoint is None:
            self._checkpoint = resolve_checkpoint(self.cfg.checkpoint)
        return load_runtime(self.cfg, self._checkpoint)

    @property
    def latent_path(self) -> Path:
        if self._latent_path is None:
            self._latent_path = encode_voice(
                self.runtime, self.voice_paths, self.latent_cache_dir, self.cfg
            )
        return self._latent_path

    def _request(self, text: str, **overrides):
        s = self.cfg.sampling
        params = {f: getattr(s, f) for f in _SAMPLING_FIELDS}
        params.update(overrides)
        return _sampling_request(
            text=tts_text(text),
            ref_latent=str(self.latent_path),
            max_ref_seconds=self.cfg.max_ref_seconds,
            **params,
        )

    def synthesize(self, text: str, force: bool = False) -> Path:
        key = cache_key(self.cfg, text, self.latent_key)
        path = self.segments_dir / f"{key}.wav"
        if path.exists() and not force:
            self.last_warnings = []
            return path
        res = self.runtime.synthesize(self._request(text))
        tmp = path.with_name(f"{key}.tmp.wav")
        _save_wav(tmp, res.audio, res.sample_rate)
        tmp.replace(path)
        self.last_warnings = collect_warnings(list(res.messages))
        for w in self.last_warnings:
            print(f"[synth] 警告: {w}")
        if self.last_warnings and self.warnings_path:
            self.warnings_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.warnings_path, "a", encoding="utf-8") as f:
                for w in self.last_warnings:
                    f.write(f"{key}\t{text}\t{w}\n")
        self._count += 1
        interval = self.cfg.empty_cache_interval
        if interval > 0 and self._count % interval == 0 and self.cfg.device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()
        return path

    def generate_once(self, text: str, **overrides) -> tuple[int, np.ndarray, list[str]]:
        unknown = set(overrides) - set(_SAMPLING_FIELDS)
        if unknown:
            raise TypeError(f"generate_once: 不明な引数 {sorted(unknown)}")
        res = self.runtime.synthesize(self._request(text, **overrides))
        audio = np.asarray(res.audio.detach().cpu().float().numpy(), dtype=np.float32).reshape(-1)
        return int(res.sample_rate), audio, collect_warnings(list(res.messages))
