from __future__ import annotations

import hashlib
from pathlib import Path

# 上流がアダプタなしと解釈する値
NO_LORA = ("none", "off", "base")

_CONFIG = "adapter_config.json"
_WEIGHTS = ("adapter_model.safetensors", "adapter_model.bin")


def normalize_lora(value: str | None) -> str | None:
    value = (value or "").strip()
    if not value or value.lower() in NO_LORA:
        return None
    return value


def resolve_lora_adapter(value: str | None) -> Path | None:
    value = normalize_lora(value)
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_dir():
        raise ValueError(f"LoRA のアダプタのディレクトリが見つかりません: {value}")
    if not (path / _CONFIG).is_file():
        raise ValueError(f"LoRA のアダプタに {_CONFIG} がありません: {value}")
    if not any((path / w).is_file() for w in _WEIGHTS):
        raise ValueError(f"LoRA のアダプタに {_WEIGHTS[0]}（または {_WEIGHTS[1]}）がありません: {value}")
    return path


def lora_stamp(path: Path) -> tuple:
    """設定と重みの (名前, サイズ, mtime_ns)。上書きの検出用で、中身は読まない。"""
    stamp = []
    for name in (_CONFIG, *_WEIGHTS):
        f = Path(path) / name
        if f.is_file():
            st = f.stat()
            stamp.append((name, st.st_size, st.st_mtime_ns))
    return tuple(stamp)


def lora_key(path: Path) -> str:
    """設定と重みのファイル名と中身の sha1（mtime は使わない）。"""
    h = hashlib.sha1()
    for name in (_CONFIG, *_WEIGHTS):
        f = Path(path) / name
        if not f.is_file():
            continue
        h.update(name.encode("utf-8") + b"\0")
        with open(f, "rb") as fp:
            for chunk in iter(lambda: fp.read(1 << 20), b""):
                h.update(chunk)
        h.update(b"\0")
    return h.hexdigest()
