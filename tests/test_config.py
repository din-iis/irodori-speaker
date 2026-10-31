from pathlib import Path

import pytest

from irodori_speaker.config import EMOTIONS, load_config

ROOT = Path(__file__).resolve().parent.parent


def test_defaults():
    cfg = load_config(None)
    assert cfg.checkpoint == "Aratako/Irodori-TTS-v4.1-Small"
    assert cfg.llm.model == "qwen3.5:4b-q8_0"
    assert cfg.split.max_chars == 40
    assert cfg.sampling.duration_scale == 0.8
    assert list(cfg.emotion_emoji) == EMOTIONS
    assert len(EMOTIONS) == 10
    assert all(len(v) == 3 for v in cfg.emotion_emoji.values())


def test_default_emoji_is_a_copy():
    a, b = load_config(None), load_config(None)
    a.emotion_emoji["happy"][1] = "X"
    assert b.emotion_emoji["happy"][1] == "😊"


def test_example_yaml_loads():
    cfg = load_config(ROOT / "config.example.yaml")
    assert cfg.voice == "narrator"
    assert cfg.llm.api == "ollama"
    assert cfg.sampling.seed == 42
    assert cfg.emotion_emoji == load_config(None).emotion_emoji


def test_unknown_top_level_key(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("bogus: 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(p)


def test_unknown_llm_key(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("llm:\n  bogus: 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(p)


def test_emotion_emoji_requires_three_items(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text('emotion_emoji:\n  happy: ["", "😊"]\n', encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(p)


def test_server_command_is_list(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text('llm:\n  server_command: ["llama-server", "--port", "8080"]\n', encoding="utf-8")
    cfg = load_config(p)
    assert cfg.llm.server_command == ["llama-server", "--port", "8080"]


@pytest.mark.parametrize("value", ["0", "-1", "true", "'40'", "1.5"])
def test_max_chars_は_1_以上の整数でなければ拒否する(tmp_path, value):
    p = tmp_path / "c.yaml"
    p.write_text(f"split:\n  max_chars: {value}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="max_chars"):
        load_config(p)


def test_bgm_defaults():
    cfg = load_config(None)
    assert cfg.bgm.ambient_db == -20.0 and cfg.bgm.bridge_seconds == 8.0
    assert cfg.bgm.music_fade_in == 2.0 and cfg.bgm.music_fade_out == 3.0


def test_bgm_section_parsed_and_unknown_key_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("bgm:\n  ambient_db: -24\n", encoding="utf-8")
    assert load_config(p).bgm.ambient_db == -24
    p.write_text("bgm:\n  nope: 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(p)


def test_bgm_bridge_must_fit_fades(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("bgm:\n  bridge_seconds: 4\n", encoding="utf-8")  # 2 + 3 > 4
    with pytest.raises(ValueError):
        load_config(p)


def test_bgm_intro_must_fit_fades(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("bgm:\n  intro_seconds: 4\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(p)
