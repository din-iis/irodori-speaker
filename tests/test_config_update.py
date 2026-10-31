from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from irodori_speaker.cli import main
from irodori_speaker.config import Config, _build
from irodori_speaker.config_update import update_config_text

EXAMPLE = Path(__file__).resolve().parents[1] / "config.example.yaml"


def example_text() -> str:
    return EXAMPLE.read_text(encoding="utf-8")


def load_text(text: str) -> Config:
    return _build(Config, yaml.safe_load(text) or {})


def drop_lines(text: str, *prefixes: str) -> str:
    """指定した接頭辞で始まる行を取り除く。"""
    return "".join(
        line for line in text.splitlines(keepends=True) if not line.startswith(prefixes)
    )


def drop_section(text: str, name: str) -> str:
    """最上位の節（name: とその中の行）を取り除く。"""
    out, skipping = [], False
    for line in text.splitlines(keepends=True):
        if line.startswith(f"{name}:"):
            skipping = True
            continue
        if skipping and (line.startswith((" ", "\t")) or not line.strip()):
            continue
        skipping = False
        out.append(line)
    return "".join(out)


# ---- update_config_text ----


def test_missing_section_is_added_whole_at_end():
    text = drop_section(example_text(), "export")
    new, added, manual = update_config_text(text, example_text())
    assert added == ["export"]
    assert manual == []
    assert new.startswith(text.rstrip("\n"))
    assert new.endswith(
        "export:\n"
        "  format: m4b               # m4b（章マーカー付き AAC。Apple Books 向け）| mp3\n"
        "  bitrate: 64k              # 朗読はモノラル 64k で十分。BGM 付きは 96k 以上を推奨\n"
        "  title: null               # 本の題名。null なら小説ファイル名（stem）\n"
    )
    # 末尾との間に空行が 1 つ
    assert "\n\nexport:\n" in new


def test_missing_top_level_key_brings_preceding_comment():
    text = drop_lines(
        example_text(),
        "# 未統合の話者 LoRA",
        "lora_adapter:",
    )
    new, added, _ = update_config_text(text, example_text())
    assert added == ["lora_adapter"]
    assert "# 未統合の話者 LoRA のアダプタのディレクトリ" in new
    lines = new.splitlines()
    i = next(i for i, ln in enumerate(lines) if ln.startswith("lora_adapter:"))
    assert lines[i - 1].startswith("# 未統合の話者 LoRA")
    assert "例: adapters/narrator" in lines[i]  # 行末コメントも付く


def test_missing_keys_in_section_go_to_end_of_that_section():
    text = drop_lines(example_text(), "  title: 1.0", "  scene:")
    new, added, _ = update_config_text(text, example_text())
    assert added == ["silence.title", "silence.scene"]
    lines = new.splitlines()
    i_ellipsis = next(i for i, ln in enumerate(lines) if ln.startswith("  ellipsis:"))
    assert lines[i_ellipsis + 1].startswith("  title: 1.0")
    assert "章題の後" in lines[i_ellipsis + 1]
    assert lines[i_ellipsis + 2].startswith("  scene: 2.0")
    assert lines[i_ellipsis + 3] == ""
    assert lines[i_ellipsis + 4].startswith("bgm:")


def test_user_edits_comments_and_order_are_preserved():
    text = example_text().replace(
        "  ambient_db: -20.0         # 朗読に対する環境音の音量",
        "  ambient_db: -26.0 # 自分で変えた",
    )
    text = text.replace("llm:\n", "# 自分のメモ\nllm:\n")
    text = drop_lines(text, "  title: 1.0")
    new, added, _ = update_config_text(text, example_text())
    assert added == ["silence.title"]
    assert "  ambient_db: -26.0 # 自分で変えた\n" in new
    assert "# 自分のメモ\nllm:\n" in new
    # 足した行を除くと元のテキストと完全に一致する（並び順も同じ）
    rest = "".join(
        ln for ln in new.splitlines(keepends=True) if not ln.startswith("  title: 1.0")
    )
    assert rest == text
    assert load_text(new).bgm.ambient_db == -26.0


def test_voice_is_not_added_but_reported():
    text = drop_lines(example_text(), "voice:")
    new, added, manual = update_config_text(text, example_text())
    assert added == []
    assert manual == [("voice", "narrator")]
    assert not re.search(r"^voice:", new, re.M)
    assert new == text


def test_commented_out_lines_are_not_added():
    new, added, _ = update_config_text("", example_text())
    assert "split" in added and "llm" in added
    assert "chapter_regex" not in new
    assert "# server_command:" not in new
    assert '"llama-server", "-m"' not in new
    # 説明のコメントは残る
    assert "# 感情 × 強度(1/2/3)" in new


def test_commented_out_lines_are_not_added_for_single_key():
    text = drop_lines(example_text(), "  max_chars:", "  server_start_timeout:")
    new, added, _ = update_config_text(text, example_text())
    assert added == ["llm.server_start_timeout", "split.max_chars"]
    assert "  server_start_timeout: 300\n" in new
    # 元から config にあったコメントアウト行は 1 つだけ（増えていない）
    assert new.count("# server_command:") == 1
    assert new.count("# chapter_regex:") == 1


def test_asdict_is_unchanged_after_update():
    text = drop_section(drop_lines(example_text(), "voice:", "  title: 1.0"), "check")
    before = asdict(load_text(text))
    new, added, manual = update_config_text(text, example_text())
    assert added == ["silence.title", "check"]
    assert asdict(load_text(new)) == before


def test_nothing_missing_returns_text_unchanged():
    text = example_text()
    new, added, manual = update_config_text(text, text)
    assert new == text
    assert added == []
    assert manual == []


def test_indent_of_section_is_respected():
    text = (
        "checkpoint: x\n"
        "silence:\n"
        "    sentence: 0.25\n"
        "    paragraph: 0.6\n"
        "output_dir: out\n"
    )
    new, added, _ = update_config_text(text, example_text())
    assert "silence.title" in added and "silence.chapter" in added
    assert "\n    title: 1.0" in new
    assert "\n  title: 1.0" not in new
    lines = new.splitlines()
    i = lines.index("    paragraph: 0.6")
    assert lines[i + 1].startswith("    chapter: 1.0")
    # 節の中の行は全部 4 スペース
    j = lines.index("output_dir: out")
    assert all(ln.startswith("    ") for ln in lines[i + 1 : j])
    assert load_text(new).silence.title == 1.0


@pytest.mark.parametrize("empty", ["", "{}\n", "# コメントだけ\n"])
def test_empty_config_gets_everything_but_voice(empty):
    new, added, manual = update_config_text(empty, example_text())
    assert manual == [("voice", "narrator")]
    assert "voice" not in added
    assert {"checkpoint", "sampling", "llm", "export", "output_dir"} <= set(added)
    assert not re.search(r"^voice:", new, re.M)
    assert asdict(load_text(new)) == asdict(Config())
    assert not new.startswith("\n")


def test_section_with_null_body_gets_keys():
    text = "export:\noutput_dir: out\n"
    new, added, _ = update_config_text(text, example_text())
    assert {"export.format", "export.bitrate", "export.title"} <= set(added)
    assert "export:\n  format: m4b" in new
    assert asdict(load_text(new)) == asdict(Config())


def test_value_differing_from_default_in_section_is_not_added(tmp_path):
    example = tmp_path / "ex.yaml"
    example.write_text("silence:\n  sentence: 0.25\n  title: 9.9\n", encoding="utf-8")
    new, added, manual = update_config_text("silence:\n  sentence: 0.25\n", example.read_text())
    assert added == []
    assert manual == [("silence.title", 9.9)]
    assert new == "silence:\n  sentence: 0.25\n"


def test_invalid_existing_config_raises():
    with pytest.raises(ValueError):
        update_config_text("not_a_key: 1\n", example_text())


# ---- CLI ----


def write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_cli_dry_run_does_not_modify(tmp_path, capsys):
    text = drop_section(example_text(), "export")
    path = write_config(tmp_path, text)
    rc = main(["config", "--update", "-c", str(path), "--dry-run"])
    assert rc == 0
    assert path.read_text(encoding="utf-8") == text
    assert not (tmp_path / "config.yaml.bak").exists()
    assert "export" in capsys.readouterr().out


def test_cli_update_writes_file_and_backup(tmp_path, capsys):
    text = drop_section(drop_lines(example_text(), "voice:"), "export")
    path = write_config(tmp_path, text)
    rc = main(["config", "--update", "-c", str(path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert (tmp_path / "config.yaml.bak").read_text(encoding="utf-8") == text
    new = path.read_text(encoding="utf-8")
    assert "export:\n  format: m4b" in new
    assert "export" in out
    assert "手で確かめてください: voice（例の値 narrator）" in out
    assert not list(tmp_path.glob("*.tmp"))


def test_cli_nothing_to_add(tmp_path, capsys):
    path = write_config(tmp_path, example_text())
    rc = main(["config", "--update", "-c", str(path)])
    assert rc == 0
    assert "足りない設定はありません" in capsys.readouterr().out
    assert not (tmp_path / "config.yaml.bak").exists()
    assert path.read_text(encoding="utf-8") == example_text()


def test_cli_missing_config_file_exits_1(tmp_path, capsys):
    rc = main(["config", "--update", "-c", str(tmp_path / "none.yaml")])
    assert rc == 1
    assert "none.yaml" in capsys.readouterr().err


def test_cli_invalid_config_exits_1(tmp_path, capsys):
    path = write_config(tmp_path, "not_a_key: 1\n")
    rc = main(["config", "--update", "-c", str(path)])
    assert rc == 1
    assert capsys.readouterr().err != ""
    assert path.read_text(encoding="utf-8") == "not_a_key: 1\n"


def test_cli_default_config_path_is_config_yaml(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    text = drop_section(example_text(), "export")
    path = write_config(tmp_path, text)
    assert main(["config", "--update"]) == 0
    assert "export:" in path.read_text(encoding="utf-8")


def test_cli_custom_example(tmp_path, capsys):
    example = tmp_path / "my.example.yaml"
    example.write_text("output_dir: out\n", encoding="utf-8")
    path = write_config(tmp_path, "device: cpu\n")
    rc = main(["config", "--update", "-c", str(path), "--example", str(example)])
    assert rc == 0
    assert path.read_text(encoding="utf-8") == "device: cpu\n\noutput_dir: out\n"


def test_cli_missing_example_exits_1(tmp_path, capsys):
    path = write_config(tmp_path, "device: cpu\n")
    rc = main(["config", "--update", "-c", str(path), "--example", str(tmp_path / "x.yaml")])
    assert rc == 1


def test_cli_config_takes_no_novel():
    with pytest.raises(SystemExit) as e:
        main(["config", "--update", "novel.txt"])
    assert e.value.code == 2


def test_cli_config_help_is_japanese(capsys):
    with pytest.raises(SystemExit) as e:
        main(["config", "--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "--update" in out and "--dry-run" in out and "--example" in out
    assert re.search(r"[ぁ-んァ-ン一-龥]", out.split("options:")[-1].split("optional arguments:")[-1])
    assert "novel" not in out


def test_unindented_comment_inside_section_does_not_end_it():
    text = (
        "silence:\n"
        "# メモ\n"
        "    sentence: 0.25\n"
        "    paragraph: 0.6\n"
        "output_dir: out\n"
    )
    new, added, _ = update_config_text(text, example_text())
    assert "silence.title" in added
    lines = new.splitlines()
    i = lines.index("    paragraph: 0.6")
    assert lines[i + 1].startswith("    chapter: 1.0")
    assert "\n  title: 1.0" not in new
    assert "# メモ" in new
    assert asdict(load_text(new)) == asdict(Config())


@pytest.mark.parametrize("mode", [0o600, 0o644])
def test_cli_update_keeps_file_permissions(tmp_path, mode):
    import os

    path = write_config(tmp_path, drop_section(example_text(), "export"))
    path.chmod(mode)
    old = os.umask(0o022)
    try:
        assert main(["config", "--update", "-c", str(path)]) == 0
    finally:
        os.umask(old)
    assert "export:" in path.read_text(encoding="utf-8")
    assert path.stat().st_mode & 0o777 == mode
    assert sorted(p.name for p in tmp_path.iterdir()) == ["config.yaml", "config.yaml.bak"]


def test_型の違う値は既定値と同じとみなさず足さない():
    ex = example_text().replace("narration_intensity_shift: 1", "narration_intensity_shift: 1.0", 1)
    cfg = drop_lines(example_text(), "narration_intensity_shift:")
    new_text, added, manual = update_config_text(cfg, ex)
    assert "narration_intensity_shift" not in added
    assert any(k == "narration_intensity_shift" for k, _ in manual)
    assert new_text == cfg


def test_シンボリックリンクの設定はリンク先を書き換えてリンクを残す(tmp_path):
    real = tmp_path / "shared.yaml"
    real.write_text(drop_section(example_text(), "export"), encoding="utf-8")
    link = tmp_path / "config.yaml"
    link.symlink_to(real)
    assert main(["config", "--update", "-c", str(link)]) == 0
    assert link.is_symlink()
    assert "export:" in real.read_text(encoding="utf-8")


def test_CRLF_の設定は既存の行の改行を保ったまま足す(tmp_path):
    path = tmp_path / "config.yaml"
    body = drop_section(example_text(), "export").replace("\n", "\r\n")
    path.write_bytes(body.encode("utf-8"))
    assert main(["config", "--update", "-c", str(path)]) == 0
    raw = path.read_bytes().decode("utf-8")
    assert raw.startswith(body)
    assert "\n" not in raw.replace("\r\n", "")
