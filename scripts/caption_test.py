"""キャプションで感情を指示できるかを試す。

本ツールの config.yaml（声・sampling・seed）と文の加工（文末の句点を外す）をそのまま使い、
キャプションと cfg_scale_caption だけを変えて生成する。--checkpoint を複数渡すと順に比べる。
比べるため、キャプションの代わりに emotion_emoji の強度 3 の絵文字を文頭に付けた音声も作る。
--no-ref は参照音声を外す（参照音声が感情を抑えているかの切り分け用）。

  uv run --no-sync python scripts/caption_test.py -c config.yaml \
    --checkpoint Aratako/Irodori-TTS-v4.1-Small --checkpoint /path/to/merged.safetensors
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from irodori_speaker.config import load_config
from irodori_speaker.synth import Synthesizer, _save_wav
from irodori_speaker.voices import resolve_voice

TEXTS = [
    "どうして何も言ってくれなかったの",
    "そう言って、彼は静かに部屋を出ていった",
]
CAPTIONS = {
    "none": None,
    "angry": "激しく怒って、強い口調で相手を問い詰めている",
    "sad": "悲しみをこらえ、声を震わせながら話している",
    "happy": "嬉しくて、明るく弾んだ声で話している",
    "whisper": "相手の耳元で、ささやくように小声で話している",
}
SCALES = [3.0, 5.0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-c", "--config", default="config.yaml")
    p.add_argument("--checkpoint", action="append", help="比べる checkpoint（複数可）。省略時は config の checkpoint")
    p.add_argument("--out", default=None, help="出力先（既定: <output_dir>/caption_test）")
    p.add_argument("--no-ref", action="store_true", help="参照音声を渡さずに生成する")
    args = p.parse_args()

    cfg = load_config(args.config)
    voice_paths = resolve_voice(Path(cfg.voices_dir), cfg.voice)
    out = Path(args.out) if args.out else Path(cfg.output_dir) / "caption_test"
    for i, ckpt in enumerate(args.checkpoint or [cfg.checkpoint]):
        synth = Synthesizer(
            dataclasses.replace(cfg, checkpoint=ckpt),
            out / ".segments", voice_paths, Path(cfg.output_dir) / ".voices",
        )
        d = out / f"{i}_{Path(ckpt).name.removesuffix('.safetensors')}{'_noref' if args.no_ref else ''}"
        d.mkdir(parents=True, exist_ok=True)

        def generate(path: Path, text: str, caption: str | None, scale: float) -> None:
            req = synth._request(text, caption=caption, cfg_scale_caption=scale)
            if args.no_ref:
                req.ref_latent, req.no_ref = None, True
            res = synth.runtime.synthesize(req)
            _save_wav(path, res.audio, res.sample_rate)
            print(f"[caption_test] {path}  text={text!r} caption={caption!r}")

        for t, text in enumerate(TEXTS, 1):
            for name, caption in CAPTIONS.items():
                if caption is None:
                    generate(d / f"{t}_{name}.wav", text, None, 3.0)
                    continue
                for scale in SCALES:
                    generate(d / f"{t}_{name}_cap{scale:g}.wav", text, caption, scale)
                emoji = cfg.emotion_emoji.get(name, ["", "", ""])[2]
                if emoji:
                    generate(d / f"{t}_{name}_emoji.wav", emoji + text, None, 3.0)


if __name__ == "__main__":
    main()
