# irodori-speaker

小説のテキストファイルを、[Irodori-TTS](https://github.com/Aratako/Irodori-TTS)（日本語特化のオープンソース TTS）で読み上げます。
声は参照音声（ゼロショットの声クローン）から作り、感情はローカル LLM（Ollama）が句ごとに推定して、Irodori-TTS が解釈する絵文字として本文に挿入します。

- 合成は GPU マシン（Windows + WSL2、RTX 3060 Ti 8GB を想定）で行います。Mac は開発・テストと、LLM だけを使う `script` ステップ用です。
- 設計: `docs/superpowers/specs/2026-10-24-irodori-speaker-design.md`

## 仕組み

処理は独立した 2 つのステップに分かれています。

1. **台本（`script`）**: 小説を章・段落・文に分け、LLM が句ごとの感情を推定して、絵文字入りの読み上げ本文を `script.tsv` に書き出します。LLM だけを使い、torch は要りません。終わると LLM を VRAM から解放します。
2. **音声化（`synth`）**: `script.tsv` の `text` 列を文ごとに Irodori-TTS で合成し、章ごとの WAV と全体の WAV に結合します。TTS だけを使い、LLM には接続しません。

`run` は 1 → 2 を続けて実行します。8GB の GPU には LLM と TTS を同時に載せられないので、台本を作り終えて LLM を解放してから TTS を読み込みます。2 つを分けてあるので、台本を手で直してから音声化できます。

分割では、章の見出し行は 1 文の `title`、文字の無い行（`＊`・`◇　◇　◇`・`……` など）は `break` になります。`title` は既定で読み上げ（`read_chapter_title`）、`break` は合成せず無音の間にします。

### 句読点ごとの絵文字

Irodori-TTS は、本文中の絵文字（😊 😠 👂 など）で演技を制御します。文頭に 1 個だけ置くのではなく、**句読点（「、。！？…」）で区切った句ごとに**感情を推定して絵文字を置きます。

```
入力:   「やった！ついに見つけたぞ！！」
台本:   「😲やった！😆ついに見つけたぞ！！
```

（実際の絵文字は LLM の推定と `emotion_emoji` の表で決まります。上は例です。）

- 感情は 10 種（neutral / happy / sad / angry / surprised / fearful / tender / whisper / excited / serious）× 強度 1〜3 です。`emotion_emoji` の表で絵文字に変換します。強度 1 は絵文字なしです。
- 地の文は強度を `narration_intensity_shift`（既定 1）だけ下げます。地の文はほぼ素の朗読になり、台詞に感情が乗ります。
- 絵文字は句の先頭（句が「『（で始まるなら開き括弧の直後）に置きます。同じ文の中で直前の句と絵文字が同じなら置きません。文は別々に合成するので、文の先頭の句には必ず置きます。
- この配置は本ツールの仮定です（上流の絵文字の説明は配置ルールを定めていません）。効きは WebUI の「感情プリセット総当たり」で確かめてください。

## セットアップ

### GPU マシン（合成）

前提: Windows + WSL2（Ubuntu など）、NVIDIA GPU（CUDA 12.8 対応のドライバー）、[uv](https://docs.astral.sh/uv/)。

```bash
# 1. FFmpeg（torchcodec が必要とする）
sudo apt install ffmpeg

# 2. 依存関係（Mac と同じ uv.lock のコミットを使うため --frozen）
uv sync --frozen --extra tts --extra cu128 --extra webui
# BGM も付けるなら --extra bgm も足す（WebUI の「BGM を付ける」にも要る）

# 3. WSL2 側に Ollama と LLM（Q8_0 で約 5.2GB）
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3.5:4b-q8_0

# 4. 設定
cp config.example.yaml config.yaml    # voice などを編集
```

以後は `uv run --no-sync python -m irodori_speaker ...` で実行します（`--no-sync` を付けると、`uv run` が環境を同期し直さず、上で入れた GPU 用の torch をそのまま使います）。

初回の合成時に、Irodori-TTS のチェックポイントとコーデック（`Aratako/Semantic-DACVAE-Japanese-32dim`）を Hugging Face から自動でダウンロードします。

### 参照音声（声）の置き方

`voices/` に参照音声を置き、`config.yaml` の `voice` で選びます。次のどちらでも構いません。

```
voices/narrator.wav          # 1 ファイル → voice: narrator
voices/narrator/             # フォルダ内の複数クリップ → voice: narrator
  01.wav                     #   名前順に連結して 1 つの声にする
  02.wav
```

- 対応する拡張子: `.wav .flac .mp3 .m4a .ogg .opus .aac .webm`
- 1 つの声で地の文も台詞も全部読みます（登場人物ごとの声の割り当ては未対応です）。
- 参照音声は初回の合成で一度だけ潜在表現（`out/.voices/<sha1>.pt`）に変換してキャッシュします。ファイル名と中身が同じなら、`voices/` を rsync し直してもキャッシュは有効です。
- `voices/` の中身は git に入れません（`.gitignore` 済み）。GPU マシンに直接置くか、rsync で送ってください。

### 参照音声を使わない（`voice: none`）

`voice` に予約語 `none` を書く（CLI は `--voice none`、WebUI は声の欄の先頭の「none」）と、参照音声を渡さずに合成します（上流の `no_ref=True`）。

- **話者 LoRA を統合した checkpoint と組み合わせる設定です。** v4.1-Small は参照音声を渡すと、絵文字でもキャプションでも感情がほとんど出ませんでした。話者 LoRA を統合した checkpoint を参照音声なしで使うと、声が文ごとにぶれず、絵文字の感情も効きました。
- **ベースモデルでは使わないでください。** 参照音声がないと声が定まらず、文ごとに別人になります。
- 参照音声の変換（潜在表現のキャッシュ）は行いません。文のキャッシュと章 WAV の署名は参照音声ありと別になるので、`none` に切り替えると作り直しになります。
- `none` は予約語です。`voices/none.wav` があっても参照音声としては使われず、WebUI の「声を追加」も ID `none` を拒否します。`voice` を未指定（`null`）にしたときは、従来どおり案内を出して止まります。

### Mac（開発・テスト）

Mac には torch も GPU も入れません。`script` ステップ（Ollama があれば LLM 付きで、無ければ `--no-emotion`）とテストが動きます。

```bash
uv sync --extra dev
uv run pytest -q
```

WebUI の画面だけ確認するなら（生成はできません）:

```bash
uv run --with gradio python -m irodori_speaker.webui
```

## 使い方（CLI）

どのサブコマンドも `novel.txt` を受け取り、作業ディレクトリは `out/<novel の拡張子を除いた名前>/` です。順に実行してください。

```bash
# 1. 台本: 分割 + 句ごとの感情 → out/<book>/script.tsv, chapters.json
uv run --no-sync python -m irodori_speaker script novel.txt -c config.yaml

# （script.tsv の text 列を手で直す）

# 2. 音声化: script.tsv → 章ごとの WAV と全体の WAV
uv run --no-sync python -m irodori_speaker synth novel.txt -c config.yaml

# 一括: 1 → 2 を続けて実行（間で LLM を解放してから TTS を読み込む）
uv run --no-sync python -m irodori_speaker run novel.txt -c config.yaml

# 3. 聴く形式に書き出す: 章マーカー付き M4B（または MP3）→ out/<book>/export/
uv run --no-sync python -m irodori_speaker export novel.txt -c config.yaml
```

`script` は先頭に「N 章 / N 段落 / N 文」を、`synth` は使う声（クリップ数と合計秒数）を表示し、終了時に `warnings.log` の件数を表示します。

### オプション

共通（`script` / `synth` / `run`）:

| オプション | 内容 |
|---|---|
| `-c`, `--config` | 設定 YAML。省略すると組み込みの既定値 |
| `--chapters 1-3,5` | 対象の章（1 始まり） |

`script` と `run`:

| オプション | 内容 |
|---|---|
| `--no-emotion` | LLM を使わず全て neutral にする（`text` = 原文）。Mac で LLM なしに動かすときに使う |
| `--reannotate` | script.tsv の手直しを捨てて、感情と絵文字を作り直す |
| `--llm-model NAME` | LLM のモデル名（config の `llm.model` を上書き） |
| `--think` | LLM の思考モードを有効にする（精度は上がるかもしれないが数倍遅い） |

`synth` と `run`:

| オプション | 内容 |
|---|---|
| `--force` | 合成済みの文も作り直す |
| `--voice ID` | 声の ID（config の `voice` を上書き）。`voices/` 内の ID のほか、ファイルやフォルダのパスも指定できる。`none` なら参照音声を使わない |
| `--checkpoint ID` | TTS のチェックポイント（config の `checkpoint` を上書き） |

```bash
# 例: 1〜3 章だけ台本を作る / 別の LLM を使う / 声を変えて作り直す
uv run --no-sync python -m irodori_speaker script novel.txt -c config.yaml --chapters 1-3
uv run --no-sync python -m irodori_speaker script novel.txt -c config.yaml --llm-model qwen3.5:4b-q4_K_M
uv run --no-sync python -m irodori_speaker synth novel.txt -c config.yaml --voice narrator2 --force
```

`check`（[品質チェック](#品質チェックと取り直しcheck)）:

| オプション | 内容 |
|---|---|
| `--asr` | 音声認識（kotoba-whisper）で文字誤り率も調べる。transformers が要る |
| `--voice ID` | `synth --voice` で作った本の鍵に合わせる（config の `voice` を上書き）。`none` も指定できる |
| `--checkpoint ID` | `synth --checkpoint` で作った本の鍵に合わせる（config の `checkpoint` を上書き） |
| `--lora-adapter DIR` | `synth --lora-adapter` で作った本の鍵に合わせる。読み辞書（`readings`・`readings.yaml`）も `synth` と同じものを使う |

`--help` で各サブコマンドのオプションを確認できます。

### 出力

```
out/
  .voices/<sha1>.pt            参照音声の潜在表現のキャッシュ
  <book>/
    chapters.json              章のタイトル（script が書き、synth が読む）
    script.tsv                 台本。手で直すと次回の音声化に反映される
    annotations.jsonl          LLM の推定結果のキャッシュ（成功した段落のみ）
    segments/<sha1>.wav        文ごとの合成結果のキャッシュ（途中で止めても続きから再開できる）
    warnings.log               合成時の警告（文が長すぎて切り詰められた等）
    chapters/NN_タイトル.wav   章ごとの WAV
    <book>_full.wav            全体の WAV
    export/                    export の出力（<book>.m4b、または chapters/ と <book>.mp3。--bgm は _bgm 付き）
```

`--chapters` で一部の章だけ合成したときは、全章の WAV が揃っていて台本と一致している場合にだけ `<book>_full.wav` を作ります。揃っていなければ古い通し WAV は削除されます。

### script.tsv を手直しする

`script.tsv` はタブ区切りで、列は次のとおりです。

| 列 | 内容 |
|---|---|
| `chapter` / `para` / `seg` | 章・段落・文の番号（0 始まり）。行のキー |
| `kind` | `narration`（地の文）・`dialogue`（台詞）・`title`（章題）・`break`（場面の区切り。`text` は常に空で、手直ししても読まれない） |
| `labels` | 句ごとの `感情:強度` を `\|` で並べた参考情報（例: `happy:2\|neutral:1`）。音声には使われない |
| `text` | **絵文字入りの読み上げ本文。手直しはここで行う** |
| `source` | 分割直後の原文。小説本文との一致判定に使うので**編集しない** |
| `seed` | その文だけの合成 seed。空なら `sampling.seed`。文のキャッシュキーに入るので、**直して `synth` すると直した文だけ作り直します**（`check` で見つけた悪い文の取り直しに使う） |

`text` 列では、絵文字の移動・削除・追加と、読み仮名の修正ができます。直したら `synth` を再実行すると、**直した文だけ**が再合成されます（文ごとのキャッシュは `text` を含むキーで引くため）。

**手直しが正です。** `script` を再実行しても、`source` が小説本文と一致する行は、保存済みの `labels` と `text` をそのまま使います（絵文字を作り直さず、段落の全行が一致するなら LLM も呼びません）。小説本文を直した文だけが再推定されます。`--chapters` で一部の章だけ処理しても、他の章の行は残ります。

`emotion_emoji` の表を変えて、その結果を台本に反映したいときは `--reannotate` を付けます。手直しを捨てて、感情と絵文字を作り直します。LLM の推定は `annotations.jsonl` にキャッシュされているので、LLM を呼ぶのはキャッシュに無い段落だけです。

```bash
uv run --no-sync python -m irodori_speaker script novel.txt -c config.yaml --reannotate
```

### 読み辞書

漢字の読み間違いを、本文や `script.tsv` を直さずに辞書で直せます。`漢字表記: よみ` の平らな YAML です。

```yaml
紫陽花: あじさい
一寸: ちょっと
```

置き場所は 2 つで、後に読んだものが勝ちます。

1. `config.yaml` の `readings:` に並べたファイル（上から順）
2. `out/<book>/readings.yaml`（その本だけ）

- 辞書は**モデルに渡す直前に、最長一致で全文に**適用します（1 回の置換なので、`花: はな` の後に `はな: …` が連鎖して当たることはありません）。
- `script.tsv`・`warnings.log`・`source` は、辞書を通す前の本文のままです。
- 文脈で読みが変わる語（例: 「行った」）は辞書に向きません。小説本文のルビ（`|漢字《よみ》`）で書いてください。`紫陽花火` のように語が重なる場合も、左端の一致が優先されるのでルビを使います。
- 辞書を変えると、**モデルに渡る本文が変わる文だけ**が再合成されます。章 WAV は、辞書があるときは署名が変わるため組み直しになります（文ごとのキャッシュは効くので、合成し直すのは変わった文だけです）。`script.tsv` は変わりません。
- 辞書ファイルが無い、または `漢字表記: よみ` の形でないときは、日本語のエラーで止まります。空のファイルは空の辞書として扱います。

## BGM を付ける

`synth` で作った朗読に、Freesound から自動で選んだ環境音と音楽を重ねて、章ごとと通しの MP3 を作ります（`bgm` サブコマンド）。朗読の WAV は読むだけで、書き換えません。`run` には含まれません（WebUI では「小説生成」タブのボタンで付けられます）。

流れは、LLM で章を場面に区切る → Freesound を検索する → LLM で候補をふるい分ける → LLM を解放する → CLAP で順位を付けて選ぶ → 重ね合わせ、です。合う音が見つからない場面は、ずれた音を流さず無音にして `missing.tsv` に残します。

### 前提

- `synth` を済ませ、章の WAV と `.times.json`・`.sig` があること。どちらかが無い章（`synth` が途中で止まった章など）があれば、`synth --chapters N` を案内して止まります（文のキャッシュがあるので音声合成はやり直しません）。
- 台本を更新したのに `synth` し直していない章は、段落の並びが `.times.json` と違うときに同じ案内で止まります。段落の中の文言だけを直した場合は検出できないので、台本を直したら `synth` を実行し直してください。
- GPU マシンで `uv sync --frozen --extra tts --extra cu128 --extra bgm` を実行する。Ollama が動いていること（`script` と同じ）。FFmpeg が必要。
- Freesound の API キーを、**自分の端末で**ファイルに書く。キーを環境変数やシェルの設定ファイルに置かないのは、AI のエージェントが読めてしまうため。`bgm` はこのファイルだけを読み、キーを画面・ログ・キャッシュに書かない。ファイルが無いかキーの行が無ければ、手順を表示して終了コード 2 で止まる。

```bash
mkdir -p ~/.config/irodori-speaker
read -rsp 'Freesound API key: ' k; echo
printf 'FREESOUND_API_KEY=%s\n' "$k" > ~/.config/irodori-speaker/secrets.env; unset k
chmod 600 ~/.config/irodori-speaker/secrets.env
```

ファイルの場所は設定 `bgm.secrets_file` で変えられます。

### コマンド

```bash
uv run --no-sync python -m irodori_speaker bgm novel.txt -c config.yaml
uv run --no-sync python -m irodori_speaker bgm novel.txt -c config.yaml --chapters 1-3
```

| オプション | 内容 |
|---|---|
| `--chapters` | 対象の章（1 始まり）。一部の章だけのときは、通しの MP3 を作らない |
| `--replan` | 場面の区切りをやり直す。場面の説明が変わるので、ふるい分けと採点もやり直しになる |
| `--llm-model` / `--think` | `script` と同じ（config の `llm` を上書き） |

WebUI では、小説を選んで「小説生成」タブの **BGM を付ける** を押します（`--replan`・途中で止める機能・`exclude.txt` の編集は CLI と手書きのままです）。章の範囲の欄が空なら全章、入っていればその章だけで、LLM は画面上部の LLM 設定を使います。押すと最初に TTS を GPU から外してから LLM と CLAP を動かします。終わると通しの MP3 を再生でき、章ごとの MP3 をダウンロードできます。通しは全章を処理したときだけ作られます。

各段の結果は本ごとのキャッシュに残るので、途中で止めても再実行で続きから進みます。重ね合わせは、朗読・選んだ音・`bgm` 設定が同じ章を飛ばします。音量や長さは `config.yaml` の `bgm` 節で調整します。

### 出力

```
out/<book>/bgm/
  scenes.json                  章ごとの場面（段落の範囲、説明、検索語、雰囲気）
  selection.json               場面ごとに選んだ音の ID・名前・作者・URL・ライセンス・点
  gate.tsv                     ふるい分けの採否と理由（外し方の確認用）
  missing.tsv                  見つからなかった音（理由: no_result / all_rejected / llm_error / error）
  CREDITS.txt                  残っている章 MP3 で使った音の題名・作者・URL・ライセンス
  exclude.txt                  手で書く除外リスト（下記）
  chapters/NN_タイトル.mp3     章ごとの BGM 付き MP3（.sig・.sounds.json は重ね直しの判定と CREDITS.txt の元）
  <book>_bgm_full.mp3          通しの MP3（全章が揃うときだけ）
  cache/                       検索・ふるい分け・採点・場面のキャッシュ
```

### 試聴 MP3 の共有キャッシュ

Freesound から取得した試聴 MP3 は、本をまたいで共有する `out/.freesound/previews/<Freesound の ID>.mp3` に残し、次からはダウンロードしません（採点にも重ね合わせにも同じものを使います）。1 本が数 MB なので、長編 1 冊（数百本）でおよそ数百 MB〜1 GB が目安です。要らなければ `rm -rf out/.freesound` で消せます（次の `bgm` で必要な分だけ取り直します）。Freesound の規約は必要な範囲の一時的なコピーまでを認めているため、このキャッシュは私的な利用の範囲でだけ使い、他人に渡さないでください。

`CREDITS.txt` は、`bgm/chapters/` に実在する章 MP3 が使っている音から作ります（章 MP3 ごとの `.sounds.json` に使った音を記録しています）。途中で失敗した章の MP3 は消すので、残っている MP3 とクレジットは食い違いません。

### 違う音を替える（exclude.txt）

聴いて「この音は違う」と思ったら、`CREDITS.txt` にある Freesound の ID を `out/<book>/bgm/exclude.txt` に 1 行に 1 つ書きます（`#` 以降はコメント）。次の `bgm` で、その音を選んでいた場面はふるい分けで残った次点を選び直し、次点が無ければ無音になります。除外した ID は同じ本の他の場面でも選びません。`bgm` はこのファイルを読むだけで書き換えません。

### 1 日の上限で止まったとき

Freesound の API は 1 日 2000 回までです。上限に達すると「1 日の上限に達しました」と表示して終了コード 1 で止まり、MP3 は書きません（無音だらけの MP3 を作らないため）。それまでの検索・ふるい分け・採点はキャッシュに保存してあるので、**翌日に同じコマンドをもう一度実行**してください。

### 私的な利用に限る

出力は私的な利用に限り、配布しないでください。小説の著作権があり、Freesound の音には CC0・CC BY・CC BY-NC などのライセンスがあるためです（クレジットは `CREDITS.txt`）。

## 品質チェックと取り直し（check）

通しで聴き直さなくても、悪い文を見つけて直せるようにする工程です。`synth` が済んだ本の文（`segments/` の WAV）を調べて、怪しい文に印を付けた `out/<book>/check.tsv` を作ります。

```bash
uv run --no-sync python -m irodori_speaker check novel.txt -c config.yaml
# 声や checkpoint を指定して synth したときは、同じものを渡す（キャッシュのキーに入るため）
uv run --no-sync python -m irodori_speaker check novel.txt -c config.yaml --voice none --checkpoint /path/to/model.safetensors
# 音声認識で読み間違いも調べる
uv run --no-sync python -m irodori_speaker check novel.txt -c config.yaml --asr --chapters 1-2
```

LLM も TTS のモデルも使わないので、GPU は要りません（`--asr` のときだけ使います）。終了時に「N 文中 M 件に印」と表示します。印があっても終了コードは 0 です。

### check.tsv の列と印

列は `chapter / para / seg`（0 始まり）、`key`（`segments/<key>.wav`）、`seconds`、`chars`（モデルが読む文字数。絵文字と文末の句点を除く）、`sec_per_char`、`z`、`flags`、`offset`、`text` です。`--asr` のときだけ `cer`（文字誤り率）と `transcript`（認識結果）が増えます。

`z` は 1 文字あたりの秒数が本全体の中央値からどれだけ外れているかです（中央値と MAD によるロバストな値）。`flags` には次の印が `|` 区切りで入ります。

| 印 | 意味 |
|---|---|
| `long` / `short` | `z` が `check.z_threshold`（既定 3.0）以上 / 以下。引き延ばし・早口・途切れの疑い |
| `clip` | 波形が振り切れているサンプルがある |
| `tail` | 文末に、長い無音を挟んだ短い孤立音がある（余計な息や声が末尾に入った疑い） |
| `silent` | 本文があるのに、発話がほとんど無い（発話が 0.1 秒以下。無音の音声） |
| `missing` | 音声のキャッシュが無い（まだ合成していないか、`--voice` / `--checkpoint` が違う） |
| `warned` | `warnings.log` に記録がある（`max_seconds` での切り詰めなど） |
| `cer` | `--asr` で、文字誤り率が `check.cer_threshold`（既定 0.3）以上 |

`offset` は章の WAV の中でその文が始まる秒です（`synth` が書く `.times.json` から求めます）。印の付いた文へそのまま飛んで聴けます。

`--asr` は `uv sync --frozen --extra tts --extra cu128` で入る transformers を使い、初回に `check.asr_model`（既定 `kotoba-tech/kotoba-whisper-v2.0`）をダウンロードします。認識結果は漢字とかなの表記が原文と違うことが多く、文字誤り率は高めに出るので、閾値は緩め（0.3）にしてあります。

### 悪い文を取り直す

悪い文は `script.tsv` の `seed` 列を直して `synth` を再実行します（`seed` は文のキャッシュキーに入るので、直した文だけが作り直しになります）。気に入る声になるまで `seed` を変えて繰り返してください。WebUI ならボタン 1 つで次の `seed` を試せます（後述）。

## 聴く形式に書き出す（export）

章 WAV を、iPhone の Apple Books などで聴きやすい形に書き出します。既定は章マーカー付きの M4B（AAC）で、章の頭に飛べます。MP3 も選べます。Mac は FFmpeg が必要です。

```bash
uv run --no-sync python -m irodori_speaker export novel.txt -c config.yaml
uv run --no-sync python -m irodori_speaker export novel.txt --format mp3 --bitrate 96k --bgm
```

書き出したファイルは AirDrop か「ファイル」アプリ経由で iPhone に移し、Apple Books で開きます。GPU マシンで作った場合は、先に Mac にダウンロードしてください。

| オプション | 内容 |
|---|---|
| `--format {m4b,mp3}` | 書き出す形式（`export.format` を上書き）。m4b は `export/<book>.m4b`、mp3 は `export/chapters/NN_タイトル.mp3` と通しの `export/<book>.mp3` |
| `--bitrate` | ビットレート 例: `64k`（`export.bitrate` を上書き） |
| `--bgm` | `bgm` で作った BGM 付きの章 MP3 から書き出す（出力は `<book>_bgm.m4b` / `chapters_bgm/` と `<book>_bgm.mp3`、ステレオ） |

設定は `export.*` です。`export.format`（既定 `m4b`）、`export.bitrate`（既定 `64k`。朗読はモノラルなのでサイズ重視にしています）、`export.title`（本の題名。`null` なら小説のファイル名）。章題は `chapters.json` の見出しで、空なら「第N章」にします。

- 前提: 全章の `synth` が済んでいて、章 WAV が今の `script.tsv` と一致していること。足りなければ `synth --chapters N` を案内して止まります。`--bgm` では全章の `bgm` も済んでいる必要があります（`bgm --chapters N`）。常に全章を書き出します。
- 再実行: 入力・形式・ビットレート・題名が前回と同じなら「変更なし」で、再エンコードしません。書き出しに失敗したときや、章 WAV の欠落などで入力の検証に失敗したときは、今回の形式の古い出力と署名を残しません（他の形式の出力は消しません）。mp3 は、書き出しに成功すると、章題の変更や章の削除で残った旧い章 MP3 を `chapters/`（`--bgm` は `chapters_bgm/`）から消します。
- BGM 付きの MP3（既定 192k）は再エンコードになるため、音質が落ちます。`--bitrate 96k` 以上を勧めます。
- 章マーカーの位置は AAC の先頭の遅れで数十ミリ秒ずれることがありますが、章の頭出しには影響しません。
- 私的な利用に限ります（前の節）。

## WebUI

試聴（短い文で声・絵文字・パラメータを確認）と、小説全体の生成の両方に使えます。GPU マシンで起動し、Mac のブラウザから開きます。

```bash
uv run --no-sync python -m irodori_speaker.webui -c config.yaml --host 0.0.0.0
```

| オプション | 内容 |
|---|---|
| `-c`, `--config` | 設定 YAML。画面の初期値（チェックポイント・声・LLM・サンプリングの値）になる。省略可 |
| `--host` | 既定は `127.0.0.1`。Mac のブラウザから開くには `0.0.0.0` |
| `--port` | 既定は `7860` |

Mac のブラウザで `http://<GPU マシンの IP>:7860` を開きます（`--host` を省略した場合は GPU マシン自身の `http://127.0.0.1:7860`）。

### TTS 設定（画面上部）

- チェックポイント（候補から選ぶか、HF の ID・ローカルのパスを自由入力）、device、precision、声を選びます。
- 声は「更新」ボタンで `voices/` の一覧を読み直せます。
- 「声を追加」: 「参照音声をアップロード」に音声を入れ、「声の ID」を書いて押すと、`voices/<ID>.<元ファイルの拡張子>` に保存して一覧に加えます（対応外の拡張子は `.wav` になります）。**既にある ID は上書きせず、エラーにします**。ID には `/` `\` を含められず、`.` で始められません。
- 「モデル読み込み」: TTS モデルを読み込み、参照音声を潜在表現に変換します。押さなくても、最初の生成時に自動で読み込みます。チェックポイント・device・precision・声を変えると、次の生成の前に読み直します。読み込み前に空き VRAM を確認し、足りなければ状態欄に案内が出ます。

### LLM 設定（折りたたみ）

- api（ollama / openai）、base_url、モデル、思考モード、並列数を指定します。ここでの値は config の `llm` を上書きして、台本作成と「LLM で絵文字を付ける」に使います。空欄の項目は config の値になります。
- 「一覧を取得」ボタンで、サーバーにあるモデルの一覧をモデル欄に入れます（Ollama は `ollama list` と同じ一覧）。一覧に無いモデル名も自由入力できます。

### 8GB での LLM と TTS の切り替え

「LLM で絵文字を付ける」「台本を作る（script.tsv）」「一括実行」の台本の部分では、LLM を動かす前に TTS のモデルを GPU から解放します。LLM と TTS を同時に載せないためです。解放した TTS は、次の生成のときに読み直します（そのぶん最初の生成に時間がかかります）。LLM は注釈のあとに解放します。

### タブ「試聴」

- 本文を入れて「生成」を押すと、音声と使用した seed・所要時間・警告が出ます。num_steps・cfg_scale_text・cfg_scale_speaker・duration_scale・seed（`-1` でランダム）を指定できます。
- 長い文は小説生成と同じく `split.max_chars` で分けて合成し（分けた文には直前の絵文字を引き継ぎ）、文の間・段落の間に `silence` の無音を挟んで 1 本にします。
- 絵文字の入れ方は 2 通りです。`irodori_tts` がある環境（GPU マシン）では Irodori-TTS の絵文字パレットが出ます。無い環境では、感情と強度を選んで「絵文字を文頭に挿入」を押します（「『（で始まる本文では括弧の直後に入ります）。
- 「LLM で絵文字を付ける」: 本文を LLM に通し、句ごとの絵文字入りの本文に置き換えます。
- 「総当たり生成」: 同じ seed・同じ本文で、文頭に 10 感情それぞれの絵文字（強度はスライダーで指定、既定 3）を付けて 10 本生成し、並べて聞き比べます。`emotion_emoji` の調整に使います。差が聞こえなければ、その絵文字は効いていません。

### タブ「小説生成」

小説ファイル（アップロード、またはパスを入力）、章の範囲、オプション（感情を付けない / 感情を作り直す（reannotate）/ 作り直す（force））を指定して、次のボタンを**それぞれ独立に**押せます。小説を選ぶと、既に `script.tsv` があれば表に読み込みます。

- **台本を作る（script.tsv）**: `script` と同じ処理です。結果は表に出ます。
- **音声にする**: `synth` と同じ処理です。表を編集していれば、**その編集を `script.tsv` に保存してから**合成します。表が空のときは、`script.tsv` をそのまま使います。`script.tsv` が無ければ「先に台本を作ってください」と案内して止まります。完了後に章ごとの WAV と全体の WAV を表示します。
- **一括実行**: 台本を作り、続けて音声にします。音声にする段階では、直前に作った `script.tsv` をそのまま使います（表の編集は使いません）。
- **停止**: 現在の文の合成が終わり次第、止まります。文ごとにキャッシュしているので、同じ設定で「音声にする」を押し直すと続きから進みます。一括実行で台本の作成中に押したときは、台本を作り終えたところで止まります。
- **script.tsv を保存**: 表の編集を `script.tsv` に書き戻します。編集できるのは `text` 列だけで、他の列は読み込み時の値のままです。
- **BGM を付ける**: `synth` 済みの小説に `bgm` と同じ処理で環境音と音楽を重ねます（章の範囲の欄が空なら全章）。Freesound のキーなど、準備は「BGM を付ける」の節のとおりです（`--extra bgm` が要ります）。進捗は状態欄に出て、終わると BGM 付きの通し MP3 と章ごとの MP3 を表示します。
- **書き出し**: 画面下の「書き出し形式」（m4b / mp3）と「BGM 付き」を選んで押すと、`export` と同じ処理をします（章の範囲は無視して全章）。完了すると書き出したファイルをダウンロードできます。GPU は使わないので、他の生成と並行して押せます。

表の `text` 列と `seed` 列の手直しは、CLI の `script.tsv` の手直しと同じ扱いです（前の節を参照）。

**悪い文を直す手順**:

1. 表の行をクリックすると、その文の音声（`segments/` のキャッシュ）が表の下に出ます。音声が無ければ「まだありません」と出ます（モデルは読み込みません）。
2. **取り直し**: 選んだ文を、今の `seed` に 1 を足した `seed` で合成します（`seed` が空なら `sampling.seed` + 1）。表の `seed` 列が更新され、新しい音声が表示されます。気に入るまで繰り返せます。この時点では `script.tsv` に保存していません。
3. **音声にする**: 表の `seed` を `script.tsv` に保存して合成するので、直した文が章の WAV に反映されます。

**チェック**: `check` と同じ調べ方（`--asr` なし、GPU なし）で、印の付いた文だけを下の表に出します（全行は `check.tsv`）。表を編集していれば、先に `script.tsv` に保存します。

## 設定

`config.example.yaml` をコピーして `config.yaml` を作ります（`config.yaml` は git に入れません）。主なキーは次のとおりです。

| キー | 既定値 | 内容 |
|---|---|---|
| `checkpoint` | `Aratako/Irodori-TTS-v4.1-Small` | Hugging Face の ID / ID + サブフォルダ / ローカルの `.safetensors` |
| `codec_repo` | `Aratako/Semantic-DACVAE-Japanese-32dim` | コーデック |
| `device` / `precision` | `cuda` / `bf16` | `cuda`・`mps`・`cpu` と `fp32`・`bf16`。CUDA は bf16 が速く VRAM も半分 |
| `min_free_vram_gb` | `3.0` | TTS を読み込む前の空き VRAM の下限。足りなければ LLM の停止を案内して止まる（`0` で無効） |
| `empty_cache_interval` | `10` | N 文ごとに `torch.cuda.empty_cache()`（`0` で無効） |
| `voices_dir` / `voice` | `voices` / 既定なし（例: `narrator`） | 参照音声の置き場所と、使う声の ID。`none` は参照音声を使わない予約語（話者 LoRA 統合 checkpoint 向け）。`voice` が未設定だと `synth` は案内を出して止まる（`--voice` でも指定できる） |
| `readings` | `[]` | 読み辞書の YAML ファイルのパスのリスト（cwd 相対、`~` 可）。本ごとの `out/<book>/readings.yaml` があれば後勝ちで重ねる。「読み辞書」を参照 |
| `lora_adapter` | なし | 未統合の話者 LoRA のアダプタのディレクトリ（`adapter_config.json` と `adapter_model.safetensors`）。`--lora-adapter` や WebUI の欄でも指定でき、`none` で無効。「話者 LoRA をアダプタのまま使う」を参照 |
| `ref_normalize_db` | `-16.0` | 参照音声のラウドネス正規化（`null` で無効） |
| `max_ref_seconds` | `null` | 参照音声の上限秒数。`null` はチェックポイントの既定（v4 は 120 秒） |
| `sampling.*` | | `num_steps`・`cfg_scale_text`（3.0）・`cfg_scale_speaker`（5.0）・`duration_scale`（0.8）・`t_schedule_mode`・`sway_coeff`・`max_seconds`（30.0）・`seed`（42） |
| `emotion_mode` | `emoji` | `emoji` か `none`（`none` は LLM を使わない） |
| `narration_intensity_shift` | `1` | 地の文の強度を下げる量 |
| `emotion_emoji` | | 感情 × 強度（1/2/3）→ 挿入する絵文字。空文字は挿入なし。要素は必ず 3 つ |
| `llm.*` | | 下記 |
| `split.max_chars` | `40` | 1 回に合成する最大文字数。これより長い文は読点の位置で分けて別々に合成する。長い文は後半で参照音声の声から離れやすいため短めにしている |
| `split.chapter_regex` | 「第N章/話/節」「プロローグ」「エピローグ」 | 章の見出しの正規表現。見出しが無ければ全体を 1 章にする |
| `silence.*` | 文 0.25 / 段落 0.6 / 章 1.0 / title 1.0 / scene 2.0 / turn 0.4 / ellipsis 0.6 | 結合時に挟む無音（秒）。文・段落の間と特別な間（title: 章題の後、scene: 区切り行の位置、turn: 同じ段落で台詞と地の文が切り替わる所、ellipsis: 前の文が `…‥―—` で終わる所）のうち**長い方**を入れる |
| `read_chapter_title` | `true` | 章の見出しを読み上げる（`false` で読まない） |
| `check.*` | `z_threshold` 3.0 / `cer_threshold` 0.3 / `asr_model` `kotoba-tech/kotoba-whisper-v2.0` / `asr_device` `cuda` | `check` の印の閾値と、`--asr` のモデル・デバイス |

| `export.*` | m4b / 64k / `null` | `export` の形式（`m4b` か `mp3`）・ビットレート・本の題名（`null` は小説のファイル名） |
| `output_dir` | `out` | 出力先 |

- 設定キーの綴りを間違えると、起動時にエラーで止まります（不明なキーは無視されません）。
- `sampling.seed` は固定を勧めます（`null` だと文ごとに声がぶれます）。
- `sampling` の値・`checkpoint`・参照音声・文の本文のどれかを変えると、文ごとのキャッシュのキーが変わり、自動で作り直しになります。`device` と `precision` はキーに含まれません。

### 8GB で OOM になるとき

`precision: bf16` で OOM になる場合は、int8 版のチェックポイントに切り替えます。

```yaml
checkpoint: Aratako/Irodori-TTS-v4.1-Small-Quantized/int8-weight-only
```

また、Ollama を止め忘れて空き VRAM が足りないときは、`synth` の読み込み前に案内が出て止まります。`ollama stop <モデル名>` で解放してから再実行してください。

### LLM を選ぶ・切り替える

既定は Ollama の `qwen3.5:4b-q8_0`（約 5.2GB、思考モード無効）です。日本語 LLM の評価（Nejumi Leaderboard 4）で 10B 未満のモデルの中では上位で、Q8_0 なら 8GB に載ります。軽くしたいときは `qwen3.5:4b-q4_K_M`（約 3.3GB）にします。`script --llm-model NAME` か WebUI の LLM 設定でも切り替えられます。

Ollama の代わりに llama-server / vLLM / LM Studio を使うときは、`llm.api: openai` にして `base_url` を OpenAI 互換の `/v1` に向けます。

```yaml
llm:
  api: openai
  base_url: http://localhost:8080/v1
  model: Qwen3.5-4B-Q8_0
  # server_command を書くと、台本作成の前にツールが起動し、終わったら停止する
  server_command: ["llama-server", "-m", "models/Qwen3.5-4B-Q8_0.gguf", "-ngl", "99", "-c", "16384", "-np", "4", "--port", "8080"]
  concurrency: 4          # llama-server の -np に合わせる
```

- `server_command` を省略すると、起動済みのサーバーを使います（停止もしません）。指定してサーバーがすでに応答していれば、起動しません。
- 注釈が終わるとモデルを解放します（`llm.unload: true`）。Ollama は `keep_alive: 0` を送り、`openai` は `server_command` で起動したサーバーを停止します。
- `llm.api_key` を指定すると `Authorization: Bearer` ヘッダーを付けます。

## 声の作り方

- 参照音声は、**複数の短いクリップで合計 30 秒程度**が推奨です（上限は 120 秒）。長すぎる 1 ファイルより、雑音の無い短いクリップを数本そろえるほうが安定します。
- クリップは `voices/<id>/` に入れると名前順に連結されます。順序を決めたいときは `01.wav`・`02.wav` のように番号を付けてください。
- BGM・残響・複数人の声が混ざっていない、一定の調子で話した音声を選びます。
- 日本語のみに対応しています。
- 声を変えたら（ファイルの中身を差し替えたら）、参照音声の潜在表現と文ごとのキャッシュが自動で作り直されます。

### 話者 LoRA をアダプタのまま使う

統合した checkpoint を作らなくても、学習したアダプタのディレクトリをそのまま渡せます。

```text
adapters/narrator/
├── adapter_config.json
└── adapter_model.safetensors   （または adapter_model.bin）
```

- `config.yaml` の `lora_adapter: adapters/narrator`、CLI の `--lora-adapter adapters/narrator`、WebUI の「話者 LoRA のアダプタ」欄のどれでも指定できます。優先は CLI / WebUI > config で、`none`（`off` / `base` も同じ）と書くと config の指定を打ち消してアダプタなしにします。
- 統合 checkpoint と同じく、**`voice: none`（参照音声なし）との組み合わせを推奨**します。参照音声ありでも動きます。
- アダプタのディレクトリやファイルが足りないときは、足りないファイル名を添えて、モデルを読み込む前に止まります。
- 上流は一度読んだアダプタをモデルから外せないので、アダプタを変える（外す）と、モデルごと読み直します（VRAM に積み上がりません）。
- 文のキャッシュと章 WAV の署名には、アダプタの設定と重みの中身が入ります。アダプタを学習し直すと、その文は再合成になります（ファイルの更新時刻は見ません）。
- 統合 checkpoint を作る手順は、これまでどおり使えます。

### キャプションで感情を試す（実験）

v4.1-Small は、参照音声と一緒にキャプション（「激しく怒って、強い口調で相手を問い詰めている」のような文）を受け取れます。本ツールの合成はまだキャプションを使いませんが、`scripts/caption_test.py` で効き方を試せます。GPU マシンで、Ollama を止めてから実行します。

```bash
uv run --no-sync python scripts/caption_test.py -c config.yaml \
  --checkpoint Aratako/Irodori-TTS-v4.1-Small --checkpoint /path/to/merged.safetensors
```

- 声・sampling・seed は `config.yaml` の値を使い、キャプション（なし / 怒り / 悲しみ / 喜び / ささやき）と `cfg_scale_caption`（3 / 5）だけを変えて、台詞と地の文の 2 文を生成します。比べるため、キャプションの代わりに `emotion_emoji` の強度 3 の絵文字を文頭に付けたものも作ります。
- `--checkpoint` を複数渡すと順に比べます。省略すると `config.yaml` の `checkpoint` を使います。
- `--no-ref` を付けると参照音声を渡さずに生成します。参照音声があると感情が出にくいのかを切り分けるためのもので、ベースモデルでは声が参照音声と別人になります。
- 出力は `out/caption_test/<番号>_<checkpoint 名>/`（`--no-ref` のときは末尾に `_noref`）に、`1_none.wav`（1 文目・キャプションなし）、`1_angry_cap3.wav`、`1_angry_emoji.wav` のような名前でできます。文とキャプションはスクリプト冒頭の `TEXTS` と `CAPTIONS` で変えられます。

## 注意

- **透かし**: 出力される音声には、上流の仕様で SilentCipher の透かしが入ります。
- **声の同意**: 他人の声を、本人の同意なく複製しないでください（モデルカードの倫理規定）。参照音声には、自分の声か、利用許諾を得た声だけを使ってください。
- **漢字の読み間違い**: 青空文庫形式のルビ（`|漢字《かんじ》`）で小説本文に読みを書くか、`script.tsv` の `text` 列を直してください。ルビは読み仮名に置き換えて読み上げます。青空文庫の注記（`［＃…］`）は取り除きます。 本文を変えたくないときは、上の「読み辞書」に `漢字表記: よみ` を書くと、全文に同じ読みを当てられます。
- **8GB の制約**: LLM と TTS を同時に載せません。`script`（`run` の前半）の終了後に LLM を解放してから `synth` を行います。
- **文末の句点・読点**: モデルに渡す前に外します（句点・読点で終わると文末に余計な音が入るため）。`script.tsv` の `text` 列は変えません。
- **区切り行と章題**: 文字（かな・漢字・英数字）も括弧も絵文字も含まない行は区切り（`break`）になります。連続する区切りは 1 つにまとめ、章の先頭と末尾の区切りは捨てます。区切りや章題を足すと後ろの段落番号がずれますが、同じ章で原文が同じ行の手直しは引き継ぎます（同じ文が複数あるときは文書順に対応づけるので、1 文を消すと隣の行に手直しが移ることがあります）。
- **既存の本**: 間の設定が章 WAV の署名に入るため、更新後の最初の `synth` は章 WAV の組み直しになります（文ごとのキャッシュは残るので、新しく合成するのは章題の文だけです）。
- 1 回に合成するのは 40 文字までです。長い文は読点の位置で分けて合成します（長い文は後半で参照音声の声から離れやすいため）。長い文が 30 秒を超えて切り詰められたときは、`warnings.log` に記録されます。

## GPU マシンでの動作確認

書き出し・品質チェック・場面区切り・読み辞書・話者 LoRA アダプタは、Mac のテストだけで確かめています。GPU マシンで初めて使うときは、短い試験用の本で全機能を通してから、実際の本で書き出してください。

### 準備

```bash
git pull
uv sync --frozen --extra tts --extra cu128 --extra webui
```

新しい設定はすべて既定値で動くので、`config.yaml` は書き換えなくて構いません。試験用に、見出しと区切り行を含む短い小説を作ります。

```bash
printf '第一章 朝\n\n彼は走った。空は高かった。\n\n　　＊\n\n「待って……」と彼女は言った。\n「どうしたの」\n彼は振り返った――\n◇　◇　◇\n朝が来た。\n\n第二章 夜\n夜になった。\n' > smoke.txt
```

### 試験用の本で確かめる

1. **台本**: Ollama を動かして `script` を実行し、`out/smoke/script.tsv` を開きます。各章の先頭が `title` の行、`＊` と `◇　◇　◇` が `text` の空いた `break` の行になっていれば合格です。

   ```bash
   uv run --no-sync python -m irodori_speaker script smoke.txt -c config.yaml
   ```

2. **合成と間**: `ollama stop <モデル名>` で LLM を止めてから `synth` を実行し、`out/smoke/chapters/01_第一章_朝.wav` を聴きます。章題が読まれること、区切りの位置に約 2 秒の間があること、台詞と地の文の切り替わりと「――」の後に間があることを確かめます。長さは `silence.scene` / `title` / `turn` / `ellipsis` で調整します。

   ```bash
   uv run --no-sync python -m irodori_speaker synth smoke.txt -c config.yaml
   ```

3. **品質チェック**: `check` を実行し、`out/smoke/check.tsv` の印を見ます。正常な文に印がほとんど付かなければ合格です。続けて `--asr` も試します。初回は約 1.5 GB のモデルを取得します。`transcript` 列が本文とおおむね合い、正しい文に `cer` の印が付かないことを確かめます。

   ```bash
   uv run --no-sync python -m irodori_speaker check smoke.txt -c config.yaml
   uv run --no-sync python -m irodori_speaker check smoke.txt -c config.yaml --asr
   ```

4. **取り直し**: `out/smoke/script.tsv` の 1 行の `seed` 列に `100` と書き、`synth` をもう一度実行します。1 文だけ合成して終わり、その文の声が変わっていれば合格です。
5. **読み辞書**: `out/smoke/readings.yaml` に `朝: あさ` のように 1 行書き、`synth` を実行します。辞書に当たる文だけが合成し直され、その文の読みが変わることを確かめます。
6. **話者 LoRA アダプタ**: 統合前のアダプタのディレクトリを渡して合成し、統合 checkpoint で作った声と同じに聞こえれば合格です。続けて `--lora-adapter none` で合成し、モデルを読み直しても VRAM が足りなくならないことも確かめます。

   ```bash
   uv run --no-sync python -m irodori_speaker synth smoke.txt -c config.yaml --voice none --lora-adapter <アダプタのディレクトリ> --force
   uv run --no-sync python -m irodori_speaker synth smoke.txt -c config.yaml --voice none --lora-adapter none --force
   ```

7. **WebUI**: `--host 0.0.0.0` で起動し、Mac のブラウザで開きます。「小説生成」タブで smoke.txt を選び、表の行をクリックするとその文の音声が出ること、「取り直し」で seed が 1 増えて声が変わること、「チェック」で印の表が出ること、「書き出し」で m4b がダウンロードできることを確かめます。行のクリックは Mac のテストでは確かめられていません。

   ```bash
   uv run --no-sync python -m irodori_speaker.webui -c config.yaml --host 0.0.0.0
   ```

### 実際の本で書き出して iPhone で聴く

既存の本は、更新後の最初の `synth` で章 WAV の組み直しになります。区切り・章題・間の設定が章の署名に入るためで、新しく合成するのは章題の文だけです。

```bash
uv run --no-sync python -m irodori_speaker script novel.txt -c config.yaml   # Ollama を動かして
ollama stop <モデル名>
uv run --no-sync python -m irodori_speaker synth  novel.txt -c config.yaml
uv run --no-sync python -m irodori_speaker export novel.txt -c config.yaml  # out/novel/export/novel.m4b
```

- WebUI の「書き出し」ボタンを使うと、Mac のブラウザに直接ダウンロードできます。AirDrop で iPhone に送り、Apple Books で開きます。
- 章の一覧が見出しどおりに並ぶこと、章の頭に飛べること、続きから再生できることを確かめます。サイズは 64k のモノラルで 1 時間あたり約 30 MB が目安です。
- MP3 も試すなら `--format mp3` を付けます。`export/chapters/` に章ごとの MP3 ができます。
- `bgm` も GPU マシンではまだ通していません。試すなら `bgm` の後に `export novel.txt -c config.yaml --bgm --bitrate 96k` を実行します。

### 確かめた結果として残すこと

- 区切り・章題・台詞の切り替わりの間が自然だったか。長すぎる・短すぎる箇所があれば、その種類。
- `check` の印のうち、聴いて本当に悪かった文と、問題なかった文の例（`tail` などの定数を実データで調整するため）。
- `--asr` と `--lora-adapter` がエラーなく動いたか。エラーが出たら、その全文。
- Apple Books で章が並んだか。

## テスト

```bash
uv sync --extra dev
uv run pytest -q
```

テストは、LLM（Ollama）・TTS（torch / `irodori_tts`）・GPU を使わずに動きます。
