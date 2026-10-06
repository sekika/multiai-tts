# 仕様書： Gemini 3.8 TTS 対応

## 目的

既存の `Prompt.save_tts(text, ..., prompt=...)` API を維持したまま、
Google Gemini 3.8 TTS（`gemini-3.8-flash-tts` および
`gemini-3.8-flash-lite-tts`）へ正しいリクエスト形式で送信できるようにする。

Gemini 3.8 TTS は、入力の `text` を逐語的な読み上げ原稿として扱う。このため、従来の
Gemini TTS 向けに行っていた「スタイル指示 (`prompt`) と原稿 (`text`) を連結して一つの
テキストとして送る」処理は使えない。連結すると、指示そのものが音声化される。

Gemini 3.8 では、原稿は `text`、発話全体に継続して適用する指示は
`speech_metadata.style` として分離して送る。

## 対象と非対象

対象:

- Google プロバイダーで、`tts_prompt_mode="speech_metadata"` が指定された単一話者 TTS。
  Gemini 3.8 はその利用例である。
- 既存の `prompt`、チャンク分割、WAV 保存、エラー報告との互換性。

非対象:

- `multiai-tts` の公開 API を呼び出し側に破壊的変更させること。
- Voice design、Voice replication、複数話者 TTS の新規公開 API。
- 原稿文をモデル側で編集・要約・翻訳すること。

## 公開 API と互換性

`Prompt.save_tts()` のシグネチャおよび意味は変更しない。

```python
client.save_tts(
    text,
    wav_path,
    prompt="落ち着いて明瞭に読む",
    chunk_size=None,
    split_chars="。．.!！?？\\n",
    chunk_overflow="extend",
)
```

- `text` は常に読み上げ原稿であり、Gemini 3.8 に対しては `text` フィールドへそのまま渡す。
- `prompt` は既存どおり任意のスタイル指示文字列である。Gemini 3.8 に対しては
  `speech_metadata.style` へ渡す。
- 空の `prompt` は `style` を省略するか空文字列として送る。原稿を加工してはならない。
- 旧 Gemini モデル、OpenAI、Azure、VOICEVOX の既存挙動は変更しない。

既存利用者が `prompt` に原稿の見出しや区切り（例: `"\\n\\n## 原稿\\n"`）を含めていても、
それは `speech_metadata` 形式ではスタイルではない。ライブラリはそれを原稿へ再連結してはならない。
必要なら呼び出し側がその形式を選ぶ際に区切りを空にする。`multiai-tts` は警告を出してよいが、
指示文を推測・削除・変更してはならない。

## モデル能力の指定

モデル名から API 契約を推測してはならない。特に `gemini-3.8-` のような接頭辞・接尾辞で
分岐する実装は採用しない。モデル名は提供者が変更・追加できる識別子であり、名前と TTS の
プロンプト形式は別の関心事だからである。

代わりに、Google TTS のプロンプト形式を明示する `tts_prompt_mode` を `Prompt` の設定として
追加する。許容値は次のとおりとする。

| 値 | 意味 |
| --- | --- |
| `legacy_inline` | `prompt` と原稿を従来形式で扱う。既存の Gemini Preview モデルとの互換用。 |
| `speech_metadata` | 原稿を `text`、継続的なスタイル指示を `speech_metadata.style` として送る。Gemini 3.8 など、構造化メタデータを受け付けるモデル用。 |

公開設定の例:

```python
client = multiai_tts.Prompt()
client.set_tts_model("google", "任意のモデルID")
client.tts_prompt_mode = "speech_metadata"
```

または、既存の設定メソッドを拡張してもよい。

```python
client.set_tts_model(
    "google", "任意のモデルID", tts_prompt_mode="speech_metadata"
)
```

後者を採用する場合も、二引数の既存呼び出しは維持する。両方を実装する必要はなく、少なくとも
一方の明示的な設定方法を公開すること。

既定値は `legacy_inline` として後方互換性を保つ。`speech_metadata` は利用者または上位
ライブラリが明示的に指定する。ライブラリが既知モデルのために既定値を補助設定することは可能だが、
その表は利便性のための初期値にとどめ、モデル名マッチを唯一の判定根拠にしてはならない。

未知の `tts_prompt_mode` は、警告付きのフォールバックではなく `ValueError`（または既存の設定
エラー型）として、API 呼び出し前に失敗させる。

## Gemini 3.8 のリクエスト変換

Google GenAI SDK の Interactions API を使う場合、各チャンクについて次の形でリクエストする。

Gemini 3.8 TTS の現行 Interactions API スキーマには
`google-genai>=2.25.0` が必要である。
この SDK 系列は Python 3.10 以上を必要とする。
既存環境で旧 SDK が残っている場合は、`pip install -U "google-genai>=2.25.0"`
で更新する。

```python
content = {
    "type": "text",
    "text": chunk,  # 原稿だけ。一字も追加・削除しない
}
if prompt:
    content["annotations"] = [{
        "type": "speech_metadata",
        "style": prompt,
    }]

interaction = client.interactions.create(
    model=model_name,
    input=[{"type": "user_input", "content": [content]}],
    response_format={"type": "audio"},
    generation_config={"speech_config": [{"voice": voice_name}]},
)
audio_bytes = base64.b64decode(interaction.output_audio.data)
```

REST を使う実装でも、同じ意味の `text` と `speech_metadata.style` を送ること。

### チャンク処理

- 既存の分割ロジックは原稿 `text` だけを対象にする。
- 分割後の**各**チャンクに同じ `speech_metadata.style` を付与する。
- `prompt` はトークン数・チャンク境界の算定対象に含めない。
- 返却された各音声チャンクを既存の順序で連結する。

### 原稿中の局所指示

特定箇所だけの休止・息継ぎは `prompt` ではなく、呼び出し側が原稿中に英語のタグとして置く。

```text
重要な結論です。<short pause> したがって、…
```

`<short pause>`、`<long pause>` 等のタグは Gemini 3.8 の機能であり、
`multiai-tts` はタグをエスケープ、除去、翻訳してはならない。

## 音声形式

Gemini 3.8 の非ストリーミング応答の既定は 24 kHz・モノラル・16-bit PCM を含む WAV である。

- 応答が `audio/wav` の場合、そのバイト列を WAV として直接保存または既存の WAV 結合処理へ渡す。
- 旧 Gemini 経路用の「生 PCM に WAV ヘッダーを追加する」処理を 3.8 応答へ適用してはならない。
- ライブラリが内部で常に PCM に正規化して結合する設計なら、WAV を正しくデコードしてから
  既存の正規化処理に渡す。

## エラーと診断

- `speech_metadata` が未対応の古い SDK を検出した場合、`legacy_inline` へ黙ってフォールバック
  してはならない。指示を読んでしまうためである。
- 代わりに、構造化 TTS メタデータ対応版の `google-genai` への更新を促す明確なエラーを返す。
- API から返るエラー本文・ステータスは既存の `error` / `error_message` 契約に沿って保持する。
- ログの debug レベルでは、モデル名、原稿文字数、`style` の有無を記録してよい。ただし原稿・
  プロンプト全文や API キーを通常ログへ出さない。

## テスト要件

外部 API を呼ばないモックテストを追加する。

1. `tts_prompt_mode="speech_metadata"` で `text="原稿"`、`prompt="落ち着いて読む"` を渡すと、SDK に渡す本文が
   `"原稿"` のみであり、annotation の `style` が後者である。
2. `prompt=""` の場合、本文は不変で、`style` annotation を省略（または空として送信）する。
3. `prompt` に `"\\n\\n## 原稿\\n"` を含めても、それが `text` に混入しない。
4. 分割時は、各本文チャンクが従来の分割結果と一致し、各リクエストに同一の `style` が付く。
5. `tts_prompt_mode="legacy_inline"` は、モデル名によらず既存のプレーンテキスト・プロンプト
   形式を維持する。
6. OpenAI、Azure、VOICEVOX の送信内容が変わらない。
7. 3.8 の WAV 応答へ PCM 用 WAV ヘッダーが二重に追加されない。

## 呼び出し側への移行案内

Gemini 3.8 利用者は、`prompt` を純粋な話し方の指示だけにする。

```python
client.set_tts_model("google", "gemini-3.8-flash-tts")
client.tts_prompt_mode = "speech_metadata"
client.save_tts(
    script,
    output_path,
    prompt="大学教員が学生に語りかけるように、落ち着いた自然な抑揚で読む。",
)
```

`slidemovie` などの上位ライブラリは、この設定を自らの設定値として公開し、`Prompt` に転送する。
その上で `prompt_separator` は空文字列にする。既存の `tts_use_prompt` は変更せず、`prompt` を
送るかどうかのスイッチとして引き続き使える。

これにより、原稿以外を音声化せず、従来どおり `prompt` で全体の話し方を指定できる。

## 受け入れ基準

- Gemini 3.8 でスタイル指示が音声に含まれない。
- 原稿は API に送る `text` としてそのまま保持される。
- 同じ `prompt` が長文原稿の全チャンクに適用される。
- 既存プロバイダーおよび旧 Gemini モデルの回帰テストが通る。
- 3.8 で生成した音声ファイルが有効な WAV として再生できる。

## 参照

- [Google Gemini API: Text-to-speech generation](https://ai.google.dev/gemini-api/docs/speech-generation)
  — Gemini 3.8 の `text` は逐語原稿、継続的な話し方は `speech_metadata.style`、局所イベントは
  インラインタグとする公式仕様。
