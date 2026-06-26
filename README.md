# multiai-tts

`multiai-tts` is an extension library for [multiai](https://sekika.github.io/multiai/) that provides Text-to-Speech (TTS) capabilities using OpenAI, Google GenAI, and Azure Speech.

- One simple interface (`speak()` / `save_tts()`) across OpenAI, Google GenAI, and Azure Speech.
- Save to WAV, MP3, and other formats (via `ffmpeg`).
- Automatic chunking and joining of long text that exceeds API length limits.

## Installation

```bash
pip install multiai-tts
```

## Quick example

```python
import multiai_tts

client = multiai_tts.Prompt()
client.set_tts_model('openai', 'gpt-4o-mini-tts')

client.speak("Hello, this is a test from OpenAI model.")
if client.error:
    print(client.error_message)
```

## Documentation

Full documentation — prerequisites, API key configuration, per-provider
examples, and the long-text chunking feature — is available at:

**<https://sekika.github.io/multiai-tts/>**
