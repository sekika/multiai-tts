# multiai-tts

`multiai-tts` is an extension library for [multiai](https://sekika.github.io/multiai/) that provides Text-to-Speech (TTS) capabilities using OpenAI and Google GenAI.

## Prerequisites

**API Key Configuration**

This library relies on the configuration provided by `multiai`. You must set up your API keys (OpenAI API Key, Google API Key) using `multiai`'s configuration files or environment variables before using this library.

For details on how to configure API keys, please refer to the **[multiai documentation](https://sekika.github.io/multiai/)**.

## Installation

You also need to install `ffmpeg` on your system if you want to save audio in formats other than WAV (e.g., MP3).

```bash
pip install multiai-tts
```

## Usage

### Google GenAI Example

```python
import sys
import multiai_tts

provider = 'google'
model = 'gemini-2.5-flash-preview-tts'

client = multiai_tts.Prompt()
client.set_tts_model(provider, model)
client.tts_voice_google = 'charon'

# Speak directly
client.speak("Please speak the following. Hello, this is a test from Google model.")
if client.error:
    print(client.error_message)
    sys.exit(1)

# Save to file
client.save_tts("Please speak the following. Saving this audio to mp3.", "output_google.mp3")
if client.error:
    print(client.error_message)
    sys.exit(1)
```

### OpenAI Example

```python
import sys
import multiai_tts

provider = 'openai'
model = 'gpt-4o-mini-tts'

client = multiai_tts.Prompt()
client.set_tts_model(provider, model)
client.tts_voice_openai = 'marin'

# Speak directly
client.speak("Hello, this is a test from OpenAI model.")
if client.error:
    print(client.error_message)
    sys.exit(1)

# Save to file
client.save_tts("Saving this audio to mp3.", "output_openai.mp3")
if client.error:
    print(client.error_message)
    sys.exit(1)
```