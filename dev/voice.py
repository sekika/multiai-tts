#!/usr/bin/env python3
"""List Gemini TTS voices grouped by perceived gender.

The API key is read from the existing ``multiai`` configuration through
``multiai_tts.Prompt``. The voice ID in the first column can be assigned directly to
``multiai_tts.Prompt().tts_voice_google``.
"""

import argparse
import sys

from google import genai
from multiai_tts import Prompt


def _value(voice, name, default="-"):
    """Read an SDK field without making the output depend on its exact type."""
    value = getattr(voice, name, None)
    if not value:
        return default
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value)


def list_voices(client, gender, language_code=None):
    """Return up to 1,000 matching prebuilt voices."""
    kwargs = {
        "gender": [gender],
        "type_": ["prebuilt"],
        "page_size": 1000,
    }
    if language_code:
        kwargs["language_code"] = [language_code]
    response = client.voices.list(**kwargs)
    return response.voices or []


def print_voices(gender, voices):
    print(f"\n[{gender}] {len(voices)} voices")
    if not voices:
        return
    print("id\tdisplay name\tlanguage\taccent\tpitch\tdescription")
    for voice in voices:
        print(
            f"{_value(voice, 'id')}\t{_value(voice, 'display_name')}\t"
            f"{_value(voice, 'language_code')}\t{_value(voice, 'accent')}\t"
            f"{_value(voice, 'pitch')}\t{_value(voice, 'description')}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="List Gemini TTS prebuilt voices grouped by gender.")
    parser.add_argument(
        "-l", "--language", "--language-code",
        dest="language_code",
        metavar="BCP47",
        help="Optional BCP-47 language filter, for example ja-JP or en-US.")
    args = parser.parse_args()

    prompt = Prompt()
    api_key = getattr(prompt, "google_api_key", None)
    if not api_key:
        print("Google API key is not configured in multiai.", file=sys.stderr)
        return 2

    client = genai.Client(api_key=api_key)
    for gender in ("female", "male"):
        try:
            voices = list_voices(client, gender, args.language_code)
        except Exception as exc:
            print(f"Failed to list {gender} voices: {exc}", file=sys.stderr)
            return 1
        print_voices(gender, voices)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
