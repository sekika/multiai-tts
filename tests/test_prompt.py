import unittest
import io
from unittest.mock import MagicMock, patch
from multiai_tts import Prompt, TTS_Provider

class TestTTSPrompt(unittest.TestCase):
    def setUp(self):
        self.client = Prompt()
        # Dummy API keys
        self.client.openai_api_key = "dummy-openai-key"
        self.client.google_api_key = "dummy-google-key"
        self.client.azure_tts_api_key = "dummy-azure-key"
        # Default Azure settings
        self.client.tts_region_azure = "japaneast"
        self.client.tts_voice_azure = "en-US-JennyNeural"

    def test_initialization(self):
        """Check default voice settings."""
        self.assertEqual(self.client.tts_voice_openai, 'marin')
        self.assertEqual(self.client.tts_voice_google, 'charon')
        self.assertIsNone(getattr(self.client, 'tts_provider', None))

    def test_set_tts_model_invalid(self):
        """Check error handling for invalid provider."""
        self.client.set_tts_provider('invalid')
        self.assertTrue(self.client.error)
        self.assertIn('not available', self.client.error_message)

    @patch('multiai_tts.prompt.OpenAI')
    def test_get_wav_openai(self, MockOpenAI):
        """Mock OpenAI TTS."""
        mock_client = MockOpenAI.return_value
        mock_response = MagicMock()
        mock_response.content = b'openai_wav'
        mock_client.audio.speech.create.return_value = mock_response

        self.client.set_tts_model('openai', 'tts-model')
        wav = self.client.get_wav("Hello OpenAI", fmt='wav')

        self.assertFalse(self.client.error)
        self.assertEqual(wav, b'openai_wav')
        mock_client.audio.speech.create.assert_called_with(
            model='tts-model',
            voice='marin',
            input='Hello OpenAI',
            response_format='wav'
        )

    @patch('multiai_tts.prompt.genai')
    @patch('wave.open')
    def test_get_wav_google(self, mock_wave, mock_genai):
        """Mock Google GenAI TTS."""
        mock_client = mock_genai.Client.return_value
        mock_response = MagicMock()
        mock_part = MagicMock()
        mock_part.inline_data.data = b'google_pcm'
        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]
        mock_response.candidates = [mock_candidate]
        mock_client.models.generate_content.return_value = mock_response

        self.client.set_tts_model('google', 'google-model')
        wav = self.client.get_wav("Hello Google")

        self.assertFalse(self.client.error)
        self.assertIsNotNone(wav)
        mock_client.models.generate_content.assert_called()

    @patch('multiai_tts.prompt.speechsdk')
    def test_get_wav_azure(self, mock_speechsdk):
        """Mock Azure TTS."""

        # ---- required attributes ----
        self.client.azure_tts_api_key = "dummy-azure-key"
        self.client.azure_tts_region = "japaneast"
        self.client.tts_voice_azure = "en-US-JennyNeural"

        mock_speechsdk.SpeechConfig.return_value = MagicMock()

        mock_audio = MagicMock()
        mock_speechsdk.audio = mock_audio
        mock_audio.PullAudioOutputStream.return_value = MagicMock()
        mock_audio.AudioOutputConfig.return_value = MagicMock()

        mock_synth = MagicMock()
        mock_speechsdk.SpeechSynthesizer.return_value = mock_synth

        mock_result = MagicMock()
        mock_result.reason = mock_speechsdk.ResultReason.SynthesizingAudioCompleted
        mock_result.audio_data = b'azure_wav'
        mock_synth.speak_text_async.return_value.get.return_value = mock_result

        # ---- execute ----
        self.client.set_tts_provider('azure')
        wav = self.client.get_wav("Hello Azure")

        # ---- verify ----
        self.assertFalse(self.client.error)
        self.assertEqual(wav, b'azure_wav')

        mock_synth.speak_text_async.assert_called_once_with("Hello Azure")

    def test_split_text_basic(self):
        """Split at the rightmost split char within chunk_size."""
        text = "あいう。えお。かきくけこ。"
        chunks = self.client.split_text(text, chunk_size=5)
        self.assertFalse(self.client.error)
        # "あいう。" fits within 5 and ends at the rightmost "。"
        self.assertEqual(chunks[0], "あいう。")
        self.assertEqual("".join(chunks), text)

    def test_split_text_no_split_remaining_fits(self):
        """Text shorter than chunk_size yields a single chunk."""
        chunks = self.client.split_text("hello", chunk_size=100)
        self.assertFalse(self.client.error)
        self.assertEqual(chunks, ["hello"])

    def test_split_text_overflow_extend(self):
        """No split char within chunk_size: read until the next one."""
        text = "abcdefghij.kl"
        chunks = self.client.split_text(
            text, chunk_size=5, split_chars=".", chunk_overflow="extend")
        self.assertFalse(self.client.error)
        self.assertEqual(chunks[0], "abcdefghij.")
        self.assertEqual("".join(chunks), text)

    def test_split_text_overflow_extend_to_end(self):
        """No split char anywhere: whole text is one chunk."""
        text = "abcdefghij"
        chunks = self.client.split_text(
            text, chunk_size=5, split_chars=".", chunk_overflow="extend")
        self.assertFalse(self.client.error)
        self.assertEqual(chunks, ["abcdefghij"])

    def test_split_text_overflow_error(self):
        """No split char within chunk_size and overflow='error' sets error."""
        chunks = self.client.split_text(
            "abcdefghij", chunk_size=5, split_chars=".",
            chunk_overflow="error")
        self.assertIsNone(chunks)
        self.assertTrue(self.client.error)
        self.assertIn("split character", self.client.error_message)

    def test_split_text_invalid_overflow(self):
        """Invalid chunk_overflow value sets error."""
        chunks = self.client.split_text(
            "abc", chunk_size=5, chunk_overflow="bogus")
        self.assertIsNone(chunks)
        self.assertTrue(self.client.error)

    def test_split_text_invalid_chunk_size(self):
        """Non-positive chunk_size sets error."""
        chunks = self.client.split_text("abc", chunk_size=0)
        self.assertIsNone(chunks)
        self.assertTrue(self.client.error)

    @patch('multiai_tts.prompt.OpenAI')
    def test_no_prompt_backward_compat(self, MockOpenAI):
        """Without a prompt, the body text is sent unchanged (backward compat)."""
        mock_client = MockOpenAI.return_value
        mock_client.audio.speech.create.return_value = MagicMock(
            content=b'wav')

        self.client.set_tts_model('openai', 'tts-model')
        self.client.get_wav("Hello", fmt='wav')

        self.assertFalse(self.client.error)
        _, kwargs = mock_client.audio.speech.create.call_args
        self.assertEqual(kwargs['input'], 'Hello')

    @patch('multiai_tts.prompt.OpenAI')
    def test_prompt_prepended_to_text(self, MockOpenAI):
        """The prompt is prepended to the body text, uniformly per provider."""
        mock_client = MockOpenAI.return_value
        mock_client.audio.speech.create.return_value = MagicMock(
            content=b'wav')

        self.client.set_tts_model('openai', 'tts-model')
        self.client.get_wav("Body text", fmt='wav', prompt="Speak cheerfully.")

        self.assertFalse(self.client.error)
        _, kwargs = mock_client.audio.speech.create.call_args
        sent = kwargs['input']
        self.assertIn("Speak cheerfully.", sent)
        self.assertIn("Body text", sent)
        self.assertTrue(sent.index("Speak cheerfully.") < sent.index("Body text"))

    @patch('multiai_tts.prompt.OpenAI')
    def test_prompt_applied_to_every_chunk(self, MockOpenAI):
        """Chunking splits only the body; the prompt is prepended to each chunk."""
        import io as _io
        import wave

        def make_wav(nframes):
            buf = _io.BytesIO()
            with wave.open(buf, 'wb') as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(24000)
                w.writeframes(b'\x00\x00' * nframes)
            return buf.getvalue()

        mock_client = MockOpenAI.return_value
        mock_client.audio.speech.create.return_value = MagicMock(
            content=make_wav(10))

        self.client.set_tts_model('openai', 'tts-model')
        # A long prompt must NOT count against chunk_size: body "abc.def.ghi"
        # splits into 3 chunks regardless of prompt length.
        prompt = "Speak cheerfully."
        self.client.save_tts(
            "abc.def.ghi", "out_prompt.wav", prompt=prompt,
            chunk_size=4, split_chars=".")

        self.assertFalse(self.client.error)
        # "abc." / "def." / "ghi" => 3 chunks regardless of prompt length.
        self.assertEqual(mock_client.audio.speech.create.call_count, 3)
        bodies = ["abc.", "def.", "ghi"]
        for call, body in zip(
                mock_client.audio.speech.create.call_args_list, bodies):
            _, kwargs = call
            # Every chunk request carries the prompt followed by that chunk.
            self.assertTrue(kwargs['input'].startswith(prompt))
            self.assertTrue(kwargs['input'].endswith(body))
        import os as _os
        self.assertTrue(_os.path.exists("out_prompt.wav"))
        _os.remove("out_prompt.wav")

    @patch('multiai_tts.prompt.OpenAI')
    def test_chunk_size_none_with_prompt_single_call(self, MockOpenAI):
        """chunk_size=None + prompt == one synth call with the prompt prepended."""
        mock_client = MockOpenAI.return_value
        mock_client.audio.speech.create.return_value = MagicMock(
            content=b'wav')

        self.client.set_tts_model('openai', 'tts-model')
        self.client.save_tts(
            "Body text", "out_single.wav", prompt="Speak calmly.")

        self.assertFalse(self.client.error)
        self.assertEqual(mock_client.audio.speech.create.call_count, 1)
        _, kwargs = mock_client.audio.speech.create.call_args
        self.assertEqual(kwargs['input'], "Speak calmly.\n\nBody text")
        import os as _os
        if _os.path.exists("out_single.wav"):
            _os.remove("out_single.wav")

    @patch('multiai_tts.prompt.OpenAI')
    def test_save_tts_chunked(self, MockOpenAI):
        """Chunked save_tts synthesizes each chunk and concatenates WAVs."""
        import io as _io
        import wave

        def make_wav(nframes):
            buf = _io.BytesIO()
            with wave.open(buf, 'wb') as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(24000)
                w.writeframes(b'\x00\x00' * nframes)
            return buf.getvalue()

        mock_client = MockOpenAI.return_value
        # Each call returns a small WAV
        mock_client.audio.speech.create.return_value = MagicMock(
            content=make_wav(10))

        self.client.set_tts_model('openai', 'tts-model')
        text = "abc.def.ghi"
        self.client.save_tts(
            text, "out.wav", chunk_size=4, split_chars=".")

        self.assertFalse(self.client.error)
        # "abc." / "def." / "ghi" => 3 chunks => 3 synth calls
        self.assertEqual(mock_client.audio.speech.create.call_count, 3)
        import os as _os
        self.assertTrue(_os.path.exists("out.wav"))
        _os.remove("out.wav")

if __name__ == '__main__':
    unittest.main()
