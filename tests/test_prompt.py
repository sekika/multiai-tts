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

if __name__ == '__main__':
    unittest.main()
