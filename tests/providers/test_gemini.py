import json
from unittest.mock import patch, MagicMock

import pytest

from jaime.providers.gemini import GeminiProvider


def _mock_urlopen_response(data_bytes):
    """Create a mock urlopen return that works as a context manager."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = data_bytes
    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = mock_resp
    mock_cm.__exit__.return_value = None
    return mock_cm


class TestGeminiProvider:
    def test_init_default_model(self):
        provider = GeminiProvider("test-token")
        assert provider._api_token == "test-token"
        assert provider._model == "gemini-2.0-flash"

    def test_init_custom_model(self):
        provider = GeminiProvider("test-token", "gemini-2.0-pro")
        assert provider._model == "gemini-2.0-pro"

    @patch("jaime.providers.gemini.urllib.request.urlopen")
    def test_generate_success(self, mock_urlopen):
        payload = json.dumps({
            "candidates": [
                {"content": {"parts": [{"text": "Hello from Gemini"}]}}
            ]
        }).encode()
        mock_urlopen.return_value = _mock_urlopen_response(payload)

        provider = GeminiProvider("test-token")
        result = provider.generate("say hello")

        assert result == "Hello from Gemini"

    @patch("jaime.providers.gemini.urllib.request.urlopen")
    def test_generate_no_candidates_raises(self, mock_urlopen):
        payload = json.dumps({"candidates": []}).encode()
        mock_urlopen.return_value = _mock_urlopen_response(payload)

        provider = GeminiProvider("test-token")
        with pytest.raises(RuntimeError, match="no candidates"):
            provider.generate("test")

    @patch("jaime.providers.gemini.urllib.request.urlopen")
    def test_generate_no_parts_raises(self, mock_urlopen):
        payload = json.dumps({
            "candidates": [{"content": {"parts": []}}]
        }).encode()
        mock_urlopen.return_value = _mock_urlopen_response(payload)

        provider = GeminiProvider("test-token")
        with pytest.raises(RuntimeError, match="no parts"):
            provider.generate("test")

    @patch("jaime.providers.gemini.urllib.request.urlopen")
    def test_generate_http_error_raises(self, mock_urlopen):
        import urllib.error

        http_error = urllib.error.HTTPError(
            url="http://example.com",
            code=400,
            msg="Bad Request",
            hdrs={},
            fp=MagicMock(),
        )
        http_error.read.return_value = b'{"error": "invalid"}'
        mock_urlopen.side_effect = http_error

        provider = GeminiProvider("test-token")
        with pytest.raises(urllib.error.HTTPError):
            provider.generate("test")

    @patch("jaime.providers.gemini.urllib.request.urlopen")
    def test_generate_url_error_raises(self, mock_urlopen):
        import urllib.error

        mock_urlopen.side_effect = urllib.error.URLError("connection failed")

        provider = GeminiProvider("test-token")
        with pytest.raises(urllib.error.URLError):
            provider.generate("test")
