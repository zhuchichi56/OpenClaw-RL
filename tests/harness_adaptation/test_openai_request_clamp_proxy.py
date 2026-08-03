import unittest

from scripts.harness_adaptation.openai_request_clamp_proxy import (
    normalize_chat_payload,
)


class NormalizeChatPayloadTest(unittest.TestCase):
    def test_overwrites_causal_controls_without_mutating_input(self) -> None:
        source = {
            "model": "wrong-model",
            "temperature": 0.9,
            "max_completion_tokens": 100,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        }

        normalized, original = normalize_chat_payload(
            source,
            model="frozen-model",
            temperature=0.0,
            max_tokens=8192,
        )

        self.assertEqual(source["model"], "wrong-model")
        self.assertEqual(normalized["model"], "frozen-model")
        self.assertEqual(normalized["temperature"], 0.0)
        self.assertEqual(normalized["max_tokens"], 8192)
        self.assertNotIn("max_completion_tokens", normalized)
        self.assertEqual(original["max_completion_tokens"], 100)


if __name__ == "__main__":
    unittest.main()
