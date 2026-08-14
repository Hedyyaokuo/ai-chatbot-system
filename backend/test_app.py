import os
import unittest

os.environ.pop("GROQ_API_KEY", None)

from app import app  # noqa: E402


class ChatApiTest(unittest.TestCase):
    def setUp(self):
        app.config.update(TESTING=True)
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/api/health")
        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["mode"], "extractive-fallback")

    def test_chat_returns_grounded_sources_and_trace(self):
        response = self.client.post(
            "/api/chat",
            json={
                "message": "Hybrid Retrieval 为什么比 text-only 更好？",
                "session_id": "unittest_session_001",
            },
            headers={"Origin": "https://hedyyaokuo.github.io"},
        )
        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["query_family"], "retrieval")
        self.assertEqual(payload["verification_result"], "passed")
        self.assertGreater(len(payload["sources"]), 0)
        self.assertIn("读取会话记忆", payload["trace"])
        self.assertEqual(
            response.headers["Access-Control-Allow-Origin"],
            "https://hedyyaokuo.github.io",
        )

    def test_invalid_session_is_rejected(self):
        response = self.client.post(
            "/api/chat",
            json={"message": "测试", "session_id": "bad"},
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
