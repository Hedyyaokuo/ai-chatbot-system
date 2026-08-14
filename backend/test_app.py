import os
import unittest
from pathlib import Path

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
        self.assertEqual(payload["knowledge_base"]["records"], 1813)
        self.assertEqual(payload["knowledge_base"]["text_chunks"], 1786)
        self.assertEqual(payload["knowledge_base"]["image_captions"], 27)
        self.assertEqual(payload["knowledge_base"]["chunk_size"], 650)
        self.assertEqual(payload["knowledge_base"]["chunk_overlap"], 100)

    def test_generation_prompt_requires_natural_synthesis(self):
        from agent import SYSTEM_PROMPT

        self.assertIn("不是搜索结果展示器", SYSTEM_PROMPT)
        self.assertIn("自然、连贯的中文", SYSTEM_PROMPT)
        self.assertIn("不要逐段复制知识块", SYSTEM_PROMPT)
        self.assertIn("不要输出 HTML", SYSTEM_PROMPT)
        self.assertIn("心理健康专业人士", SYSTEM_PROMPT)

    def test_all_image_captions_have_public_assets(self):
        from agent import KNOWLEDGE_INDEX

        image_directory = Path(__file__).resolve().parents[1] / "assets" / "knowledge-images"
        image_records = [
            record
            for record in KNOWLEDGE_INDEX.records
            if record.get("modality") == "image_caption"
        ]
        self.assertEqual(len(image_records), 27)
        missing = [
            record["source_file"]
            for record in image_records
            if not (image_directory / record["source_file"]).is_file()
        ]
        self.assertEqual(missing, [])

    def test_chat_returns_grounded_sources_and_trace(self):
        response = self.client.post(
            "/api/chat",
            json={
                "message": "EventNow 的 organiser 可以创建和管理哪些内容？",
                "session_id": "unittest_session_001",
            },
            headers={"Origin": "https://hedyyaokuo.github.io"},
        )
        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["query_family"], "factual_retrieval")
        self.assertEqual(payload["selected_tool"], "text_retrieval")
        self.assertEqual(
            payload["verification_result"],
            "passed: original knowledge evidence retrieved",
        )
        self.assertGreater(len(payload["sources"]), 0)
        self.assertIn("读取会话记忆", payload["trace"])
        self.assertTrue(
            any(source["document_family"] == "eventnow_project" for source in payload["sources"])
        )
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

    def test_cors_safelisted_text_request(self):
        response = self.client.post(
            "/api/chat",
            data=(
                '{"message":"EventNow 的登录页面是什么样的？",'
                '"session_id":"simple_cors_session_001"}'
            ),
            content_type="text/plain;charset=UTF-8",
            headers={"Origin": "https://hedyyaokuo.github.io"},
        )
        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["query_family"], "cross_modal")
        self.assertEqual(payload["selected_tool"], "image_caption_retrieval")
        self.assertTrue(any(source["modality"] == "image_caption" for source in payload["sources"]))
        self.assertTrue(
            all(
                source["image_url"].startswith(
                    "https://hedyyaokuo.github.io/ai-chatbot-system/assets/knowledge-images/"
                )
                for source in payload["sources"]
                if source["modality"] == "image_caption"
            )
        )
        self.assertEqual(
            response.headers["Access-Control-Allow-Origin"],
            "https://hedyyaokuo.github.io",
        )

    def test_session_memory_personalises_follow_up(self):
        session_id = "memory_session_001"
        first = self.client.post(
            "/api/chat",
            json={
                "message": "我喜欢日本夜市和灯笼图片。",
                "session_id": session_id,
            },
        )
        self.assertEqual(first.status_code, 200)

        follow_up = self.client.post(
            "/api/chat",
            json={"message": "再推荐一个适合我的。", "session_id": session_id},
        )
        payload = follow_up.get_json()
        self.assertEqual(follow_up.status_code, 200)
        self.assertEqual(payload["query_family"], "personalised_context")
        self.assertEqual(payload["selected_tool"], "personalised_hybrid_retrieval")
        self.assertTrue(any("food_street_culture" in step for step in payload["trace"]))

    def test_chinese_query_retrieves_original_anxiety_workbook(self):
        response = self.client.post(
            "/api/chat",
            json={
                "message": "焦虑管理资料里有哪些呼吸练习？",
                "session_id": "anxiety_session_001",
            },
        )
        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            any(
                source["source_file"] == "AnxietyManagmentWorkbook.pdf"
                for source in payload["sources"]
            )
        )

    def test_chinese_visual_query_retrieves_original_image_caption(self):
        response = self.client.post(
            "/api/chat",
            json={
                "message": "请找到最符合日本夜市和灯笼主题的图片。",
                "session_id": "japan_image_session_001",
            },
        )
        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["selected_tool"], "image_caption_retrieval")
        self.assertEqual(payload["sources"][0]["source_file"], "Japan culture.jpg")
        self.assertTrue(payload["sources"][0]["image_url"].endswith("Japan%20culture.jpg"))
        self.assertTrue(
            all(
                source["document_family"] == "japan_travel"
                for source in payload["sources"]
            )
        )


if __name__ == "__main__":
    unittest.main()
