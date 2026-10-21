"""控制线索在新旧流程中的端到端传递及模型响应协议；全部离线。"""
import unittest
from unittest.mock import patch

from agent_schema import response_schema
import test_fact_driven as fact_fixtures
import test_generic_flow as generic_fixtures
import test_source_agent_v4 as source_fixtures


def withdrawal(session):
    return {
        "kind": "withdrawal", "subject_key": None, "attribute_key": None,
        "scope": "session", "target_type": "session", "target_text": "",
        "evidence": [{"evidence_id": session["turns"][0]["evidence_id"],
                      "speaker": "user", "start": None, "end": None}],
        "switch_basis": "none", "switch_reference": None,
    }


class RouterSchemaTests(unittest.TestCase):
    def test_generic_router_accepts_evidence_backed_controls(self):
        for flow in ("generic-flow-v2", "generic-flow-v3-relations"):
            with self.subTest(flow=flow):
                schema = response_schema("source_router", {"flow_version": flow})
                self.assertEqual(set(schema["required"]), {"skills", "signals", "reason"})
                signal = schema["properties"]["signals"]["items"]["properties"]
                reference = signal["evidence"]["items"]["properties"]
                self.assertEqual(set(reference), {"evidence_id", "speaker", "start", "end"})
                self.assertIn("switch_reference", signal)

    def test_fact_router_and_control_have_separate_response_contracts(self):
        data = {"flow_version": "fact-driven-v1"}
        router = response_schema("source_router", data)
        self.assertEqual(set(router["properties"]), {"skills", "reason"})
        control = response_schema("source_control", data)
        self.assertEqual(set(control["required"]), {"signals"})
        reference = control["properties"]["signals"]["items"]["properties"]["evidence"]["items"]
        self.assertEqual(reference["properties"]["speaker"]["enum"], ["user"])
        self.assertEqual(reference["properties"]["start"], {"type": "null"})

    def test_fact_relation_schema_retains_explicit_update_direction(self):
        schema = response_schema("source_relations", {"flow_version": "fact-driven-v1"})
        result = schema["properties"]["results"]["items"]
        self.assertIn("direction", result["required"])
        self.assertIn("changed_with_direction", result["properties"]["relation"]["enum"])


class GenericControlTests(unittest.TestCase):
    def test_control_only_window_preserves_withdrawal_without_extracting(self):
        harness = generic_fixtures.GenericTests()
        harness.setUp()
        self.addCleanup(harness.doCleanups)
        session = source_fixtures.session("control-only", text="这次聊天不要保存")
        pipeline = generic_fixtures.GenericPipeline(harness.h.agent, session)
        calls = []

        def respond(role, prompt, data):
            calls.append(role)
            self.assertEqual(role, "source_router")
            self.assertIn("signals", response_schema(role, data)["required"])
            self.assertIn("输出skills,signals,reason", prompt)
            return {"skills": [], "signals": [withdrawal(session)], "reason": "仅有保存控制"}

        with patch.object(harness.fake, "ask", side_effect=respond):
            candidates, signals = pipeline.extract_window(session)
        self.assertEqual(candidates, [])
        self.assertEqual(calls, ["source_router"])
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0]["kind"], "withdrawal")
        self.assertEqual(signals[0]["evidence"][0]["quote"], "这次聊天不要保存")


class FactControlTests(unittest.TestCase):
    def setUp(self):
        self.harness = fact_fixtures.FactPipelineTests()
        self.harness.setUp()
        self.addCleanup(self.harness.doCleanups)

    def test_independent_control_blocks_memory_with_router_only_contract(self):
        session = fact_fixtures.session("do-not-save")
        session["turns"][0]["user_text"] = "我喜欢恐龙，这次聊天不要保存"
        model = self.harness.agent.client.client
        original = model.ask
        calls = []

        def respond(role, prompt, data):
            calls.append(role)
            if role == "source_control":
                self.assertEqual(set(response_schema(role, data)["required"]), {"signals"})
                return {"signals": [withdrawal(session)]}
            if role == "source_router":
                self.assertEqual(set(response_schema(role, data)["required"]), {"skills", "reason"})
                return {"skills": ["interests_preferences"], "reason": "提取偏好"}
            return original(role, prompt, data)

        with patch.object(model, "ask", side_effect=respond):
            result = self.harness.agent.run_session(session)
        self.assertEqual(result["approved"], [])
        self.assertTrue(any(item.get("storage_control") == "blocked" for item in result["maintenance"]))
        self.assertEqual(self.harness.store.db.execute(
            "SELECT COUNT(*) FROM fact_assertions WHERE state='accepted'"
        ).fetchone()[0], 0)
        self.assertLess(calls.index("source_control"), calls.index("source_router"))

    def test_router_cannot_inject_controls_into_fact_flow(self):
        session = fact_fixtures.session("router-control")
        model = self.harness.agent.client.client
        original = model.ask

        def respond(role, prompt, data):
            if role == "source_control":
                return {"signals": []}
            if role == "source_router":
                return {"skills": ["interests_preferences"],
                        "signals": [{"invalid": "不是已核验的控制线索"}], "reason": "旧式路由返回"}
            return original(role, prompt, data)

        with patch.object(model, "ask", side_effect=respond):
            result = self.harness.agent.run_session(session)
        self.assertEqual(result["signals"], [])
        self.assertEqual(len(result["approved"]), 1)


if __name__ == "__main__":
    unittest.main()
