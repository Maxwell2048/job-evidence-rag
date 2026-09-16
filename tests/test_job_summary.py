import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from job_summary import generate_summary, summary_markdown, validate_summary
from local_llm import LLMError


def match(rid="R1", verdict="related", **kwargs):
    return {"requirement_id": rid, "text": "React", "importance": "required",
            "category": "technical_skill", "processing_status": "ok", "verdict": verdict,
            "evidence": [{"quote": "A project", "project": "Synthetic"}],
            "missing_aspects": [], "needs_review": False, **kwargs}


def advice(rid="R1", **kwargs):
    return {"requirement_id": rid, "analysis": "当前资料支持相关经历，需补充 React 证据。",
            "next_step": "先核实是否做过组件开发。", "knowledge_points": ["props 与 state"],
            "exercise": "编写一个可筛选列表组件。", "acceptance_check": "筛选结果正确且有测试。",
            "urgency": "before_apply", **kwargs}


class SummaryTests(unittest.TestCase):
    def test_unknown_missing_duplicate_ids_rejected(self):
        for entries in [[], [advice("R2")], [advice(), advice()]]:
            self.assertTrue(validate_summary({"overview": "总结", "advice": entries}, [match()], []))

    def test_direct_cannot_be_urgent_gap(self):
        self.assertTrue(validate_summary({"overview": "总结", "advice": [advice()]},
                                         [match(verdict="direct")], []))

    def test_failed_requirement_cannot_be_learning_gap(self):
        self.assertTrue(validate_summary({"overview": "总结", "advice": [advice()]},
                                         [match(processing_status="error")], []))

    def test_satisfied_or_does_not_require_all_options(self):
        groups = [{"operator": "OR", "requirement_ids": ["R1", "R2"], "source_span": {"start": 0}}]
        data = {"overview": "总结", "advice": [advice(), advice("R2", urgency="later")]}
        self.assertTrue(validate_summary(data, [match(), match("R2", "direct")], groups))

    def test_qualifications_are_not_crash_courses(self):
        self.assertTrue(validate_summary({"overview": "总结", "advice": [advice()]},
                                         [match(category="work_authorization")], []))

    def test_model_failure_preserves_readable_fallback(self):
        class Offline:
            def generate_json(self, *args, **kwargs):
                raise LLMError("offline")
        result = generate_summary(Offline(), [match()])
        self.assertEqual(result["status"], "error")
        self.assertIn("逐项匹配结果仍可查看", "\n".join(summary_markdown(result, [match()])))

    def test_generated_summary_is_validated_and_rendered(self):
        class Local:
            model = "synthetic"
            def generate_json(self, task, prompt, payload, schema, validate):
                data = {"overview": "先整理相关项目，再补 React 演示。", "advice": [advice()]}
                assert not validate(data)
                return data
        result = generate_summary(Local(), [match()], run_status="partial")
        self.assertEqual(result["status"], "complete")
        text = "\n".join(summary_markdown(result, [match()]))
        self.assertIn("本次匹配未全部完成", text)
        self.assertIn("完成标准", text)
        self.assertIn("[R1]", text)

    def test_pending_claim_is_replaced_with_confirmation_not_an_advantage(self):
        class Local:
            def generate_json(self, *args, **kwargs):
                return {"overview": "仍需核实。", "advice": [advice(
                    analysis="完全掌握 React。", urgency="confirm_first", knowledge_points=[], exercise="")]}
        result = generate_summary(Local(), [match(needs_review=True)])
        self.assertEqual(result["status"], "complete")
        self.assertNotIn("完全掌握", result["advice"][0]["analysis"])
        self.assertIn("暂不据此评价能力", result["advice"][0]["analysis"])


if __name__ == "__main__":
    unittest.main()
