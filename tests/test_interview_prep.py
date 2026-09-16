import json
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import interview_prep
import tailor_cv
from local_llm import LLMError
try:
    from test_tailor_cv import EXPERIENCE, FakeLLM, PROFILE
except ModuleNotFoundError:  # run as tests.test_interview_prep
    from tests.test_tailor_cv import EXPERIENCE, FakeLLM, PROFILE

STAR = """
## STAR 案例 标注一批地图
Situation：团队需要标注数据。
Task：我负责一批地图。
Action：我用 LabelMe 标注了 12 幅地图。
Result：提交了 12 幅地图的标注。
"""


def make_match(rid, verdict, candidates=True):
    cand = [{"candidate_id": "E001", "project": "Synthetic Project", "section": "我的贡献 图片标注",
             "source": "synthetic_project.md", "line_start": 7, "line_end": 9,
             "section_id": "s", "score": 0.8}] if candidates else []
    return {"requirement_id": rid, "text": f"Requirement {rid}", "importance": "required",
            "processing_status": "ok", "verdict": verdict, "source_quote": f"req {rid}",
            "missing_aspects": [], "questions_for_user": [], "candidates": cand,
            "evidence": [] if verdict == "insufficient" else [
                {"candidate_id": "E001", "quote": "12 幅", "project": "Synthetic Project",
                 "section": "我的贡献 图片标注", "source": "synthetic_project.md",
                 "line_start": 7, "line_end": 9}]}


def load(tmp):
    data = Path(tmp) / "experiences"
    data.mkdir()
    (data / "synthetic_project.md").write_text(EXPERIENCE + STAR, encoding="utf-8")
    (data / "profile.md").write_text(PROFILE, encoding="utf-8")
    materials, _, resume = tailor_cv.load_materials(data)
    assert resume is None
    return materials


class SelectionTests(unittest.TestCase):
    def test_materials_include_candidates_star_and_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            materials = load(tmp)
            chosen = interview_prep.select_materials([make_match("R001", "direct")], materials)
            sections = {m["section"] for m in chosen}
            self.assertIn("我的贡献 图片标注", sections)
            self.assertIn("STAR 案例 标注一批地图", sections)
            self.assertIn("学历 Education", sections)
            self.assertNotIn("英文简历表述", sections)

    def test_batches_skip_unprocessed(self):
        matches = [make_match(f"R{i:03d}", "direct") for i in range(1, 8)]
        matches[2]["processing_status"] = "error"
        self.assertEqual([len(b) for b in interview_prep.batches(matches, 5)], [5, 1])


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.by_id = {"M001": {"text": "I annotated 12 maps in LabelMe."}}

    def test_items_must_cover_batch_and_cite(self):
        data = {"items": [
            {"requirement_id": "R001", "questions": [{"question": "q", "intent": "", "note": "",
                "answer": [{"part": "Action", "text": "Annotated 12 maps",
                            "basis": [{"material_id": "M001", "quote": "annotated 12 maps"}]}]}]},
            {"requirement_id": "R999", "questions": []}]}
        errors = interview_prep.validate_items(data, ["R001", "R002"], self.by_id, ["M001"], "JD")
        self.assertEqual(len(errors), 2)
        self.assertIn("R999", errors[0])
        self.assertIn("缺少要求 R002", errors[1])

    def test_honest_point_rules(self):
        ok = [{"part": "Honest", "text": "I have not worked in an IT support role yet.", "basis": []}]
        self.assertEqual(interview_prep.validate_answer(ok, self.by_id, ["M001"], "JD"), [])
        bad_number = [{"part": "Honest", "text": "I supported 300 users.", "basis": []}]
        self.assertIn("'300'", interview_prep.validate_answer(bad_number, self.by_id, ["M001"], "JD")[0])
        bad_claim = [{"part": "Honest", "text": "I am proficient in ITIL.", "basis": []}]
        self.assertIn("不能声称能力", interview_prep.validate_answer(bad_claim, self.by_id, ["M001"], "JD")[0])
        star_without_basis = [{"part": "Result", "text": "Done.", "basis": []}]
        self.assertIn("缺少 basis", interview_prep.validate_answer(star_without_basis, self.by_id, ["M001"], "JD")[0])


class ScopeTests(unittest.TestCase):
    def test_star_answer_must_cite_personal_material(self):
        by_id = {"T1": {"text": "The team built the web system.", "scope": "team"},
                 "P1": {"text": "I annotated 57 maps.", "scope": "personal"}}
        team_only = {"items": [{"requirement_id": "R001", "questions": [{"question": "q", "intent": "", "note": "",
            "answer": [{"part": "Situation", "text": "The team built the web system.",
                        "basis": [{"material_id": "T1", "quote": "built the web system"}]}]}]}]}
        errors = interview_prep.validate_items(team_only, ["R001"], by_id, ["T1", "P1"], "JD")
        self.assertEqual(len(errors), 1)
        self.assertIn("scope=team", errors[0])
        team_only["items"][0]["questions"][0]["answer"].append(
            {"part": "Action", "text": "I annotated 57 maps.", "basis": [{"material_id": "P1", "quote": "annotated 57 maps"}]})
        self.assertEqual(interview_prep.validate_items(team_only, ["R001"], by_id, ["T1", "P1"], "JD"), [])
        honest_only = {"items": [{"requirement_id": "R001", "questions": [{"question": "q", "intent": "", "note": "",
            "answer": [{"part": "Honest", "text": "No direct experience.", "basis": []}]}]}]}
        self.assertEqual(interview_prep.validate_items(honest_only, ["R001"], by_id, ["T1", "P1"], "JD"), [])


class EndToEndTests(unittest.TestCase):
    def test_prepare_writes_markdown_and_handles_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            materials = load(tmp)
            star_id = next(m["material_id"] for m in materials if m["section"].startswith("STAR"))
            matches = [make_match("R001", "direct"), make_match("R002", "insufficient")]
            good = {"items": [
                {"requirement_id": "R001", "questions": [{"question": "How did you annotate?",
                    "intent": "细节", "note": "",
                    "answer": [{"part": "Action", "text": "I annotated 12 maps with LabelMe.",
                                "basis": [{"material_id": star_id, "quote": "LabelMe 标注了 12 幅地图"}]}]}]},
                {"requirement_id": "R002", "questions": [{"question": "Any IT support experience?",
                    "intent": "有无经验", "note": "",
                    "answer": [{"part": "Honest", "text": "Not directly; I am keen to learn.", "basis": []}]}]}]}
            bad = json.loads(json.dumps(good))
            bad["items"][0]["questions"][0]["answer"][0]["basis"][0]["quote"] = "attention to detail"
            llm = FakeLLM([bad, good, LLMError("boom", error_type="timeout")])
            result = interview_prep.prepare(llm, matches, "JD wants attention to detail", materials)
            self.assertEqual(result["status"], "partial")
            self.assertEqual([e["step"] for e in result["errors"]], ["general"])
            self.assertEqual(len(result["items"]), 2)
            self.assertEqual(result["items"][0]["questions"][0]["answer"][0]["basis"][0]["section"],
                             "STAR 案例 标注一批地图")
            meta = {"run_id": "r", "input": {"jd_path": "jd.txt"}, "models": {"generator": {"name": "fake"}}}
            out = Path(tmp) / "interview_prep.md"
            interview_prep.write_markdown(result, meta, out)
            md = out.read_text(encoding="utf-8")
            self.assertIn("## 一、会被深入追问的（直接支持）", md)
            self.assertIn("### R001 Requirement R001", md)
            self.assertIn("**Honest**：Not directly", md)
            self.assertIn("（无引用：坦诚说明，不含事实主张）", md)
            self.assertIn("## 失败步骤", md)


if __name__ == "__main__":
    unittest.main()
