import json
import tempfile
import time
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_resume
import tailor_cv
from local_llm import LLMError
try:
    from test_tailor_cv import EXPERIENCE, FakeLLM, PROFILE, RESUME
except ModuleNotFoundError:  # run as tests.test_build_resume
    from tests.test_tailor_cv import EXPERIENCE, FakeLLM, PROFILE, RESUME


def setup_dir(tmp):
    data = tmp / "experiences"
    data.mkdir()
    (data / "synthetic_project.md").write_text(EXPERIENCE, encoding="utf-8")
    (data / "profile.md").write_text(PROFILE, encoding="utf-8")
    materials, _by_project, resume_entry = tailor_cv.load_materials(data, RESUME)
    suggestions = {
        "projects": [{"project": "Synthetic Project", "bullets": [
            {"text": "Annotated 12 sample maps in LabelMe format",
             "basis": [{"material_id": "M004", "quote": "Annotated 12 sample maps",
                        "kind": "experience"}]}]}],
        "summary": {"text": "IT graduate.", "basis": [{"material_id": "RESUME", "quote": "Master of IT"}]},
        "resume_review": [{"quote": "Python · SQL", "action": "keep", "reason": "相关"}],
        "gaps": [{"requirement_id": "R002", "text": "x", "kind": "insufficient"}],
    }
    return materials, resume_entry, suggestions


class PayloadTests(unittest.TestCase):
    def test_allowed_ids_and_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            materials, resume_entry, suggestions = setup_dir(Path(tmp))
            allowed = build_resume.allowed_material_ids(suggestions, materials)
            self.assertEqual(allowed, ["RESUME", "M001", "M004"])
            payload = build_resume.build_payload(suggestions, materials, resume_entry, "JD")
            self.assertEqual(payload["tailored_projects"][0]["bullets"][0],
                             {"id": "B01", "text": "Annotated 12 sample maps in LabelMe format"})
            self.assertEqual(payload["summary"], {"id": "SUMMARY", "text": "IT graduate."})
            self.assertEqual(payload["profile"][0]["material_id"], "M001")

    def test_latest_suggestions_picks_newest(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            with self.assertRaises(ValueError):
                build_resume.latest_suggestions(run)
            (run / "cv_suggestions.json").write_text("{}", encoding="utf-8")
            time.sleep(0.01)
            (run / "cv_suggestions-2.json").write_text("{}", encoding="utf-8")
            self.assertEqual(build_resume.latest_suggestions(run).name, "cv_suggestions-2.json")


class BuildTests(unittest.TestCase):
    GOOD = {"sections": [
        {"title": "Header", "lines": [
            {"kind": "text", "text": "ALEX SAMPLE", "basis": [{"material_id": "RESUME", "quote": "ALEX SAMPLE"}]}]},
        {"title": "Summary", "lines": [
            {"kind": "text", "text": "IT graduate.", "basis": [{"material_id": "RESUME", "quote": "Master of IT"}]}]},
        {"title": "Key Projects", "lines": [
            {"kind": "heading", "text": "Synthetic Project",
             "basis": [{"material_id": "M004", "quote": "Annotated 12 sample maps"}]},
            {"kind": "bullet", "text": "Annotated 12 sample maps in LabelMe format",
             "basis": [{"material_id": "M004", "quote": "Annotated 12 sample maps"}]}]},
        {"title": "Skills", "lines": [
            {"kind": "text", "text": "Python · SQL", "basis": [{"material_id": "RESUME", "quote": "Python · SQL"}]}]},
    ]}

    def test_build_validates_then_renders(self):
        with tempfile.TemporaryDirectory() as tmp:
            materials, resume_entry, suggestions = setup_dir(Path(tmp))
            bad = json.loads(json.dumps(self.GOOD))
            bad["sections"][2]["lines"][1]["basis"][0]["quote"] = "attention to detail"
            llm = FakeLLM([bad, self.GOOD])
            sections = build_resume.build(llm, suggestions, materials, resume_entry,
                                          "JD wants attention to detail")
            self.assertEqual(len(llm.calls), 1)
            self.assertEqual(sections[2]["lines"][1]["basis"][0]["section"], "英文简历表述")
            md = build_resume.render(sections, "cv_suggestions.json", "base.docx", {"run_id": "r"})
            self.assertTrue(md.startswith("# ALEX SAMPLE\n"))
            self.assertIn("### Synthetic Project", md)
            self.assertIn("- Annotated 12 sample maps in LabelMe format", md)
            self.assertIn("## 来源附录", md)
            self.assertIn("依据未核对材料（简历底稿、待核对经历）的引用共 3 处", md)
            self.assertIn("[已核对经历] synthetic_project.md › 英文简历表述", md)

    def test_refs_are_expanded_from_suggestions(self):
        with tempfile.TemporaryDirectory() as tmp:
            materials, resume_entry, suggestions = setup_dir(Path(tmp))
            refs = build_resume.referenced_lines(suggestions)
            self.assertEqual(sorted(refs), ["B01", "SUMMARY"])
            payload = build_resume.build_payload(suggestions, materials, resume_entry, "JD")
            self.assertEqual(payload["tailored_projects"][0]["bullets"][0]["id"], "B01")
            self.assertNotIn("basis", payload["tailored_projects"][0]["bullets"][0])
            doc = {"sections": [
                {"title": "Header", "lines": [{"kind": "text", "text": "ALEX SAMPLE",
                                               "basis": [{"material_id": "RESUME", "quote": "ALEX SAMPLE"}]}]},
                {"title": "Summary", "lines": [{"kind": "text", "ref": "SUMMARY"}]},
                {"title": "Projects", "lines": [{"kind": "bullet", "ref": "B01"},
                                                {"kind": "bullet", "ref": "B99"},
                                                {"kind": "text"}]}]}
            by_id = {m["material_id"]: m for m in materials}
            required = build_resume.required_lines(suggestions, resume_entry["text"])
            errors = build_resume.validate_document(doc, by_id, ["RESUME", "M001", "M004"], "JD", required, refs)
            self.assertEqual(len(errors), 2)
            self.assertIn("B99", errors[0])
            self.assertIn("既没有 ref", errors[1])
            expanded = doc["sections"][2]["lines"][0]
            self.assertEqual(expanded["text"], "Annotated 12 sample maps in LabelMe format")
            self.assertEqual(expanded["basis"][0]["material_id"], "M004")
            self.assertEqual(doc["sections"][1]["lines"][0]["text"], "IT graduate.")

    def test_jd_quote_and_missing_material_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            materials, resume_entry, suggestions = setup_dir(Path(tmp))
            by_id = {m["material_id"]: m for m in materials}
            doc = {"sections": [{"title": "S", "lines": [
                {"kind": "text", "text": "x", "basis": [{"material_id": "M002", "quote": "课程项目"}]},
                {"kind": "text", "text": "y" * 600, "basis": [{"material_id": "RESUME", "quote": "ALEX"}]}]}]}
            errors = build_resume.validate_document(doc, by_id, ["RESUME", "M004"], "JD")
            self.assertEqual(len(errors), 2)
            self.assertIn("M002", errors[0])
            self.assertIn("超过", errors[1])

    def test_name_and_summary_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            materials, resume_entry, suggestions = setup_dir(Path(tmp))
            by_id = {m["material_id"]: m for m in materials}
            required = build_resume.required_lines(suggestions, resume_entry["text"])
            self.assertEqual([r[1] for r in required], ["ALEX SAMPLE", "IT graduate."])
            doc = json.loads(json.dumps(self.GOOD))
            doc["sections"][0]["lines"][0]["text"] = "AI Engineer"
            summary_section = doc["sections"].pop(1)
            errors = build_resume.validate_document(doc, by_id, ["RESUME", "M004"], "JD", required)
            self.assertEqual(len(errors), 2)
            self.assertIn("第 1 节缺少底稿第一行（姓名）", errors[0])
            self.assertIn("summary", errors[1])
            doc["sections"][0]["lines"][0]["text"] = "ALEX SAMPLE"
            doc["sections"].append(summary_section)
            self.assertEqual(build_resume.validate_document(doc, by_id, ["RESUME", "M004"], "JD", required), [])

    def test_failure_propagates(self):
        with tempfile.TemporaryDirectory() as tmp:
            materials, resume_entry, suggestions = setup_dir(Path(tmp))
            llm = FakeLLM([LLMError("boom", error_type="timeout")])
            with self.assertRaises(LLMError):
                build_resume.build(llm, suggestions, materials, resume_entry, "JD")


if __name__ == "__main__":
    unittest.main()
