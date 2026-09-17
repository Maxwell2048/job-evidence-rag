import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tailor_cv
from local_llm import LLMError

EXPERIENCE = """# Synthetic Project

## 基本信息
- 项目类型：课程项目，团队 4 人。

## 我的贡献 图片标注
我完成并提交 12 幅示例地图的标注，以 LabelMe 格式记录制图元素。
证据：[标注提交](https://github.com/example/commit/aaa)。

## 英文简历表述
Annotated 12 sample maps in LabelMe format for an object detection dataset.
"""

PROFILE = """# 个人基本信息（自述事实）

## 学历 Education
- 学位：信息技术硕士（Master of Information Technology）。
"""

RESUME = ("ALEX SAMPLE\nEDUCATION\nMaster of IT · AI Track\n"
          "KEY PROJECTS\nGeoMind | trained YOLO11n on 10 classes\n"
          "SKILLS\nPython · SQL\n")


def make_match(rid, verdict, project="Synthetic Project", source="synthetic_project.md",
               importance="required", missing=None, status="ok"):
    return {
        "requirement_id": rid, "text": f"Requirement {rid}", "category": "experience",
        "importance": importance, "processing_status": status, "verdict": verdict,
        "reason": "r", "missing_aspects": missing or [], "questions_for_user": [],
        "evidence": [] if verdict == "insufficient" else [
            {"candidate_id": "E001", "quote": "q", "project": project, "section": "s",
             "source": source, "line_start": 1, "line_end": 2}],
        "candidates": [], "errors": [],
    }


class FakeLLM:
    prompt_char_budget = 100000

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def estimate_context(self, task, system_prompt, payload, schema):
        return len(json.dumps(payload, ensure_ascii=False))

    def generate_json(self, task, system_prompt, payload, schema=None,
                      timeout_seconds=None, validate=None):
        self.calls.append((task, payload))
        last = None
        for _ in range(3):
            if not self.responses:
                break
            response = self.responses.pop(0)
            if isinstance(response, Exception):
                raise response
            errors = list(validate(response)) if validate else []
            if not errors:
                return response
            last = errors
        raise LLMError("；".join(last or ["no response"]), error_type="structure")


def build_docx(path, paragraphs):
    body = ""
    for text in paragraphs:
        runs = "".join(f"<w:t xml:space=\"preserve\">{part}</w:t><w:br/>"
                       for part in text.split("\n"))
        body += f"<w:p><w:r>{runs}</w:r></w:p>"
    xml = ('<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="x"><w:body>'
           '<w:tbl><w:tr><w:tc><w:tcPr><w:tcW w:w="9866" w:type="dxa"/></w:tcPr>'
           f"{body}</w:tc></w:tr></w:tbl></w:body></w:document>")
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", xml)


class ResumeReadingTests(unittest.TestCase):
    def test_docx_paragraphs_and_breaks_without_table_tags(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.docx"
            build_docx(path, ["ALEX SAMPLE\nAI Engineer", "Skills &amp; Tools"])
            text = tailor_cv.read_resume(path)
        self.assertEqual(text, "ALEX SAMPLE\nAI Engineer\nSkills & Tools\n")
        self.assertNotIn("tcW", text)

    def test_markdown_and_unknown_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp) / "r.md"
            md.write_text("hello", encoding="utf-8")
            self.assertEqual(tailor_cv.read_resume(md), "hello")
            with self.assertRaises(ValueError):
                tailor_cv.read_resume(Path(tmp) / "r.pdf")


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.by_id = {"M001": {"text": "I annotated 12 maps in LabelMe format."},
                      "RESUME": {"text": "trained YOLO11n on 10 classes"}}

    def test_quote_must_be_verbatim_and_material_allowed(self):
        items = [{"text": "Annotated 12 maps", "basis": [{"material_id": "M001", "quote": "Annotated 21 maps"}]},
                 {"text": "x", "basis": [{"material_id": "M999", "quote": "x"}]},
                 {"text": "y", "basis": []}]
        errors = tailor_cv.validate_basis(items, self.by_id, ["M001"])
        self.assertEqual(len(errors), 3)
        self.assertIn("逐字连续原文", errors[0])
        self.assertIn("M999", errors[1])
        self.assertIn("缺少 basis", errors[2])

    def test_numbers_must_exist_in_material_pool(self):
        good = [{"text": "Annotated 12 maps", "basis": [{"material_id": "M001", "quote": "12 maps"}]}]
        bad = [{"text": "Annotated 120 maps", "basis": [{"material_id": "M001", "quote": "12 maps"}]}]
        self.assertEqual(tailor_cv.validate_basis(good, self.by_id, ["M001"]), [])
        self.assertIn("'120'", tailor_cv.validate_basis(bad, self.by_id, ["M001"])[0])

    def test_jd_wording_is_named_in_error(self):
        items = [{"text": "Strong attention to detail",
                  "basis": [{"material_id": "M001", "quote": "attention to detail"}]}]
        errors = tailor_cv.validate_basis(items, self.by_id, ["M001"], label="summary ",
                                          jd_text="We need Attention to Detail.")
        self.assertEqual(len(errors), 1)
        self.assertTrue(errors[0].startswith("summary 对 M001"))
        self.assertIn("JD 的措辞", errors[0])

    def test_resume_review_quotes(self):
        errors = tailor_cv.validate_quotes_in(
            [{"quote": "YOLO11n on 10 classes"}, {"quote": "made up"}], self.by_id["RESUME"]["text"])
        self.assertEqual(len(errors), 1)


class AnalysisTests(unittest.TestCase):
    def test_gaps_and_ranking(self):
        matches = [make_match("R001", "direct"), make_match("R002", "insufficient"),
                   make_match("R003", "related", missing=["production use"]),
                   make_match("R004", "direct", project="Other", source="other.md"),
                   make_match("R005", "direct", status="error"),
                   make_match("R006", "direct", project="个人基本信息（自述事实）", source="profile.md")]
        by_project = {"Synthetic Project": [{"kind": "experience"}],
                      "Other": [{"kind": "experience"}],
                      "个人基本信息（自述事实）": [{"kind": "profile"}]}
        order = tailor_cv.rank_projects(matches, by_project)
        self.assertEqual([p["project"] for p in order], ["Synthetic Project", "Other"])
        self.assertEqual(order[0]["evidence_score"], 4)
        kinds = {g["requirement_id"]: g["kind"] for g in tailor_cv.gaps(matches)}
        self.assertEqual(kinds, {"R002": "insufficient", "R003": "partial", "R005": "unprocessed"})


class EndToEndTests(unittest.TestCase):
    def run_tailor(self, responses):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data = tmp / "experiences"
            data.mkdir()
            (data / "synthetic_project.md").write_text(EXPERIENCE, encoding="utf-8")
            (data / "profile.md").write_text(PROFILE, encoding="utf-8")
            run = tmp / "run"
            run.mkdir()
            matches = [make_match("R001", "direct"), make_match("R002", "insufficient")]
            meta = {"run_id": "run", "scope": "full", "input": {"jd_path": "jd.txt"},
                    "models": {"generator": {"name": "fake"}}}
            (run / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")
            (run / "matches.json").write_text(json.dumps({"matches": matches}), encoding="utf-8")
            (run / "jd.txt").write_text("JD text", encoding="utf-8")
            meta, matches, jd_text = tailor_cv.load_run(run)
            materials, by_project, resume_entry = tailor_cv.load_materials(data, RESUME)
            llm = FakeLLM(responses)
            result = tailor_cv.tailor(llm, meta, matches, jd_text, materials, by_project, resume_entry)
            out = run / "cv_suggestions.md"
            tailor_cv.write_markdown(result, meta, matches, "base.docx", out)
            return result, out.read_text(encoding="utf-8"), llm, materials

    def test_full_flow_with_retry_on_bad_quote(self):
        good_bullet = {"text": "Annotated 12 sample maps in LabelMe format",
                       "basis": [{"material_id": "M004", "quote": "Annotated 12 sample maps"}],
                       "requirement_ids": ["R001"]}
        bad = {"bullets": [{**good_bullet, "basis": [{"material_id": "M004", "quote": "fabricated"}]}],
               "note": ""}
        responses = [
            bad, {"bullets": [good_bullet], "note": "ok"},
            {"items": [{"quote": "trained YOLO11n on 10 classes", "action": "keep", "reason": "相关"}]},
            {"summary": {"text": "IT graduate.", "basis": [{"material_id": "RESUME", "quote": "Master of IT"}]}},
            {"cover_letter": [{"text": "Dear team, I annotated 12 sample maps.",
                               "basis": [{"material_id": "M004", "quote": "Annotated 12 sample maps"}]}]},
        ]
        result, md, llm, materials = self.run_tailor(responses)
        self.assertEqual(result["status"], "complete")
        self.assertEqual([m["material_id"] for m in materials][-1], "RESUME")
        self.assertEqual(result["projects"][0]["bullets"][0]["basis"][0]["section"], "英文简历表述")
        self.assertEqual(result["gaps"][0]["requirement_id"], "R002")
        self.assertIn("## 6. 不要硬写的缺口", md)
        self.assertIn("R002", md)
        self.assertIn("[已核对经历] synthetic_project.md › 英文简历表述", md)
        self.assertIn("[简历底稿（未核对）] 简历底稿", md)
        self.assertIn("保留并强调", md)
        self.assertEqual(llm.calls[0][0], "cv_bullets:Synthetic Project")
        self.assertIn("resume_draft", llm.calls[0][1])

    def test_step_failure_is_partial_not_fatal(self):
        responses = [LLMError("boom", error_type="timeout"),
                     {"items": []},
                     {"summary": {"text": "IT graduate.", "basis": [{"material_id": "RESUME", "quote": "Master of IT"}]}},
                     LLMError("boom2", error_type="timeout")]
        result, md, llm, _ = self.run_tailor(responses)
        self.assertEqual(result["status"], "partial")
        self.assertEqual([e["step"] for e in result["errors"]],
                         ["bullets:Synthetic Project", "cover_letter"])
        # a failed cover letter no longer takes the summary down with it
        self.assertEqual(result["summary"]["text"], "IT graduate.")
        self.assertEqual([c[0] for c in llm.calls][-2:], ["cv_summary", "cv_cover_letter"])
        self.assertIn("## 失败步骤", md)
        self.assertIn("（未生成）", md)


class MaterialKindTests(unittest.TestCase):
    def test_unverified_files_are_labelled_and_still_ranked(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            (data / "verified.md").write_text(EXPERIENCE, encoding="utf-8")
            (data / "imported.md").write_text(
                "# Imported\n\n## 基本信息\n- 事实状态：待核对。来源仅为简历底稿。\n\n"
                "## 我的工作 x\nDid x.\n", encoding="utf-8")
            (data / "profile.md").write_text(PROFILE, encoding="utf-8")
            materials, by_project, resume = tailor_cv.load_materials(data)
            self.assertIsNone(resume)
            kinds = {m["source"]: m["kind"] for m in materials}
            self.assertEqual(kinds, {"verified.md": "experience", "imported.md": "unverified",
                                     "profile.md": "profile"})
            matches = [make_match("R001", "direct", project="Imported", source="imported.md")]
            order = tailor_cv.rank_projects(matches, by_project)
            self.assertEqual([p["project"] for p in order], ["Imported", "Synthetic Project"])
            self.assertEqual(tailor_cv.KIND_LABELS["unverified"], "待核对经历（仅底稿）")


class ScopeTests(unittest.TestCase):
    def test_sections_get_personal_or_team_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            (data / "p.md").write_text(
                "# P\n\n## 项目背景与团队成果\n团队做了系统。\n\n## 项目内容与团队成果 方法\n团队方法。\n\n"
                "## 我的贡献 标注\n我标注。\n\n## STAR 案例 x\nS。\n\n## 英文简历表述\nLine.\n\n"
                "## 我的贡献 本人自述补充\n本人自述参与。\n", encoding="utf-8")
            (data / "profile.md").write_text(PROFILE, encoding="utf-8")
            materials, _, _ = tailor_cv.load_materials(data, RESUME)
            scopes = {m["section"]: m["scope"] for m in materials}
            self.assertEqual(scopes["项目背景与团队成果"], "team")
            self.assertEqual(scopes["项目内容与团队成果 方法"], "team")
            self.assertEqual(scopes["我的贡献 标注"], "personal")
            self.assertEqual(scopes["STAR 案例 x"], "personal")
            self.assertEqual(scopes["英文简历表述"], "personal")
            self.assertEqual(scopes["我的贡献 本人自述补充"], "personal")
            self.assertEqual(scopes["学历 Education"], "personal")
            self.assertEqual(scopes["全文"], "personal")

    def test_bullet_citing_only_team_sections_is_rejected(self):
        by_id = {"T1": {"text": "The team built a Docker stack.", "scope": "team"},
                 "P1": {"text": "I annotated 57 maps.", "scope": "personal"}}
        only_team = [{"text": "Built a Docker stack", "basis": [{"material_id": "T1", "quote": "Docker stack"}]}]
        mixed = [{"text": "Annotated 57 maps for a Docker stack",
                  "basis": [{"material_id": "T1", "quote": "Docker stack"},
                            {"material_id": "P1", "quote": "57 maps"}]}]
        self.assertEqual(tailor_cv.validate_basis(only_team, by_id, ["T1", "P1"]), [])
        errors = tailor_cv.validate_basis(only_team, by_id, ["T1", "P1"], require_personal=True)
        self.assertEqual(len(errors), 1)
        self.assertIn("scope=team", errors[0])
        self.assertEqual(tailor_cv.validate_basis(mixed, by_id, ["T1", "P1"], require_personal=True), [])


class NumberSupportTests(unittest.TestCase):
    def test_thousands_separator_is_the_only_tolerated_difference(self):
        pool = "保存输出显示共有 1288 张图像；98,619 条记录；accuracy 0.5117。"
        self.assertTrue(tailor_cv.number_supported("1288", pool))
        self.assertTrue(tailor_cv.number_supported("1,288", pool))
        self.assertTrue(tailor_cv.number_supported("98619", pool))
        self.assertTrue(tailor_cv.number_supported("98,619", pool))
        self.assertTrue(tailor_cv.number_supported("0.5117", pool))
        self.assertFalse(tailor_cv.number_supported("1,289", pool))
        self.assertFalse(tailor_cv.number_supported("51.17", pool))  # no unit conversion
        self.assertFalse(tailor_cv.number_supported("12,88", pool))  # not a thousands group

    def test_validate_basis_accepts_reformatted_thousands(self):
        by_id = {"P1": {"text": "共有 1288 张 62×47 灰度图像", "scope": "personal"}}
        item = [{"text": "Processed 1,288 grayscale images", "basis": [{"material_id": "P1", "quote": "1288 张"}]}]
        self.assertEqual(tailor_cv.validate_basis(item, by_id, ["P1"]), [])


class LooseQuoteTests(unittest.TestCase):
    def test_punctuation_case_and_spacing_are_ignored(self):
        text = "I took a proactive problem solving approach (see notes)."
        self.assertTrue(tailor_cv.quote_in("proactive problem-solving", text))
        self.assertTrue(tailor_cv.quote_in("Proactive  Problem Solving approach", text))
        self.assertTrue(tailor_cv.quote_in("see notes", text))
        self.assertFalse(tailor_cv.quote_in("proactive solving", text))
        self.assertFalse(tailor_cv.quote_in("", text))


class CourseCodeTests(unittest.TestCase):
    def test_course_codes_rejected_only_when_asked(self):
        by_id = {"P1": {"text": "Finance Planner · CITS5505 group project, 30 commits", "scope": "personal"}}
        item = [{"text": "Finance Planner (CITS5505): 30 commits",
                 "basis": [{"material_id": "P1", "quote": "30 commits"}]}]
        self.assertEqual(tailor_cv.validate_basis(item, by_id, ["P1"]), [])
        errors = tailor_cv.validate_basis(item, by_id, ["P1"], no_course_codes=True)
        self.assertEqual(len(errors), 1)
        self.assertIn("CITS5505", errors[0])
        clean = [{"text": "Finance Planner (group project): 30 commits",
                  "basis": [{"material_id": "P1", "quote": "30 commits"}]}]
        self.assertEqual(tailor_cv.validate_basis(clean, by_id, ["P1"], no_course_codes=True), [])


class RunGuardTests(unittest.TestCase):
    def test_extract_only_run_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "run_meta.json").write_text(json.dumps({"run_id": "x", "scope": "extraction"}),
                                               encoding="utf-8")
            (run / "matches.json").write_text(json.dumps({"matches": []}), encoding="utf-8")
            (run / "jd.txt").write_text("x", encoding="utf-8")
            with self.assertRaises(ValueError):
                tailor_cv.load_run(run)


if __name__ == "__main__":
    unittest.main()
