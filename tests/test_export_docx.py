import json
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import export_docx
from docx import Document
from docx.oxml.ns import qn

MD = """# ALEX SAMPLE
IT Graduate · Data Analysis
📍 Perth, WA
✉ someone@example.com
📞 +61 400 000 000

## SUMMARY
Master of IT graduate with full working rights.

## EDUCATION

### University of Western Australia
Master of IT · AI Track | 2024.02 – 2026.06
Coursework: Data Warehousing · RDBMS

## KEY PROJECTS

### Finance Planner | Flask Web App
- Model: Built the login pages.
- Delivered the forum module.

## ADDITIONAL PROJECTS

### GeoMind | Object Detection
- Annotated 57 maps.

### CyberWin | Requirements
- Recorded interviews.

## TECHNICAL SKILLS
Languages: Python · SQL
Data & DB: PostgreSQL

## SELF-ASSESSMENT
Profile: Strong self-learner.

---

## 来源附录（发送前删除本节及以下）

- 1.1 “ALEX SAMPLE” ← [简历底稿] ...
"""


def all_paragraphs(doc):
    """Every paragraph in body and (nested) tables, in document order."""
    found = []

    def walk_cell(cell):
        for para in cell.paragraphs:
            found.append(para)
        for table in cell.tables:
            for row in table.rows:
                for c in row.cells:
                    walk_cell(c)
    for para in doc.paragraphs:
        found.append(para)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                walk_cell(cell)
    return found


class ParseTests(unittest.TestCase):
    def test_resume_body_and_parse(self):
        lines = export_docx.resume_body(MD)
        self.assertFalse(any("来源附录" in l for l in lines))
        parsed = export_docx.parse_resume(lines)
        self.assertEqual(parsed["name"], "ALEX SAMPLE")
        self.assertEqual(parsed["title"], "IT Graduate · Data Analysis")
        self.assertEqual(len(parsed["contact"]), 3)
        self.assertEqual([s["title"] for s in parsed["sections"]],
                         ["SUMMARY", "EDUCATION", "KEY PROJECTS", "ADDITIONAL PROJECTS", "TECHNICAL SKILLS", "SELF-ASSESSMENT"])
        edu = parsed["sections"][1]["blocks"][0]
        self.assertEqual(edu["heading"], "University of Western Australia")
        self.assertEqual(export_docx._split_degree_line(edu["lines"][0]["text"]),
                         ("Master of IT · AI Track", "2024.02 – 2026.06"))
        proj = parsed["sections"][2]["blocks"][0]
        self.assertEqual([l["kind"] for l in proj["lines"]], ["bullet", "bullet"])


class ExportTests(unittest.TestCase):
    def test_education_keeps_parenthesized_dates_with_each_school(self):
        md = """# Test Applicant
## EDUCATION
### University of Western Australia
Master of IT · AI Track (2024.02 – 2026.06)
Coursework: Machine Learning
### Wuhan Polytechnic University, China
B.Eng. Food Science (2020.09 – 2024.06)
"""
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "resume.docx"
            export_docx.markdown_to_docx(md.splitlines(), output)
            doc = Document(output)
            table = doc.tables[-1]
            self.assertEqual([c.text for c in table.rows[0].cells],
                             ["2024.02 – 2026.06", "University of Western Australia",
                              "Master of IT · AI Track"])
            self.assertEqual(table.rows[1].cells[0].text, "Coursework: Machine Learning")
            self.assertEqual([c.text for c in table.rows[2].cells],
                             ["2020.09 – 2024.06", "Wuhan Polytechnic University, China",
                              "B.Eng. Food Science"])
            self.assertFalse(any("Master of IT" in p.text or "B.Eng." in p.text
                                 for p in doc.paragraphs))
            self.assertEqual([c.width for c in table.columns],
                             [c.width for c in table.rows[0].cells])

    def test_degree_without_dates_stays_in_education_table(self):
        doc = Document()
        export_docx._education(doc, {"blocks": [{"heading": "Test University",
            "lines": [{"kind": "text", "text": "Master of IT (AI Track)"}]}]})
        self.assertEqual([c.text for c in doc.tables[0].rows[0].cells],
                         ["", "Test University", "Master of IT (AI Track)"])

    def test_export_reproduces_original_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "resume_tailored.md").write_text(MD, encoding="utf-8")
            (run / "cv_suggestions.json").write_text(json.dumps({"cover_letter": [
                {"text": "Dear Hiring Manager,\n\nI am writing about the role."}]}), encoding="utf-8")
            (run / "run_meta.json").write_text(json.dumps({"input": {
                "jd_path": "D:\\x\\jobs\\20260912-0930-acme service_desk!.txt"}}), encoding="utf-8")
            outputs = export_docx.export_run(run, with_letter=True)
            # each run's files carry the person, the JD name and the run time
            self.assertRegex(outputs["resume"].name, r"^Alex_Sample_Resume_acme-service-desk_.+\.docx$")
            self.assertRegex(outputs["cover_letter"].name, r"^Alex_Sample_Cover_Letter_acme-service-desk_.+\.docx$")
            self.assertEqual(export_docx.latest_docx(run, "resume"), outputs["resume"])
            doc = Document(outputs["resume"])
            paras = all_paragraphs(doc)
            texts = [p.text for p in paras if p.text.strip()]
            # header band: name, title, contacts
            name = next(p for p in paras if p.text == "ALEX SAMPLE")
            self.assertEqual(name.runs[0].font.size.pt, 22)
            self.assertEqual(str(name.runs[0].font.color.rgb), "FFFFFF")
            self.assertIn("📍 Perth, WA", texts)
            # navy banners with cyan text
            banner = next(p for p in paras if p.text == "▐  EDUCATION")
            self.assertEqual(str(banner.runs[0].font.color.rgb), "00A8CC")
            self.assertTrue(banner.runs[0].bold)
            # education table: dates | school | degree
            self.assertIn("2024.02 – 2026.06", texts)
            degree = next(p for p in paras if p.text == "Master of IT · AI Track")
            self.assertEqual(str(degree.runs[0].font.color.rgb), "1F5FA6")
            self.assertTrue(any(p.text.startswith("Coursework: ") for p in paras))
            # project bullet with bold navy label
            bullet = next(p for p in paras if p.text.startswith("Model: "))
            self.assertEqual(bullet.style.name, "List Bullet")
            self.assertTrue(bullet.runs[0].bold)
            self.assertEqual(str(bullet.runs[0].font.color.rgb), "1A3A5C")
            plain = next(p for p in paras if p.text == "Delivered the forum module.")
            self.assertEqual(plain.style.name, "List Bullet")
            # skills lines get the diamond marker
            self.assertIn("◆ Languages  Python · SQL", texts)
            # two-column additional projects
            self.assertIn("GeoMind | Object Detection", texts)
            self.assertIn("CyberWin | Requirements", texts)
            self.assertFalse(any("来源附录" in t for t in texts))
            # cover letter
            letter_texts = [p.text for p in Document(outputs["cover_letter"]).paragraphs if p.text.strip()]
            self.assertEqual(letter_texts[0], "ALEX SAMPLE")
            self.assertIn("I am writing about the role.", letter_texts)

    def test_file_names_tell_runs_apart(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "20260917-145916-63174a"
            second = Path(tmp) / "20260918-091500-0a1b2c"
            for run, jd in ((first, "20260917-1459-acme-service-desk.txt"), (second, "20260918-0915-关于 这个岗位.txt")):
                run.mkdir()
                (run / "run_meta.json").write_text(json.dumps({"input": {"jd_path": "jobs/" + jd}}), encoding="utf-8")
            self.assertEqual(export_docx.docx_name("resume", "ALEX SAMPLE", first),
                             "Alex_Sample_Resume_acme-service-desk_0917-1459.docx")
            self.assertEqual(export_docx.docx_name("cover_letter", "ALEX SAMPLE", second, 2),
                             "Alex_Sample_Cover_Letter_关于-这个岗位_0918-0915-2.docx")
            bare = Path(tmp) / "20260919-101010-ffffff"  # no metadata: the time alone still separates runs
            bare.mkdir()
            self.assertEqual(export_docx.docx_name("resume", None, bare), "Resume_0919-1010.docx")
            (first / "resume_tailored.docx").write_bytes(b"PK")  # exported before the naming change
            self.assertEqual(export_docx.latest_docx(first, "resume").name, "resume_tailored.docx")
            self.assertIsNone(export_docx.latest_docx(first, "cover_letter"))

    def test_other_projects_flow_in_short_rows(self):
        blocks = [{"heading": f"Project {i}", "lines": [{"kind": "bullet", "text": "Did a thing."}]} for i in range(3)]
        doc = Document()
        table = export_docx._two_columns(doc, blocks, lambda cell, block: export_docx._project_block(
            cell, block, keep_heading=False))
        self.assertEqual(len(table.rows), 2)  # two projects per row, not one tall unbreakable row
        self.assertEqual([c.paragraphs[0].text for c in table.rows[0].cells], ["Project 0", "Project 1"])
        last = table.rows[1]  # the odd block spans both columns
        self.assertEqual(last.cells[0]._tc, last.cells[1]._tc)
        self.assertEqual(last.cells[0].paragraphs[0].text, "Project 2")
        for row in table.rows:
            self.assertIsNotNone(row._tr.trPr.find(qn("w:cantSplit")))
            for cell in row.cells:
                self.assertFalse(any(p.paragraph_format.keep_with_next for p in cell.paragraphs))

    def test_missing_resume_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                export_docx.export_run(tmp)


if __name__ == "__main__":
    unittest.main()
