import json
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_site


class MarkdownTests(unittest.TestCase):
    def test_headings_lists_tables_and_inline(self):
        md = ("# Title\n\nPara with **bold**, `code` and [link](x.html).\n"
              "second line\n\n- a\n  - nested\n- b\n\n1. one\n2. two\n\n"
              "| ID | 要求 |\n| --- | --- |\n| R001 | a \\| b |\n\n---\n")
        out = build_site.md_to_html(md)
        self.assertIn("<h1>Title</h1>", out)
        self.assertIn("<strong>bold</strong>", out)
        self.assertIn("<code>code</code>", out)
        self.assertIn('<a href="x.html">link</a>', out)
        self.assertIn("Para with", out)
        self.assertIn("<br>second line", out)
        self.assertIn("<ul>\n<li>a</li>\n<ul>\n<li>nested</li>\n</ul>\n<li>b</li>\n</ul>", out)
        self.assertIn("<ol>\n<li>one</li>\n<li>two</li>\n</ol>", out)
        self.assertIn("<th>ID</th><th>要求</th>", out)
        self.assertIn("<td>R001</td><td>a | b</td>", out)
        self.assertIn("<hr>", out)

    def test_html_is_escaped(self):
        self.assertIn("&lt;script&gt;", build_site.md_to_html("<script>x</script>"))


class SiteTests(unittest.TestCase):
    def test_index_and_pages(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs = Path(tmp) / "outputs"
            run = outputs / "20260911-000000-abc"
            run.mkdir(parents=True)
            (run / "run_meta.json").write_text(json.dumps({
                "run_id": run.name, "run_status": "complete", "scope": "full",
                "started_at": "2026-09-11T00:00:00", "input": {"jd_path": "jobs/x.txt"}}),
                encoding="utf-8")
            (run / "matches.json").write_text(json.dumps({"matches": [
                {"processing_status": "ok", "verdict": "direct"},
                {"processing_status": "ok", "verdict": "insufficient"},
                {"processing_status": "error", "verdict": "insufficient"}]}), encoding="utf-8")
            (run / "report.md").write_text("# 报告\n\n- ok\n", encoding="utf-8")
            (run / "cv_suggestions.md").write_text("# old\n", encoding="utf-8")
            (run / "cv_suggestions-2.md").write_text("# new\n", encoding="utf-8")
            (outputs / "junk").mkdir()
            site = Path(tmp) / "site"
            count = build_site.build(outputs, site)
            self.assertEqual(count, 1)
            index = (site / "index.html").read_text(encoding="utf-8")
            self.assertIn("x.txt", index)
            self.assertIn("直接 1", index)
            self.assertIn("错误 1", index)
            self.assertIn(f'{run.name}/cv_suggestions-2.html', index)
            self.assertNotIn(f'"{run.name}/cv_suggestions.html"', index)
            page = (site / run.name / "report.html").read_text(encoding="utf-8")
            self.assertIn("<h1>报告</h1>", page)
            self.assertIn("(旧)", page)
            self.assertIn('href="../index.html"', page)


if __name__ == "__main__":
    unittest.main()
