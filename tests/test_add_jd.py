import datetime
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import add_jd

LONG = "Senior Data Engineer (Perth)\n\nRequired\n" + "Experience with SQL. " * 10


class SlugTests(unittest.TestCase):
    def test_first_line_becomes_slug(self):
        self.assertEqual(add_jd.make_slug(LONG), "senior-data-engineer-perth")

    def test_chinese_kept_and_explicit_name_wins(self):
        self.assertEqual(add_jd.make_slug("数据工程师 (珀斯)\n..."), "数据工程师-珀斯")
        self.assertEqual(add_jd.make_slug(LONG, "Acme ML"), "acme-ml")

    def test_slug_truncated_and_never_empty(self):
        self.assertLessEqual(len(add_jd.make_slug("x" * 200)), add_jd.SLUG_MAX)
        self.assertEqual(add_jd.make_slug("!!! ???"), "jd")


class SaveTests(unittest.TestCase):
    def test_saves_utf8_with_timestamp_and_no_overwrite(self):
        now = datetime.datetime(2026, 9, 11, 14, 5)
        with tempfile.TemporaryDirectory() as tmp:
            first = add_jd.save_jd(LONG, Path(tmp), now=now)
            second = add_jd.save_jd(LONG, Path(tmp), now=now)
            self.assertEqual(first.name, "20260911-1405-senior-data-engineer-perth.txt")
            self.assertEqual(second.name, "20260911-1405-senior-data-engineer-perth-2.txt")
            self.assertEqual(first.read_text(encoding="utf-8"), add_jd.normalise(LONG))

    def test_crlf_and_bom_normalised(self):
        text = "﻿Title\r\nline two  \r\n" + "x" * 100
        self.assertEqual(add_jd.normalise(text), "Title\nline two\n" + "x" * 100 + "\n")

    def test_short_text_rejected_unless_forced(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                add_jd.save_jd("too short", Path(tmp))
            path = add_jd.save_jd("too short", Path(tmp), min_chars=0)
            self.assertTrue(path.exists())

    def test_cli_stdin_prints_next_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_stdin = sys.stdin
            sys.stdin = io.StringIO(LONG)
            out = io.StringIO()
            try:
                with redirect_stdout(out):
                    code = add_jd.main(["--stdin", "--jobs", tmp])
            finally:
                sys.stdin = old_stdin
            self.assertEqual(code, 0)
            self.assertIn("已保存", out.getvalue())
            self.assertIn("match_job.py --jd", out.getvalue())
            self.assertEqual(len(list(Path(tmp).glob("*.txt"))), 1)


if __name__ == "__main__":
    unittest.main()
