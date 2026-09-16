import os
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from search_experience import (read_sections, make_chunks, select_sections,
                               prepare_search_chunks, rank_sections)


class CharacterTokenizer:
    """Deterministic tokenizer for testing window/source logic only."""
    def __call__(self, text, **kwargs):
        return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}


class SourceTests(unittest.TestCase):
    def test_personal_sections_default_and_full_search_opt_in(self):
        records = [{"source": "a.md", "section": name} for name in
                   ["基本信息", "我的贡献 图片标注", "英文简历表述", "STAR 案例"]]
        records.append({"source": "plain.txt", "section": "项目概况"})
        self.assertEqual([r["section"] for r in select_sections(records)],
                         ["我的贡献 图片标注", "项目概况"])
        self.assertEqual(select_sections(records, all_sections=True), records)

    def test_long_citation_is_retained_but_not_embedded(self):
        url = "https://github.com/example/project/blob/" + "a" * 1000
        body = "我完成了合成样例地图的图片标注，并参与目标检测。"
        source_text = body + "\n相关能力：Image Annotation\n证据：[提交](" + url + ")。"
        record = {"source": "a.md", "section": "我的贡献 图片标注",
                  "project": "Synthetic Project", "text": source_text}
        chunks = prepare_search_chunks([record], CharacterTokenizer())
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["text"], body)
        result = rank_sections(chunks, [0.8], 3)[0]
        self.assertEqual(result["text"], source_text)
        self.assertEqual(result["matched_text"], body)
        self.assertIn(url, result["text"])
        self.assertEqual(result["evidence"], ["证据：[提交](" + url + ")。"])

    def test_overlapping_matches_return_one_complete_section(self):
        original = "模型训练与数据标注。" * 100
        records = [{"source": "a.md", "project": "Synthetic Project", "section": "我的贡献",
                    "text": original},
                   {"source": "b.md", "project": "Other", "section": "我的工作",
                    "text": "其他完整经历。"}]
        chunks = prepare_search_chunks(records, CharacterTokenizer())
        results = rank_sections(chunks, [0.9] * (len(chunks) - 1) + [0.7], 3)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["text"], original)
        self.assertEqual(len({r["section_id"] for r in results}), 2)

    def test_evidence_only_sections_do_not_create_search_chunks(self):
        records = [{"source": "a.md", "section": "证据", "text":
                    "证据：https://github.com/example/project\nhttps://example.com"}]
        self.assertEqual(prepare_search_chunks(records, CharacterTokenizer()), [])

    def test_utf8_bom_headings_sources_and_drafts(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "real.md").write_text(
                "# 校园项目\n\n## 我的贡献\n使用 SQL 查询。\n", encoding="utf-8-sig"
            )
            (folder / "_draft.md").write_text("草稿不得进入索引", encoding="utf-8")
            (folder / "empty.txt").write_text("", encoding="utf-8")
            records = read_sections(folder)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["project"], "校园项目")
            self.assertEqual(records[0]["section"], "我的贡献")
            self.assertEqual(records[0]["text"], "使用 SQL 查询。")
            self.assertEqual(records[0]["source"], "real.md")
            self.assertEqual(records[0]["line_start"], 4)

    def test_long_section_preserves_all_characters_and_stable_ids(self):
        original = "真实经历ABC。 " * 100
        records = [{"text": original, "source": "a.md", "section": "贡献"}]
        chunks = make_chunks(records, CharacterTokenizer())
        self.assertGreater(len(chunks), 1)
        reconstructed = chunks[0]["text"]
        for previous, current in zip(chunks, chunks[1:]):
            overlap = previous["char_end"] - current["char_start"]
            reconstructed += current["text"][overlap:]
        self.assertEqual(reconstructed, original)
        self.assertEqual(chunks, make_chunks(records, CharacterTokenizer()))
        self.assertEqual(len({c["chunk_id"] for c in chunks}), len(chunks))
        for chunk in chunks:
            self.assertEqual(chunk["text"], original[chunk["char_start"]:chunk["char_end"]])

    def test_example_check_without_model_dependencies(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-X", "utf8", str(root / "search_experience.py"),
             "--data", str(root / "examples"), "--check"],
            capture_output=True, encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "ascii"}
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("虚构", result.stdout)

    def test_invalid_query_and_top_k(self):
        root = Path(__file__).resolve().parents[1]
        for options in [["   "], ["SQL", "--top-k", "0"]]:
            result = subprocess.run(
                [sys.executable, "-X", "utf8", str(root / "search_experience.py"),
                 "--data", str(root / "examples"), *options],
                capture_output=True, encoding="utf-8"
            )
            self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
