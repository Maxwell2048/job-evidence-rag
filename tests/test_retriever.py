import contextlib
import io
import tempfile
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from search_experience import ExperienceRetriever, search_experience


class CharacterTokenizer:
    """Deterministic tokenizer for testing load/encode counting only."""
    def __call__(self, text, **kwargs):
        return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}

    def encode(self, text):
        return list(text)


class FakeModel:
    """Stands in for SentenceTransformer; records load/encode call sizes."""
    def __init__(self, max_seq_length=100000):
        self.tokenizer = CharacterTokenizer()
        self.max_seq_length = max_seq_length
        self.load_count = 0
        self.encode_lengths = []

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        self.encode_lengths.append(len(texts))
        return np.array([[float(len(t) % 7 + 1)] for t in texts])


def make_corpus(folder):
    (folder / "proj_a.md").write_text(
        "# 项目 A\n\n## 我的贡献 图片标注\n"
        "我完成了 12 幅示例地图的图片标注，并参与目标检测训练。\n"
        "相关能力：Image Annotation\n",
        encoding="utf-8",
    )
    (folder / "proj_b.md").write_text(
        "# 项目 B\n\n## 我的工作 参数搜索\n"
        "我使用 GridSearchCV 对 SVM 的 C 与 gamma 执行参数搜索。\n"
        "相关能力：Hyperparameter Search\n",
        encoding="utf-8",
    )


class RetrieverTests(unittest.TestCase):
    def build_retriever(self, folder, fake):
        def load_model(self=None):
            fake.load_count += 1
            return fake
        retriever = ExperienceRetriever(folder)
        retriever._load_model = load_model
        return retriever

    def quiet(self, fn, *args, **kwargs):
        with contextlib.redirect_stderr(io.StringIO()):
            return fn(*args, **kwargs)

    def test_one_model_load_and_one_corpus_encode_per_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            make_corpus(folder)
            fake = FakeModel()
            retriever = self.build_retriever(folder, fake)
            self.quiet(retriever.build)
            results = [self.quiet(retriever.search, q, top_k=3)
                       for q in ("标注", "参数搜索", "yolo")]
            self.assertEqual(fake.load_count, 1)
            self.assertEqual([n for n in fake.encode_lengths if n > 1], [2])
            self.assertEqual([len(r) for r in results], [2, 2, 2])

    def test_search_keeps_convention_fields_and_full_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            make_corpus(folder)
            fake = FakeModel()
            retriever = self.build_retriever(folder, fake)
            results = self.quiet(retriever.search, "标注", top_k=3)
            self.assertEqual(len(results), 2)
            for result in results:
                for key in ("project", "section", "source", "line_start", "line_end",
                            "text", "matched_text", "match_char_start",
                            "match_char_end", "score", "section_id", "chunk_id",
                            "evidence", "search_keywords"):
                    self.assertIn(key, result)
            self.assertEqual(len({r["section_id"] for r in results}), 2)
            project_a = next(r for r in results if r["project"] == "项目 A")
            self.assertIn("相关能力：Image Annotation", project_a["text"])
            self.assertNotIn("相关能力：Image Annotation", project_a["matched_text"])

    def test_search_before_build_auto_builds(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            make_corpus(folder)
            fake = FakeModel()
            retriever = self.build_retriever(folder, fake)
            results = self.quiet(retriever.search, "标注", top_k=1)
            self.assertEqual(fake.load_count, 1)
            self.assertLessEqual(len(results), 1)

    def test_compatibility_wrapper_builds_fresh_retriever_per_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            make_corpus(folder)
            fake = FakeModel()
            def load_model(self=None):
                fake.load_count += 1
                return fake
            with mock.patch.object(ExperienceRetriever, "_load_model", load_model):
                first = self.quiet(search_experience, "标注", folder, top_k=2)
                second = self.quiet(search_experience, "标注", folder, top_k=2)
            self.assertEqual(fake.load_count, 2)
            self.assertEqual(first, second)

    def test_empty_corpus_error_raised_before_model_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "_draft.md").write_text("草稿", encoding="utf-8")
            fake = FakeModel()
            retriever = self.build_retriever(folder, fake)
            with self.assertRaisesRegex(ValueError, "没有可检索的内容"):
                retriever.build()
            self.assertEqual(fake.load_count, 0)

    def test_no_encodable_text_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "proj.md").write_text(
                "# 项目\n\n## 我的贡献\n证据：https://example.com/x\n", encoding="utf-8"
            )
            fake = FakeModel()
            retriever = self.build_retriever(folder, fake)
            with self.assertRaisesRegex(ValueError, "资料没有可编码的文本"):
                self.quiet(retriever.build)

    def test_oversized_query_rejected_without_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            make_corpus(folder)
            fake = FakeModel()
            retriever = self.build_retriever(folder, fake)
            self.quiet(retriever.build)
            fake.max_seq_length = 5
            with self.assertRaisesRegex(ValueError, "查询或项目标题过长"):
                self.quiet(retriever.search, "一个明显更长的查询字符串", top_k=1)


if __name__ == "__main__":
    unittest.main()
