"""Search local project Markdown/TXT files with multilingual E5 embeddings."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

MODEL = "intfloat/multilingual-e5-small"


def read_sections(folder):
    """Keep source text, headings and one-based line ranges for citations."""
    records = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
            continue
        if any(part.startswith(("_", ".")) for part in path.relative_to(folder).parts):
            continue  # Keep templates such as _template.md out of the index.
        lines = path.read_text(encoding="utf-8-sig").splitlines(keepends=True)
        project, section, start = path.stem, "项目概况", 0

        def append(end):
            text = "".join(lines[start:end]).strip()
            if text:
                records.append({
                    "project": project,
                    "section": section,
                    "source": path.relative_to(folder).as_posix(),
                    "line_start": start + 1,
                    "line_end": end,
                    "text": text,
                })

        for index, line in enumerate(lines):
            match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
            if match:
                append(index)
                if len(match[1]) == 1:
                    project = match[2]
                section, start = match[2], index + 1
        append(len(lines))
    return records


def make_chunks(records, tokenizer, size=320, overlap=40):
    """Token-aware windows; slice ORIGINAL text so citations stay faithful."""
    chunks = []
    for record in records:
        text = record["text"]
        offsets = tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True
        )["offset_mapping"]
        for start in range(0, len(offsets), size - overlap):
            end = min(start + size, len(offsets))
            char_start, char_end = offsets[start][0], offsets[end - 1][1]
            chunk = {
                **record,
                "text": text[char_start:char_end],
                "char_start": char_start,
                "char_end": char_end,
            }
            identity = json.dumps(chunk, ensure_ascii=False, sort_keys=True)
            chunk["chunk_id"] = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
            chunks.append(chunk)
            if end == len(offsets):
                break
    return chunks


def select_sections(records, all_sections=False):
    """Prefer personal evidence over derived CV/STAR text in structured files."""
    if all_sections:
        return records
    personal = re.compile(r"^(我的贡献|我的工作|个人贡献|个人工作|My Contributions?\b|My Work\b)", re.I)
    structured_sources = {r["source"] for r in records if personal.match(r["section"])}
    return [r for r in records
            if r["source"] not in structured_sources or personal.match(r["section"])]


def prepare_search_chunks(records, tokenizer):
    """Embed narrative only; retain the full original section for display."""
    prepared = []
    evidence_line = re.compile(r"^(?:[-*]\s*)?(证据|来源|相关能力|相关技能|Evidence|Sources?|Skills)\s*[:：]", re.I)
    for record in records:
        narrative, evidence, keywords = [], [], []
        for line in record["text"].splitlines():
            match = evidence_line.match(line.strip())
            if match:
                if match[1].lower() not in {"相关能力", "相关技能", "skills"}:
                    evidence.append(line)
                else:
                    keywords.append(line.strip())
                continue
            # Keep link labels in prose but never embed URLs or file paths.
            line = re.sub(r"\[([^\]]+)\]\((?:<[^>]*>|[^)]+)\)", r"\1", line)
            line = re.sub(r"https?://\S+", "", line)
            if re.search(r"[\w\u4e00-\u9fff]", line):
                narrative.append(line)
        search_text = "\n".join(narrative).strip()
        if not search_text:
            continue
        parent_id = hashlib.sha256(json.dumps(
            record, ensure_ascii=False, sort_keys=True
        ).encode("utf-8")).hexdigest()[:16]
        prepared.append({**record, "text": search_text,
                         "parent_text": record["text"],
                         "search_keywords": " ".join(keywords),
                         "section_id": parent_id, "evidence": evidence})
    return make_chunks(prepared, tokenizer)


def rank_sections(chunks, scores, top_k):
    """Rank by best matching child, returning each full source section once."""
    ranked = sorted(range(len(chunks)), key=lambda i: float(scores[i]), reverse=True)
    results, seen = [], set()
    for i in ranked:
        chunk = chunks[i]
        if chunk["section_id"] in seen:
            continue
        seen.add(chunk["section_id"])
        result = {k: v for k, v in chunk.items()
                  if k not in {"parent_text", "char_start", "char_end"}}
        result.update(text=chunk["parent_text"], matched_text=chunk["text"],
                      match_char_start=chunk["char_start"],
                      match_char_end=chunk["char_end"], score=round(float(scores[i]), 4))
        results.append(result)
        if len(results) == top_k:
            break
    return results


class ExperienceRetriever:
    """Reusable E5 retriever: one model load and one corpus encode per task."""

    def __init__(self, data_dir, offline=False, all_sections=False):
        self.data_dir = Path(data_dir)
        self.offline = offline
        self.all_sections = all_sections
        self.model = None
        self.chunks = None
        self.passages = None
        self.vectors = None

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "缺少依赖。请运行：python -m pip install -r requirements.txt"
            ) from exc
        return SentenceTransformer(MODEL, device="cpu", local_files_only=self.offline)

    def build(self):
        if self.model is not None:
            return self
        records = select_sections(read_sections(self.data_dir), self.all_sections)
        if not records:
            raise ValueError("没有可检索的内容；请添加 UTF-8 的 .md 或 .txt 项目文件。")
        print("正在加载本地模型并检索……", file=sys.stderr)
        self.model = self._load_model()
        self.chunks = prepare_search_chunks(records, self.model.tokenizer)
        if not self.chunks:
            raise ValueError("资料没有可编码的文本。")
        self.passages = [
            f"passage: {c['project']} | {c['section']}\n{c['search_keywords']}\n{c['text']}"
            for c in self.chunks
        ]
        # Reject oversized inputs instead of silently losing text to truncation.
        for text in self.passages:
            if len(self.model.tokenizer.encode(text)) > self.model.max_seq_length:
                raise ValueError("查询或项目标题过长，请缩短查询/标题后重试。")
        self.vectors = self.model.encode(
            self.passages, normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False, batch_size=16
        )
        return self

    def search(self, query, top_k=5):
        if self.model is None:
            self.build()
        query_text = "query: " + query.strip()
        if len(self.model.tokenizer.encode(query_text)) > self.model.max_seq_length:
            raise ValueError("查询或项目标题过长，请缩短查询/标题后重试。")
        query_vector = self.model.encode(
            [query_text], normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False
        )[0]
        scores = self.vectors @ query_vector
        return rank_sections(self.chunks, scores, top_k)


def search_experience(query, folder, top_k=5, offline=False, all_sections=False):
    """Return ranked CANDIDATES, not verified skill claims or match probabilities."""
    return ExperienceRetriever(folder, offline=offline, all_sections=all_sections)\
        .build().search(query, top_k)


def main():
    # Windows pipes may default to a code page that cannot encode Chinese.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", help="例如 Python backend development")
    parser.add_argument(
        "--data", type=Path,
        default=Path(__file__).resolve().parent / "experiences"
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json", action="store_true", help="输出 JSON，便于后续接入 Flask")
    parser.add_argument("--offline", action="store_true", help="仅使用已缓存模型")
    parser.add_argument("--all-sections", action="store_true", help="同时搜索基本信息、CV、STAR 等章节")
    parser.add_argument("--check", action="store_true", help="只检查资料读取，不加载模型")
    args = parser.parse_args()
    folder = args.data.resolve()
    if not folder.is_dir():
        parser.error(f"资料文件夹不存在：{folder}")
    if args.top_k < 1:
        parser.error("--top-k 必须大于 0")
    if not args.check and not (args.query or "").strip():
        parser.error("请输入查询，例如：python search_experience.py \"SQL\"")
    try:
        if args.check:
            records = read_sections(folder)
            if not records:
                raise ValueError("没有可读取的项目内容。")
            print(json.dumps(records, ensure_ascii=False, indent=2))
            return
        results = search_experience(args.query, folder, args.top_k, args.offline, args.all_sections)
        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
            return
        print("\n以下为相关候选；分数不是能力匹配概率，仍需检查原文。")
        for rank, result in enumerate(results, 1):
            print(f"\n{rank}. {result['project']} / {result['section']}")
            print(f"相似度：{result['score']} | ID：{result['chunk_id']}")
            print(f"来源：{result['source']}，所在章节行号 "
                  f"{result['line_start']}-{result['line_end']}")
            print(result["text"])
    except (OSError, UnicodeError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"错误：{exc}\n")


if __name__ == "__main__":
    main()
