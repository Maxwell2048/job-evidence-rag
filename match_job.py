"""Match a saved job description (JD) against local project experiences.

Exit codes: 0 = requested stage fully processed, 1 = failed or partial,
2 = CLI argument error. The boundary is whether a run started: argparse
rejects a malformed command with 2 and nothing is written; once a run
directory and run_meta.json exist the run happened, so every later failure
(unreadable JD, invalid config content, empty corpus, model unavailable) is 1
and the specific error_type carries the diagnosis.
"""
import argparse
import datetime
import hashlib
import json
import sys
import time
import uuid
from pathlib import Path

import jd_parser
from local_llm import LLMError, build_llm, load_config
from search_experience import MODEL as EMBEDDING_MODEL, ExperienceRetriever
from evidence_matcher import (JUDGMENT_PROMPT_VERSION, EvidenceMatcher)
from report_writer import (overall_status, write_matches, write_report)
from job_summary import generate_summary, save_summary, PROMPT_VERSION as SUMMARY_PROMPT_VERSION

SCHEMA_VERSION = "1.0"
PROJECT_ROOT = Path(__file__).resolve().parent


def build_parser():
    parser = argparse.ArgumentParser(
        description="Extract JD requirements and match them against local experiences.")
    parser.add_argument("--jd", required=True,
                        help="保存的 JD 原文（UTF-8 或 UTF-8 BOM 的 .txt/.md）")
    parser.add_argument("--config", required=True,
                        help="本地生成模型配置文件路径")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "experiences",
                        help="经历资料目录（默认项目内 experiences）")
    parser.add_argument("--top-k", type=int, default=5,
                        help="每项要求检索的不同章节数（默认 5）")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs",
                        help="输出根目录（每次运行创建独立子目录）")
    parser.add_argument("--extract-only", action="store_true",
                        help="仅提取并验证 JD 要求，不启动向量检索")
    parser.add_argument("--offline", action="store_true",
                        help="向量模型仅使用本地缓存；生成模型仍可通过本机接口调用")
    parser.add_argument("--json", action="store_true",
                        help="stdout 仅输出最终 JSON；进度与错误写 stderr")
    return parser


def _embedding_version():
    try:
        import sentence_transformers
        return getattr(sentence_transformers, "__version__", None)
    except ImportError:
        return None


def _experience_hashes(data_dir):
    files = []
    for path in sorted(data_dir.rglob("*")):
        if path.is_file():
            files.append({
                "path": path.relative_to(data_dir).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
    return files


def _judgment_errors(matches):
    """Per-requirement failures, summarised into run metadata.

    A requirement whose judgment errored is a processing problem, so it is
    recorded as an error with its ID; matches.json keeps the full text.
    """
    summary = []
    for match in matches:
        if match["processing_status"] == "ok" or not match.get("errors"):
            continue
        summary.append({
            "error_type": ("judgment_incomplete"
                           if match["processing_status"] == "incomplete"
                           else "judgment_failed"),
            "message": (f"{match['requirement_id']}"
                        f"（{match['processing_status']}）："
                        + "；".join(match["errors"])),
        })
    return summary


def create_run_dir(out_root):
    """outputs/<timestamp>-<random suffix> that never overwrites an old run."""
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    for _ in range(8):
        candidate = Path(out_root) / (stamp + "-" + uuid.uuid4().hex[:6])
        try:
            candidate.mkdir(parents=True)
            return candidate
        except FileExistsError:
            continue
    raise OSError(f"无法在 {out_root} 创建唯一运行目录")


def execute(args, jd_path, config_path, run_dir, log):
    """Run the requested stage. Returns (exit_code, stdout document)."""
    started = time.monotonic()
    meta = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_dir.name,
        "run_status": "failed",
        "scope": "extraction" if args.extract_only else "full",
        "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "input": {"jd_path": str(jd_path), "jd_sha256": None, "jd_chars": None},
        "experience_files": None,
        "models": {"generator": None, "embedding": None},
        "prompts": {
            "extraction": jd_parser.EXTRACTION_PROMPT_VERSION,
            "judgment": JUDGMENT_PROMPT_VERSION if not args.extract_only else None,
        },
        "parameters": {
            "data_dir": str(Path(args.data)),
            "top_k": args.top_k,
            "offline": args.offline,
            "extract_only": bool(args.extract_only),
        },
        "errors": [],
        "warnings": [],
    }
    exit_code = 1
    result_doc = None
    try:
        jd = jd_parser.read_jd(jd_path)
        meta["input"]["jd_sha256"] = jd["raw_sha256"]
        meta["input"]["jd_chars"] = len(jd["text"])
        (run_dir / "jd.txt").write_text(jd["text"], encoding="utf-8")
        config = load_config(config_path)
        llm = build_llm(config)
        meta["models"]["generator"] = {**llm.describe_model(),
                                       "parameters": llm.describe_parameters()}
        log("[match-job] 正在提取 JD 要求……")
        doc = jd_parser.extract_requirements(jd["text"], llm)
        meta["warnings"].extend(doc["warnings"])
        failed_batches = doc["failed_batches"]
        for failed in failed_batches:
            meta["errors"].append({
                "error_type": failed["error_type"],
                "message": (f"第 {failed['index']}/{failed['count']} 批提取失败"
                            f"（JD 第 {failed['line_start']}-{failed['line_end']} 行"
                            f"未处理）：{failed['message']}"),
            })
        requirements_doc = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_dir.name,
            "jd_sha256": jd["raw_sha256"],
            "requirements": doc["requirements"],
            "requirement_groups": doc["requirement_groups"],
            "batches": doc["batches"],
            "failed_batches": failed_batches,
            "review_count": sum(1 for r in doc["requirements"]
                                if r["needs_review"]),
            "warnings": doc["warnings"],
        }
        (run_dir / "requirements.json").write_text(
            json.dumps(requirements_doc, ensure_ascii=False, indent=2),
            encoding="utf-8")
        result_doc = dict(requirements_doc)
        if not doc["requirements"]:
            meta["run_status"] = "failed"
            if not meta["errors"]:
                meta["errors"].append({"error_type": "extraction",
                                       "message": "未提取到任何要求"})
        elif args.extract_only:
            meta["run_status"] = "partial" if failed_batches else "complete"
        else:
            log("[match-job] 正在加载向量模型并检索证据……")
            retriever = ExperienceRetriever(Path(args.data),
                                            offline=args.offline)
            retriever.build()
            meta["models"]["embedding"] = {
                "name": EMBEDDING_MODEL,
                "version": _embedding_version(),
            }
            meta["experience_files"] = _experience_hashes(Path(args.data))
            matcher = EvidenceMatcher(llm, retriever, top_k=args.top_k)
            matches = matcher.match_all(doc["requirements"])
            meta["errors"].extend(_judgment_errors(matches))
            meta["run_status"] = overall_status(matches)
            if failed_batches and meta["run_status"] == "complete":
                meta["run_status"] = "partial"
            write_matches(run_dir, SCHEMA_VERSION, matches)
            log("[match-job] 正在生成优势、缺口与优先补学总结……")
            summary = generate_summary(llm, matches, doc["requirement_groups"], meta["run_status"])
            save_summary(run_dir, summary)
            meta["prompts"]["summary"] = SUMMARY_PROMPT_VERSION
            meta["summary_status"] = summary["status"]
            if summary["status"] != "complete":
                meta["warnings"].append("求职总结未完成，逐项匹配结果保留；可用 job_summary.py 单独重试。")
            write_report(run_dir, meta, matches, doc["requirement_groups"], summary)
            result_doc["job_summary"] = summary
            result_doc["matches"] = matches
        exit_code = 0 if meta["run_status"] == "complete" else 1
    except LLMError as exc:
        # ConfigError, ContextBudgetError, timeouts: each carries its error_type.
        meta["errors"].append({"error_type": exc.error_type, "message": str(exc)})
        exit_code = 1
    except RuntimeError as exc:
        meta["errors"].append({"error_type": "runtime", "message": str(exc)})
        exit_code = 1
    except ValueError as exc:
        meta["errors"].append({"error_type": "input", "message": str(exc)})
        exit_code = 1
    except OSError as exc:
        meta["errors"].append({"error_type": "io", "message": str(exc)})
        exit_code = 1
    finally:
        meta["elapsed_seconds"] = round(time.monotonic() - started, 1)
        (run_dir / "run_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if result_doc is None:
        result_doc = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_dir.name,
            "run_status": meta["run_status"],
            "scope": meta["scope"],
            "errors": list(meta["errors"]),
        }
    else:
        result_doc["run_status"] = meta["run_status"]
        result_doc["scope"] = meta["scope"]
        result_doc["errors"] = list(meta["errors"])
    return exit_code, result_doc


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args(argv)
    jd_path = Path(args.jd)
    if not jd_path.is_file():
        parser.error(f"JD 文件不存在：{jd_path}")
    if jd_path.suffix.lower() not in {".txt", ".md", ".markdown"}:
        parser.error(f"JD 文件必须是 .txt 或 .md：{jd_path}")
    config_path = Path(args.config)
    if not config_path.is_file():
        parser.error(f"配置文件不存在：{config_path}")
    if args.top_k < 1:
        parser.error("--top-k 必须大于 0")
    if not args.extract_only and not Path(args.data).is_dir():
        parser.error(f"经历资料目录不存在：{args.data}")
    try:
        run_dir = create_run_dir(Path(args.out))
    except OSError as exc:
        parser.error(str(exc))
    log = (lambda message: None) if args.json else \
          (lambda message: print(message, file=sys.stderr))
    exit_code, result_doc = execute(args, jd_path, config_path, run_dir, log)
    if args.json:
        print(json.dumps(result_doc, ensure_ascii=False))
    else:
        meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
        print(f"运行目录：{run_dir}")
        print(f"状态：{meta['run_status']}（范围：{meta['scope']}）")
        if "requirements" in result_doc:
            print(f"要求：{len(result_doc['requirements'])} 条"
                  f"（待复核 {result_doc['review_count']} 条）")
            if result_doc["requirement_groups"]:
                print(f"AND/OR 组：{len(result_doc['requirement_groups'])} 个")
        for warning in meta["warnings"][:5]:
            print(f"警告：{warning}")
        for error in meta["errors"]:
            print(f"错误：[{error['error_type']}] {error['message']}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
