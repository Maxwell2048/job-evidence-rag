"""Eval runner for the fictional public evaluation set.

Phases:
  retrieval  real E5 model, section-level Recall@K (always runs)
      python tests/run_eval.py [--data examples/experiences] [--top-k 5] [--offline]
  extraction + judgment  real generator model (optional, needs a local
  OpenAI-compatible config):
      python tests/run_eval.py --config config.local.json

Exit code: 0 when every executed check passes, 1 otherwise.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from search_experience import ExperienceRetriever


def _line_for(jd_text, line):
    for raw in jd_text.splitlines():
        if line and line in raw:
            return raw
    return ""


def run_retrieval(evaluation, retriever, top_k):
    rows = []
    for case in evaluation["cases"]:
        hits = retriever.search(case["search_query"], top_k=top_k)
        recall = case.get("recall") or []
        recall_sections = case.get("recall_sections") or []
        hit = False
        for h in hits:
            if Path(h["source"]).stem not in recall:
                continue
            if not recall_sections or any(
                    s in h["section"] for s in recall_sections):
                hit = True
                break
        rows.append({
            "id": case["id"],
            "text": case["text"],
            "category": case["category"],
            "recall_expected": bool(recall),
            "recall_hit": hit,
            "top": [f"{h['source']} :: {h['section']} ({h['score']})"
                    for h in hits],
        })
    return rows


def run_extraction(evaluation, llm, set_dir=None):
    import jd_parser
    jd_path = (set_dir or Path(__file__).parent) / evaluation["extraction_jd"]
    jd_text = jd_path.read_text(encoding="utf-8")
    doc = jd_parser.extract_requirements(jd_text, llm)
    rows = []
    for check in evaluation.get("extraction_checks", []):
        line = _line_for(jd_text, check["line"])
        requirements = [r for r in doc["requirements"]
                        if r.get("source_quote") and r["source_quote"] in line]
        ok = bool(requirements)
        detail = []
        if not requirements:
            detail.append("该行未提取到要求")
        if check.get("expect_importance"):
            actual = sorted({r["importance"] for r in requirements})
            ok = ok and check["expect_importance"] in actual
            detail.append("实际优先级=" + ("/".join(actual) if actual else "—"))
        if check.get("expect_group"):
            found = any(g["operator"] == check["expect_group"]
                        and g["source_quote"] in line
                        for g in doc["requirement_groups"])
            ok = ok and found
            detail.append(f"期望组={check['expect_group']}")
        if check.get("expect_category"):
            actual = sorted({r["category"] for r in requirements})
            ok = ok and check["expect_category"] in actual
            detail.append("实际类别=" + ("/".join(actual) if actual else "—"))
        if check.get("expect_qualifier_type"):
            found = any(q["type"] == check["expect_qualifier_type"]
                        for r in requirements
                        for q in r.get("qualifiers", []))
            ok = ok and found
            detail.append(f"期望限定={check['expect_qualifier_type']}")
        rows.append({"line": check["line"], "ok": ok,
                     "detail": "；".join(detail),
                     "note": check.get("note", "")})
    return rows, doc, jd_text


def run_judgment(evaluation, llm, retriever, top_k, extraction_doc, jd_text):
    from evidence_matcher import EvidenceMatcher
    matcher = EvidenceMatcher(llm, retriever, top_k=top_k)
    rows = []
    for case in evaluation["cases"]:
        line = _line_for(jd_text, case.get("jd_line", ""))
        requirement = next(
            (r for r in extraction_doc.get("requirements", [])
             if r.get("source_quote") and r["source_quote"] in line), None)
        synthetic = requirement is None
        if synthetic:
            requirement = {
                "requirement_id": f"EVAL-{case['id']}",
                "text": case["text"],
                "category": "technical_skill",
                "importance": case.get("expected_importance", "unspecified"),
                "source_quote": case["text"],
                "source_span": None,
                "importance_source_quote": None,
                "importance_source_span": None,
                "qualifiers": [],
                "search_queries": [case["search_query"]],
                "needs_review": False,
                "review_notes": ["提取未返回该行要求，用合成要求判断"],
            }
        match = matcher.match_requirement(requirement)
        expected = case.get("expected_verdict")
        acceptable = case.get("acceptable_verdicts")
        if expected is not None:
            ok = match["processing_status"] == "ok" \
                 and match["verdict"] == expected
        elif acceptable:
            ok = match["processing_status"] == "ok" \
                 and match["verdict"] in acceptable
        else:
            ok = match["processing_status"] == "ok"
        ok = ok and not synthetic
        target = expected or ("/".join(acceptable) if acceptable
                              else "无标签（仅要求处理成功）")
        rows.append({"id": case["id"], "ok": ok, "verdict": match["verdict"],
                     "status": match["processing_status"], "target": target,
                     "requirement_id": requirement["requirement_id"],
                     "synthetic": synthetic})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the small human eval set "
                    "(retrieval recall, optional extraction + judgment)")
    parser.add_argument("--set", type=Path,
                        default=Path(__file__).with_name("public_eval_set.json"))
    parser.add_argument("--data", type=Path, default=ROOT / "examples" / "experiences")
    parser.add_argument("--offline", action="store_true", help="Use cached embeddings only")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--config", type=Path, default=None,
                        help="OpenAI 兼容本地模型配置；提供后追加提取与判断阶段")
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    try:
        evaluation = json.loads(args.set.read_text(encoding="utf-8"))
        if not isinstance(evaluation.get("cases"), list) or not evaluation["cases"]:
            raise ValueError("Evaluation set must contain nonempty cases")
        if args.config and not (args.set.parent / evaluation["extraction_jd"]).is_file():
            raise ValueError("Evaluation JD is missing beside the set file")
    except (OSError, ValueError, KeyError, AttributeError) as exc:
        parser.error(f"Cannot load evaluation set: {exc}")
    top_k = args.top_k
    retriever = ExperienceRetriever(args.data, offline=args.offline).build()
    failures = 0
    rows = run_retrieval(evaluation, retriever, top_k)
    checked = [r for r in rows if r["recall_expected"]]
    hits = sum(1 for r in checked if r["recall_hit"])
    print(f"检索 Recall@{top_k}（章节级，有召回目标的要求）：{hits}/{len(checked)}")
    for row in rows:
        mark = "OK " if (not row["recall_expected"] or row["recall_hit"]) else "MISS"
        if mark == "MISS":
            failures += 1
        print(f"[{mark}] {row['id']} {row['text']}")
        for item in row["top"][:3]:
            print(f"      {item}")
    if args.config:
        from local_llm import build_llm, load_config
        config = load_config(args.config)
        llm = build_llm(config)
        extraction_rows, extraction_doc, jd_text = run_extraction(
            evaluation, llm, args.set.parent)
        failures += len(extraction_doc.get("failed_batches", []))
        print(f"\n提取检查（{evaluation['extraction_jd']}）："
              f"{sum(1 for r in extraction_rows if r['ok'])}"
              f"/{len(extraction_rows)}")
        for row in extraction_rows:
            mark = "OK " if row["ok"] else "BAD"
            if not row["ok"]:
                failures += 1
            print(f"[{mark}] {row['line']} {row['detail']}")
        judgment_rows = run_judgment(evaluation, llm, retriever, top_k,
                                     extraction_doc, jd_text)
        good = sum(1 for r in judgment_rows if r["ok"])
        print(f"\n判断检查（{args.config}）：{good}/{len(judgment_rows)}")
        for row in judgment_rows:
            mark = "OK " if row["ok"] else "BAD"
            if not row["ok"]:
                failures += 1
            note = "（合成要求）" if row["synthetic"] else ""
            print(f"[{mark}] {row['id']} 要求={row['requirement_id']}{note} "
                  f"预期={row['target']} 实际={row['verdict']} "
                  f"状态={row['status']}")
    print(f"\n结果：{'全部通过' if failures == 0 else f'{failures} 项未通过'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
