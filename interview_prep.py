"""Interview questions and STAR answers for one JD, from a match_job run.

    python interview_prep.py --run outputs/<run> --config config.local.json

For every requirement the model proposes the questions an interviewer is
likely to ask and a STAR-shaped answer built only from the candidate's own
material (contribution sections, STAR sections, profile facts). Every answer
point cites a material id with a verbatim quote, validated by the program
(substring, numbers, JD wording named) inside the adapter's retry budget.
Requirements judged insufficient get an "Honest" point: acknowledge the gap
without inventing experience. Output: interview_prep.md / .json in the run
directory. Nothing here is a script to memorise; it is evidence-linked
preparation notes for the candidate to review.
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

from local_llm import LLMError, build_llm, load_config
from tailor_cv import (IMPORTANCE_LABELS, KIND_LABELS, NUMBER, VERDICT_LABELS,
                       jd_name_number, latest_output, load_materials, load_run, number_supported, personal_ids,
                       unique_output, validate_basis)

PROJECT_ROOT = Path(__file__).resolve().parent
INTERVIEW_PROMPT_VERSION = "1"
PARTS = ["Situation", "Task", "Action", "Result", "Honest"]
MAX_QUESTIONS = 2
CANDIDATES_PER_REQUIREMENT = 3
DEFAULT_BATCH = 5

SYSTEM_PROMPT = """你为求职者准备英文面试。对每条 JD 要求，写出面试官最可能问的 1-2 个问题，并用求职者自己的材料组织 STAR 回答。
材料分 experience（已核对或待核对的项目经历）和 profile（本人自述）；scope=personal 是本人的工作或自述，scope=team 是项目背景或团队成果。
回答的每个 point 都要在 basis 里给出 material_id 和该材料中逐字连续的 quote；team 材料只能交代背景，STAR 回答至少一个 point 必须引用 personal 材料，且不能把团队成果说成本人完成。
规则：
1. verdict=direct 的要求：问题偏深入追问（细节、取舍、结果），回答按 Situation/Task/Action/Result 四个 part。
2. verdict=related 的要求：问题会探究缺口（missing_aspects），回答用 STAR 讲已有的部分，再加一个 part=Honest 的 point，坦诚说明材料没覆盖的部分和学习计划，不编造经历。
3. verdict=insufficient 的要求：问题是"你有没有…经验"，回答只给 part=Honest：坦诚没有直接经验，可提及材料里最接近的可迁移经历（要有 basis），不要声称具备。
4. Honest 类 point 可以没有 basis，但不得包含任何数字、工具名或成果主张；其他 part 必须有 basis。
5. 不得新增材料里没有的事实、数字、工具、职位或成果。JD 措辞不能当 quote。
6. question 与 point.text 用英文；intent（面试官意图）与 note 用中文。输出符合 schema 的紧凑 JSON（不缩进、不换行、无代码围栏）。"""

GENERAL_PROMPT = """你为求职者准备两道通用英文面试题的回答："Tell me about yourself"（60 秒自我介绍）和 "Why are you interested in this role"。
只能使用 materials（项目经历的英文简历表述、STAR 案例）和 profile（本人自述）里的事实；每个 point 都要给 basis（material_id 与逐字 quote）。
不要声称 gaps 里的要求；对岗位的兴趣要落到 JD 里真实存在的职责或要求上，可引用 jd_text 的内容作为动机来源但不能把 JD 当作自己的经历。
point.text 用英文；note 用中文。输出符合 schema 的 JSON。"""

_BASIS = {"type": "array", "items": {
    "type": "object", "required": ["material_id", "quote"], "additionalProperties": False,
    "properties": {"material_id": {"type": "string"}, "quote": {"type": "string", "minLength": 1}}}}

_POINT = {"type": "object", "required": ["part", "text", "basis"], "additionalProperties": False,
          "properties": {"part": {"type": "string", "enum": PARTS},
                         "text": {"type": "string", "minLength": 1},
                         "basis": _BASIS}}

PREP_SCHEMA = {
    "type": "object", "required": ["items"], "additionalProperties": False,
    "properties": {"items": {"type": "array", "items": {
        "type": "object", "required": ["requirement_id", "questions"], "additionalProperties": False,
        "properties": {
            "requirement_id": {"type": "string"},
            "questions": {"type": "array", "minItems": 1, "maxItems": MAX_QUESTIONS, "items": {
                "type": "object", "required": ["question", "intent", "answer", "note"],
                "additionalProperties": False,
                "properties": {"question": {"type": "string", "minLength": 1},
                               "intent": {"type": "string"},
                               "answer": {"type": "array", "minItems": 1, "items": _POINT},
                               "note": {"type": "string"}}}}}}}},
}

GENERAL_SCHEMA = {
    "type": "object", "required": ["questions"], "additionalProperties": False,
    "properties": {"questions": {"type": "array", "minItems": 2, "maxItems": 2, "items": {
        "type": "object", "required": ["question", "answer", "note"], "additionalProperties": False,
        "properties": {"question": {"type": "string", "minLength": 1},
                       "answer": {"type": "array", "minItems": 1, "items": _POINT},
                       "note": {"type": "string"}}}}},
}


# ---------------------------------------------------------------- material selection


def material_index(materials):
    """(source, line_start, line_end) -> material, to map match candidates back."""
    return {(m["source"], m["line_start"], m["line_end"]): m for m in materials}


def select_materials(matches_batch, materials, per_requirement=CANDIDATES_PER_REQUIREMENT):
    """Materials a batch may cite: top candidate sections of each requirement,
    STAR sections of the projects involved, and all profile facts."""
    index = material_index(materials)
    chosen, projects = {}, set()
    for match in matches_batch:
        for cand in match.get("candidates", [])[:per_requirement]:
            m = index.get((cand["source"], cand["line_start"], cand["line_end"]))
            if m is not None:
                chosen[m["material_id"]] = m
                projects.add(m["project"])
        for ev in match.get("evidence", []):
            m = index.get((ev["source"], ev["line_start"], ev["line_end"]))
            if m is not None:
                chosen[m["material_id"]] = m
                projects.add(m["project"])
    for m in materials:
        if m["kind"] == "profile" or (m["project"] in projects and m["section"].startswith("STAR")):
            chosen[m["material_id"]] = m
    return [chosen[k] for k in sorted(chosen)]


def batches(matches, size=DEFAULT_BATCH):
    ok = [m for m in matches if m["processing_status"] == "ok"]
    return [ok[i:i + size] for i in range(0, len(ok), size)]


# ---------------------------------------------------------------- validation

HONEST_FORBIDDEN = ("experience with", "proficient", "expert", "years")


def validate_items(data, batch_ids, by_id, allowed, jd_text):
    errors = []
    items = data.get("items", []) if isinstance(data, dict) else []
    seen = set()
    for item in items:
        rid = item.get("requirement_id")
        if rid not in batch_ids:
            errors.append(f"requirement_id {rid!r} 不在本批要求里（本批：{', '.join(batch_ids)}）")
            continue
        seen.add(rid)
        for q_index, question in enumerate(item.get("questions", []), 1):
            points = question.get("answer", [])
            errors += validate_answer(points, by_id, allowed, jd_text,
                                      label=f"{rid} 问题 {q_index} 第 {{index}} 个 point")
            star_points = [p for p in points if p.get("part") != "Honest"]
            own = personal_ids(by_id, allowed)
            if star_points and not any(b.get("material_id") in own
                                       for p in star_points for b in p.get("basis", [])):
                errors.append(f"{rid} 问题 {q_index} 的 STAR 回答只引用了 scope=team 的团队/背景章节；"
                              "至少一个 point 必须引用 scope=personal 的本人材料")
    for rid in batch_ids:
        if rid not in seen:
            errors.append(f"缺少要求 {rid} 的题目")
    return errors


def validate_answer(points, by_id, allowed, jd_text, label="第 {index} 个 point"):
    errors = []
    with_basis = [p for p in points if p.get("part") != "Honest" or p.get("basis")]
    errors += validate_basis(with_basis, by_id, allowed, label=label, jd_text=jd_text,
                             allow_jd_names=True)
    pool = "\n".join(by_id[i]["text"] for i in allowed)
    for index, point in enumerate(points, 1):
        if point.get("part") == "Honest" and not point.get("basis"):
            text = point.get("text", "")
            for number in NUMBER.findall(text):
                if jd_name_number(number, text, jd_text):
                    continue
                if not number_supported(number, pool):
                    errors.append(f"{label.format(index=index)}（Honest）含材料中没有的数字 {number!r}")
            lowered = text.lower()
            for phrase in HONEST_FORBIDDEN:
                if phrase in lowered:
                    errors.append(f"{label.format(index=index)}（Honest）不能声称能力（含“{phrase}”）；"
                                  "只承认缺口并说明可迁移经历或学习计划")
                    break
    return errors


# ---------------------------------------------------------------- generation


def _visible(entries):
    return [{"material_id": e["material_id"], "kind": e["kind"], "scope": e.get("scope", "personal"),
             "project": e["project"], "section": e["section"], "text": e["text"]} for e in entries]


def _requirement_payload(match):
    return {"requirement_id": match["requirement_id"], "text": match["text"],
            "importance": match["importance"], "verdict": match["verdict"],
            "jd_source_quote": match.get("source_quote", ""),
            "missing_aspects": match.get("missing_aspects", []),
            "questions_for_user": match.get("questions_for_user", []),
            "evidence_quotes": [e["quote"] for e in match.get("evidence", [])]}


def prepare(llm, matches, jd_text, materials, timeout_seconds=None, batch_size=DEFAULT_BATCH,
            summary=None):
    by_id = {m["material_id"]: m for m in materials}
    result = {"schema_version": "1.0", "prompt_version": INTERVIEW_PROMPT_VERSION,
              "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
              "items": [], "general": [], "errors": [], "warnings": []}
    for number, batch in enumerate(batches(matches, batch_size), 1):
        chosen = select_materials(batch, materials)
        allowed = [m["material_id"] for m in chosen]
        batch_ids = [m["requirement_id"] for m in batch]
        payload = {"jd_requirements": [_requirement_payload(m) for m in batch],
                   "materials": _visible(chosen)}
        task = f"interview_batch_{number}"
        while llm.estimate_context(task, SYSTEM_PROMPT, payload, PREP_SCHEMA) > llm.prompt_char_budget \
                and len(payload["materials"]) > len([m for m in chosen if m["kind"] == "profile"]):
            payload["materials"] = payload["materials"][:-1]
            allowed = [m["material_id"] for m in payload["materials"]]
            result["warnings"].append(f"第 {number} 批材料超出预算，已减少可引用章节")
        try:
            data = llm.generate_json(task, SYSTEM_PROMPT, payload, PREP_SCHEMA, timeout_seconds,
                                     validate=lambda d, ids=batch_ids, al=allowed:
                                     validate_items(d, ids, by_id, al, jd_text))
        except LLMError as exc:
            result["errors"].append({"step": task, "requirement_ids": batch_ids,
                                     "error_type": exc.error_type, "message": str(exc)})
            continue
        by_rid = {m["requirement_id"]: m for m in batch}
        for item in data["items"]:
            match = by_rid[item["requirement_id"]]
            result["items"].append({
                "requirement_id": match["requirement_id"], "text": match["text"],
                "importance": match["importance"], "verdict": match["verdict"],
                "source_quote": match.get("source_quote", ""),
                "questions": [{**q, "answer": [_attach(p, by_id) for p in q["answer"]]}
                              for q in item["questions"]]})
    try:
        result["general"] = general_questions(llm, jd_text, matches, materials, timeout_seconds,
                                              summary)
    except LLMError as exc:
        result["errors"].append({"step": "general", "requirement_ids": [],
                                 "error_type": exc.error_type, "message": str(exc)})
    result["status"] = "complete" if not result["errors"] else "partial"
    return result


def general_questions(llm, jd_text, matches, materials, timeout_seconds=None, summary=None):
    by_id = {m["material_id"]: m for m in materials}
    chosen = [m for m in materials if m["kind"] == "profile"
              or m["section"].startswith(("英文简历表述", "STAR"))]
    allowed = [m["material_id"] for m in chosen]
    gaps = [{"requirement_id": m["requirement_id"], "text": m["text"]} for m in matches
            if m["processing_status"] == "ok" and m["verdict"] == "insufficient"]
    payload = {"jd_text": jd_text, "gaps": gaps, "materials": _visible(chosen)}
    if summary:
        payload["cv_summary"] = summary
    data = llm.generate_json(
        "interview_general", GENERAL_PROMPT, payload, GENERAL_SCHEMA, timeout_seconds,
        validate=lambda d: sum((validate_answer(q.get("answer", []), by_id, allowed, jd_text,
                                                label=f"问题 {i} 第 {{index}} 个 point")
                                for i, q in enumerate(d.get("questions", []), 1)), []))
    return [{**q, "answer": [_attach(p, by_id) for p in q["answer"]]} for q in data["questions"]]


def _attach(point, by_id):
    basis = []
    for b in point.get("basis", []):
        m = by_id[b["material_id"]]
        basis.append({**b, "kind": m["kind"], "project": m["project"], "section": m["section"],
                      "source": m["source"], "line_start": m["line_start"],
                      "line_end": m["line_end"]})
    return {**point, "basis": basis}


# ---------------------------------------------------------------- output

GROUPS = [("direct", "一、会被深入追问的（直接支持）"),
          ("related", "二、会被追问缺口的（相关但不足）"),
          ("insufficient", "三、需要坦诚回答的（证据不足）")]


def _sources(basis):
    if not basis:
        return "（无引用：坦诚说明，不含事实主张）"
    return "；".join(
        f"[{KIND_LABELS.get(b['kind'], b['kind'])}] {b['source']} › {b['section']}"
        f"（行 {b['line_start']}-{b['line_end']}）：“{b['quote']}”" for b in basis)


def write_markdown(result, meta, out_path):
    lines = ["# 面试准备", ""]
    lines.append(f"- 匹配运行：{result.get('run_id', meta['run_id'])}（JD：{meta['input']['jd_path']}）")
    lines.append(f"- 生成模型：{meta.get('models', {}).get('generator', {}).get('name')}；"
                 f"提示词版本 {result['prompt_version']}；生成时间 {result['generated_at']}")
    lines.append(f"- 状态：{result['status']}")
    lines.append("- 说明：每个回答要点都附来源与逐字引用；标为 Honest 的要点是坦诚说明，"
                 "不含事实主张。回答是素材，不是背诵稿；形容词与语境请自行核对。")
    lines += ["", "## 通用问题", ""]
    if not result["general"]:
        lines.append("（未生成）")
    for q in result["general"]:
        lines += [f"### {q['question']}", ""]
        for p in q["answer"]:
            lines.append(f"- **{p['part']}**：{p['text']}")
            lines.append(f"  - 来源：{_sources(p['basis'])}")
        if q.get("note"):
            lines.append(f"- 备注：{q['note']}")
        lines.append("")
    for verdict, title in GROUPS:
        items = [i for i in result["items"] if i["verdict"] == verdict]
        lines += [f"## {title}", ""]
        if not items:
            lines += ["（无）", ""]
        for item in items:
            lines += [f"### {item['requirement_id']} {item['text']}", "",
                      f"- 优先级：{IMPORTANCE_LABELS.get(item['importance'], item['importance'])}；"
                      f"判断：{VERDICT_LABELS.get(item['verdict'], item['verdict'])}；"
                      f"JD 原文：“{item['source_quote']}”", ""]
            for n, q in enumerate(item["questions"], 1):
                lines += [f"**Q{n}. {q['question']}**", ""]
                if q.get("intent"):
                    lines.append(f"- 面试官意图：{q['intent']}")
                for p in q["answer"]:
                    lines.append(f"- **{p['part']}**：{p['text']}")
                    lines.append(f"  - 来源：{_sources(p['basis'])}")
                if q.get("note"):
                    lines.append(f"- 备注：{q['note']}")
                lines.append("")
    if result["warnings"]:
        lines += ["## 警告", ""] + [f"- {w}" for w in result["warnings"]] + [""]
    if result["errors"]:
        lines += ["## 失败步骤", ""] + [
            f"- {e['step']}（{e['error_type']}；要求 {', '.join(e['requirement_ids']) or '—'}）：{e['message']}"
            for e in result["errors"]] + [""]
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def latest_summary(run_dir):
    path = latest_output(run_dir, "cv_suggestions", ".json")
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return (data.get("summary") or {}).get("text")


# ---------------------------------------------------------------- CLI


def main(argv=None):
    parser = argparse.ArgumentParser(description="生成面试题与 STAR 回答（Markdown）")
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "experiences")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    args = parser.parse_args(argv)
    try:
        meta, matches, jd_text = load_run(args.run)
        config = load_config(args.config)
        llm = build_llm(config)
    except (OSError, ValueError, LLMError, KeyError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    materials, _by_project, _resume = load_materials(args.data)
    print(f"[interview-prep] 要求 {len(matches)} 条，材料 {len(materials)} 条；生成中……",
          file=sys.stderr)
    result = prepare(llm, matches, jd_text, materials, config.get("timeout_seconds"),
                     args.batch_size, summary=latest_summary(args.run))
    result["run_id"] = meta["run_id"]
    out_md = unique_output(args.run, "interview_prep")
    write_markdown(result, meta, out_md)
    out_md.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    total = sum(len(i["questions"]) for i in result["items"])
    print(f"已写入：{out_md}")
    print(f"状态：{result['status']}；要求 {len(result['items'])} 条，问题 {total} 个，"
          f"通用问题 {len(result['general'])} 个")
    for e in result["errors"]:
        print(f"失败步骤 {e['step']}：{e['message']}")
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
