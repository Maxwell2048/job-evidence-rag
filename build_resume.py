"""Assemble a complete tailored resume (Markdown) from a match_job run.

    python build_resume.py --run outputs/<run> --resume resume/base_resume.docx --config config.local.json

Inputs: the run's latest cv_suggestions*.json (from tailor_cv.py), the base
resume, experiences/ materials and profile.md. One model call arranges the
document as sections of lines; every line carries a material id and a
verbatim quote, checked by the program exactly as in tailor_cv (substring,
numbers, JD wording named). The body is written clean; a source appendix
maps every line back to its evidence so a reviewer can delete it before
sending. Output: resume_tailored.md / .json next to the suggestions.
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

from local_llm import LLMError, build_llm, load_config
from tailor_cv import (RESUME_ID, KIND_LABELS, UNVERIFIED_KINDS, CVTailor, _attach_sources, latest_output,
                       load_materials, load_run, non_english, read_resume, requirement_summary, validate_basis,
                       unique_output)

PROJECT_ROOT = Path(__file__).resolve().parent
BUILD_PROMPT_VERSION = "1"
MAX_LINE_CHARS = 500
# Page budget enforced by the program: models ignore 'keep it short' guidance.
MAX_KEY_PROJECTS, MAX_KEY_BULLETS = 5, 4
MAX_ADDITIONAL_PROJECTS, MAX_ADDITIONAL_BULLETS = 3, 2

SYSTEM_PROMPT = """你把已经写好的素材组装成一份完整的英文简历，不是重新创作。
素材：base_resume（现有简历底稿全文，material_id=RESUME）、profile（本人自述基本事实）、summary（已生成的摘要）、
tailored_projects（针对本岗位已生成并核对过来源的项目要点，附 basis）、review（对底稿每段的处理建议：keep/move_up/move_down/trim/remove）、gaps（不要声称具备的要求）。
规则：
1. 输出 sections 列表，按简历顺序；每个 section 有 title 和 lines；每行 kind 为 heading（项目名/学校等子标题）、bullet（要点）或 text（普通行）。
2. 来自 tailored_projects 的要点和 summary 只写 ref（要点的 id，如 "B07"；摘要为 "SUMMARY"），不要重抄 text 和 basis，程序会回填。
   其他行（底稿或 profile 里的内容）必须给 text 与 basis：material_id 与该材料中逐字连续的 quote；底稿里保留的行 quote 就是该行原文；trim 的行改写后 quote 用原句开头。
3. review 标 remove 的段落不要出现；trim 的段落缩短但不新增事实；move_up/move_down 体现在顺序上。
4. 不得新增素材里没有的事实、数字、工具或职位；gaps 里的要求不要声称具备。JD 措辞不能当 quote。
5. 第一个 section 的第一行必须是底稿的第一行（姓名）原文；联系方式、学历、语言等信息来自底稿或 profile，逐字引用。
   summary 若提供，必须作为一个 section 出现，text 原样照抄并沿用其 basis。
6. 同一项目不要重复出现：底稿中与 tailored_projects 同名的项目用 tailored 版本替换。
   tailored_projects 的子标题（heading）逐字使用它的 title；material_name 只是材料里的中文名，绝不能写进简历。
   title 为空时，自己给出英文标题。
7. 全文英文，任何一行都不得出现中文或全角标点（程序会逐行检查）；整份简历控制在两页以内：KEY PROJECTS 最多 5 个项目、每个最多 4 条要点，ADDITIONAL/OTHER PROJECTS 最多 3 个项目、每个最多 2 条要点；按与岗位的相关度排序，最相关的放前面。超出的部分程序会按顺序裁掉。
8. 任何一行都不出现课程代码（CITS5505、GENG5505 之类）；项目标题只保留名称与类型，底稿标题里的课程代码要去掉。
9. 输出紧凑 JSON：不缩进、不换行、不加 Markdown 代码围栏，避免输出过长被截断。"""

RESUME_SCHEMA = {
    "type": "object",
    "required": ["sections"],
    "additionalProperties": False,
    "properties": {"sections": {"type": "array", "minItems": 2, "items": {
        "type": "object", "required": ["title", "lines"], "additionalProperties": False,
        "properties": {
            "title": {"type": "string", "minLength": 1},
            "lines": {"type": "array", "minItems": 1, "items": {
                "type": "object", "required": ["kind"],
                "additionalProperties": False,
                "properties": {
                    "kind": {"type": "string", "enum": ["heading", "bullet", "text"]},
                    "ref": {"type": "string", "minLength": 1},
                    "text": {"type": "string", "minLength": 1},
                    "basis": {"type": "array", "minItems": 1, "items": {
                        "type": "object", "required": ["material_id", "quote"],
                        "additionalProperties": False,
                        "properties": {"material_id": {"type": "string"},
                                       "quote": {"type": "string", "minLength": 1}}}},
                }}},
        }}}},
}


def latest_suggestions(run_dir):
    path = latest_output(run_dir, "cv_suggestions", ".json")
    if path is None:
        raise ValueError(f"{run_dir} 里没有 cv_suggestions*.json；先运行 tailor_cv.py")
    return path


def referenced_lines(suggestions):
    """id -> {text, basis} for every tailored bullet (B01, B02...) and SUMMARY,
    so the model can cite them by id and the program fills the rest in."""
    refs = {}
    counter = 0
    for p in suggestions.get("projects", []):
        for b in p["bullets"]:
            counter += 1
            refs[f"B{counter:02d}"] = {"text": b["text"], "basis": [
                {"material_id": x["material_id"], "quote": x["quote"]} for x in b["basis"]]}
    summary = suggestions.get("summary")
    if summary:
        refs["SUMMARY"] = {"text": summary["text"], "basis": [
            {"material_id": x["material_id"], "quote": x["quote"]} for x in summary["basis"]]}
    return refs


def build_payload(suggestions, materials, resume_entry, jd_text):
    profile = [{"material_id": m["material_id"], "section": m["section"], "text": m["text"]}
               for m in materials if m["kind"] == "profile"]
    refs = referenced_lines(suggestions)
    projects, counter = [], 0
    for p in suggestions.get("projects", []):
        bullets = []
        for b in p["bullets"]:
            counter += 1
            bullets.append({"id": f"B{counter:02d}", "text": b["text"]})
        # "title" is the English heading from tailor_cv; "project" is the material's own
        # (often Chinese) name and must never be copied into the resume.
        projects.append({"title": p.get("title") or None, "material_name": p["project"], "bullets": bullets})
    return {
        "task_note": "组装最终英文简历。tailored_projects 的要点与 summary 用 ref 引用 id，程序回填文本与来源。",
        "jd_text": jd_text,
        "base_resume": {"material_id": RESUME_ID, "text": resume_entry["text"]},
        "profile": profile,
        "summary": ({"id": "SUMMARY", "text": refs["SUMMARY"]["text"]} if "SUMMARY" in refs else None),
        "tailored_projects": projects,
        "review": [{"quote": r["quote"], "action": r["action"], "reason": r["reason"][:120]}
                   for r in suggestions.get("resume_review", [])],
        "gaps": [{"requirement_id": g["requirement_id"], "text": g["text"], "kind": g["kind"]}
                 for g in suggestions.get("gaps", [])],
    }


def allowed_material_ids(suggestions, materials):
    ids = [RESUME_ID] + [m["material_id"] for m in materials if m["kind"] == "profile"]
    for p in suggestions.get("projects", []):
        for b in p["bullets"]:
            ids += [x["material_id"] for x in b["basis"]]
    if suggestions.get("summary"):
        ids += [x["material_id"] for x in suggestions["summary"]["basis"]]
    return list(dict.fromkeys(ids))


def required_lines(suggestions, resume_text):
    """Exact lines the assembled resume must contain: the base resume's first
    line (the candidate's name) and the generated summary, if any."""
    required = []
    first = next((line.strip() for line in resume_text.splitlines() if line.strip()), "")
    if first:
        required.append(("底稿第一行（姓名）", first, 1))
    summary = suggestions.get("summary")
    if summary and summary.get("text"):
        required.append(("已生成的 summary 原文", summary["text"].strip(), None))
    return required


def expand_refs(data, refs):
    """Replace {ref: id} lines with the referenced text and basis, in place.
    Returns error strings for unknown ids or lines with neither ref nor text."""
    errors = []
    sections = data.get("sections", []) if isinstance(data, dict) else []
    for s_index, section in enumerate(sections, 1):
        for l_index, line in enumerate(section.get("lines", []), 1):
            if not isinstance(line, dict):
                continue
            ref = line.pop("ref", None)
            if ref is not None:
                if ref not in refs:
                    errors.append(f"第 {s_index} 节第 {l_index} 行引用了不存在的 ref {ref!r}")
                    continue
                line["text"] = refs[ref]["text"]
                line["basis"] = [dict(b) for b in refs[ref]["basis"]]
                # Filled in by the program from suggestions that tailor_cv already
                # validated against the project's full material; the model cannot
                # change it, so re-checking it here could only cause doomed retries.
                line["_from_ref"] = ref
            elif not line.get("text") or not line.get("basis"):
                errors.append(f"第 {s_index} 节第 {l_index} 行既没有 ref 也没有完整的 text+basis")
    return errors


def validate_document(data, by_id, allowed, jd_text, required=(), refs=None):
    errors = expand_refs(data, refs or {})
    sections = data.get("sections", []) if isinstance(data, dict) else []
    for label, text, section_index in required:
        scope = sections[section_index - 1:section_index] if section_index else sections
        texts = [line.get("text", "").strip() for s in scope for line in s.get("lines", [])]
        if text not in texts and section_index == 1 and sections \
                and isinstance(sections[0].get("lines"), list):
            # The name is a fixed fact from the base resume: insert it rather than
            # spend one of three attempts asking the model to copy it.
            sections[0]["lines"].insert(0, {"kind": "text", "text": text, "_from_ref": "NAME",
                                            "basis": [{"material_id": RESUME_ID, "quote": text}]})
            continue
        if text not in texts:
            where = f"第 {section_index} 节" if section_index else "任一节"
            errors.append(f"{where}缺少{label}：必须有一行 text 与之完全一致：{text[:80]!r}")
    for s_index, section in enumerate(sections, 1):
        lines = [l for l in section.get("lines", []) if isinstance(l, dict) and l.get("text")]
        for l_index, line in enumerate(lines, 1):
            if line.get("_from_ref"):
                continue  # program-filled and already validated upstream
            errors += validate_basis([line], by_id, allowed,
                                     label=f"第 {s_index} 节第 {l_index} 行", jd_text=jd_text,
                                     no_course_codes=True, english_only=True)
        for l_index, line in enumerate(lines, 1):
            if line.get("_from_ref"):
                continue  # the model cannot shorten program-filled text; asking it to only breaks the ref
            if len(line.get("text", "")) > MAX_LINE_CHARS:
                errors.append(f"第 {s_index} 节第 {l_index} 行超过 {MAX_LINE_CHARS} 字符，请拆分或精简")
    return errors


def ensure_summary(llm, suggestions, matches, jd_text, materials, by_project, resume_entry,
                   timeout_seconds=None):
    """A resume needs its SUMMARY. When tailor_cv could not produce one, try
    once more here, on its own. Returns a note for the appendix, or None."""
    if suggestions.get("summary"):
        return None
    tailor = CVTailor(llm, materials, by_project, resume_entry, timeout_seconds)
    try:
        data = tailor.summary(requirement_summary(matches), suggestions.get("projects", []),
                              suggestions.get("gaps", []), jd_text)
    except LLMError as exc:
        return f"SUMMARY 未生成：定制建议里没有摘要，组装时补生成也失败（{str(exc)[:120]}）；请手动补写"
    suggestions["summary"] = _attach_sources(data["summary"], tailor.by_id)
    return "SUMMARY 是组装时补生成的（定制建议那一步没有产出摘要）"


def build(llm, suggestions, materials, resume_entry, jd_text, timeout_seconds=None):
    by_id = {m["material_id"]: m for m in materials}
    allowed = allowed_material_ids(suggestions, materials)
    payload = build_payload(suggestions, materials, resume_entry, jd_text)

    def fits():
        return llm.estimate_context("build_resume", SYSTEM_PROMPT, payload, RESUME_SCHEMA) \
            <= llm.prompt_char_budget
    if not fits():
        # Shed the least essential context first: review reasons, then the JD body.
        payload["review"] = [{k: v for k, v in r.items() if k != "reason"} for r in payload["review"]]
    if not fits():
        payload["jd_text"] = jd_text[:3000] + ("…" if len(jd_text) > 3000 else "")
    if not fits():
        raise LLMError("组装简历的请求超出提示词预算，请调大 context_budget 或减少项目数",
                       error_type="context_budget")
    required = required_lines(suggestions, resume_entry["text"])
    refs = referenced_lines(suggestions)
    data = llm.generate_json("build_resume", SYSTEM_PROMPT, payload, RESUME_SCHEMA,
                             timeout_seconds,
                             validate=lambda d: validate_document(d, by_id, allowed, jd_text,
                                                                  required, refs))
    sections = []
    for section in data["sections"]:
        lines = []
        for line in section["lines"]:
            basis = []
            for b in line["basis"]:
                m = by_id[b["material_id"]]
                basis.append({**b, "kind": m["kind"], "source": m["source"],
                              "section": m["section"], "line_start": m["line_start"],
                              "line_end": m["line_end"]})
            clean = {k: v for k, v in line.items() if k != "_from_ref"}
            lines.append({**clean, "basis": basis})
        sections.append({"title": section["title"], "lines": lines})
    return sections


def _project_limits(title):
    key = title.strip().upper()
    if "PROJECT" not in key and "EXPERIENCE" not in key:
        return None
    if "ADDITIONAL" in key or "OTHER" in key:
        return MAX_ADDITIONAL_PROJECTS, MAX_ADDITIONAL_BULLETS
    return MAX_KEY_PROJECTS, MAX_KEY_BULLETS


def trim_projects(sections):
    """Enforce the page budget in project sections, keeping the model's order.
    Returns (trimmed sections, notes describing what was dropped)."""
    notes, result = [], []
    for section in sections:
        limits = _project_limits(section["title"])
        if limits is None:
            result.append(section)
            continue
        max_projects, max_bullets = limits
        kept, projects, bullets, dropping = [], 0, 0, False
        dropped_projects, dropped_bullets = [], 0
        for line in section["lines"]:
            if line["kind"] == "heading":
                projects += 1
                bullets = 0
                dropping = projects > max_projects
                if dropping:
                    dropped_projects.append(line["text"])
                    continue
            elif dropping:
                continue
            elif line["kind"] == "bullet":
                bullets += 1
                if bullets > max_bullets:
                    dropped_bullets += 1
                    continue
            kept.append(line)
        if dropped_projects:
            notes.append(f"篇幅裁剪：{section['title']}：超过 {max_projects} 个项目，已略去 "
                         + "；".join(dropped_projects))
        if dropped_bullets:
            notes.append(f"篇幅裁剪：{section['title']}：每个项目最多 {max_bullets} 条要点，"
                         f"已略去 {dropped_bullets} 条")
        result.append({**section, "lines": kept})
    return result, notes


def render(sections, suggestions_path, resume_path, meta, notes=()):
    body, appendix = [], []
    unverified = 0
    for s_index, section in enumerate(sections, 1):
        if s_index != 1:
            body += ["", f"## {section['title']}"]
        for l_index, line in enumerate(section["lines"], 1):
            key = f"{s_index}.{l_index}"
            if s_index == 1 and l_index == 1:
                body.append(f"# {line['text']}")
            elif line["kind"] == "heading":
                body += ["", f"### {line['text']}"]
            elif line["kind"] == "bullet":
                body.append(f"- {line['text']}")
            else:
                body.append(line["text"])
            cites = []
            for b in line["basis"]:
                if b["kind"] in UNVERIFIED_KINDS:
                    unverified += 1
                if b["kind"] == "resume":
                    where = "简历底稿"
                else:
                    where = f"{b['source']} › {b['section']}（行 {b['line_start']}-{b['line_end']}）"
                cites.append(f"[{KIND_LABELS.get(b['kind'], b['kind'])}] {where}：“{b['quote']}”")
            appendix.append(f"- {key} “{line['text'][:50]}{'…' if len(line['text']) > 50 else ''}” ← "
                            + "；".join(cites))
    lines = body + ["", "---", "", "## 来源附录（发送前删除本节及以下）", "",
                    f"- 匹配运行：{meta['run_id']}；定制建议：{Path(suggestions_path).name}；"
                    f"底稿：{resume_path}；组装提示词版本 {BUILD_PROMPT_VERSION}；"
                    f"生成时间 {datetime.datetime.now().isoformat(timespec='seconds')}",
                    f"- 依据未核对材料（简历底稿、待核对经历）的引用共 {unverified} 处，其余来自已核对经历或自述事实。",
                    "- 编号为 节.行；每行的改写是否忠于引用，请逐条核对，尤其是形容词与程度副词。",
                    *[f"- {note}" for note in notes],
                    ""] + appendix
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description="从匹配结果和定制建议组装完整简历（Markdown）")
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--resume", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--suggestions", type=Path, default=None,
                        help="cv_suggestions*.json，默认取运行目录里最新的一份")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "experiences")
    args = parser.parse_args(argv)
    try:
        meta, matches, jd_text = load_run(args.run)
        suggestions_path = args.suggestions or latest_suggestions(args.run)
        suggestions = json.loads(Path(suggestions_path).read_text(encoding="utf-8"))
        resume_text = read_resume(args.resume)
        config = load_config(args.config)
        llm = build_llm(config)
    except (OSError, ValueError, LLMError, KeyError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    materials, by_project, resume_entry = load_materials(args.data, resume_text)
    print(f"[build-resume] 使用 {Path(suggestions_path).name}，材料 {len(materials)} 条；组装中……",
          file=sys.stderr)
    summary_note = ensure_summary(llm, suggestions, matches, jd_text, materials, by_project, resume_entry,
                                  config.get("timeout_seconds"))
    if summary_note:
        print(f"[build-resume] {summary_note}", file=sys.stderr)
    try:
        sections = build(llm, suggestions, materials, resume_entry, jd_text,
                         config.get("timeout_seconds"))
    except LLMError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    out_md = unique_output(args.run, "resume_tailored")
    sections, trim_notes = trim_projects(sections)
    if summary_note:
        trim_notes = [summary_note, *trim_notes]
    out_md.write_text(render(sections, suggestions_path, args.resume, meta, trim_notes),
                      encoding="utf-8")
    out_md.with_suffix(".json").write_text(json.dumps(
        {"prompt_version": BUILD_PROMPT_VERSION, "run_id": meta["run_id"],
         "suggestions": str(suggestions_path), "resume_path": str(args.resume),
         "sections": sections}, ensure_ascii=False, indent=2), encoding="utf-8")
    total = sum(len(s["lines"]) for s in sections)
    print(f"已写入：{out_md}（{len(sections)} 节，{total} 行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
