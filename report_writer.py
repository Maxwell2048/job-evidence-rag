"""Report writer: matches.json and the human-readable report.md."""
import json

VERDICT_LABELS = {
    "direct": "直接支持",
    "related": "相关但不足",
    "insufficient": "当前资料证据不足",
}
IMPORTANCE_LABELS = {
    "required": "必需",
    "preferred": "优先",
    "unspecified": "未说明",
}
STATUS_LABELS = {
    "complete": "complete（全部要求已处理）",
    "partial": "partial（部分要求未处理完成，单列见下）",
    "failed": "failed（本阶段未处理完成）",
}
GROUP_OPERATOR_LABELS = {"OR": "二选一", "AND": "同时需要"}


def _group_labels(requirement_id, groups):
    labels = [f"{g['group_id']} {GROUP_OPERATOR_LABELS.get(g['operator'], g['operator'])}"
              for g in groups
              if requirement_id in g.get("requirement_ids", [])]
    return "、".join(labels) if labels else "—"


def overall_status(matches):
    """complete / partial / failed from per-requirement processing_status."""
    if not matches:
        return "failed"
    ok = sum(1 for m in matches if m["processing_status"] == "ok")
    if ok == len(matches):
        return "complete"
    if ok == 0:
        return "failed"
    return "partial"


def verdict_label(match):
    if match["processing_status"] == "ok":
        return VERDICT_LABELS.get(match["verdict"], "未判断")
    if match["processing_status"] == "incomplete":
        base = VERDICT_LABELS.get(match["verdict"], "无判断")
        return f"未完成（{base}）"
    return "处理错误（无判断）"


def _primary_project(match):
    for evidence in match.get("evidence", []):
        if evidence.get("project"):
            return evidence["project"]
    for candidate in match.get("candidates", []):
        if candidate.get("project"):
            return candidate["project"]
    return "—"


def write_matches(run_dir, schema_version, matches):
    document = {"schema_version": schema_version, "matches": matches}
    (run_dir / "matches.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")


def write_report(run_dir, meta, matches, requirement_groups=None, summary=None):
    groups = requirement_groups or []
    text_by_id = {m["requirement_id"]: m["text"] for m in matches}
    lines = []
    lines.append("# JD 匹配报告")
    lines.append("")
    lines.append(f"- 输入 JD：{meta['input']['jd_path']}"
                 f"（SHA-256：{meta['input']['jd_sha256']}）")
    lines.append("- 分析范围：JD 要求 → 个人项目贡献（默认知识库）。"
                 "学历、工作资格、签证、工作地点、办公软件等基本事实仅来自 "
                 "experiences/profile.md 的本人自述（报告中标为“个人基本信息（自述事实）”），"
                 "不因项目中提到某事实而推断其成立；没有该文件时这些要求通常判为证据不足。")
    lines.append(f"- 运行状态：{STATUS_LABELS.get(meta['run_status'], meta['run_status'])}"
                 f"（范围：{meta['scope']}）")
    review = [m["requirement_id"] for m in matches if m.get("needs_review")]
    incomplete = [m["requirement_id"] for m in matches
                  if m["processing_status"] != "ok"]
    if review:
        lines.append(f"- **待复核 {len(review)} 条**：{', '.join(review)}"
                     "（原文定位或优先级需人工确认，见详情）")
    if incomplete:
        lines.append(f"- **未完成/出错 {len(incomplete)} 条**："
                     f"{', '.join(incomplete)}（处理问题不等于证据不足，见详情）")
    for warning in meta.get("warnings", [])[:10]:
        lines.append(f"- 警告：{warning}")
    lines.append("")
    if summary is not None:
        from job_summary import summary_markdown
        lines.extend(summary_markdown(summary, matches, groups))
    lines.append("## 要求概览")
    lines.append("")
    lines.append("| ID | 要求 | 优先级 | 证据判断 | 主要项目 | 组关系 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for match in matches:
        lines.append(
            f"| {match['requirement_id']} | {match['text']} "
            f"| {IMPORTANCE_LABELS.get(match['importance'], match['importance'])} "
            f"| {verdict_label(match)} | {_primary_project(match)} "
            f"| {_group_labels(match['requirement_id'], groups)} |")
    lines.append("")
    if groups:
        lines.append("## AND/OR 关系")
        lines.append("")
        for group in groups:
            members = [f"{rid} {text_by_id.get(rid, '')}".strip()
                       for rid in group.get("requirement_ids", [])]
            label = GROUP_OPERATOR_LABELS.get(group["operator"], group["operator"])
            lines.append(f"- {group['group_id']} {label}："
                         + "；".join(m for m in members if m)
                         + f" — JD 原文：“{group['source_quote']}”")
            span = group.get("source_span")
            if span:
                lines.append(f"  - 位置：JD 第 {span['line_start']}-{span['line_end']} 行")
            else:
                lines.append("  - 位置：原文定位待复核")
            for missing_text in group.get("missing_texts", []):
                lines.append(f"  - 组引用了未解析的要求 text：{missing_text}")
        lines.append("- 注意：组关系表示招聘方在这几条要求间的选择/叠加关系，"
                     "不代表每条要求本身成立")
    lines.append("")
    lines.append("## 逐项详情")
    for match in matches:
        lines.append("")
        lines.append(f"### {match['requirement_id']} {match['text']}")
        lines.append("")
        lines.append(f"- 优先级：{IMPORTANCE_LABELS.get(match['importance'], match['importance'])}")
        if match.get("category"):
            lines.append(f"- 类别：{match['category']}")
        if match.get("qualifiers"):
            parts = [f"{q['type']}={q['value']}" for q in match["qualifiers"]]
            lines.append(f"- 限定条件：{'；'.join(parts)}")
        quote = match.get("source_quote")
        span = match.get("source_span")
        if quote:
            location = (f"（JD 第 {span['line_start']}-{span['line_end']} 行）"
                        if span else "（原文定位待复核）")
            lines.append(f"- JD 原文：“{quote}”{location}")
        if match.get("needs_review"):
            notes = "；".join(match.get("review_notes", []))
            lines.append(f"- **待复核**：{notes or '提取模型标记为需人工确认'}")
        lines.append(f"- 判断：{verdict_label(match)}")
        if match.get("reason"):
            lines.append(f"- 理由：{match['reason']}")
        for error in match.get("errors", []):
            lines.append(f"- 处理问题：{error}")
        if match.get("evidence"):
            lines.append("- 个人经历证据：")
            for evidence in match["evidence"]:
                lines.append(f"  > {evidence['quote']}")
                lines.append(f"  - 来源：{evidence['source']} 章节行 "
                             f"{evidence['line_start']}-{evidence['line_end']}"
                             f"（候选 {evidence['candidate_id']}，"
                             f"章节 {evidence['section_id']}；行号为整个章节范围）")
                for evidence_line in evidence.get("evidence_lines", []):
                    lines.append(f"  - 已有证据行（既有来源记录，本次未重新核查）："
                                 f"{evidence_line}")
        if match.get("missing_aspects"):
            lines.append("- 未覆盖条件：" + "；".join(match["missing_aspects"]))
        if match.get("questions_for_user"):
            lines.append("- 可向用户确认的问题：" + "；".join(match["questions_for_user"]))
        if match.get("candidates"):
            lines.append("- 检索候选（分数仅用于检索排序，不证明技能存在）：")
            lines.append("")
            lines.append("| 候选 | 项目 / 章节 | 分数（仅用于检索排序） | 源文件 |")
            lines.append("| --- | --- | --- | --- |")
            for candidate in match["candidates"]:
                lines.append(f"| {candidate['candidate_id']} "
                             f"| {candidate['project']} / {candidate['section']} "
                             f"| {candidate['score']} | {candidate['source']} |")
    lines.append("")
    lines.append("## 待补充资料 / 待确认问题")
    lines.append("")
    questions = [(m["requirement_id"], q) for m in matches
                 for q in m.get("questions_for_user", [])]
    missing = [(m["requirement_id"], a) for m in matches
               for a in m.get("missing_aspects", [])]
    if questions or missing:
        for requirement_id, question in questions:
            lines.append(f"- [{requirement_id}] 待确认：{question}")
        for requirement_id, aspect in missing:
            lines.append(f"- [{requirement_id}] 未覆盖：{aspect}")
    else:
        lines.append("- （无）")
    lines.append("")
    lines.append("---")
    lines.append("注：“相关但不足”不否定个人能力；“当前资料证据不足”指当前检索资料未提供证据；"
                 "待复核项是提取模型的推断，不代表招聘方明示要求。")
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
