"""模型侧严格JSON Schema；本地仍执行独立业务校验。"""
from __future__ import annotations

from typing import Any

from agent_data import SKILL_NAMES


def obj(properties: dict) -> dict:
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def arr(items: dict) -> dict:
    return {"type": "array", "items": items}


def response_schema(role: str, data: dict) -> dict | None:
    """按中心/提取/复核角色生成严格结构，不允许新增动态键。"""
    if role == "dimension_mapper":
        string = {"type": "string"}
        dimension = {"type": "string", "enum": [r["dimension_id"] for r in data["catalog"]]}
        return obj({"results": arr(obj({
            "fact_id": {"type": "string", "enum": [r["fact_id"] for r in data["facts"]]},
            "matches": arr(obj({"dimension_id": dimension, "reason": string, "evidence_ids": arr(string)})),
            "deferred": arr(obj({"dimension_id": dimension, "reason": string})),
            "reason": string,
        }))})
    if role.startswith("source_") or role == "history_compare":
        if data.get("flow_version") in ("generic-flow-v1", "generic-flow-v2", "generic-flow-v3-relations", "fact-driven-v1"):
            from agent_flow_schema import schema
            return schema(role, data)
        if data.get("policy") == "source-maintenance-v4":
            from agent_v4_schema import schema
            return schema(role, data)
        from agent_v3_schema import schema
        return schema(role, data)
    string = {"type": "string"}
    skill = {"type": "string", "enum": list(SKILL_NAMES)}
    if role == "reviewer":
        from agent_review import CHECKS
        return obj({"decision": {"type": "string", "enum": ["approve", "revise", "defer", "reject"]},
                    "checks": obj({key: {"type": "string", "enum": ["pass", "fail", "uncertain"]}
                                   for key in CHECKS}),
                    "reason": string, "evidence_ids": arr(string)})
    if role == "router":
        return obj({"skills": arr(skill), "reason": string})
    if role == "revision_router":
        return obj({"dimension_ids": arr({"type": "string", "enum":
                    [d for entry in data["catalog"] for d in entry["dimensions"]]})})
    if role == "planner":
        return obj({"action": {"type": "string", "enum": ["execute_skill", "search_history", "finish"]},
                    "skill_id": {"type": "string", "enum": [""] + list(SKILL_NAMES)},
                    "query": string, "reason": string})
    if role.startswith("extractor:"):
        dimension = {"type": "string", "enum": [d["dimension_id"] for d in data["skill_rules"]]}
        evidence = obj({"evidence_id": string, "quote": string})
        candidate = obj({"dimension_id": dimension, "subject": string, "claim": string,
                         "scope": string, "kind": {"type": "string", "enum": ["self_report", "observation", "hypothesis"]},
                         "keywords": arr(string), "evidence": arr(evidence),
                         "operation": {"type": "string", "enum": ["new", "reinforce", "supersede", "conflict", "control"]},
                         "target_memory_id": string})
        return obj({"candidates": arr(candidate), "notes": string})
    if role == "auditor":
        dimensions = [d for entry in data["catalog"] for d in entry["dimensions"]]
        dimension = {"type": "string", "enum": dimensions}
        return obj({"approved_candidate_ids": arr(string),
                    "rejected": arr(obj({"candidate_id": string, "reason": string})),
                    "followup_skills": arr(skill),
                    "dimension_statuses": obj({d: {"type": "string", "enum":
                        ["with_candidate", "no_evidence", "insufficient", "missing_source", "not_assessed"]}
                        for d in dimensions}),
                    "notes": string})
    return None
