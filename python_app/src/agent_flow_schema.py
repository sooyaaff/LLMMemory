"""通用流程模型协议；标签判断和来源判断分开。"""
from agent_schema import obj, arr
from agent_v4_schema import schema as legacy_schema


def schema(role: str, data: dict) -> dict:
    s = {"type": "string"}
    if role == "source_router" and data.get("flow_version") == "fact-driven-v1":
        return obj({"skills": arr(s), "reason": s})
    if role == "source_control":
        result = legacy_schema("source_router", data)
        ref = obj(dict(evidence_id=s, speaker={"type":"string","enum":["user"]},
                       start={"type":"null"}, end={"type":"null"}))
        signal = result["properties"]["signals"]["items"]
        signal["properties"]["evidence"] = arr(ref)
        signal["properties"].update(switch_basis={"type":"string","enum":["none","handoff","self_identification"]},
                                    switch_reference={"anyOf":[ref,{"type":"null"}]})
        signal["required"] = list(signal["properties"])
        return obj({"signals": arr(signal)})
    if role == 'source_relations':
        if data.get("flow_version") == "fact-driven-v1":
            relations = ["equivalent", "related_but_independent", "changed_with_direction", "contradictory", "insufficient_evidence"]
        else:
            from agent_relations import RELATIONS
            relations = list(RELATIONS)
        ref=obj(dict(evidence_id=s,speaker={'type':'string','enum':['user']},start={'type':['integer','null']},end={'type':['integer','null']}))
        if data.get("flow_version") == "fact-driven-v1":
            return obj({'results':arr(obj(dict(pair_id=s,relation={'type':'string','enum':relations},reason=s,basis_refs=arr(ref),
                                              direction={'type':'string','enum':['none','left_replaces_right','right_replaces_left']})))})
        return obj({'results':arr(obj(dict(pair_id=s,relation={'type':'string','enum':relations},reason=s,basis_refs=arr(ref))))})
    if role == "source_classifier":
        return obj({"results": arr(obj({"candidate_id": s, "dimension_ids": arr(s), "reason": s}))})
    if role == "source_consistency":
        return obj({"results": arr(obj({"pair_id": s, "relation": {"type":"string", "enum":["equivalent","distinct","incompatible","uncertain"]}, "reason": s}))})
    result = legacy_schema(role, data)
    ref = obj({"evidence_id": s, "speaker": {"type":"string","enum":["user","assistant"]},
               "start":{"type":["integer","null"]}, "end":{"type":["integer","null"]}})
    if role == "source_router":
        signal = result["properties"]["signals"]["items"]
        signal["properties"]["evidence"] = arr(ref)
        signal["properties"].update(switch_basis={"type":"string","enum":["none","handoff","self_identification"]},
                                    switch_reference={"anyOf":[ref,{"type":"null"}]})
        signal["required"] = list(signal["properties"])
    elif role == "history_compare":
        row = result["properties"]["results"]["items"]
        row["properties"].pop("basis")
        row["properties"]["basis_refs"] = arr(ref)
        row["required"] = list(row["properties"])
    if role.startswith("source_extractor:") or role == "source_repair":
        if role == "source_repair":
            candidate = result["properties"]["results"]["items"]["properties"]["candidates"]["items"]
        else:
            candidate = result["properties"]["candidates"]["items"]
        candidate["properties"]["evidence"] = arr(ref)
        for field in ("valid_from", "valid_to"):
            candidate["properties"][field] = {"type":"null"}
        candidate["properties"]["dimension_id"] = {"type":"string","enum":["UNCLASSIFIED"]}
        fact = candidate["properties"]["fact"]
        fact["properties"].pop("time_text")
        fact["properties"].pop("time_value")
        fact["properties"]["time_reference"] = {"anyOf":[ref,{"type":"null"}]}
        fact["required"] = list(fact["properties"])
        candidate["properties"]["structure"]["properties"]["steps"] = arr(obj({
            "kind":{"type":"string","enum":["attempt","hint","feedback","contribution","event","context"]}, "reference": ref}))
    return result
