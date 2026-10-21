"""单一记忆流程：提取、审核、维护与原子提交，不继承旧业务引擎。"""
from __future__ import annotations

import copy
import logging
import json
from collections import Counter

from agent_data import digest, exact_keys, SKILL_NAMES
from agent_context import windows, pages, setting, bisect_window, ContextCapacityError, compact_turn, size
from agent_contracts import (VERSION, CHECKS, FORBIDDEN, validate_candidate, validate_signal,
                               source_result, validate_comparison, verify_evidence, seal, ModelOutputError, normalize_signal, subject_id, explicit_control, NonMemoryControl, explicit_subject_switch)
from agent_maintenance import V4Store, relevant, apply_decision, suspend, applies, StaleHistory, storage_control
from agent_fact_policy import REVISION, same_fact, relation_guard, latest_candidates, comparable, deduplicate_candidates, retention
from agent_llm import TransientModelError
from agent_memory_structure import rebuild_summaries
from agent_source_consistency import base_preference_support
from agent_fact_policy import compatible_scope


from agent_flow_data import FLOW, PROCESSING_REVISION, materialize, materialize_references, compact_fact, content_hash, source_pairs, semantic_config, EvidenceReferenceError
from agent_prompts import COMMON, GENERIC_ROUTER, COMPARE, EXTRACT, REVIEW, CLASSIFY, CONSISTENCY, PHOTO_NOTICE
from agent_stage_runtime import StageRuntime
REPAIR = EXTRACT

def failure_details(exc):
    origin = exc
    while origin.__cause__ is not None:
        origin = origin.__cause__
    return dict(code=getattr(origin,"code",type(origin).__name__), reason=str(exc),
                evidence_ids=[origin.evidence_id] if isinstance(origin,EvidenceReferenceError) else [])

class GenericPipeline:
    def extract_page(self, w, wid, skill, rules, signals):
        """先校验批次封套，再逐候选整理；只重试坏候选，保存其原始输出。"""
        role = "source_extractor:" + skill
        data = {"session": w, "skill_rules": rules, "signals": signals}
        def envelope(r):
            exact_keys(r, {"candidates", "notes"})
            if not isinstance(r["candidates"], list) or not isinstance(r["notes"], str):
                raise ValueError("提取结果类型非法")
        def normalize(raw):
            value = copy.deepcopy(raw)
            if isinstance(value, dict) and value.get("dimension_id") in FORBIDDEN:
                value["dimension_id"] = "UNCLASSIFIED"
            c = self.normalize_candidate(value, skill, w, wid)
            if c["dimension_id"] != "UNCLASSIFIED" and c["dimension_id"] not in {x["dimension_id"] for x in rules}:
                raise ValueError("候选不属于本页规则")
            return c
        batch_unit = digest([role, wid, skill, "envelope"])
        result = self.ask(role, EXTRACT, data, envelope, validation_label="envelope_valid", unit_key=batch_unit)
        if self.a.config.get('extraction_execution') == 'isolated-skill-batch-repair-v1':
            from agent_extraction_repair import repair_rows
            return repair_rows(self,w,wid,skill,result['candidates'],normalize,batch_unit)
        good = []
        for index, raw in enumerate(result["candidates"]):
            try:
                good.append(normalize(raw))
            except (ValueError, TypeError, KeyError) as exc:
                failure = str(exc)
                attempts = [dict(attempt=1, **failure_details(exc))]
                repaired = None
                row_unit = digest([batch_unit, index, raw])
                self.runtime.attempts[row_unit] = max(self.runtime.attempts[row_unit], self.runtime.attempts[batch_unit])
                retries = self.a.config.get("stage_max_attempts", 3) - 1
                for attempt in range(retries):
                    def check(r):
                        envelope(r)
                        for value in r["candidates"]:
                            normalize(value)
                    try:
                        row = self.ask(role, EXTRACT + "\n仅修订candidate_to_fix，其他合格候选不得重新提取；允许返回空数组表示不生成记忆。",
                                       dict(data, candidate_to_fix=raw, validation_feedback=failure, row_index=index, row_attempt=attempt + 1),
                                       check, attempt_limit=1, unit_key=row_unit)
                        repaired = [normalize(value) for value in row["candidates"]]
                        break
                    except (ModelOutputError, ContextCapacityError, TransientModelError) as err:
                        failure = str(err)
                        attempts.append(dict(attempt=attempt+2, **failure_details(err)))
                if repaired is not None:
                    good.extend(repaired)
                    self.stats["extraction_rows_repaired"] += 1
                    self.record_validation(role + ":row", 1, "repaired", f"row={index}")
                else:
                    issue = {"issue_id": digest([wid, skill, index, raw])[:24], "source_window": wid,
                             "skill": skill, "row_index": index, "raw_candidate": raw,
                             "reason": failure, "status": "pending_extraction",
                             "code": attempts[-1]["code"], "evidence_ids": attempts[-1]["evidence_ids"], "attempt_errors": attempts}
                    self.extraction_issues.append(issue)
                    self.stats["extraction_rows_pending"] += 1
                    self.record_validation(role + ":row", retries + 1, "pending_extraction", failure)
                    logging.warning("Session=%s Skill=%s 候选=%s整理未完成；保留问题记录，其他候选继续：%s", self.s["session_id"], skill, index, failure)
        return good

    def pack(self, candidates, extra=()):
        ids = ({e["evidence_id"] for c in candidates for e in c["evidence"]}
               | {s["evidence_id"] for c in candidates for s in c.get("structure", {}).get("steps", [])} | set(extra))
        turns = self.s["turns"]
        positions = [i for i, t in enumerate(turns) if t["evidence_id"] in ids]
        if ids - {t["evidence_id"] for t in turns}:
            raise ValueError("补充上下文引用不存在")
        neighbors = {j for i in positions for j in range(max(0, i - 1), min(len(turns), i + 2))}
        return [compact_turn(turns[i]) for i in sorted(neighbors)]

    def proof(self, c, result, pack, signals):
        return seal(dict(result, policy=VERSION, candidate_version=c["candidate_version"], candidate_hash=digest(c),
            source_content_hash=content_hash(c),
            device_id=self.s["device_id"], session_id=self.s["session_id"], input_hash=self.s["input_hash"],
            evidence_pack=pack, context_hash=digest(pack), signals_hash=digest(signals), rule_hash=digest(self.a.registry.dimensions.get(c["dimension_id"], {}))))

    def pending(self, c, reason, signals, decision="pending"):
        return self.proof(c, {"candidate_id": c["candidate_id"], "decision": decision,
            "checks": {k: "uncertain" for k in CHECKS}, "reason": reason,
            "evidence_ids": [], "needs_context": False, "context_ids": [], "repair": {"kind": "none", "reason": ""}}, [], signals)

    def _review_core(self, batch, signals, supplement=False, extras=()):
        pack = self.pack(batch, extras)
        data = {"candidates": batch, "evidence_pack": pack, "signals": signals,
                "candidate_rules": [self.a.registry.dimensions.get(d, {"dimension_id": "UNCLASSIFIED", "rules": "未分类候选保留待整理"}) for d in sorted({d for c in batch for d in c.get("dimension_ids", [c["dimension_id"]])})],
                "supplement_used": supplement}
        try:
            result = self.ask("source_reviewer", REVIEW, data,
                lambda r: source_result(r, batch, {t["evidence_id"] for t in pack}, normalize=True))
        except ContextCapacityError:
            if len(batch) > 1:
                n = len(batch) // 2
                return self.review(batch[:n], signals, supplement, extras) + self.review(batch[n:], signals, supplement, extras)
            self.task("source_reviewer", [batch[0]["candidate_id"]], "来源证据包超过请求容量", code="context_capacity")
            return [self.pending(batch[0], "来源证据包超过请求容量，未裁剪证据；待调整容量后复核", signals)]
        except (TransientModelError, ModelOutputError) as exc:
            self.task("source_reviewer", [c["candidate_id"] for c in batch], exc)
            self.stats["source_review_incomplete_batches"] += 1
            logging.warning("Session=%s 来源审核批次未完成，仅本批待处理：%s", self.s["session_id"], exc)
            return [self.pending(c, "来源审核服务/协议未完成：" + str(exc), signals) for c in batch]
        byid = {c["candidate_id"]: c for c in batch}
        proofs, retry = [], []
        requested = set()
        all_ids = {t["evidence_id"] for t in self.s["turns"]}
        for r in result["results"]:
            c = byid[r["candidate_id"]]
            if r["needs_context"] and not supplement and r["context_ids"] and set(r["context_ids"]).issubset(all_ids):
                retry.append(c)
                requested.update(r["context_ids"])
            else:
                if r["needs_context"]:
                    r = dict(r, decision="pending", needs_context=False, reason=r["reason"] + "；补充后仍不足或请求的上下文不存在")
                proofs.append(self.proof(c, r, pack, signals))
        if retry:
            proofs.extend(self.review(retry, signals, True, requested))
        return proofs

    def _compare_core(self, c, history, signals, state):
        base = {"candidate_id": c["candidate_id"], "action": "new", "targets": [], "reason": "完整相关历史为空", "basis": []}
        if not history:
            return base
        weaker = [m for m in history if comparable(c, m) and compatible_scope(c, m) and base_preference_support(c, m)]
        if len(history) == len(weaker) == 1:
            return dict(base, action="reinforce", targets=[{"memory_id": weaker[0]["memory_id"], "revision": weaker[0]["revision"]}],
                        support_kind="base_only", reason="支持已有基础喜好，不强化或改变最喜欢的程度")
        # 完全相同结构免语义调用，但仍检查其他历史，不能漏掉同属性冲突方。
        matches, remaining = [], []
        fields = ("dimension_id", "subject_key", "attribute_key", "claim", "scope", "kind", "valid_from", "valid_to")
        for m in history:
            if same_fact(c, m) or not c.get("fact") and not m.get("fact") and all(c.get(k) == m.get(k) for k in fields):
                matches.append(dict(base, action="reinforce", targets=[{"memory_id": m["memory_id"], "revision": m["revision"]}], reason="主体、内容及范围结构完全一致"))
            else:
                remaining.append(self.history_record(m))
        # 已有冲突的一方被重复支持，只强化该方；不按支持次数解决冲突。
        if len(matches) == 1 and any(m["memory_id"] == matches[0]["targets"][0]["memory_id"] and m["state"] == "conflicted" for m in history):
            return matches[0]
        try:
            for page in self.history_pages(remaining) if remaining else []:
                data = {"candidates": [c], "history": page, "signals": signals,
                        "group_context": [g for g in state["groups"].values() if g["status"] == "open" and any(t["memory_id"] in {m["memory_id"] for m in page} for t in g["members"])]}
                matches.extend(self.compare_page(data,c,page))
        except ContextCapacityError:
            self.task("history_compare", [c["candidate_id"]], "历史分页/候选超过容量", code="context_capacity")
            return dict(base, action="pending", reason="历史分页/候选超过容量，完整比对未完成")
        except (ValueError, TypeError, KeyError, TransientModelError) as exc:
            self.task("history_compare", [c["candidate_id"]], exc)
            return dict(base, action="pending", reason="历史返回未通过契约校验，未完成比较：" + str(exc))
        meaningful = [r for r in matches if r["action"] != "new"]
        if not meaningful:
            return dict(base, reason="已比对全部相关历史页，无相同或互斥信息")
        kinds = {r["action"] for r in meaningful}
        # 可并存关系无需改变历史，不阻断另一个明确关系。
        decisive = [r for r in meaningful if r["action"] != "coexist"] or meaningful
        kinds = {r["action"] for r in decisive}
        if "pending" in kinds or len(kinds) != 1:
            return dict(base, action="pending", reason="历史各页维护关系不一致，保留候选，不自动覆盖或创建冲突")
        action = decisive[0]["action"]
        targets = list({(t["memory_id"], t["revision"]): t for r in decisive for t in r["targets"]}.values())
        return dict(base, action=action, targets=targets, semantic_verified=all(r.get("semantic_verified") or r["reason"] == "主体、内容及范围结构完全一致" for r in decisive), reason="；".join(r["reason"] for r in decisive),
                    basis=list({(e["evidence_id"], e["quote"]): e for r in decisive for e in r["basis"]}.values()))

    def _compare_rows_core(self, data, histories):
        """严格核对批次ID；每条分别校验与定向重试，保留无关合格结果。"""
        candidates = data["candidates"]
        ready, pending, item_keys = [], [], {}
        for c in candidates:
            dependency = dict(candidate=compact_fact(c), history=[self.history_record(m) for m in histories[c["candidate_id"]]],
                              signals=data["signals"], group_context=data.get("group_context", []))
            def check_cached(row):
                expected = dict(row)
                marker = expected.pop("semantic_verified", None)
                if type(marker) is not bool:
                    raise ValueError("历史缓存缺少程序校验标记")
                validate_comparison({"results":[expected]}, [c], histories[c["candidate_id"]], self.s)
            key, cached = self.runtime.load_item("history_compare", COMPARE, dependency, check_cached)
            item_keys[c["candidate_id"]] = (key, dependency)
            if cached is None:
                pending.append(c)
            else:
                ready.append(cached)
        if not pending:
            return {"results":ready}
        candidates = pending
        data = dict(data,candidates=candidates)
        def envelope(r):
            exact_keys(r, {"results"})
            if not isinstance(r["results"], list) or any(not isinstance(x, dict) or not isinstance(x.get("candidate_id"), str) for x in r["results"]):
                raise ValueError("维护结果结构非法")
            if sorted(x["candidate_id"] for x in r["results"]) != sorted(c["candidate_id"] for c in candidates):
                raise ValueError("历史比对缺失、重复或额外候选")
        batch_unit = digest(["history_compare", data])
        result = self.ask("history_compare", COMPARE, data, envelope, validation_label="batch_ids_valid", unit_key=batch_unit)
        rows = []
        byid = {r["candidate_id"]: r for r in result["results"]}
        for c in candidates:
            row = byid[c["candidate_id"]]
            history = histories[c["candidate_id"]]
            def check(r):
                validate_comparison(r, [c], history, self.s)
            try:
                check({"results": [row]})
            except (ValueError, TypeError, KeyError) as exc:
                self.stats["history_row_retries"] += 1
                self.record_validation("history_compare:row", 1, "retrying", c["candidate_id"] + ": " + str(exc))
                try:
                    row_unit = digest([batch_unit, c["candidate_id"]])
                    self.runtime.attempts[row_unit] = max(self.runtime.attempts[row_unit], self.runtime.attempts[batch_unit])
                    retries = self.a.config.get("stage_max_attempts", 3) - 1
                    if retries <= 0:
                        raise ModelOutputError(str(exc)) from exc
                    row = self.ask("history_compare", COMPARE,
                        dict(data, candidates=[c], history=history, validation_feedback=str(exc)), check, attempt_limit=retries, unit_key=row_unit)["results"][0]
                except (ValueError, TypeError, KeyError, ContextCapacityError, TransientModelError) as failure:
                    self.task("history_compare", [c["candidate_id"]], failure)
                    row = {"candidate_id": c["candidate_id"], "action": "pending", "targets": [], "basis": [],
                           "reason": "维护协议失败，仅本候选待处理：" + str(failure)}
                    self.stats["history_row_failures"] += 1
                    self.record_validation("history_compare:row", 1, "failed", c["candidate_id"] + ": " + str(failure))
            else:
                self.record_validation("history_compare:row", 1, "valid", c["candidate_id"])
            # 标记只能由程序在协议校验后产生，模型不能通过额外字段注入。
            row = dict(row, semantic_verified=row["action"] != "pending")
            rows.append(row)
            if not any(t["stage"] == "history_compare" and c["candidate_id"] in t["candidate_ids"] for t in self.retry_tasks):
                key, dependency = item_keys[c["candidate_id"]]
                self.runtime.save_item(key,"history_compare",dependency,row)
        return {"results": ready + rows}

    def repair_once(self, candidates, proofs, signals, limit):
        """仅整理有明确建议的父候选，最多一轮；父子各保留独立凭证。"""
        from agent_source_repair import repair
        return repair(self, candidates, proofs, signals, limit)

    def compare_batch(self, candidates, state, signals):
        """互不修改同一属性的候选可同批比较；有关联的候选放到下一波。"""
        decisions, semantic, histories = {}, [], {}
        fields = ("dimension_id", "subject_key", "attribute_key", "claim", "scope", "kind", "valid_from", "valid_to")
        for c in candidates:
            history = relevant(state, c)
            exact = [m for m in history if same_fact(c, m) or not c.get("fact") and not m.get("fact") and all(c.get(k) == m.get(k) for k in fields)]
            if not history or len(history) == 1 and base_preference_support(c, history[0]) or len(exact) == 1 and (len(history) == 1 or exact[0]["state"] == "conflicted"):
                decisions[c["candidate_id"]] = self.compare(c, history, signals, state)
            else:
                semantic.append(c)
                histories[c["candidate_id"]] = history
        if not semantic:
            return decisions
        keys = ("memory_id", "revision", "device_id", "dimension_id", "subject_key", "attribute_key", "claim", "scope", "kind", "valid_from", "valid_to", "last_seen", "state", "fact", "origin_session_id", "source_order", "event", "evidence", "retention", "assertion_mode", "structure")
        union = list({m["memory_id"]: self.history_record(m) for h in histories.values() for m in h}.values())
        if size(union) <= self.history_page_budget():
            try:
                result = self.compare_rows(
                    {"candidates": semantic, "history": union, "signals": signals,
                     "group_context": [g for g in state["groups"].values() if g["status"] == "open" and any(t["memory_id"] in {m["memory_id"] for m in union} for t in g["members"])]}, histories)
                decisions.update({r["candidate_id"]: r for r in result["results"]})
                return decisions
            except ContextCapacityError:
                pass
            except (ValueError, TypeError, KeyError, TransientModelError) as exc:
                self.task("history_compare", [c["candidate_id"] for c in semantic], exc)
                for c in semantic:
                    decisions[c["candidate_id"]] = {"candidate_id": c["candidate_id"], "action": "pending", "targets": [], "basis": [], "reason": "历史批量返回无效：" + str(exc)}
                return decisions
        # 超出单批容量时对完整相关历史分页；不能截取 top-N 后宣称没有冲突。
        for c in semantic:
            decisions[c["candidate_id"]] = self.compare(c, histories[c["candidate_id"]], signals, state)
        return decisions

    def _maintain_core(self, candidates, proofs, signals, coverage):
        source = {p["candidate_id"]: p for p in proofs}
        for attempt in range(3):
            state = self.store.snapshot(self.s["device_id"])
            base_hash, decisions = digest(state), []
            remaining = list(candidates)
            while remaining:
                wave = []
                for c in remaining:
                    if len(wave) >= self.a.config.get("source_review_batch_size", 8):
                        break
                    overlap = {m["memory_id"] for m in relevant(state, c)}
                    if not any(comparable(c, x)
                               or overlap.intersection(m["memory_id"] for m in relevant(state, x)) for x in wave):
                        wave.append(c)
                eligible = [c for c in wave if c.get("retention") != "discard" and source[c["candidate_id"]]["decision"] == "pass" and source[c["candidate_id"]]["repair"]["kind"] == "none" and storage_control(c, signals)["status"] != "blocked"]
                compared = self.compare_batch(eligible, state, signals)
                for c in wave:
                    p = source[c["candidate_id"]]
                    history = relevant(state, c)
                    d = compared.get(c["candidate_id"], {"candidate_id": c["candidate_id"], "action": "pending", "targets": [], "basis": [], "reason": p["reason"]})
                    if p["repair"]["kind"] != "none":
                        d.update(action="pending", targets=[], basis=[], reason="内容需要整理：" + p["repair"]["reason"] + "；" + p["reason"])
                    if c.get("retention") == "discard":
                        d.update(action="pending", targets=[], basis=[], disposition="discarded", reason="不生成记忆：" + c["retention_reason"])
                    d = relation_guard(c, d, history)
                    if d["action"] == "supersede" and c.get("fact", {}).get("type") in ("event", "request") and not any(s["kind"] == "correction" and applies(s, c) for s in signals):
                        d.update(action="coexist", basis=[], reason="事件变化按先后分别记录，没有明确更正，不替代先前事件")
                    control = storage_control(c, signals)
                    if control["status"] == "blocked":
                        d.update(action="pending", targets=[], basis=[], reason=control["reason"])
                    d["storage_control"] = control
                    d.update(history_hash=digest(history), source_proof=p["proof_hash"])
                    trial = copy.deepcopy(state)
                    try:
                        apply_decision(trial, c, d, p, self.s, signals)
                    except ValueError as exc:
                        d.update(action="pending", targets=[], basis=[], reason=str(exc))
                    else:
                        state = trial
                    decisions.append(d)
                    remaining.remove(c)
            suspend(state, signals)
            summary_config = {"days": self.a.config.get("summary_window_days", 30),
                              "min_sessions": self.a.config.get("summary_min_sessions", 2)}
            rebuild_summaries(state, self.s["end"], **summary_config)
            plan = seal({"policy": VERSION, "input_hash": self.s["input_hash"], "base_hash": base_hash,
                         "signals_hash": digest(signals), "signals": signals, "decisions": decisions,
                         "summary_config": summary_config, "extraction_issues": self.extraction_issues, "result_hash": digest(state)})
            try:
                return self.commit_plan(candidates, proofs, plan, signals, coverage), decisions
            except StaleHistory:
                self.stats["history_revision_retries"] += 1
                if attempt == 2:
                    raise

    def _run_core(self):
        result_key = digest([VERSION, "session_result", self.s["input_hash"], self.config_hash])
        cached = self.a.store.load_work(result_key)
        if cached and self.store.receipt(self.s):
            return cached
        receipt = self.store.receipt(self.s)
        if receipt:
            # 事务成功而报告检查点写入前崩溃：直接从原子回执恢复，绝不再次调用模型。
            from agent_recovery import committed_result
            result = committed_result(self.store.db, self.s)
            if result is None:
                raise ValueError('事务回执存在但Session缺失，禁止重新提交')
            self.a.store.save_work(result_key, self.s, result)
            return result
        candidates, signals = {}, {}
        count = 0
        char_budget = self.a.config.get("session_chunk_chars")
        for w in windows(self.s, char_budget if char_budget is not None else setting(self.a.config, "session_chunk_tokens"),
                         unit="chars" if char_budget is not None else "bytes"):
            cs, ss = self.extract_window(w)
            count += 1
            for c in cs:
                cid = c["candidate_id"]
                if cid in candidates:
                    c["source_windows"] = sorted(set(c["source_windows"] + candidates[cid]["source_windows"]))
                candidates[cid] = c
            for s in ss:
                signals[digest(s)] = s
        candidates, signals = list(candidates.values()), list(signals.values())
        # 全Session收集后处理跨窗口的明确换人，不让后续窗口回到默认儿童主体。
        positions = {t["evidence_id"]: i for i, t in enumerate(self.s["turns"])}
        switches = sorted((s for s in signals if s["kind"] == "subject_switch"),
                          key=lambda s: min(positions[e["evidence_id"]] for e in s["evidence"]))
        for c in candidates:
            if c["subject_role"] != "current_user":
                continue
            at = min(positions[e["evidence_id"]] for e in c["evidence"])
            prior = [s for s in switches if min(positions[e["evidence_id"]] for e in s["evidence"]) <= at]
            if prior:
                switch = prior[-1]
                if switch["subject_key"] == subject_id(self.s["device_id"]):
                    if c["assertion_mode"] in ("request", "temporary") or c.get("fact",{}).get("type") in ("request","event"):
                        continue  # 未识别换人只阻断依赖稳定身份的属性，不污染当次需求。
                    c.update(subject_role="unresolved", subject_key=None)
                else:
                    c.update(subject_role="other", subject_key=switch["subject_key"], subject=switch["target_text"] or "已明确切换的发言者")
                c["candidate_id"] = digest([VERSION, "speaker", c])[:20]
        before = len(candidates)
        candidates = deduplicate_candidates(candidates)
        self.stats["session_exact_duplicates"] += before - len(candidates)
        proofs = []
        grouped = {}
        for c in candidates:
            if c.get("retention") == "discard":
                proofs.append(self.pending(c, "价值筛选，不执行来源审核：" + c["retention_reason"], signals))
                self.stats["discarded_low_value"] += 1
                continue
            try:
                verify_evidence(c["evidence"], self.s)
            except ValueError as exc:
                proofs.append(self.pending(c, str(exc), signals, "reject"))
                continue
            grouped.setdefault(tuple(c["source_windows"]), []).append(c)
        limit = self.a.config.get("source_review_batch_size", 8)
        if type(limit) is not int or limit < 1:
            raise ValueError("source_review_batch_size必须为正整数")
        for group in grouped.values():
            for i in range(0, len(group), limit):
                batch = group[i:i + limit]
                related = [s for s in signals if s["kind"] == "subject_switch" or any(applies(s, c) for c in batch)]
                proofs.extend(self.review(batch, related))
        candidates, proofs = self.repair_once(candidates, proofs, signals, limit)
        # 完整审核轨迹仍留检查点；当前业务结果仅保留成功产生的新版本。
        candidates = latest_candidates(candidates)
        leaf_ids = {c["candidate_id"] for c in candidates}
        proofs = [p for p in proofs if p["candidate_id"] in leaf_ids]
        candidates.sort(key=lambda c: min(positions[e["evidence_id"]] for e in c["evidence"]))
        coverage = {"with_candidate": sorted({c["dimension_id"] for c in candidates}), "no_evidence": [],
                    "insufficient": [], "missing_source": sorted(FORBIDDEN),
                    "not_assessed": sorted(set(self.a.registry.dimensions) - FORBIDDEN - {c["dimension_id"] for c in candidates})}
        receipt, decisions = self.maintain(candidates, proofs, signals, coverage)
        self.stats["history_reinforcements"] += receipt["counts"]["reinforce"]
        self.stats["history_merged_records"] += receipt.get("merged_records", 0)
        for c in candidates:
            self.stats["retention:" + c.get("retention", "memory")] += 1
        logging.info("Session=%s 事实归并/价值统计=%s", self.s["session_id"], dict(self.stats))
        admitted = {m["candidate_id"]: m for m in receipt["mapping"]}
        reviews = [dict(p, source_decision=p["decision"], decision={"pass": "approve", "pending": "defer", "reject": "reject"}[p["decision"]],
                        candidate=next(c for c in candidates if c["candidate_id"] == p["candidate_id"])) for p in proofs]
        result = {"session_id": self.s["session_id"], "turns": len(self.s["turns"]), "status": "completed", "policy": VERSION,
                  "candidates": candidates, "approved": [dict(c, operation=admitted[c["candidate_id"]]["operation"]) for c in candidates if c["candidate_id"] in admitted],
                  "reviews": reviews, "maintenance": decisions, "signals": signals, "coverage": coverage,
                  "commit": receipt, "window_count": count, "batch_metrics": dict(self.stats), "implementation_revision": REVISION,
                  "processing_revision": self.processing_revision, "cache_invalidations": self.runtime.invalidations,
                  "extraction_issues": self.extraction_issues, "needs_attention": bool(self.extraction_issues),
                  "data_completeness": self.s.get("source_kind", "dialogue") + "_only"}
        self.a.store.save_work(result_key, self.s, result)
        logging.info("Session=%s 来源审核批次=%s 历史比对批次=%s 统一关系批次=%s 候选=%s 维护=%s", self.s["session_id"],
                     self.stats["source_reviewer"], self.stats["history_compare"], self.stats['source_relations'], len(candidates), receipt["counts"])
        return result

    def __init__(self, agent, session):
        self.a, self.s = agent, session
        self.flow = agent.config.get('flow_version', FLOW)
        self.store = V4Store(agent.store)
        self.stats = Counter()
        self.extraction_issues = []
        self.runtime = StageRuntime(self)
        from agent_fact_policy import REVISION
        self.config_hash=digest([semantic_config(agent.config),REVISION,FLOW])
        self.retry_tasks = []
        self.shared = []
        self.processing_revision = 'bge-relations-2026-10-15-1' if self.flow == 'generic-flow-v3-relations' else PROCESSING_REVISION
        if agent.config.get('relation_policy') == 'sparse-relations-v2':
            self.processing_revision = 'bge-sparse-relations-2026-10-16-1'
        if agent.config.get('extraction_execution') == 'isolated-skill-batch-repair-v1':
            self.processing_revision = 'isolated-extraction-2026-10-16-1'
        if self.flow == 'generic-flow-v3-relations':
            self.processing_revision = 'resume-source-identity-2026-10-17-1'
        self.consistency_audit = []
        existing = self.store.db.execute("SELECT data FROM v3_plans LIMIT 1").fetchone()
        if existing:
            import json
            if json.loads(existing[0]).get("flow_version") != self.flow:
                raise ValueError("通用流程禁止续写旧流程输出；请使用独立目录")
            if bool(json.loads(existing[0]).get('batch_recall_manifest')) != (agent.config.get('relation_policy') == 'sparse-relations-v2') and self.flow == 'generic-flow-v3-relations':
                raise ValueError('关系召回策略变化，请使用独立目录；禁止混合续写旧关系计划')

    def task(self, stage, ids, reason, code=None, evidence_ids=()):
        code = code or ("transient_service" if isinstance(reason,TransientModelError) else "context_capacity" if isinstance(reason,ContextCapacityError) else "model_protocol")
        row = dict(task_id=digest([stage, sorted(ids), code])[:24], stage=stage,
                   session_id=self.s["session_id"], candidate_ids=sorted(ids), evidence_ids=sorted(evidence_ids),
                   status="retryable", retryable=True, code=code, reason=str(reason))
        if not any(t["task_id"] == row["task_id"] for t in self.retry_tasks): self.retry_tasks.append(row)

    def ask(self, role, prompt, data, validate, **kwargs):
        data = dict(data, flow_version=self.flow)
        converting = role.startswith("source_extractor:") or role == "source_repair"
        if converting:
            prompt = EXTRACT + ("\n仅修订给出的parents或candidate_to_fix，不重做合格内容。" if role == "source_repair" or "candidate_to_fix" in data else "")
            if role == "source_repair":
                prompt += "\n返回results，每项parent_candidate_id,candidates,reason。"
            data = dict(data, skill_rules=[], strategy=role.split(":")[-1], shared_facts=[compact_fact(c) for c in self.shared])
            if data.get('repair_mode') == 'extraction_rows_v1':
                from agent_extraction_repair import PROMPT as ROW_REPAIR
                prompt = EXTRACT + '\n' + ROW_REPAIR
                data = dict(data,strategy=data['repair_skill'],shared_facts=[])
            elif data.get('repair_mode') == 'source_parents_v2':
                from agent_source_repair import PROMPT as SOURCE_REPAIR
                prompt = EXTRACT + '\n' + SOURCE_REPAIR
                data = dict(data, shared_facts=[])
        elif role == "source_reviewer":
            prompt = self.source_review_prompt()
            data = {k:v for k,v in data.items() if k != "candidate_rules"}
        elif role == "history_compare":
            pack = self.pack(data["candidates"], {e["evidence_id"] for s in data["signals"] for e in s["evidence"]})
            data = dict(data, candidates=[compact_fact(c) for c in data["candidates"]],
                        history=[compact_fact(m) if "context" not in m else m for m in data["history"]], evidence_pack=pack)
        def convert(r):
            if role in ("source_router", "history_compare"):
                r = copy.deepcopy(r)
                session = data.get("session", dict(self.s, turns=data.get("evidence_pack", [])))
                if role == "source_router":
                    if self.flow == "fact-driven-v1":
                        # 新流程的控制线索来自独立 source_control，不接受路由附带的线索。
                        r["signals"] = []
                    else:
                        for signal in r.get("signals", []):
                            signal["evidence"] = materialize_references(signal["evidence"], session)
                            ref = signal.pop("switch_reference")
                            signal["switch_evidence"] = materialize_references([ref], session)[0] if ref is not None else None
                else:
                    for row in r.get("results", []):
                        if "basis" in row:
                            raise ValueError("模型必须返回basis_refs，不得复制basis引文")
                        row["basis"] = materialize_references(row.pop("basis_refs"), session)
            if role == "source_reviewer":
                r=copy.deepcopy(r)
                for row in r.get("results",[]):
                    if row.get("decision")=="pass" and row.get("repair",{}).get("kind")=="reclassify" and all(row.get("checks",{}).get(k)=="pass" for k in ("evidence","semantics","subject")):
                        row["repair"]={"kind":"none","reason":"产品标签交独立分类处理"}
            if not converting: return r
            r = copy.deepcopy(r)
            session = data.get("session", dict(self.s, turns=data.get("evidence_pack", self.s["turns"])))
            groups = r.get("results", []) if role == "source_repair" else [r]
            for group in groups:
                converted = []
                for raw in group.get("candidates", []):
                    # 单条定位错误由提取的逐条隔离处理，不丢失同批有效内容。
                    try:
                        converted.append(materialize(raw, session))
                    except (ValueError, TypeError, KeyError, IndexError) as exc:
                        error = dict(code=getattr(exc, "code", "invalid_reference"),
                                     evidence_id=getattr(exc, "evidence_id", ""), reason=str(exc))
                        converted.append(dict(raw, reference_error=error) if isinstance(raw,dict) else raw)
                group["candidates"] = converted
            return r
        def check_raw(r):
            converted = convert(r)
            validate(converted)
            # 审核的保守归一必须作用于返回和缓存对象，而不只修改临时副本。
            if role == "source_reviewer":
                r.clear(); r.update(converted)
        result = self.runtime.ask(role, prompt, data, check_raw, **kwargs)
        converted = convert(result)
        validate(converted)
        return converted

    def normalize_candidate(self, raw, skill, session, wid):
        if isinstance(raw,dict) and "reference_error" in raw:
            error = raw["reference_error"]
            raise EvidenceReferenceError(error["code"], error["evidence_id"], error["reason"])
        c = validate_candidate(raw, skill, self.a.registry, session, wid)
        # 产品规则版本只属于分类，不影响事实来源身份和审核缓存。
        c["skill_version"] = digest([FLOW, EXTRACT, skill])
        if len(c["claim"]) > 600 or any(isinstance(v,str) and len(v)>400 for v in c.get("fact",{}).values()):
            raise ValueError("事实节点过长；按独立子事件整理，完整过程保留在步骤中")
        c["flow_version"] = self.flow
        c["fact_id"] = digest([session["device_id"], session["session_id"], content_hash(c)])[:24]
        return c

    def source_review_prompt(self):
        return REVIEW

    def review(self,batch,signals,supplement=False,extras=()):
        ready=[]; remaining=[]; keys={}
        for c in batch:
            from agent_stage_runtime import model_signature, stage_version
            key=digest([FLOW,"source-proof",stage_version("source_reviewer"),self.s["input_hash"],model_signature(self.a.config),getattr(self.a.client.client,"model",None),self.source_review_prompt(),c,self.pack([c],extras),signals,supplement,sorted(extras)])
            keys[c["candidate_id"]]=key
            cached=self.a.store.load_work(key)
            if cached and "proof" in cached:
                from agent_contracts import verify_source
                try:
                    verify_source(cached["proof"],c,self.s)
                except (ValueError, TypeError, KeyError) as exc:
                    self.runtime.invalidate(key,"source_reviewer",cached,exc)
                else:
                    ready.append(cached["proof"]); self.stats["source_proof:cached"]+=1
                    continue
            remaining.append(c)
        if remaining:
            proofs=self._review_core(remaining,signals,supplement,extras)
            for p in proofs:
                if p["evidence_pack"] and not any(p["candidate_id"] in t["candidate_ids"] and t["stage"] == "source_reviewer" for t in self.retry_tasks):
                    from agent_contracts import verify_source
                    verify_source(p, next(c for c in remaining if c["candidate_id"]==p["candidate_id"]), self.s)
                    self.a.store.save_work(keys[p["candidate_id"]],self.s,{"proof":p})
            ready.extend(proofs)
        return ready

    def extract_window(self, w, depth=0):
        def check(r):
            exact_keys(r, {"skills", "signals", "reason"})
            if not isinstance(r["skills"],list) or any(s not in SKILL_NAMES for s in r["skills"]) or len(set(r["skills"])) != len(r["skills"]) or not isinstance(r["reason"],str) or not isinstance(r["signals"],list):
                raise ValueError("路由结构非法")
            safe = []
            for signal in r["signals"]:
                try:
                    validate_signal(signal, w)
                except NonMemoryControl as exc:
                    self.record_validation("source_router:control", 1, "ignored_non_memory_command", str(exc))
                    continue
                if signal["kind"] == "subject_switch" and not explicit_subject_switch(signal):
                    self.record_validation("source_router:subject", 1, "ignored_unproven_switch", "没有明确交接依据，不改变候选主体")
                    continue
                safe.append(signal)
            r["signals"] = safe
            if w.get("source_kind") == "photo" and r["signals"]:
                raise ValueError("图片不能产生保存控制")
        try:
            route = self.ask("source_router", GENERIC_ROUTER, {"session":w,"catalog":[{"skill_id":k,"name":v} for k,v in SKILL_NAMES.items()]},check)
            found = []
            self.shared = []
            for skill in route["skills"]:
                if self.a.config.get('extraction_execution') == 'isolated-skill-batch-repair-v1':
                    cs = self.extract_skill_window(w,skill,route['signals'])
                else:
                    cs = self.extract_page(w,digest(w)[:20],skill,[],route["signals"])
                found.extend(cs); self.shared.extend(cs)
            return found,[normalize_signal(s,self.s) for s in route["signals"]]
        except ContextCapacityError:
            if depth >= setting(self.a.config,"context_split_retries"): raise
            a,b = bisect_window(w)
            ca,sa = self.extract_window(a,depth+1); cb,sb = self.extract_window(b,depth+1)
            return ca+cb,sa+sb

    def extract_skill_window(self, w, skill, signals, depth=0):
        """仅拆失败Skill；同级成功结果保留，不重跑路由。"""
        try:
            return self.extract_page(w,digest(w)[:20],skill,[],signals)
        except ContextCapacityError as exc:
            if depth >= setting(self.a.config,'context_split_retries'):
                self.task('source_extractor:'+skill,[],exc);return []
            try:
                a,b=bisect_window(w)
            except ContextCapacityError as terminal:
                self.task('source_extractor:'+skill,[],terminal);return []
            logging.warning('Session=%s Skill=%s 容量拆分depth=%s；不重跑其他Skill或路由',self.s['session_id'],skill,depth+1)
            self.stats['skill_capacity_splits']+=1
            def local(window):
                ids={t['evidence_id'] for t in window['turns']}
                return [s for s in signals if all(e['evidence_id'] in ids for e in s['evidence'])]
            return self.extract_skill_window(a,skill,local(a),depth+1)+self.extract_skill_window(b,skill,local(b),depth+1)
        except (ModelOutputError,TransientModelError) as exc:
            self.task('source_extractor:'+skill,[],exc);return []

    def history_record(self, m):
        return compact_fact(m)

    def history_pages(self, items):
        batch = []
        budget = self.history_page_budget()
        for item in items:
            if batch and size(batch+[item]) > budget:
                yield batch; batch=[]
            batch.append(item)
            if size(batch)>budget:
                yield batch; batch=[]  # 单节点不截断，最终发送容量门禁仍生效。
        if batch: yield batch

    def history_page_budget(self):
        # 显式字节单位；仍由完整请求容量门禁做最终检查，不截断历史。
        return self.a.config.get('history_page_bytes',setting(self.a.config,'history_context_tokens'))

    def compare_rows(self, data, histories):
        return self._compare_rows_core(data,histories)

    def compare_page(self, data, candidate, page):
        """按真实请求容量拆历史页，不跳过任何历史记录。"""
        try:
            return self.compare_rows(data,{candidate["candidate_id"]:page})["results"]
        except ContextCapacityError:
            if len(page) <= 1:
                raise
            middle=len(page)//2
            return (self.compare_page(dict(data,history=page[:middle]),candidate,page[:middle])
                    + self.compare_page(dict(data,history=page[middle:]),candidate,page[middle:]))

    def compare(self,c,history,signals,state):
        return self._compare_core(c,history,signals,state)

    def reconcile(self,candidates,proofs):
        byid = {p["candidate_id"]:p for p in proofs}
        pairs = {}
        for a,b in source_pairs(candidates):
            if a.get("retention")=="discard" or b.get("retention")=="discard": continue
            pair_id=digest([a["candidate_id"],b["candidate_id"]])[:24]
            pairs[pair_id]=(a,b)
        items=[dict(pair_id=pid,left=compact_fact(a),right=compact_fact(b)) for pid,(a,b) in pairs.items()]
        limit=self.a.config.get("source_review_batch_size",8)
        for start in range(0,len(items),limit):
            self.reconcile_batch(items[start:start+limit],pairs,byid)
        # 应用关系之前先完成全部裁决，避免前一对的修改污染后一对的输入。
        for _ in range(len(proofs)+1):
            changed=False
            for row in self.consistency_audit:
                pa,pb=[byid[i] for i in row["candidate_ids"]]
                disagree = any(p["decision"]!="pass" and any(p["checks"][k]!="pass" for k in ("evidence","semantics","subject")) for p in (pa,pb))
                if row["relation"] in ("incompatible","uncertain") or row["relation"]=="equivalent" and disagree:
                    for p in (pa,pb):
                        if p["decision"]=="pass":
                            p.update(decision="pending",checks=dict(p["checks"],semantics="uncertain"),reason="跨维度同源结论需确认："+row["reason"])
                            p["proof_hash"]=digest({k:v for k,v in p.items() if k!="proof_hash"})
                            changed=True
            if not changed: break

    def reconcile_batch(self,items,pairs,byid):
        pending, cached_rows, item_keys = [], [], {}
        for item in items:
            pair = pairs[item["pair_id"]]
            dependency = dict(pair=item, source_reviews=[{k:byid[c["candidate_id"]][k] for k in ("candidate_id","decision","checks","reason")} for c in pair], evidence_pack=self.pack(list(pair)))
            def check_cached(row):
                exact_keys(row,{"pair_id","relation","reason"})
                if row["pair_id"] != item["pair_id"] or row["relation"] not in ("equivalent","distinct","incompatible","uncertain") or not isinstance(row["reason"],str):
                    raise ValueError("同源关系缓存协议错误")
            key, cached = self.runtime.load_item("source_consistency", CONSISTENCY, dependency, check_cached)
            item_keys[item["pair_id"]] = (key, dependency)
            if cached is None:
                pending.append(item)
            else:
                cached_rows.append(dict(cached,candidate_ids=[c["candidate_id"] for c in pair]))
        self.consistency_audit.extend(cached_rows)
        items = pending
        if not items:
            return
        cs={c["candidate_id"]:c for item in items for c in pairs[item["pair_id"]]}
        data={"pairs":items,"source_reviews":[{k:byid[cid][k] for k in ("candidate_id","decision","checks","reason")} for cid in cs],"evidence_pack":self.pack(list(cs.values()))}
        def check(r):
            exact_keys(r,{"results"})
            if not isinstance(r["results"],list): raise ValueError("同源关系结果不完整")
            for row in r["results"]:
                exact_keys(row,{"pair_id","relation","reason"})
                if row["relation"] not in ("equivalent","distinct","incompatible","uncertain") or not isinstance(row["reason"],str): raise ValueError("同源关系协议错误")
            if sorted(x["pair_id"] for x in r["results"])!=sorted(x["pair_id"] for x in items): raise ValueError("同源关系ID遗漏、重复或额外")
        try:
            rows=self.ask("source_consistency",CONSISTENCY,data,check)["results"]
        except ContextCapacityError as exc:
            if len(items)>1:
                n=len(items)//2
                self.reconcile_batch(items[:n],pairs,byid); self.reconcile_batch(items[n:],pairs,byid)
            else: self.task("source_consistency",list(cs),exc)
            return
        except (ModelOutputError,TransientModelError) as exc:
            self.task("source_consistency",list(cs),exc); return
        for row in rows:
            key, dependency = item_keys[row["pair_id"]]
            self.runtime.save_item(key,"source_consistency",dependency,row)
            self.consistency_audit.append(dict(row,candidate_ids=[c["candidate_id"] for c in pairs[row["pair_id"]]]))

    def classify(self,candidates,proofs):
        rules=[dict(dimension_id=d["dimension_id"],name=d["name"],rules=d["rules"]) for d in self.a.registry.dimensions.values() if d["dimension_id"] not in FORBIDDEN]
        byid={p["candidate_id"]:p for p in proofs}
        eligible=[c for c in candidates if byid[c["candidate_id"]]["decision"]=="pass"]
        collected={c["candidate_id"]:{"labels":set(),"reasons":[]} for c in eligible}
        limit=self.a.config.get("source_review_batch_size",8)
        for start in range(0,len(eligible),limit):
            batch=eligible[start:start+limit]
            # 先用完整目录选标签，再仅核对选中标签的完整规则，避免每条扫描八页规则。
            shortlist={c["candidate_id"]:{"labels":set(),"reasons":[]} for c in batch}
            self.classify_batch(batch,[{k:d[k] for k in ("dimension_id","name")} for d in rules],shortlist)
            selected=set().union(*(v["labels"] for v in shortlist.values()))
            if selected:
                for catalog in pages([d for d in rules if d["dimension_id"] in selected],setting(self.a.config,"rule_page_tokens")):
                    self.classify_batch(batch,catalog,collected)
            else:
                for c in batch: collected[c["candidate_id"]]["reasons"].append("目录未匹配；内容仍可独立保存")
        for c in eligible:
            labels=collected[c["candidate_id"]]["labels"]; reasons=collected[c["candidate_id"]]["reasons"]
            previous=content_hash(c)
            c["dimension_ids"]=sorted(labels) or ["UNCLASSIFIED"]
            c["dimension_id"]=c["dimension_ids"][0]
            c["classification"]={"status":"classified" if labels else "unclassified","reason":"；".join(reasons)}
            assert content_hash(c)==previous
            # 标签独立变化，来源凭证保持原封不动。

    def classify_batch(self,batch,catalog,collected):
        pending, item_keys = [], {}
        for c in batch:
            dependency = dict(candidate=compact_fact(c), catalog=catalog)
            def check_cached(row):
                exact_keys(row,{"candidate_id","dimension_ids","reason"})
                if row["candidate_id"] != c["candidate_id"] or not isinstance(row["dimension_ids"],list) or any(x not in {d["dimension_id"] for d in catalog} for x in row["dimension_ids"]) or len(set(row["dimension_ids"])) != len(row["dimension_ids"]) or not isinstance(row["reason"],str):
                    raise ValueError("分类缓存协议错误")
            key, cached = self.runtime.load_item("source_classifier", CLASSIFY, dependency, check_cached)
            item_keys[c["candidate_id"]] = (key, dependency)
            if cached is None:
                pending.append(c)
            else:
                collected[c["candidate_id"]]["labels"].update(cached["dimension_ids"])
                collected[c["candidate_id"]]["reasons"].append(cached["reason"])
        batch = pending
        if not batch:
            return
        def check(r):
            exact_keys(r,{"results"})
            if not isinstance(r["results"],list): raise ValueError("分类结果不完整")
            for row in r["results"]:
                exact_keys(row,{"candidate_id","dimension_ids","reason"})
                if not isinstance(row["dimension_ids"],list) or any(x not in {d["dimension_id"] for d in catalog} for x in row["dimension_ids"]) or len(set(row["dimension_ids"]))!=len(row["dimension_ids"]) or not isinstance(row["reason"],str): raise ValueError("分类协议错误")
            if sorted(x["candidate_id"] for x in r["results"])!=sorted(c["candidate_id"] for c in batch): raise ValueError("分类ID遗漏、重复或额外")
        try:
            rows=self.ask("source_classifier",CLASSIFY,{"candidates":[compact_fact(c) for c in batch],"catalog":catalog,
                "classification_stage":"rule_check" if catalog and "rules" in catalog[0] else "shortlist"},check)["results"]
            for row in rows:
                key, dependency = item_keys[row["candidate_id"]]
                self.runtime.save_item(key,"source_classifier",dependency,row)
                collected[row["candidate_id"]]["labels"].update(row["dimension_ids"])
                collected[row["candidate_id"]]["reasons"].append(row["reason"])
        except ContextCapacityError as exc:
            if len(batch)>1:
                n=len(batch)//2; self.classify_batch(batch[:n],catalog,collected); self.classify_batch(batch[n:],catalog,collected)
            elif len(catalog)>1:
                n=len(catalog)//2; self.classify_batch(batch,catalog[:n],collected); self.classify_batch(batch,catalog[n:],collected)
            else: self.task("source_classifier",[c["candidate_id"] for c in batch],exc)
        except (ModelOutputError,TransientModelError) as exc:
            self.task("source_classifier",[c["candidate_id"] for c in batch],exc)

    def maintain(self,candidates,proofs,signals,coverage):
        if self.flow == 'generic-flow-v3-relations':
            from agent_relation_plan import maintain
            return maintain(self,candidates,proofs,signals,coverage)
        self.reconcile(candidates,proofs)
        self.classify(candidates,proofs)
        return self._maintain_core(candidates,proofs,signals,coverage)

    def commit_plan(self,candidates,proofs,plan,signals,coverage):
        for issue in self.extraction_issues:
            self.task("source_extractor", [issue["issue_id"]], issue["reason"],
                      code=issue["code"], evidence_ids=issue["evidence_ids"])
        if self.retry_tasks:
            # 不提交半个Session；成功的调用缓存保留，下次只恢复未完成阶段。
            return dict(status="deferred",mapping=[],counts={k:0 for k in ("new","reinforce","coexist","supersede","conflict","pending")})
        plan=seal(dict({k:v for k,v in plan.items() if k!="proof_hash"},flow_version=self.flow,processing_revision=self.processing_revision,consistency_audit=self.consistency_audit))
        return self.store.commit(self.s,candidates,proofs,plan,signals,coverage)

    def run(self):
        r=self._run_core()
        if r["commit"]["status"]=="deferred":
            r.update(status="partial",needs_attention=True,approved=[],retry_tasks=self.retry_tasks,
                     execution_status="awaiting_retry",commit_status="not_committed")
        else:
            r.update(execution_status="completed",commit_status="committed",retry_tasks=[])
            if not self.consistency_audit:
                import json
                row=self.store.db.execute("SELECT data FROM v3_plans WHERE batch_id=?",(r["commit"]["batch_id"],)).fetchone()
                if row: self.consistency_audit=json.loads(row[0]).get("consistency_audit",[])
        r["flow_version"]=self.flow
        r["stages"]=[{k:v for k,v in row.items() if k != "output"} for row in self.runtime.records.values()]
        r["consistency_audit"]=self.consistency_audit
        r["fact_pool"]=[dict(fact_id=c.get("fact_id"),candidate_id=c["candidate_id"],dimension_ids=c.get("dimension_ids",[c["dimension_id"]])) for c in r["candidates"]]
        r["episodes"]=[dict(episode_id=digest([self.s["session_id"],c.get("fact_id")])[:24],fact_ids=[c.get("fact_id")],structure=c["structure"]) for c in r["candidates"] if c.get("structure",{}).get("type") in ("learning_episode","shared_work","real_event")]
        return r

    def record_validation(self, role, attempt, status, reason):
        with self.a.trace_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps({"session_id": self.s["session_id"], "implementation_revision": REVISION,
                "role": role, "attempt": attempt, "schema_status": status, "reason": reason}, ensure_ascii=False) + "\n")
