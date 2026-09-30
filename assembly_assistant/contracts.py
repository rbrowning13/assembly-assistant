"""Neutral, JSON-safe contracts for procedure sources and evidence."""

from __future__ import annotations

import hashlib
import re


CONTRACT_VERSION = 1

AUTHORITY_RANK = {
    "authoritative": 0,
    "primary_evidence": 1,
    "supporting_context": 2,
    "reference_only": 3,
}

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9._/-]*", re.IGNORECASE)
SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]+|\n+|$)")
TORQUE_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(in[- ]?lb|inch[- ]?pounds?|ft[- ]?lb|foot[- ]?pounds?|n[- ]?m|nm|newton[- ]?meters?)\b",
    re.IGNORECASE,
)
DIMENSION_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(mm|millimeters?|cm|centimeters?|m|meters?|inches?|inch|in\.?|feet|foot|ft\.?)\b",
    re.IGNORECASE,
)
PART_NUMBER_RE = re.compile(
    r"\b(?:part|item|p\s*/?\s*n|pn)\s*(?:number|no\.?|#)?\s*[:=-]?\s*([a-z0-9][a-z0-9._/-]{2,})\b",
    re.IGNORECASE,
)
SAFETY_RE = re.compile(
    r"\b(?:warning|caution|danger|safety|must|shall|do not|never|keep clear|wear|disconnect|lockout)\b",
    re.IGNORECASE,
)
PROHIBITION_RE = re.compile(r"\b(?:do not|never|must not|shall not|prohibited)\b", re.IGNORECASE)

SUBJECT_STOPWORDS = {
    "a", "an", "and", "at", "before", "by", "for", "from", "in", "into", "is", "of", "on", "or",
    "the", "then", "this", "to", "with", "tighten", "torque", "set", "value", "dimension", "length",
    "must", "shall", "should", "do", "not", "never", "warning", "caution", "danger", "safety", "required",
    "part", "number", "item", "pn", "p/n",
}


def stable_contract_id(prefix, *parts):
    material = "|".join(str(part or "").strip().lower() for part in parts)
    return f"{prefix}-{hashlib.sha256(material.encode('utf-8')).hexdigest()[:16]}"


def source_contract(file_record):
    timeline_sync = file_record.get("timeline_sync") or {}
    return {
        "contract": "Source",
        "contract_version": CONTRACT_VERSION,
        "source_id": file_record.get("source_id") or stable_contract_id(
            "src",
            file_record.get("sha256"),
            file_record.get("name"),
        ),
        "kind": file_record.get("kind") or "unknown",
        "name": file_record.get("name") or "",
        "media_type": file_record.get("type") or "application/octet-stream",
        "sha256": file_record.get("sha256") or "",
        "bytes": int(file_record.get("bytes") or 0),
        "storage_ref": file_record.get("stored_path") or "",
        "role": file_record.get("source_role") or "reference",
        "label": file_record.get("source_label") or "",
        "authority_level": file_record.get("authority_level") or "supporting_context",
        "revision": file_record.get("revision") or "",
        "applicability": file_record.get("applicability") or "",
        "timeline": {
            "reference": bool(file_record.get("timeline_reference")),
            "use_for_drafting": bool(file_record.get("use_for_drafting")),
            "offset_seconds": float(file_record.get("timeline_offset_seconds") or 0),
            "scale": float(file_record.get("timeline_scale") or 1),
            "sync_mode": timeline_sync.get("mode") or "",
            "alignment": file_record.get("timeline_alignment") or {},
        },
        "derivation": {
            "parent_source_id": file_record.get("parent_source_id") or "",
            "method": file_record.get("derivation_method") or "uploaded",
        },
        "governance": {
            "classification": file_record.get("classification") or "not_set",
            "retention_policy": file_record.get("retention_policy") or "not_set",
        },
    }


def applicability_review(product, applicability, authority_level="", revision=""):
    product = re.sub(r"\s+", " ", str(product or "")).strip()
    applicability = re.sub(r"\s+", " ", str(applicability or "")).strip()
    reasons = []
    if not applicability:
        status = "unspecified"
        reasons.append("Source applicability was not provided.")
    elif re.search(r"\b(all|universal|general)\b", applicability, re.IGNORECASE):
        status = "applicable"
    else:
        generic = {"model", "product", "machine", "series", "assembly", "system"}
        product_tokens = {token.lower() for token in TOKEN_RE.findall(product) if len(token) >= 2 and token.lower() not in generic}
        applicability_tokens = {token.lower() for token in TOKEN_RE.findall(applicability) if len(token) >= 2 and token.lower() not in generic}
        direct_match = bool(product and (product.lower() in applicability.lower() or applicability.lower() in product.lower()))
        token_match = bool(product_tokens and applicability_tokens and product_tokens.intersection(applicability_tokens))
        if direct_match or token_match:
            status = "applicable"
        else:
            status = "review_required"
            reasons.append(f"Source applicability '{applicability}' was not matched to product '{product or 'unspecified'}'.")
    if authority_level == "authoritative" and not str(revision or "").strip():
        status = "review_required" if status != "review_required" else status
        reasons.append("Authoritative source revision was not provided.")
    return {
        "status": status,
        "product": product,
        "declared_applicability": applicability,
        "reasons": reasons,
    }


def evidence_item(
    *,
    evidence_type,
    source,
    locator,
    text="",
    summary="",
    tags=None,
    confidence="",
    extraction_method="",
    product="",
):
    source_id = source.get("source_id") or ""
    locator_key = "|".join(f"{key}={locator[key]}" for key in sorted(locator))
    evidence_id = stable_contract_id("ev", source_id, evidence_type, locator_key)
    return {
        "contract": "EvidenceItem",
        "contract_version": CONTRACT_VERSION,
        "evidence_id": evidence_id,
        "evidence_type": evidence_type,
        "source_id": source_id,
        "source_name": source.get("name") or "",
        "source_kind": source.get("kind") or "unknown",
        "source_role": source.get("role") or "reference",
        "authority_level": source.get("authority_level") or "supporting_context",
        "revision": source.get("revision") or "",
        "applicability": source.get("applicability") or "",
        "applicability_review": applicability_review(
            product,
            source.get("applicability"),
            source.get("authority_level"),
            source.get("revision"),
        ),
        "locator": dict(locator),
        "text": re.sub(r"[ \t]+", " ", str(text or "")).strip(),
        "summary": re.sub(r"[ \t]+", " ", str(summary or "")).strip(),
        "tags": sorted({str(tag).strip() for tag in (tags or []) if str(tag).strip()}),
        "confidence": confidence or "unknown",
        "extraction": {
            "method": extraction_method or "unknown",
        },
    }


def _subject_key(sentence, start, end):
    before = TOKEN_RE.findall(sentence[max(0, start - 90):start])[-8:]
    after = TOKEN_RE.findall(sentence[end:end + 45])[:4]
    tokens = []
    for token in before + after:
        value = token.lower().strip("._/-")
        if len(value) < 2 or value in SUBJECT_STOPWORDS or value.isdigit():
            continue
        tokens.append(value)
    return " ".join(tokens[-7:]) or "unspecified"


def _normalized_unit(unit):
    value = re.sub(r"[ .-]+", "", str(unit or "").lower())
    aliases = {
        "inchpound": "in-lb", "inchpounds": "in-lb", "inlb": "in-lb",
        "footpound": "ft-lb", "footpounds": "ft-lb", "ftlb": "ft-lb",
        "newtonmeter": "n-m", "newtonmeters": "n-m", "nm": "n-m",
        "millimeter": "mm", "millimeters": "mm", "centimeter": "cm", "centimeters": "cm",
        "meter": "m", "meters": "m", "inch": "in", "inches": "in", "feet": "ft", "foot": "ft",
    }
    return aliases.get(value, value)


def technical_facts(item):
    if item.get("evidence_type") not in {"document_page", "transcript_span"}:
        return []
    text = item.get("text") or ""
    facts = []
    for sentence_match in SENTENCE_RE.finditer(text):
        sentence = re.sub(r"\s+", " ", sentence_match.group(0)).strip()
        if not sentence:
            continue
        for fact_type, pattern in (("torque", TORQUE_RE), ("dimension", DIMENSION_RE)):
            for match in pattern.finditer(sentence):
                unit = _normalized_unit(match.group(2))
                number = f"{float(match.group(1)):g}"
                subject = _subject_key(sentence, match.start(), match.end())
                facts.append(
                    {
                        "fact_id": stable_contract_id("fact", item.get("evidence_id"), fact_type, subject, number, unit),
                        "fact_type": fact_type,
                        "subject": subject,
                        "value": f"{number} {unit}",
                        "normalized_value": f"{number}|{unit}",
                        "sentence": sentence[:500],
                        "evidence_id": item.get("evidence_id") or "",
                        "source_id": item.get("source_id") or "",
                        "source_name": item.get("source_name") or "",
                        "authority_level": item.get("authority_level") or "supporting_context",
                        "applicability_status": (item.get("applicability_review") or {}).get("status") or "unspecified",
                    }
                )
        for match in PART_NUMBER_RE.finditer(sentence):
            value = match.group(1).upper()
            subject = _subject_key(sentence, match.start(), match.end())
            facts.append(
                {
                    "fact_id": stable_contract_id("fact", item.get("evidence_id"), "part_number", subject, value),
                    "fact_type": "part_number",
                    "subject": subject,
                    "value": value,
                    "normalized_value": value,
                    "sentence": sentence[:500],
                    "evidence_id": item.get("evidence_id") or "",
                    "source_id": item.get("source_id") or "",
                    "source_name": item.get("source_name") or "",
                    "authority_level": item.get("authority_level") or "supporting_context",
                    "applicability_status": (item.get("applicability_review") or {}).get("status") or "unspecified",
                }
            )
        if SAFETY_RE.search(sentence):
            subject = " ".join(
                token.lower()
                for token in TOKEN_RE.findall(sentence)
                if token.lower() not in SUBJECT_STOPWORDS
            )[:180]
            if subject:
                value = "prohibited" if PROHIBITION_RE.search(sentence) else "required"
                facts.append(
                    {
                        "fact_id": stable_contract_id("fact", item.get("evidence_id"), "safety_directive", subject, value),
                        "fact_type": "safety_directive",
                        "subject": subject,
                        "value": value,
                        "normalized_value": value,
                        "sentence": sentence[:500],
                        "evidence_id": item.get("evidence_id") or "",
                        "source_id": item.get("source_id") or "",
                        "source_name": item.get("source_name") or "",
                        "authority_level": item.get("authority_level") or "supporting_context",
                        "applicability_status": (item.get("applicability_review") or {}).get("status") or "unspecified",
                    }
                )
    return facts


def detect_conflicts(evidence_items):
    grouped = {}
    all_facts = []
    for item in evidence_items:
        for fact in technical_facts(item):
            all_facts.append(fact)
            if fact.get("applicability_status") == "not_applicable":
                continue
            grouped.setdefault((fact["fact_type"], fact["subject"]), []).append(fact)

    conflicts = []
    for (fact_type, subject), facts in grouped.items():
        source_ids = {fact.get("source_id") for fact in facts if fact.get("source_id")}
        values = {fact.get("normalized_value") for fact in facts}
        if len(source_ids) < 2 or len(values) < 2:
            continue
        ordered = sorted(
            facts,
            key=lambda fact: (
                AUTHORITY_RANK.get(fact.get("authority_level"), 4),
                fact.get("source_name") or "",
                fact.get("value") or "",
            ),
        )
        conflict_id = stable_contract_id("conflict", fact_type, subject, *sorted(values), *sorted(source_ids))
        conflicts.append(
            {
                "conflict_id": conflict_id,
                "fact_type": fact_type,
                "subject": subject,
                "severity": "high" if fact_type in {"torque", "safety_directive"} else "medium",
                "status": "open",
                "requires_user_review": True,
                "claims": ordered,
                "evidence_ids": sorted({fact.get("evidence_id") for fact in ordered if fact.get("evidence_id")}),
                "reason": "Different source values were found for the same technical subject. No value was selected automatically.",
            }
        )
    return sorted(conflicts, key=lambda item: (item["severity"] != "high", item["fact_type"], item["subject"])), all_facts
