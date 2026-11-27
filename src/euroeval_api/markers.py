"""Parse, render, and authenticate volunteer-worker ownership markers."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
from datetime import datetime
from typing import Any

import rfc8785

from euroeval_api.dependencies import require_env

PROTOCOL_VERSION = "volunteer-worker/v1"
TITLE_PREFIX = "[MODEL EVALUATION REQUEST]"
VOLUNTEER_MARKER_RE = re.compile(r"<!--[ \t]*euroeval-volunteer-worker:v1[ \t]+([^<]*?)-->", re.I)
GROUPS = {
    "Baltic languages (Latvian, Lithuanian)": ["lv", "lt"],
    "Finnic languages (Estonian, Finnish)": ["et", "fi"],
    "Romance languages (Catalan, French, Italian, Portuguese, Romanian, Spanish)": ["ca", "fr", "it", "pt", "ro", "es"],
    "Scandinavian languages (Danish, Faroese, Icelandic, Norwegian, Swedish)": ["da", "fo", "is", "no", "sv"],
    "Slavic languages (Belarusian, Bulgarian, Bosnian, Croatian, Czech, Polish, Serbian, Slovak, Slovenian, Ukrainian)": ["be", "bg", "bs", "hr", "cs", "pl", "sr", "sk", "sl", "uk"],
    "West Germanic languages (Dutch, English, German, Luxembourgish)": ["nl", "en", "de", "lb"],
    "Albanian": ["sq"], "Greek": ["el"], "Hungarian": ["hu"],
}


def _date(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return True
    except ValueError:
        return False


def parse_volunteer_marker(body: str | None) -> dict[str, Any] | None:
    """Parse the sole valid marker, retaining allowed nested extensions."""
    if not body:
        return None
    matches = list(VOLUNTEER_MARKER_RE.finditer(body))
    if len(matches) != 1:
        return None
    try:
        item = json.loads(matches[0].group(1).strip())
        if not isinstance(item, dict) or item.get("protocol_version") != PROTOCOL_VERSION or not isinstance(item.get("coordinator"), str) or not item["coordinator"] or item.get("submission") not in {"active", "submitted", "accepted", "rejected"} or not isinstance(item.get("leases"), list):
            return None
        leases = item["leases"]
        if any(not isinstance(x, dict) or any(not isinstance(x.get(k), str) for k in ("lease_id", "language", "worker", "contributor", "expires_at")) or x.get("ownership_mode") not in (None, "assignee", "public") or not _date(x["expires_at"]) for x in leases):
            return None
        if len({x["lease_id"] for x in leases}) != len(leases) or len({x["language"] for x in leases}) != len(leases): return None
        subs = item.get("submissions")
        if subs is not None and (not isinstance(subs, list) or any(not isinstance(x, dict) or not all(isinstance(x.get(k), str) and x[k] for k in ("submission_id", "language", "manifest_path", "submitted_at", "verified_contributor")) or not _date(x["submitted_at"]) or not isinstance(x.get("result_count"), int) or isinstance(x["result_count"], bool) or x["result_count"] <= 0 or x.get("ownership_mode") not in (None, "assignee", "public") or x.get("status") not in ("submitted", "accepted", "rejected") for x in subs) or len({x["submission_id"] for x in subs}) != len(subs)): return None
        for key in ("completed_languages", "public_assignees"):
            val = item.get(key)
            if val is not None and (not isinstance(val, list) or any(not isinstance(x, str) or not x if key == "completed_languages" else not isinstance(x, str) or not x.strip() for x in val)): return None
        if "public_assignees" in item and len({x.lower() for x in item["public_assignees"]}) != len(item["public_assignees"]): return None
        keys = {"protocol_version", "coordinator", "submission", "leases", "submissions", "completed_languages", "public_assignees", "signature"}
        return {**{k: v for k, v in item.items() if k in keys}, "protocol_version": PROTOCOL_VERSION}
    except (ValueError, TypeError, KeyError):
        return None


def has_volunteer_marker(body: str | None) -> bool:
    return bool(body and VOLUNTEER_MARKER_RE.search(body))


def render_volunteer_marker(marker: dict[str, Any]) -> str:
    return "<!-- euroeval-volunteer-worker:v1 " + rfc8785.dumps(marker).decode() + " -->"


def remove_final_credit(body: str) -> str:
    return re.sub(r"<!--[ \t]*euroeval-volunteer-credit:v1[\s\S]*?-->", "", body, flags=re.I).replace("\n\n\n", "\n\n").rstrip()


def replace_volunteer_marker(body: str, marker: dict[str, Any] | None) -> str:
    without = re.sub(r"\n{3,}", "\n\n", VOLUNTEER_MARKER_RE.sub("", remove_final_credit(body))).rstrip()
    return f"{without}\n\n{render_volunteer_marker(marker)}\n" if marker else f"{without}\n"


def selected_languages(body: str | None) -> list[str]:
    if not body: return []
    return [code for group, codes in GROUPS.items() if re.search(r"-\s*\[[xX]\]\s*" + re.escape(group) + r"(?:\s|$)", body, re.M) for code in codes]


def volunteer_assignees_match(assignees: list[dict[str, str]] | None, marker: dict[str, Any] | None, now: float | None = None, include_expired_leases: bool = False, allow_expired_unassigned: bool = False) -> bool:
    now = now if now is not None else datetime.now().timestamp() * 1000
    actual = {x["login"].lower() for x in assignees or []}
    if marker is None: return not actual
    expected: set[str] = set(); optional: set[str] = set()
    expiry = lambda x: datetime.fromisoformat(x["expires_at"].replace("Z", "+00:00")).timestamp() * 1000
    for lease in marker["leases"]:
        if lease.get("ownership_mode") == "public": continue
        who = lease["contributor"].lower()
        if expiry(lease) > now: expected.add(who)
        elif include_expired_leases:
            expected.add(who)
            if allow_expired_unassigned and not any(o["lease_id"] != lease["lease_id"] and o.get("ownership_mode") != "public" and expiry(o) > now and o["contributor"].lower() == who for o in marker["leases"]) and not any(s["status"] in ("submitted", "accepted") and s.get("ownership_mode") != "public" and s["verified_contributor"].lower() == who for s in marker.get("submissions", [])): optional.add(who)
    for s in marker.get("submissions", []):
        if s["status"] in ("submitted", "accepted") and s.get("ownership_mode") != "public": expected.add(s["verified_contributor"].lower())
    expected.update(x.lower() for x in marker.get("public_assignees", []))
    public = any(x.get("ownership_mode") == "public" for x in marker["leases"]) or any(x.get("ownership_mode") == "public" for x in marker.get("submissions", []))
    return (public or actual <= expected) and expected <= actual | optional


def is_ready_for_review(marker: dict[str, Any], selected: list[str]) -> bool:
    subs = marker.get("submissions", [])
    completed = {s["language"] for s in subs if s["status"] in ("submitted", "accepted")}
    return bool(selected) and all(x in completed for x in selected) and any(s["status"] == "submitted" and s["language"] in selected for s in subs)


def claimable_languages(selected: list[str], marker: dict[str, Any] | None, now: float | None = None) -> list[str]:
    if marker is None: return selected
    now = now if now is not None else datetime.now().timestamp() * 1000
    return [lang for lang in selected if not any(x["language"] == lang and datetime.fromisoformat(x["expires_at"].replace("Z", "+00:00")).timestamp() * 1000 > now for x in marker["leases"]) and not any(x["language"] == lang and x["status"] in ("submitted", "accepted") for x in marker.get("submissions", []))]


def extract_model_id(title: str, body: str | None) -> str | None:
    m = re.search(r"(?:^|\n)#{1,6}\s*Model ID\s*\n+([^\n]+)", body or "", re.I)
    if not m and not title.startswith(TITLE_PREFIX + " "): return None
    value = (m.group(1) if m else title[len(TITLE_PREFIX):]).strip().strip("`*_")
    return value if re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+(?::[A-Z0-9_]+)?", value) else None


def sign_volunteer_marker(issue_number: int, marker: dict[str, Any]) -> dict[str, Any]:
    unsigned = {k: v for k, v in marker.items() if k != "signature"}
    payload = {"domain": "euroeval-volunteer-marker", "version": 1, "issue_number": issue_number, "marker": unsigned}
    digest = hmac.new(require_env("VOLUNTEER_MARKER_SECRET").encode(), rfc8785.dumps(payload), hashlib.sha256).digest()
    return {**marker, "signature": base64.urlsafe_b64encode(digest).decode().rstrip("=")}


def verify_volunteer_marker(issue_number: int, marker: dict[str, Any]) -> bool:
    signature = marker.get("signature")
    return isinstance(signature, str) and hmac.compare_digest(sign_volunteer_marker(issue_number, marker)["signature"], signature)
