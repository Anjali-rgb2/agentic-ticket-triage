"""
Tools used by the agentic ticket-triage system.

These functions are intentionally separated from the LLM.
The LLM decides WHICH tool to call and WHEN to call it.
The tools only retrieve/compute information from the dataset.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pandas as pd


AS_OF_DATE = datetime(2026, 10, 20)

def _clean(value: Any) -> str:
    """Convert a value to a safe stripped string."""
    if value is None:
        return ""
    return str(value).strip()


def _normalise(value: Any) -> str:
    """Normalise text for comparisons."""
    return " ".join(_clean(value).lower().split())


def _safe_date(value: Any):
    """Convert a value to a pandas timestamp."""
    value = _clean(value)
    if not value:
        return None

    try:
        return pd.to_datetime(value)
    except Exception:
        return None

# 1. ACCOUNT LOOKUP
def lookup_account(
    ctx,
    account_id: str | None = None,
    company_name: str | None = None,
    domain: str | None = None,
) -> dict:
    """
    Resolve a customer account.

    Lookup priority:
      1. Exact account_id
      2. Exact requester/company domain
      3. Exact company name
      4. Normalised company name

    Returns a compact account record instead of the entire dataframe.
    """

    accounts = ctx.accounts

    account_id = _clean(account_id)
    company_name = _clean(company_name)
    domain = _clean(domain)

    # 1. Account ID
    if account_id:
        matches = accounts[
            accounts["account_id"].astype(str).str.strip() == account_id
        ]

        if not matches.empty:
            row = matches.iloc[0]

            return {
                "found": True,
                "match_method": "account_id",
                "account_id": row["account_id"],
                "company_name": row["company_name"],
                "domain": row["domain"],
                "industry": row["industry"],
                "region": row["region"],
                "tier": row["tier"],
                "arr_band": row["arr_band"],
                "contract_start": row["contract_start"],
                "renewal_date": row["renewal_date"],
                "modules_licensed": row["modules_licensed"],
                "open_escalations": row["open_escalations"],
                "last_qbr_date": row["last_qbr_date"],
                "nps_last": row["nps_last"],
            }

    # 2. Domain
    if domain:
        domain_norm = _normalise(domain)

        matches = accounts[
            accounts["domain"]
            .astype(str)
            .map(_normalise)
            == domain_norm
        ]

        if not matches.empty:
            row = matches.iloc[0]

            return {
                "found": True,
                "match_method": "domain",
                "account_id": row["account_id"],
                "company_name": row["company_name"],
                "domain": row["domain"],
                "industry": row["industry"],
                "region": row["region"],
                "tier": row["tier"],
                "arr_band": row["arr_band"],
                "contract_start": row["contract_start"],
                "renewal_date": row["renewal_date"],
                "modules_licensed": row["modules_licensed"],
                "open_escalations": row["open_escalations"],
                "last_qbr_date": row["last_qbr_date"],
                "nps_last": row["nps_last"],
            }

    # 3. Exact company name
    if company_name:
        company_norm = _normalise(company_name)

        matches = accounts[
            accounts["company_name"]
            .astype(str)
            .map(_normalise)
            == company_norm
        ]

        if not matches.empty:
            row = matches.iloc[0]

            return {
                "found": True,
                "match_method": "company_name",
                "account_id": row["account_id"],
                "company_name": row["company_name"],
                "domain": row["domain"],
                "industry": row["industry"],
                "region": row["region"],
                "tier": row["tier"],
                "arr_band": row["arr_band"],
                "contract_start": row["contract_start"],
                "renewal_date": row["renewal_date"],
                "modules_licensed": row["modules_licensed"],
                "open_escalations": row["open_escalations"],
                "last_qbr_date": row["last_qbr_date"],
                "nps_last": row["nps_last"],
            }

    return {
        "found": False,
        "reason": "No matching account found",
        "account_id": None,
    }

# 2. USAGE TREND
def get_usage_trend(
    ctx,
    account_id: str,
) -> dict:
    """
    Calculate recent account usage trends.

    The task requires the agent to consider usage trend as part of
    account health.
    """

    account_id = _clean(account_id)

    if not account_id:
        return {
            "found": False,
            "reason": "account_id is required",
        }

    usage = ctx.usage

    rows = usage[
        usage["account_id"].astype(str).str.strip() == account_id
    ].copy()

    if rows.empty:
        return {
            "found": False,
            "account_id": account_id,
            "reason": "No usage data found",
        }

    rows["month_date"] = pd.to_datetime(
        rows["month"],
        errors="coerce"
    )

    rows = rows.sort_values("month_date")

    # Last three available months
    recent = rows.tail(3)

    def pct_change(first, last):
        try:
            first = float(first)
            last = float(last)

            if first == 0:
                return None

            return round(((last - first) / first) * 100, 2)

        except (ValueError, TypeError):
            return None

    first = recent.iloc[0]
    last = recent.iloc[-1]

    active_users_change = pct_change(
        first["active_users"],
        last["active_users"],
    )

    documents_change = pct_change(
        first["documents_processed"],
        last["documents_processed"],
    )

    login_change = pct_change(
        first["login_rate_pct"],
        last["login_rate_pct"],
    )

    tickets_change = pct_change(
        first["tickets_opened"],
        last["tickets_opened"],
    )

    return {
        "found": True,
        "account_id": account_id,
        "months_considered": recent["month"].tolist(),
        "active_users": {
            "latest": _clean(last["active_users"]),
            "change_pct": active_users_change,
        },
        "documents_processed": {
            "latest": _clean(last["documents_processed"]),
            "change_pct": documents_change,
        },
        "login_rate_pct": {
            "latest": _clean(last["login_rate_pct"]),
            "change_pct": login_change,
        },
        "tickets_opened": {
            "latest": _clean(last["tickets_opened"]),
            "change_pct": tickets_change,
        },
    }

# 3. RELATED TICKET SEARCH
def search_related_tickets(
    ctx,
    account_id: str,
    module: str | None = None,
    error_code: str | None = None,
    subject: str | None = None,
    description: str | None = None,
    days: int = 30,
) -> dict:
    """
    Find potentially related tickets for the same account.

    This tool is useful for:
      - duplicate detection
      - recurring issues
      - identifying existing escalations
    """

    account_id = _clean(account_id)
    module = _normalise(module)
    error_code = _normalise(error_code)

    if not account_id:
        return {
            "found": False,
            "reason": "account_id is required",
            "tickets": [],
        }

    tickets = ctx.tickets.copy()

    tickets = tickets[
        tickets["account_id"].astype(str).str.strip() == account_id
    ].copy()

    if tickets.empty:
        return {
            "found": True,
            "account_id": account_id,
            "tickets": [],
        }

    tickets["created_date"] = pd.to_datetime(
        tickets["created_at"],
        errors="coerce",
    )

    cutoff = AS_OF_DATE - timedelta(days=days)

    recent = tickets[
        tickets["created_date"] >= cutoff
    ].copy()

    # Do not return the entire ticket history.
    # Rank likely matches first.
    results = []

    for _, row in recent.iterrows():

        score = 0
        reasons = []

        row_module = _normalise(row.get("module", ""))
        row_error = _normalise(row.get("error_code", ""))

        if module and row_module == module:
            score += 3
            reasons.append("same module")

        if error_code and row_error == error_code:
            score += 4
            reasons.append("same error code")

        ticket_text = _normalise(
            f"{row.get('subject', '')} {row.get('description', '')}"
        )

        current_text = _normalise(
            f"{subject or ''} {description or ''}"
        )

        if current_text and ticket_text:
            current_words = set(current_text.split())
            ticket_words = set(ticket_text.split())

            overlap = current_words.intersection(ticket_words)

            if len(overlap) >= 3:
                score += 1
                reasons.append("similar wording")

        results.append({
            "ticket_id": row["ticket_id"],
            "created_at": row["created_at"],
            "module": row["module"],
            "subject": row["subject"],
            "error_code": row["error_code"],
            "status": row["status"],
            "customer_priority": row["customer_priority"],
            "match_score": score,
            "match_reasons": reasons,
        })

    results.sort(
        key=lambda x: (
            -x["match_score"],
            x["created_at"],
        )
    )

    return {
        "found": True,
        "account_id": account_id,
        "tickets": results[:10],
    }

# 4. KB SEARCH
def search_kb(
    ctx,
    query: str,
    module: str | None = None,
    error_code: str | None = None,
    limit: int = 6,
) -> dict:
    """
    Search the KB without sending the entire KB to the LLM.

    Ranking signals:
      - exact error code
      - module match
      - query token overlap
      - title overlap
      - current/policy articles
    """

    query = _normalise(query)
    module = _normalise(module)
    error_code = _normalise(error_code)

    if not query and not module and not error_code:
        return {
            "found": False,
            "articles": [],
            "reason": "Search query is empty",
        }

    query_tokens = set(query.split())

    results = []

    for article in ctx.kb:

        meta = article.get("meta", {})
        title = _normalise(meta.get("title", ""))
        body = _normalise(article.get("body", ""))

        article_module = _normalise(meta.get("module", ""))
        article_type = _normalise(meta.get("type", ""))
        article_status = _normalise(meta.get("status", ""))

        article_error_codes = meta.get("error_codes", [])

        if isinstance(article_error_codes, str):
            article_error_codes = [
                x.strip()
                for x in article_error_codes.split(",")
                if x.strip()
            ]

        article_error_codes = [
            _normalise(x)
            for x in article_error_codes
        ]

        score = 0
        reasons = []

        # Exact error-code match is highly important.
        if error_code and error_code in article_error_codes:
            score += 10
            reasons.append("error code match")

        # Module match.
        if module and article_module == module:
            score += 5
            reasons.append("module match")

        # Query overlap with title.
        title_tokens = set(title.split())
        title_overlap = query_tokens.intersection(title_tokens)

        score += min(len(title_overlap) * 3, 9)

        if title_overlap:
            reasons.append("title keyword match")

        # Body keyword overlap.
        body_tokens = set(body.split())
        body_overlap = query_tokens.intersection(body_tokens)

        score += min(len(body_overlap), 5)

        if body_overlap:
            reasons.append("body keyword match")

        # Policy articles receive a small priority boost.
        if article_type == "policy":
            score += 4
            reasons.append("policy article")

        # Prefer current / active articles.
        if article_status in {"current", "active_incident"}:
            score += 2

        # Deprecated articles should not disappear completely,
        # but should rank much lower.
        if article_status == "deprecated":
            score -= 5

        if score > 0:
            results.append({
                "id": article["id"],
                "title": meta.get("title", ""),
                "module": meta.get("module", ""),
                "type": meta.get("type", ""),
                "status": meta.get("status", ""),
                "error_codes": article_error_codes,
                "score": score,
                "match_reasons": reasons,
                "related": meta.get("related", []),
                "superseded_by": meta.get("superseded_by", ""),
            })

    results.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    return {
        "found": bool(results),
        "query": query,
        "articles": results[:limit],
    }

# 5. GET FULL KB ARTICLE
def get_kb_article(
    ctx,
    article_id: str,
) -> dict:
    """
    Retrieve one complete KB article.

    The agent can call this after search_kb identifies a relevant article.
    """

    article_id = _clean(article_id)

    for article in ctx.kb:

        if _clean(article.get("id")) == article_id:

            meta = article.get("meta", {})

            return {
                "found": True,
                "id": article["id"],
                "title": meta.get("title", ""),
                "meta": meta,
                "body": article.get("body", ""),
            }

    return {
        "found": False,
        "id": article_id,
        "reason": "KB article not found",
    }

# TOOL REGISTRY
TOOLS = {
    "lookup_account": lookup_account,
    "get_usage_trend": get_usage_trend,
    "search_related_tickets": search_related_tickets,
    "search_kb": search_kb,
    "get_kb_article": get_kb_article,
}