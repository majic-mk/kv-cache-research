"""Read-only audit of the existing append-only external cost ledger."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .common import ValidationError, nonempty_string

LEDGER_FIELDS = ["timestamp_utc", "provider", "resource", "amount", "currency", "status", "evidence"]
STATUSES = {"no_external_purchase", "estimate", "committed", "settled", "cancelled"}


def audit_ledger(path: str | Path) -> dict[str, Any]:
    totals: dict[str, dict[str, Decimal]] = {}
    rows = 0
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != LEDGER_FIELDS:
            raise ValidationError(f"Ledger header must be {LEDGER_FIELDS}")
        for line_number, row in enumerate(reader, 2):
            try:
                if set(row) != set(LEDGER_FIELDS):
                    raise ValidationError("Incorrect CSV column count")
                for name in LEDGER_FIELDS:
                    nonempty_string(row[name], f"ledger.{name}")
                timestamp = datetime.fromisoformat(row["timestamp_utc"].replace("Z", "+00:00"))
                if timestamp.utcoffset() != timezone.utc.utcoffset(timestamp):
                    raise ValidationError("Ledger timestamp must be UTC")
                amount = Decimal(row["amount"])
                if not amount.is_finite() or amount < 0:
                    raise ValidationError("Ledger amount must be finite and nonnegative")
                currency, status = row["currency"], row["status"]
                if len(currency) != 3 or not currency.isascii() or not currency.isupper() or not currency.isalpha():
                    raise ValidationError("Currency must be a three-letter uppercase code")
                if status not in STATUSES:
                    raise ValidationError(f"Unknown cost status: {status}")
                if status == "no_external_purchase" and amount != 0:
                    raise ValidationError("no_external_purchase entries must have zero amount")
                totals.setdefault(currency, {label: Decimal(0) for label in STATUSES})[status] += amount
                rows += 1
            except (ValueError, TypeError, InvalidOperation) as exc:
                raise ValidationError(f"Ledger line {line_number}: {exc}") from exc
    return {"rows": rows, "totals_by_currency_and_status": {
        currency: {status: str(amount) for status, amount in sorted(statuses.items())}
        for currency, statuses in sorted(totals.items())},
        "note": "No FX conversion or purchase authorization inferred. Distinct event rows are not deduplicated invoices."}
