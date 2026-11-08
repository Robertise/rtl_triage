#!/usr/bin/env python3
"""
RTL Failure Triage Agent - Persistent Cost Ledger & Hard Stop Guard
Tracks cumulative API token usage and USD cost across all dev and test runs.

Official Pricing (Claude Haiku 4.5 on AWS Bedrock - official model):
- Input tokens:  $1.00 / 1,000,000 tokens
- Output tokens: $5.00 / 1,000,000 tokens
- Hard stop limit: $15.00 USD cumulative across dev + test.

Dev Pricing (Amazon Nova 2 Lite):
- Input tokens:  $0.30 / 1,000,000 tokens
- Output tokens: $2.50 / 1,000,000 tokens

Checked strictly BEFORE every LLM call. If cumulative cost has reached the
hard stop threshold (or would exceed it), raises BudgetHardStopError and
prevents any further LLM calls. Runs stopped by BudgetHardStopError are
never counted as agent failures.
"""

import json
from pathlib import Path
from typing import Any

INPUT_PRICE_PER_MILLION = 1.00
OUTPUT_PRICE_PER_MILLION = 5.00
DEFAULT_HARD_STOP_USD = 15.00
PRICING_STATUS_NOTE = "Claude Haiku 4.5, official model ($1.00/1M in, $5.00/1M out)"

# Model pricing registry
MODEL_PRICING: dict[str, dict[str, Any]] = {
    "global.amazon.nova-2-lite-v1:0": {
        "input_per_million": 0.30,
        "output_per_million": 2.50,
        "note": "Amazon Nova 2 Lite dev rate",
    },
    "global.anthropic.claude-haiku-4-5-20251001-v1:0": {
        "input_per_million": 1.00,
        "output_per_million": 5.00,
        "note": "Claude Haiku 4.5, official model",
    },
}


class BudgetHardStopError(RuntimeError):
    """Raised before an LLM call when the cumulative budget hard stop is reached."""


def compute_call_cost_usd(
    input_tokens: int,
    output_tokens: int,
    model_id: str | None = None,
) -> float:
    """Computes exact USD cost for a single LLM invocation."""
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("Token counts must be non-negative.")

    mid = model_id or "global.anthropic.claude-haiku-4-5-20251001-v1:0"
    if mid not in MODEL_PRICING:
        raise ValueError(
            f"Model ID '{mid}' not found in MODEL_PRICING registry: {list(MODEL_PRICING.keys())}"
        )

    in_price = MODEL_PRICING[mid]["input_per_million"]
    out_price = MODEL_PRICING[mid]["output_per_million"]

    cost = (
        input_tokens * in_price
        + output_tokens * out_price
    ) / 1_000_000.0
    return round(cost, 8)


class BudgetManager:
    """
    Persistent budget ledger stored as JSON on disk.
    Reloads state on initialization and before every pre-call check.
    """

    def __init__(
        self,
        ledger_path: Path | str | None = None,
        hard_stop_usd: float = DEFAULT_HARD_STOP_USD,
    ) -> None:
        if ledger_path is None:
            project_root = Path(__file__).resolve().parent.parent
            self.ledger_path = project_root / "results" / "budget_ledger.json"
        else:
            self.ledger_path = Path(ledger_path).resolve()

        self.hard_stop_usd = float(hard_stop_usd)
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_cost_usd: float = 0.0
        self.total_calls: int = 0
        self.entries: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        """Loads persistent ledger state from disk if it exists."""
        if not self.ledger_path.is_file():
            return
        with open(self.ledger_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.total_input_tokens = int(data.get("total_input_tokens", 0))
        self.total_output_tokens = int(data.get("total_output_tokens", 0))
        self.total_cost_usd = round(float(data.get("total_cost_usd", 0.0)), 8)
        self.total_calls = int(data.get("total_calls", 0))
        self.entries = list(data.get("entries", []))

    def _save(self) -> None:
        """Atomically writes persistent ledger state to disk."""
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "hard_stop_usd": self.hard_stop_usd,
            "input_price_per_million_usd": INPUT_PRICE_PER_MILLION,
            "output_price_per_million_usd": OUTPUT_PRICE_PER_MILLION,
            "pricing_status_note": PRICING_STATUS_NOTE,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 8),
            "total_calls": self.total_calls,
            "entries": self.entries,
        }
        tmp_path = self.ledger_path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")
        tmp_path.replace(self.ledger_path)

    def check_can_call(self, estimated_next_cost_usd: float = 0.0) -> None:
        """
        Verifies before an LLM call that the cumulative cost has not reached
        (and with estimated_next_cost_usd will not reach/exceed) hard_stop_usd.
        Raises BudgetHardStopError if limit is reached.
        """
        self._load()
        projected = round(self.total_cost_usd + max(0.0, estimated_next_cost_usd), 8)
        if self.total_cost_usd >= self.hard_stop_usd or projected > self.hard_stop_usd:
            raise BudgetHardStopError(
                f"HARD STOP: Cumulative cost ${self.total_cost_usd:.6f} USD "
                f"(projected ${projected:.6f} USD) reached/exceeded budget limit "
                f"${self.hard_stop_usd:.2f} USD. Stopping immediately."
            )

    def record_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        run_id: str = "",
        config: str = "agent",
        split: str = "dev",
        turn: int = 1,
        model_id: str = "",
    ) -> float:
        """Records token usage and cost for a completed LLM call into the persistent ledger."""
        self._load()
        call_cost = compute_call_cost_usd(input_tokens, output_tokens, model_id=model_id)
        self.total_input_tokens += int(input_tokens)
        self.total_output_tokens += int(output_tokens)
        self.total_cost_usd = round(self.total_cost_usd + call_cost, 8)
        self.total_calls += 1

        entry = {
            "call_index": self.total_calls,
            "run_id": run_id,
            "config": config,
            "split": split,
            "turn": turn,
            "model_id": model_id,
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "cost_usd": call_cost,
            "cumulative_cost_usd": self.total_cost_usd,
        }
        self.entries.append(entry)
        self._save()
        return call_cost
