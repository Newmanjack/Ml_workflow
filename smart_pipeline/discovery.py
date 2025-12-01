from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Sequence, Tuple, Any

import pandas as pd

from .config import DiscoveryConfig, OverridesConfig, TableConfig
from .data_models import AggregationStrategy, ColumnMetadata, DiscoveryContext

logger = logging.getLogger("smart_pipeline.discovery")

try:
    from rapidfuzz import fuzz, process

    _RAPIDFUZZ_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _RAPIDFUZZ_AVAILABLE = False
    logger.warning("rapidfuzz not found. Falling back to simple string matching.")


class SmartDiscoveryEngine:
    """Schema discovery + aggregation strategy builder."""

    def __init__(
        self,
        con: Any,
        tables: TableConfig,
        overrides: Optional[OverridesConfig] = None,
        discovery_config: Optional[DiscoveryConfig] = None,
        current_line_table: Optional[str] = None,
    ):
        self.con = con
        self.tables = tables
        self.overrides = overrides or OverridesConfig()
        self.discovery_config = discovery_config or DiscoveryConfig()
        self.current_line_table = current_line_table or tables.line_table

        self.concepts = {
            "date": ["date", "time", "created", "timestamp", "day", "period", "transaction"],
            "amount": [
                "total",
                "amount",
                "price",
                "value",
                "cost",
                "revenue",
                "sales",
                "turnover",
                "gross",
                "net",
            ],
            "quantity": ["qty", "quantity", "units", "count", "volume"],
            "id": ["id", "num", "no", "key", "code", "identifier", "number"],
            "primary_date_boost": ["order", "doc", "trans", "created", "invoice"],
            "transaction_id_boost": ["order", "sales", "invoice", "trans", "doc", "header", "line"],
            "secondary_date_penalty": ["payment", "due", "delivery", "ship", "start", "end", "valid", "term", "fiscal"],
            "master_id_penalty": ["person", "customer", "vendor", "item", "product", "type", "status", "group", "batch", "worker"],
        }

        self.type_tokens = {
            "numeric": ("int", "decimal", "double", "float", "long"),
            "date": ("timestamp", "date", "time"),
        }

    def _normalize(self, name: str) -> str:
        return re.sub(r"[^A-Z0-9]", "", name.upper())

    def _quote(self, identifier: str) -> str:
        return '"' + identifier.replace('"', '""') + '"'

    def _describe_table(self, table_name: str) -> Sequence[Tuple]:
        return self.con.execute(f"DESCRIBE {table_name}").fetchall()

    def _build_catalog(self, table_name: str) -> List[ColumnMetadata]:
        desc = self._describe_table(table_name)
        return [
            ColumnMetadata(
                name=row[0],
                dtype=str(row[1]).lower(),
                normalized=self._normalize(row[0]),
                original_field=row,
            )
            for row in desc
        ]

    def _score_similarity(self, candidate: str, concepts: List[str]) -> float:
        if not _RAPIDFUZZ_AVAILABLE:
            candidate_upper = candidate.upper()
            return max((100 if c.upper() in candidate_upper else 0) for c in concepts)

        scores = []
        for c in concepts:
            s1 = fuzz.partial_ratio(c.upper(), candidate.upper())
            s2 = fuzz.token_set_ratio(c, candidate)
            scores.append(max(s1, s2))

        return max(scores) if scores else 0

    def _find_best_column(
        self,
        catalog: List[ColumnMetadata],
        concept_key: str,
        type_category: str,
        override_val: Optional[str] = None,
    ) -> Tuple[Optional[str], float]:
        if override_val:
            return override_val, 100.0

        valid_types = self.type_tokens.get(type_category, [])
        candidates = [c for c in catalog if any(t in c.dtype for t in valid_types)]

        # Fallback: if type-based filter finds nothing (e.g., dates stored as VARCHAR),
        # consider all columns and rely on semantic scoring.
        if not candidates:
            candidates = catalog

        best_col = None
        best_score = -999.0
        concepts = self.concepts.get(concept_key, [])

        for col in candidates:
            score = self._score_similarity(col.name, concepts)

            if col.normalized in [self._normalize(c) for c in concepts]:
                score += 15

            if any(col.normalized.endswith(self._normalize(c)) for c in concepts):
                score += 5

            norm_name = col.normalized

            if concept_key == "date":
                if any(b.upper() in norm_name for b in self.concepts.get("primary_date_boost", [])):
                    score += 20
                if any(p.upper() in norm_name for p in self.concepts.get("secondary_date_penalty", [])):
                    score -= 30

            if score > best_score:
                best_score = score
                best_col = col.name

        if best_score > 40:
            return best_col, best_score

        return None, best_score

    def _discover_join_keys(
        self,
        header_catalog: List[ColumnMetadata],
        line_catalog: List[ColumnMetadata],
    ) -> Tuple[Dict[str, Optional[str]], float]:
        h_ov = self.overrides.join_key.header
        l_ov = self.overrides.join_key.line
        if h_ov and l_ov:
            return {"header": h_ov, "line": l_ov}, 100.0

        h_map = {c.normalized: c for c in header_catalog}
        l_map = {c.normalized: c for c in line_catalog}

        intersection = set(h_map.keys()) & set(l_map.keys())

        if intersection:
            best_col = None
            best_score = -999

            for norm_name in intersection:
                score = self._score_similarity(norm_name, self.concepts["id"])

                if any(b.upper() in norm_name for b in self.concepts.get("transaction_id_boost", [])):
                    score += 50

                if any(p.upper() in norm_name for p in self.concepts.get("master_id_penalty", [])):
                    score -= 100

                if "CODE" in norm_name and "ORDER" not in norm_name and "SALES" not in norm_name:
                    score -= 150

                if any(prefix in norm_name for prefix in ["INTRASTAT", "TAX", "CUSTOMS", "FISCAL", "REGULATORY"]):
                    score -= 200

                if score > best_score:
                    best_score = score
                    best_col = norm_name

            if best_col and best_score > 0:
                return {"header": h_map[best_col].name, "line": l_map[best_col].name}, 95.0

        if _RAPIDFUZZ_AVAILABLE:
            best_pair = (None, None)
            best_score = -1

            h_names = [c.name for c in header_catalog]
            l_names = [c.name for c in line_catalog]

            for h_name in h_names:
                match = process.extractOne(h_name, l_names, scorer=fuzz.token_sort_ratio)
                if match and match[1] > best_score:
                    best_score = match[1]
                    best_pair = (h_name, match[0])

            if best_score > 85:
                return {"header": best_pair[0], "line": best_pair[1]}, best_score

        return {"header": None, "line": None}, 0.0

    def _resolve_line_overrides(self) -> ColumnOverride:
        """Select per-line-table overrides if provided."""
        if self.overrides.per_line and self.current_line_table in self.overrides.per_line:
            base = self.overrides.line.model_dump()
            specific = self.overrides.per_line[self.current_line_table].model_dump()
            merged = {**base, **{k: v for k, v in specific.items() if v is not None}}
            return ColumnOverride(**merged)
        return self.overrides.line

    def discover_context(self) -> DiscoveryContext:
        line_cat = self._build_catalog(self.current_line_table)
        header_cat = self._build_catalog(self.tables.header_table)

        ctx = DiscoveryContext()
        ctx.overrides_used = any(
            [
                self.overrides.header.date,
                self.overrides.header.amount,
                self.overrides.line.date,
                self.overrides.line.amount,
                self.overrides.join_key.header,
                self.overrides.join_key.line,
            ]
        )

        ctx.header_date, _ = self._find_best_column(
            header_cat, "date", "date", self.overrides.header.date
        )
        ctx.header_amount, _ = self._find_best_column(
            header_cat, "amount", "numeric", self.overrides.header.amount
        )

        line_overrides = self._resolve_line_overrides()
        ctx.line_date, _ = self._find_best_column(
            line_cat, "date", "date", line_overrides.date
        )
        ctx.line_amount, _ = self._find_best_column(
            line_cat, "amount", "numeric", line_overrides.amount
        )

        ctx.join_keys, _ = self._discover_join_keys(header_cat, line_cat)

        return ctx

    def generate_strategies(self, ctx: DiscoveryContext) -> List[AggregationStrategy]:
        strategies = []

        if ctx.header_date and ctx.header_amount:
            sql = f"""
            SELECT
                CAST(h.{self._quote(ctx.header_date)} AS DATE) AS OrderDate,
                SUM(h.{self._quote(ctx.header_amount)}) AS TotalAmount
            FROM {self.tables.header_table} h
            WHERE h.{self._quote(ctx.header_amount)} IS NOT NULL
            GROUP BY 1
            ORDER BY 1
            """
            strategies.append(
                AggregationStrategy(
                    name="header_only",
                    label="Aggregating from headers (preferred totals)",
                    sql_query=sql,
                    confidence_score=90.0,
                    required_columns={"date": ctx.header_date, "amount": ctx.header_amount},
                )
            )

        if ctx.line_date and ctx.line_amount:
            sql = f"""
            SELECT
                CAST(l.{self._quote(ctx.line_date)} AS DATE) AS OrderDate,
                SUM(l.{self._quote(ctx.line_amount)}) AS TotalAmount
            FROM {self.current_line_table} l
            WHERE l.{self._quote(ctx.line_amount)} IS NOT NULL
            GROUP BY 1
            ORDER BY 1
            """
            strategies.append(
                AggregationStrategy(
                    name="line_direct",
                    label="Aggregating from line items (self-contained)",
                    sql_query=sql,
                    confidence_score=80.0,
                    required_columns={"date": ctx.line_date, "amount": ctx.line_amount},
                )
            )

        if (
            ctx.line_amount
            and ctx.header_date
            and ctx.join_keys.get("header")
            and ctx.join_keys.get("line")
        ):
            sql = f"""
            SELECT
                CAST(h.{self._quote(ctx.header_date)} AS DATE) AS OrderDate,
                SUM(l.{self._quote(ctx.line_amount)}) AS TotalAmount
            FROM {self.current_line_table} l
            JOIN {self.tables.header_table} h
              ON l.{self._quote(ctx.join_keys['line'])} = h.{self._quote(ctx.join_keys['header'])}
            WHERE l.{self._quote(ctx.line_amount)} IS NOT NULL
            GROUP BY 1
            ORDER BY 1
            """
            strategies.append(
                AggregationStrategy(
                    name="line_join_header",
                    label="Aggregating from line items joined to headers",
                    sql_query=sql,
                    confidence_score=85.0,
                    required_columns={
                        "date": ctx.header_date,
                        "amount": ctx.line_amount,
                        "join_header": ctx.join_keys["header"],
                        "join_line": ctx.join_keys["line"],
                    },
                )
            )

        return strategies

    def select_strategy(self, strategies: List[AggregationStrategy]) -> AggregationStrategy:
        if not strategies:
            raise RuntimeError("No aggregation strategies available.")

        min_conf = self.discovery_config.min_confidence
        candidates = [s for s in strategies if s.confidence_score >= min_conf]
        if not candidates:
            raise RuntimeError(f"No strategies met minimum confidence {min_conf}.")

        pref_order = self.discovery_config.strategy_preference or []
        for pref in pref_order:
            for strategy in candidates:
                if strategy.name == pref:
                    return strategy

        return max(candidates, key=lambda s: s.confidence_score)

    def execute(self) -> Tuple[pd.DataFrame, DiscoveryContext]:
        ctx = self.discover_context()
        strategies = self.generate_strategies(ctx)
        strategy = self.select_strategy(strategies)

        logger.info("Selected strategy %s with confidence %.1f", strategy.name, strategy.confidence_score)
        logger.info("Columns used: %s", strategy.required_columns)

        df = self.con.execute(strategy.sql_query).df()
        if df.empty:
            logger.warning("Aggregation returned empty result set.")

        df["OrderDate"] = pd.to_datetime(df["OrderDate"])
        df = df.set_index("OrderDate").sort_index()

        ctx.selected_strategy = strategy.name
        ctx.selected_confidence = strategy.confidence_score
        ctx.strategy_columns = {**strategy.required_columns, "line_table": self.current_line_table}

        return df, ctx


def run_smart_discovery(
    con: Any,
    tables: TableConfig,
    overrides: Optional[OverridesConfig] = None,
    discovery_config: Optional[DiscoveryConfig] = None,
    current_line_table: Optional[str] = None,
) -> Tuple[pd.DataFrame, DiscoveryContext]:
    engine = SmartDiscoveryEngine(con, tables, overrides, discovery_config, current_line_table=current_line_table)
    return engine.execute()
