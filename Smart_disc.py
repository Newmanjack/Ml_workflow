import re
import logging
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Any, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SmartDiscovery")

try:
    from rapidfuzz import fuzz, process
    _RAPIDFUZZ_AVAILABLE = True
except ImportError:
    _RAPIDFUZZ_AVAILABLE = False
    logger.warning("rapidfuzz not found. Falling back to simple string matching.")

@dataclass
class ColumnMetadata:
    """Metadata for a column in a dataset."""
    name: str
    dtype: str
    normalized: str
    original_field: Any

@dataclass
class DiscoveryContext:
    """Holds all discovered columns and metadata for a pair of datasets."""
    header_date: Optional[str] = None
    header_amount: Optional[str] = None
    line_date: Optional[str] = None
    line_amount: Optional[str] = None
    join_keys: Dict[str, Optional[str]] = field(default_factory=dict)
    overrides_used: bool = False

@dataclass
class AggregationStrategy:
    """Represents a potential aggregation strategy."""
    name: str
    label: str
    sql_query: str
    confidence_score: float
    required_columns: Dict[str, str]

class SmartDiscoveryEngine:
    """
    A robust engine for discovering schema relationships and aggregation strategies
    between two datasets (Header and Line items).
    """

    def __init__(self, overrides: Optional[Dict] = None):
        self.overrides = overrides or {}

        # Semantic Concepts to look for
        self.concepts = {
            "date": ["date", "time", "created", "timestamp", "day", "period", "transaction"],
            "amount": ["total", "amount", "price", "value", "cost", "revenue", "sales", "turnover", "gross", "net"],
            "quantity": ["qty", "quantity", "units", "count", "volume"],
            "id": ["id", "num", "no", "key", "code", "identifier", "number"],
            # Boosts: Words that strongly suggest this is the RIGHT column
            "primary_date_boost": ["order", "doc", "trans", "created", "invoice"],
            "transaction_id_boost": ["order", "sales", "invoice", "trans", "doc", "header", "line"],
            # Penalties: Words that suggest this is a secondary/wrong column
            "secondary_date_penalty": ["payment", "due", "delivery", "ship", "start", "end", "valid", "term", "fiscal"],
            "master_id_penalty": ["person", "customer", "vendor", "item", "product", "type", "status", "group", "batch", "worker"]
        }

        # Data Type Tokens
        self.type_tokens = {
            "numeric": ("int", "decimal", "double", "float", "long"),
            "date": ("timestamp", "date", "time")
        }

    def _normalize(self, name: str) -> str:
        """Normalize column name for comparison (uppercase, alphanumeric only)."""
        return re.sub(r"[^A-Z0-9]", "", name.upper())

    def _build_catalog(self, dataset) -> List[ColumnMetadata]:
        """Build a catalog of column metadata from a PyArrow dataset."""
        if dataset is None:
            return []
        return [
            ColumnMetadata(
                name=f.name,
                dtype=str(f.type).lower(),
                normalized=self._normalize(f.name),
                original_field=f
            )
            for f in dataset.schema
        ]

    def _score_similarity(self, candidate: str, concepts: List[str]) -> float:
        """Score a candidate string against a list of concepts (0-100)."""
        if not _RAPIDFUZZ_AVAILABLE:
            candidate_upper = candidate.upper()
            return max((100 if c.upper() in candidate_upper else 0) for c in concepts)

        # IMPROVED LOGIC: Use partial_ratio to handle substrings (e.g. 'DATE' in 'PAYMENTTERMSBASEDATE')
        # token_set_ratio is good for "Order Date" vs "Date", but bad for "OrderDate" vs "Date" if not tokenized.
        # We take the max of both to be robust.
        scores = []
        for c in concepts:
            s1 = fuzz.partial_ratio(c.upper(), candidate.upper()) # Good for substrings
            s2 = fuzz.token_set_ratio(c, candidate)               # Good for reordered words
            scores.append(max(s1, s2))

        return max(scores) if scores else 0

    def _find_best_column(
        self,
        catalog: List[ColumnMetadata],
        concept_key: str,
        type_category: str,
        override_val: Optional[str] = None
    ) -> Tuple[Optional[str], float]:
        """
        Finds the best column for a concept.
        Returns: (column_name, score)
        """
        if override_val:
             return override_val, 100.0

        # 1. Filter by Type
        valid_types = self.type_tokens.get(type_category, [])
        candidates = [c for c in catalog if any(t in c.dtype for t in valid_types)]

        if not candidates:
            return None, 0.0

        # 2. Score by Concept
        best_col = None
        best_score = -999.0 # Allow negatives
        concepts = self.concepts.get(concept_key, [])

        for col in candidates:
            score = self._score_similarity(col.name, concepts)

            # Boost score for exact normalized matches
            if col.normalized in [self._normalize(c) for c in concepts]:
                score += 15

            # Boost score if concept is at the end of the name (e.g. OrderDate vs DateOrder)
            if any(col.normalized.endswith(self._normalize(c)) for c in concepts):
                score += 5

            # NEW: Contextual Boosts/Penalties
            norm_name = col.normalized

            # Boost for primary concepts (e.g. ORDERDATE > PAYMENTDATE)
            if concept_key == "date":
                 if any(b.upper() in norm_name for b in self.concepts.get("primary_date_boost", [])):
                     score += 20
                 if any(p.upper() in norm_name for p in self.concepts.get("secondary_date_penalty", [])):
                     score -= 30

            if score > best_score:
                best_score = score
                best_col = col.name

        # Threshold
        if best_score > 40: # Lowered slightly to allow for penalties to work relative to others
            return best_col, best_score

        return None, best_score

    def _discover_join_keys(
        self,
        header_catalog: List[ColumnMetadata],
        line_catalog: List[ColumnMetadata]
    ) -> Tuple[Dict[str, Optional[str]], float]:

        # Check overrides
        h_ov = self.overrides.get("join_key", {}).get("header")
        l_ov = self.overrides.get("join_key", {}).get("line")
        if h_ov and l_ov:
            return {"header": h_ov, "line": l_ov}, 100.0

        # Strategy 1: Exact Normalized Intersection
        h_map = {c.normalized: c for c in header_catalog}
        l_map = {c.normalized: c for c in line_catalog}

        intersection = set(h_map.keys()) & set(l_map.keys())

        if intersection:
            # Pick the one that looks most like a Transaction ID
            best_col = None
            best_score = -999 # Allow negatives

            for norm_name in intersection:
                # Base score: is it an ID?
                score = self._score_similarity(norm_name, self.concepts["id"])

                # STRONG Boost: Is it a Transaction ID?
                if any(b.upper() in norm_name for b in self.concepts.get("transaction_id_boost", [])):
                    score += 50

                # HARSH Penalties for bad columns
                if any(p.upper() in norm_name for p in self.concepts.get("master_id_penalty", [])):
                    score -= 100  # Increased from -50

                # Extra penalty for "CODE" (categorical/status codes)
                if "CODE" in norm_name and "ORDER" not in norm_name and "SALES" not in norm_name:
                    score -= 150  # Very harsh penalty for transaction codes

                # Extra penalty for "INTRASTAT" or other non-transactional prefixes
                if any(prefix in norm_name for prefix in ["INTRASTAT", "TAX", "CUSTOMS", "FISCAL", "REGULATORY"]):
                    score -= 200  # Extremely harsh

                logger.debug(f"  Join key candidate '{norm_name}': score={score}")

                if score > best_score:
                    best_score = score
                    best_col = norm_name

            # CONFIDENCE THRESHOLD: Only use the match if score is positive
            if best_col and best_score > 0:
                logger.info(f"Selected join key: '{best_col}' (score={best_score})")
                return {
                    "header": h_map[best_col].name,
                    "line": l_map[best_col].name
                }, 95.0
            else:
                logger.warning(f"Best exact match '{best_col}' has negative score ({best_score}). Skipping to fuzzy matching.")

        # Strategy 2: Fuzzy Intersection
        if _RAPIDFUZZ_AVAILABLE:
            best_pair = (None, None)
            best_score = -1

            h_names = [c.name for c in header_catalog]
            l_names = [c.name for c in line_catalog]

            # Heuristic: Join keys often appear first in tables, or have "ID" in them.
            # We compare all pairs.
            for h_name in h_names:
                match = process.extractOne(h_name, l_names, scorer=fuzz.token_sort_ratio)
                if match and match[1] > best_score:
                    best_score = match[1]
                    best_pair = (h_name, match[0])

            if best_score > 85:
                return {
                    "header": best_pair[0],
                    "line": best_pair[1]
                }, best_score

        return {"header": None, "line": None}, 0.0

    def _quote(self, identifier: str) -> str:
        return '"' + identifier.replace('"', '""') + '"'

    def discover_context(self, d1, d2) -> DiscoveryContext:
        """Discover all relevant columns in both datasets."""
        line_cat = self._build_catalog(d1)
        header_cat = self._build_catalog(d2)

        ctx = DiscoveryContext()
        ctx.overrides_used = bool(self.overrides)

        # Header Columns
        ctx.header_date, _ = self._find_best_column(
            header_cat, "date", "date",
            (self.overrides.get("header") or {}).get("date")
        )
        ctx.header_amount, _ = self._find_best_column(
            header_cat, "amount", "numeric",
            (self.overrides.get("header") or {}).get("amount")
        )

        # Line Columns
        ctx.line_date, _ = self._find_best_column(
            line_cat, "date", "date",
            (self.overrides.get("line") or {}).get("date")
        )
        ctx.line_amount, _ = self._find_best_column(
            line_cat, "amount", "numeric",
            (self.overrides.get("line") or {}).get("amount")
        )

        # Join Keys
        ctx.join_keys, _ = self._discover_join_keys(header_cat, line_cat)

        return ctx

    def generate_strategies(self, ctx: DiscoveryContext) -> List[AggregationStrategy]:
        strategies = []

        # Strategy A: Header Aggregation
        if ctx.header_date and ctx.header_amount:
            sql = f"""
            SELECT
                CAST(h.{self._quote(ctx.header_date)} AS DATE) AS OrderDate,
                SUM(h.{self._quote(ctx.header_amount)}) AS TotalAmount
            FROM headers h
            WHERE h.{self._quote(ctx.header_amount)} IS NOT NULL
            GROUP BY 1
            ORDER BY 1
            """
            strategies.append(AggregationStrategy(
                name="header_only",
                label="Aggregating from headers (preferred totals)",
                sql_query=sql,
                confidence_score=90.0, # High confidence if columns exist
                required_columns={"date": ctx.header_date, "amount": ctx.header_amount}
            ))

        # Strategy B: Line Direct
        if ctx.line_date and ctx.line_amount:
            sql = f"""
            SELECT
                CAST(l.{self._quote(ctx.line_date)} AS DATE) AS OrderDate,
                SUM(l.{self._quote(ctx.line_amount)}) AS TotalAmount
            FROM line_items l
            WHERE l.{self._quote(ctx.line_amount)} IS NOT NULL
            GROUP BY 1
            ORDER BY 1
            """
            strategies.append(AggregationStrategy(
                name="line_direct",
                label="Aggregating from line items (self-contained)",
                sql_query=sql,
                confidence_score=80.0,
                required_columns={"date": ctx.line_date, "amount": ctx.line_amount}
            ))

        # Strategy C: Line Join Header
        if ctx.line_amount and ctx.header_date and ctx.join_keys.get("header") and ctx.join_keys.get("line"):
            sql = f"""
            SELECT
                CAST(h.{self._quote(ctx.header_date)} AS DATE) AS OrderDate,
                SUM(l.{self._quote(ctx.line_amount)}) AS TotalAmount
            FROM line_items l
            JOIN headers h
              ON l.{self._quote(ctx.join_keys['line'])} = h.{self._quote(ctx.join_keys['header'])}
            WHERE l.{self._quote(ctx.line_amount)} IS NOT NULL
            GROUP BY 1
            ORDER BY 1
            """
            strategies.append(AggregationStrategy(
                name="line_join_header",
                label="Aggregating from line items joined to headers",
                sql_query=sql,
                confidence_score=85.0,
                required_columns={
                    "date": ctx.header_date,
                    "amount": ctx.line_amount,
                    "join_header": ctx.join_keys["header"],
                    "join_line": ctx.join_keys["line"]
                }
            ))

        return strategies

def run_smart_discovery(con: Any, d1: Any, d2: Any, overrides: Optional[Dict] = None) -> Tuple[Optional[pd.DataFrame], Dict]:
    """
    Main entry point.
    """
    if not con:
        print("⚠️ DuckDB connection not established.")
        return None, {}

    engine = SmartDiscoveryEngine(overrides)

    try:
        # 1. Discover Context
        ctx = engine.discover_context(d1, d2)

        print("\n🔎 Discovered Match Candidates:")
        print(f"  • Header Date:   '{ctx.header_date}'")
        print(f"  • Header Amount: '{ctx.header_amount}'")
        print(f"  • Line Date:     '{ctx.line_date}'")
        print(f"  • Line Amount:   '{ctx.line_amount}'")
        print(f"  • Join Keys:     {ctx.join_keys}\n")

        # 2. Generate Strategies
        strategies = engine.generate_strategies(ctx)

        if not strategies:
            # Construct error message
            missing = []
            if not ctx.header_date: missing.append("Header Date")
            if not ctx.header_amount: missing.append("Header Amount")
            if not ctx.line_date: missing.append("Line Date")
            if not ctx.line_amount: missing.append("Line Amount")
            if not ctx.join_keys.get("header"): missing.append("Join Key")

            raise RuntimeError(f"No valid aggregation strategy found. Missing concepts: {', '.join(missing)}")

        # 3. Pick Best Strategy
        # Priority: Header Only > Line Join > Line Direct (usually header totals are pre-calculated and safer)
        # But we can use confidence scores.
        best_strategy = max(strategies, key=lambda s: s.confidence_score)

        print(f"Strategy Selected: {best_strategy.label}")
        print(f"Columns Used: {best_strategy.required_columns}")

        # 4. Execute
        print("Executing Query...")
        df = con.execute(best_strategy.sql_query).df()

        # 5. Post-process
        df['OrderDate'] = pd.to_datetime(df['OrderDate'])
        df = df.set_index('OrderDate').sort_index()

        print(f"✅ Aggregation Complete. Rows: {len(df)}")

        # 6. Return Context (Compatible with Notebook)
        # We return the FULL discovery context so validation cells can work even if we didn't use those columns for aggregation
        notebook_context = {
            "strategy": best_strategy.name,
            "columns": best_strategy.required_columns,
            "header": {"date": ctx.header_date, "amount": ctx.header_amount},
            "line": {"date": ctx.line_date, "amount": ctx.line_amount},
            "join_key": ctx.join_keys,
            "overrides_used": ctx.overrides_used
        }

        return df, notebook_context

    except Exception as e:
        print(f"❌ Smart Discovery Failed: {e}")
        # Return partial context if possible for debugging
        return None, {}