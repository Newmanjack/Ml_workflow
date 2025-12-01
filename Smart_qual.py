import logging
import pandas as pd
import duckdb
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SmartQuality")

@dataclass
class ValidationResult:
    check_name: str
    status: str  # "PASS", "FAIL", "WARN"
    message: str
    details: Dict[str, Any]

class SmartValidator:
    """
    A robust engine for validating data quality, schema integrity, and reconciliation
    between datasets.
    """

    def __init__(self, con: duckdb.DuckDBPyConnection):
        self.con = con

    def _quote(self, identifier: str) -> str:
        return '"' + identifier.replace('"', '""') + '"'

    def _scan_all_columns(self, table_name: str) -> List[Dict[str, Any]]:
        """Full diagnostic scan of all columns in a table."""
        try:
            desc = self.con.execute(f"DESCRIBE {table_name}").fetchall()
            column_profiles = []

            for row in desc:
                col_name = row[0]
                col_type = row[1]
                try:
                    col_q = self._quote(col_name)
                    stats = self.con.execute(f"""
                        SELECT
                            COUNT(*) as total,
                            COUNT({col_q}) as non_null,
                            COUNT(DISTINCT {col_q}) as unique_vals
                        FROM {table_name}
                    """).fetchone()

                    total, non_null, unique = stats
                    null_pct = ((total - non_null) / total * 100) if total > 0 else 0
                    unique_pct = (unique / total * 100) if total > 0 else 0

                    column_profiles.append({
                        "name": col_name,
                        "type": col_type,
                        "total_rows": total,
                        "non_null": non_null,
                        "unique": unique,
                        "null_pct": null_pct,
                        "unique_pct": unique_pct,
                        "is_candidate_pk": (non_null == total and unique == total)
                    })
                except:
                    pass

            return column_profiles
        except Exception as e:
            logger.error(f"Column scan failed: {e}")
            return []

    def _find_better_pk(self, table_name: str, current_candidate: str) -> Optional[str]:
        """Scans the table for a valid Primary Key if the current one fails."""
        try:
            profiles = self._scan_all_columns(table_name)

            # Filter columns that are NOT the current bad candidate and qualify as PK
            pk_candidates = [
                p for p in profiles
                if p["is_candidate_pk"] and p["name"] != current_candidate
            ]

            if not pk_candidates:
                return None

            # Prioritize by naming convention
            def pk_score(p):
                name_upper = p["name"].upper()
                score = 0

                # Strong indicators
                if name_upper.endswith("ID"): score += 100
                if name_upper.endswith("KEY"): score += 90
                if name_upper.endswith("NUM") or name_upper.endswith("NUMBER"): score += 80
                if "RECID" in name_upper: score += 150  # Common ERP PK
                if "SALESID" in name_upper or "ORDERID" in name_upper: score += 120

                # Weak indicators (avoid)
                if "CODE" in name_upper: score -= 50
                if "STATUS" in name_upper: score -= 50
                if "TYPE" in name_upper: score -= 50

                return score

            # Return the best-scoring candidate
            best = max(pk_candidates, key=pk_score)
            return best["name"]

        except Exception as e:
            logger.error(f"PK discovery failed: {e}")
            return None

    def check_primary_key(self, table_name: str, candidate_key: str) -> ValidationResult:
        """
        Verifies if a column acts as a Primary Key (Unique & Not Null).
        """
        try:
            key_q = self._quote(candidate_key)
            query = f"""
            SELECT
                COUNT(*) as total_rows,
                COUNT({key_q}) as non_null_count,
                COUNT(DISTINCT {key_q}) as unique_count
            FROM {table_name}
            """
            res = self.con.execute(query).fetchone()
            total, non_null, unique = res

            details = {"total_rows": total, "non_null": non_null, "unique": unique}

            if total == 0:
                return ValidationResult("PrimaryKey", "WARN", f"Table '{table_name}' is empty.", details)

            if non_null != total or unique != total:
                # Proactive: Find a better key
                better_key = self._find_better_pk(table_name, candidate_key)
                suggestion = ""
                if better_key:
                    suggestion = f" 💡 SUGGESTION: Use '{better_key}' instead."
                    details["suggested_key"] = better_key
                else:
                    # Debugging: Why didn't we find a better key?
                    # logger.info(f"No better PK found for {table_name}. Checked candidates.")
                    pass

                duplicates = total - unique
                nulls = total - non_null
                msg = f"Column '{candidate_key}' failed PK check ({nulls} nulls, {duplicates} duplicates).{suggestion}"
                return ValidationResult("PrimaryKey", "FAIL", msg, details)

            return ValidationResult("PrimaryKey", "PASS", f"Column '{candidate_key}' is a valid Primary Key.", details)

        except Exception as e:
            return ValidationResult("PrimaryKey", "ERROR", str(e), {})

    def _find_matching_amounts(self, header_table: str, line_table: str, h_total: float, l_total: float) -> Optional[Dict[str, Any]]:
        """
        Scans ALL numeric columns to find a better matching pair.
        Returns: {"header_col": str, "line_col": str, "gap_pct": float} or None
        """
        try:
            # Get ALL numeric columns
            h_desc = self.con.execute(f"DESCRIBE {header_table}").fetchall()
            l_desc = self.con.execute(f"DESCRIBE {line_table}").fetchall()

            h_numeric = [row[0] for row in h_desc if row[1].upper() in ['DOUBLE', 'DECIMAL', 'BIGINT', 'INTEGER', 'FLOAT', 'NUMERIC']]
            l_numeric = [row[0] for row in l_desc if row[1].upper() in ['DOUBLE', 'DECIMAL', 'BIGINT', 'INTEGER', 'FLOAT', 'NUMERIC']]

            # Prioritize columns with "AMOUNT", "TOTAL", "SUM", "VALUE", "PRICE", "NET", "GROSS"
            likely_names = ["AMOUNT", "TOTAL", "SUM", "VALUE", "PRICE", "NET", "GROSS", "SALES"]

            def priority_score(name):
                name_upper = name.upper()
                score = 0
                for keyword in likely_names:
                    if keyword in name_upper:
                        score += 10
                # Penalize "UNIT" (likely unit price, not total)
                if "UNIT" in name_upper:
                    score -= 20
                return score

            h_sorted = sorted(h_numeric, key=priority_score, reverse=True)[:8]
            l_sorted = sorted(l_numeric, key=priority_score, reverse=True)[:8]

            # Compute sums for all candidates
            h_sums = {}
            for hc in h_sorted:
                try:
                    h_sums[hc] = self.con.execute(f"SELECT SUM({self._quote(hc)}) FROM {header_table}").fetchone()[0] or 0
                except:
                    pass

            l_sums = {}
            for lc in l_sorted:
                try:
                    l_sums[lc] = self.con.execute(f"SELECT SUM({self._quote(lc)}) FROM {line_table}").fetchone()[0] or 0
                except:
                    pass

            # Find the best matching pair
            best_match = None
            best_gap = float('inf')

            for hc, h_sum in h_sums.items():
                for lc, l_sum in l_sums.items():
                    if h_sum == 0 or l_sum == 0:
                        continue

                    gap = abs((h_sum - l_sum) / h_sum * 100)

                    if gap < best_gap:
                        best_gap = gap
                        best_match = {
                            "header_col": hc,
                            "line_col": lc,
                            "header_sum": h_sum,
                            "line_sum": l_sum,
                            "gap_pct": gap
                        }

            # If we found a pair with <5% gap, return it
            if best_match and best_match["gap_pct"] < 5.0:
                return best_match

            return None

        except Exception as e:
            logger.error(f"Amount matching failed: {e}")
            return None

    def reconcile_amounts(self, header_table: str, line_table: str, header_amt: str, line_amt: str) -> ValidationResult:
        """
        Reconciles the sum of amounts between Header and Line tables.
        """
        try:
            h_amt_q = self._quote(header_amt)
            l_amt_q = self._quote(line_amt)

            # Simple Sums
            h_total = self.con.execute(f"SELECT SUM({h_amt_q}) FROM {header_table}").fetchone()[0] or 0
            l_total = self.con.execute(f"SELECT SUM({l_amt_q}) FROM {line_table}").fetchone()[0] or 0

            diff = h_total - l_total
            pct_diff = (diff / h_total) * 100 if h_total != 0 else 0

            details = {"header_total": h_total, "line_total": l_total, "diff": diff, "pct_diff": pct_diff}

            if abs(pct_diff) < 1.0:
                return ValidationResult("Reconciliation", "PASS", f"Totals match within 1% ({pct_diff:.2f}%).", details)

            # Diagnosis 1: Is it a Quantity issue?
            qty_cols = [row[0] for row in self.con.execute(f"DESCRIBE {line_table}").fetchall() if 'QTY' in row[0].upper() or 'QUANTITY' in row[0].upper()]

            diagnosis = ""
            if qty_cols:
                qty_col = qty_cols[0]
                qty_q = self._quote(qty_col)
                l_total_calc = self.con.execute(f"SELECT SUM({l_amt_q} * {qty_q}) FROM {line_table}").fetchone()[0] or 0
                diff_calc = h_total - l_total_calc
                pct_diff_calc = (diff_calc / h_total) * 100 if h_total != 0 else 0

                if abs(pct_diff_calc) < 5.0:
                    diagnosis = f" (Likely Cause: Line Amount is Unit Price. Corrected using {qty_col} -> Gap: {pct_diff_calc:.2f}%)"
                    details["diagnosis"] = "Unit Price Issue"
                    details["corrected_line_total"] = l_total_calc

            # Diagnosis 2: Proactive Column Search
            if not diagnosis:
                better_match = self._find_matching_amounts(header_table, line_table, h_total, l_total)
                if better_match:
                    diagnosis = f" 💡 SUGGESTION: Use '{better_match['header_col']}' (header) ↔ '{better_match['line_col']}' (line). Gap: {better_match['gap_pct']:.2f}%"
                    details["suggested_amounts"] = better_match

            return ValidationResult("Reconciliation", "FAIL", f"Significant Gap: {pct_diff:.2f}%{diagnosis}", details)

        except Exception as e:
            return ValidationResult("Reconciliation", "ERROR", str(e), {})

    def analyze_join_relationship(self, header_table: str, line_table: str, header_key: str, line_key: str) -> ValidationResult:
        """
        Analyzes the relationship between Header and Line tables (1:N, Orphans, etc.).
        NOW WITH: Automatic FK suggestion if the provided keys produce bad joins.
        """
        try:
            hk_q = self._quote(header_key)
            lk_q = self._quote(line_key)

            # Check for Orphans (Lines without Headers)
            orphan_query = f"""
            SELECT COUNT(*)
            FROM {line_table} l
            LEFT JOIN {header_table} h ON l.{lk_q} = h.{hk_q}
            WHERE h.{hk_q} IS NULL
            """
            orphan_count = self.con.execute(orphan_query).fetchone()[0]

            # Check for Childless Headers (Headers without Lines)
            childless_query = f"""
            SELECT COUNT(*)
            FROM {header_table} h
            LEFT JOIN {line_table} l ON h.{hk_q} = l.{lk_q}
            WHERE l.{lk_q} IS NULL
            """
            childless_count = self.con.execute(childless_query).fetchone()[0]

            # Check total rows to calculate orphan percentage
            total_lines = self.con.execute(f"SELECT COUNT(*) FROM {line_table}").fetchone()[0]
            total_headers = self.con.execute(f"SELECT COUNT(*) FROM {header_table}").fetchone()[0]

            orphan_pct = (orphan_count / total_lines * 100) if total_lines > 0 else 0
            childless_pct = (childless_count / total_headers * 100) if total_headers > 0 else 0

            details = {
                "orphans": orphan_count,
                "childless_headers": childless_count,
                "orphan_pct": orphan_pct,
                "childless_pct": childless_pct
            }

            # If orphan rate > 80%, the join key is almost certainly wrong
            if orphan_pct > 80:
                # PROACTIVE: Find better join keys
                suggestion = self._find_better_join_keys(header_table, line_table, header_key, line_key)
                if suggestion:
                    details["suggested_keys"] = suggestion
                    msg = f"❌ CRITICAL: {orphan_pct:.1f}% of Lines are orphaned. Current keys are likely wrong. 💡 SUGGESTION: {suggestion}"
                    return ValidationResult("JoinAnalysis", "FAIL", msg, details)
                else:
                    msg = f"❌ CRITICAL: {orphan_pct:.1f}% of Lines are orphaned. Current keys: '{header_key}' ↔ '{line_key}' appear invalid."
                    return ValidationResult("JoinAnalysis", "FAIL", msg, details)

            msgs = []
            status = "PASS"

            if orphan_count > 0:
                status = "WARN"
                msgs.append(f"{orphan_count} Line items have no matching Header (Orphans).")

            if childless_count > 0:
                msgs.append(f"{childless_count} Headers have no Line items.")

            if not msgs:
                msgs.append("Perfect Referential Integrity (All lines have headers).")

            return ValidationResult("JoinAnalysis", status, " | ".join(msgs), details)

        except Exception as e:
            return ValidationResult("JoinAnalysis", "ERROR", str(e), {})

    def _find_better_join_keys(self, header_table: str, line_table: str, current_h_key: str, current_l_key: str) -> Optional[str]:
        """
        Scans all columns to find a better FK relationship.
        Returns a suggestion string like "Use 'SalesId' (header) ↔ 'SalesOrderNumber' (line)".
        """
        try:
            h_profiles = self._scan_all_columns(header_table)
            l_profiles = self._scan_all_columns(line_table)

            # Strategy 1: Exact Name Match (excluding bad current keys)
            h_names = {p["name"].upper(): p["name"] for p in h_profiles if p["name"] != current_h_key}
            l_names = {p["name"].upper(): p["name"] for p in l_profiles if p["name"] != current_l_key}

            common = set(h_names.keys()) & set(l_names.keys())

            # Filter out "CODE", "STATUS", "TYPE"
            common = {c for c in common if "CODE" not in c and "STATUS" not in c and "TYPE" not in c}

            if common:
                # Pick the most "ID-like" one
                def id_score(name):
                    score = 0
                    if name.endswith("ID"): score += 100
                    if name.endswith("NUM"): score += 80
                    if "SALES" in name or "ORDER" in name: score += 50
                    return score

                best = max(common, key=id_score)
                return f"Use '{h_names[best]}' (header) ↔ '{l_names[best]}' (line)"

            # Strategy 2: Fuzzy Match (look for similar names like SalesId vs SalesOrderId)
            # This is computationally expensive, so we only check top candidates
            h_id_cols = [p["name"] for p in h_profiles if re.search(r"(ID|NUM|KEY)$", p["name"].upper())][:10]
            l_id_cols = [p["name"] for p in l_profiles if re.search(r"(ID|NUM|KEY)$", p["name"].upper())][:10]

            best_pair = None
            best_match_score = 0

            for h_col in h_id_cols:
                for l_col in l_id_cols:
                    # Simple similarity: count common tokens
                    h_tokens = set(re.findall(r"[A-Z][a-z]*", h_col))
                    l_tokens = set(re.findall(r"[A-Z][a-z]*", l_col))
                    common_tokens = h_tokens & l_tokens

                    if common_tokens:
                        score = len(common_tokens)
                        if score > best_match_score:
                            best_match_score = score
                            best_pair = (h_col, l_col)

            if best_pair and best_match_score >= 2:
                return f"Use '{best_pair[0]}' (header) ↔ '{best_pair[1]}' (line)"

            return None

        except Exception as e:
            logger.error(f"Join key discovery failed: {e}")
            return None

    def discover_and_check_line_pk(self, table_name: str, exclude_cols: List[str]) -> ValidationResult:
        """
        Attempts to discover and validate a Primary Key for the line table.
        """
        try:
            profiles = self._scan_all_columns(table_name)

            if not profiles:
                return ValidationResult("LinePK", "ERROR", "Column scan failed for line table.", {})

            # Filter valid PK candidates (excluding the FK)
            pk_candidates = [
                p for p in profiles
                if p["is_candidate_pk"] and p["name"] not in exclude_cols
            ]

            if not pk_candidates:
                # Show what columns we DID find for debugging
                all_cols = [p["name"] for p in profiles[:10]]
                return ValidationResult("LinePK", "WARN", f"No candidate Primary Key found. Scanned columns: {all_cols}", {})

            # Pick the best one by naming convention
            def pk_score(p):
                name_upper = p["name"].upper()
                score = 0
                if name_upper.endswith("ID"): score += 100
                if "RECID" in name_upper: score += 150
                if "LINEID" in name_upper: score += 120
                return score

            best = max(pk_candidates, key=pk_score)

            return ValidationResult("LinePK", "PASS", f"Discovered valid Line Primary Key: '{best['name']}'", best)

        except Exception as e:
            return ValidationResult("LinePK", "ERROR", str(e), {})

def run_smart_validation(con: duckdb.DuckDBPyConnection, context: Dict) -> List[ValidationResult]:
    """
    Main entry point for validation.
    NOW WITH: Full diagnostic reporting and automatic fix suggestions.
    """
    validator = SmartValidator(con)
    results = []

    # DIAGNOSTIC: Show what Discovery actually found
    print(f"🔍 CONTEXT FROM DISCOVERY:")
    print(f"   Strategy: {context.get('strategy', 'UNKNOWN')}")
    print(f"   Join Keys: {context.get('join_key', {})}")
    print(f"   Header Columns: {context.get('header', {})}")
    print(f"   Line Columns: {context.get('line', {})}")
    print()

    # 1. Validate Keys
    join_keys = context.get("join_key", {})
    h_key = join_keys.get("header")
    l_key = join_keys.get("line")

    # NOTE: We don't validate join keys as Primary Keys anymore.
    # Join keys can have duplicates (1-to-many relationships are valid).
    # Validation will auto-discover true PKs in the diagnostic phase.

    # 2. Validate Relationship & Discover Primary Keys
    if h_key and l_key:
        results.append(validator.analyze_join_relationship("headers", "line_items", h_key, l_key))

        # Auto-discover true Primary Keys (for reference/debugging)
        results.append(validator.discover_and_check_line_pk("line_items", [l_key]))

        # Also discover header PK
        better_h_pk = validator._find_better_pk("headers", h_key)
        if better_h_pk:
            results.append(ValidationResult("HeaderPK", "PASS", f"Discovered valid Header Primary Key: '{better_h_pk}'", {"suggested_pk": better_h_pk}))
        else:
            results.append(ValidationResult("HeaderPK", "WARN", "No distinct Primary Key found for headers", {}))

    # 3. Reconcile Totals
    h_amt = context.get("header", {}).get("amount")
    l_amt = context.get("line", {}).get("amount")

    if h_amt and l_amt:
        results.append(validator.reconcile_amounts("headers", "line_items", h_amt, l_amt))

    # 4. DIAGNOSTIC SUMMARY: Collect all suggestions
    suggestions = []
    header_pk = None
    line_pk = None

    for res in results:
        # Primary Keys
        if res.check_name == "HeaderPK" and res.status == "PASS":
            header_pk = res.details.get("suggested_pk")
            if header_pk:
                suggestions.append(f"🔧 Primary Key: Use '{header_pk}' for headers.")

        if res.check_name == "LinePK" and res.status == "PASS":
            line_pk = res.details.get("name")
            if line_pk:
                suggestions.append(f"🔧 Primary Key: Use '{line_pk}' for lines.")

        # Join Keys
        if "suggested_join_keys" in res.details:
            suggestions.append(f"🔗 Join Keys: {res.details['suggested_join_keys']}")

        # Amount Columns
        if "suggested_amounts" in res.details:
            sugg = res.details["suggested_amounts"]
            suggestions.append(f"💰 Amount Columns: {sugg['header_col']} ↔ {sugg['line_col']} (Gap: {sugg['gap_pct']:.2f}%)")

    if suggestions:
        print("\n🛠️ AUTOMATED DIAGNOSIS SUGGESTIONS:")
        for s in suggestions:
            print(f"   {s}")

        # Show that PKs are informational only
        if header_pk or line_pk:
            print("\n📝 Note: Primary Keys are discovered for reference. Join keys can have duplicates (1-to-many is valid).")

        print("\n💡 To apply amount/join key fixes, update the SMART_OVERRIDES cell:")
        print("```python")
        print("SMART_OVERRIDES = {")

        # Generate override code
        for res in results:
            if "suggested_join_keys" in res.details:
                sugg_text = res.details["suggested_join_keys"]
                # Parse the suggestion (e.g., "Use 'SalesId' (header) ↔ 'SalesOrderNumber' (line)")
                import re
                match = re.search(r"'([^']+)' \(header\) ↔ '([^']+)' \(line\)", sugg_text)
                if match:
                    h_sugg, l_sugg = match.groups()
                    print(f'    "join_key": {{"header": "{h_sugg}", "line": "{l_sugg}"}},')
            if "suggested_amounts" in res.details:
                sugg = res.details["suggested_amounts"]
                print(f'    "header": {{"amount": "{sugg["header_col"]}"}},')
                print(f'    "line": {{"amount": "{sugg["line_col"]}"}},')

        print("}")
        print("```")

    return results
