import re
from typing import Any, Dict, Optional, Union

# ADVISORY DISCLAIMER: Derived flags are heuristic and not a substitute
# for the originating laboratory's own formal diagnostic flagging.


def parse_reference_range(text: str) -> Dict[str, Any]:
    """
    Parses a reference range text string into numeric boundary limits.
    Supports formats like '135-145', '0.5 - 1.2', '<5', '>10'.
    """
    result = {"low": None, "high": None, "low_inclusive": True, "high_inclusive": True}

    if not text or not isinstance(text, str):
        return result

    text = text.strip()

    # Pattern 1: Standard Range (e.g., "135-145" or "0.5 - 1.2")
    range_match = re.match(r"^([0-9.]+)\s*-\s*([0-9.]+)$", text)
    if range_match:
        result["low"] = float(range_match.group(1))
        result["high"] = float(range_match.group(2))
        return result

    # Pattern 2: Less than (e.g., "<5")
    lt_match = re.match(r"^<\s*([0-9.]+)$", text)
    if lt_match:
        result["high"] = float(lt_match.group(1))
        result["high_inclusive"] = False
        return result

    # Pattern 3: Greater than (e.g., ">10")
    gt_match = re.match(r"^>\s*([0-9.]+)$", text)
    if gt_match:
        result["low"] = float(gt_match.group(1))
        result["low_inclusive"] = False
        return result

    return result


def derive_abnormal_flag(
    value: Union[int, float, str],
    reference_range: Dict[str, Any],
    explicit_flag: Optional[str] = None,
) -> str:
    """
    Derives clinical abnormality status based on parsed bounds or explicit flags.
    """
    if explicit_flag:
        flag_upper = explicit_flag.upper().strip()
        if flag_upper in ["H", "HIGH"]:
            return "high"
        if flag_upper in ["L", "LOW"]:
            return "low"
        if flag_upper in ["CRITICAL", "C"]:
            return "critical"

    try:
        numeric_value = float(value)
    except (ValueError, TypeError):
        return "unknown"

    if not reference_range or not isinstance(reference_range, dict):
        return "unknown"

    low = reference_range.get("low")
    high = reference_range.get("high")

    if low is None and high is None:
        return "unknown"

    if low is not None:
        if reference_range.get("low_inclusive", True):
            if numeric_value < low:
                return "low"
        elif numeric_value <= low:
            return "low"

    if high is not None:
        if reference_range.get("high_inclusive", True):
            if numeric_value > high:
                return "high"
        elif numeric_value >= high:
            return "high"

    return "normal"
