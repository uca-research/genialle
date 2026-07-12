import json
import re
from typing import Any, Dict, Tuple

FINAL_MARKER = "<|channel|>final<|message|>"
END_MARKER = "<|end|>"
START_MARKER = "<|start|>"
REASONING_MARKERS = (
    "<|channel|>analysis",
    "<|channel|>commentary",
    "<|channel|>final",
)


def extract_final_content(raw_content: str) -> Tuple[str, bool]:
    text = raw_content or ""
    had_markers = any(marker in text for marker in REASONING_MARKERS)

    if FINAL_MARKER in text:
        final = text.rsplit(FINAL_MARKER, 1)[1]
        for marker in (END_MARKER, START_MARKER):
            if marker in final:
                final = final.split(marker, 1)[0]
        return final.strip(), had_markers

    match = re.search(
        r"<\|channel\|>final(?:<\|message\|>)?(.*?)(?:<\|end\|>|$)",
        text,
        flags=re.DOTALL,
    )
    if match:
        return match.group(1).strip(), had_markers
    return text.strip(), had_markers


def parse_json_object(text: str) -> Dict[str, Any]:
    candidate = (text or "").strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.I)
        candidate = re.sub(r"\s*```$", "", candidate)

    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            raise
        data = json.loads(candidate[start:end + 1])

    if not isinstance(data, dict):
        raise ValueError("Output is not a JSON object")
    return data
