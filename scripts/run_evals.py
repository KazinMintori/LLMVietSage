"""Model-generation evaluations without retrieval, ORM or browser dependencies."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = ROOT / "evals"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OFFLINE_PROBES = {"response_guidance", "sse_event_contract", "inference_prompt"}
LIVE_PROBES = {"live_provider"}
COMMON_ASSERTIONS = {"equals", "not_equals", "null", "nonempty", "contains", "not_contains", "contains_any"}
BROWSER_ASSERTIONS = set()

class EvalContractError(ValueError):
    """Raised when a JSON suite violates the executable eval contract."""


def _get(value: Any, path: str) -> Any:
    if not isinstance(path, str):
        raise EvalContractError(f"Result path must be a string, got {type(path).__name__}")
    current = value
    for part in path.split(".") if path else []:
        if isinstance(current, dict):
            if part not in current:
                raise EvalContractError(f"Unknown result path {path!r}: missing key {part!r}")
            current = current[part]
        elif isinstance(current, list) and part.isdigit():
            index = int(part)
            if index >= len(current):
                raise EvalContractError(f"Unknown result path {path!r}: index {index} is out of range")
            current = current[index]
        else:
            raise EvalContractError(f"Unknown result path {path!r}: cannot read {part!r} from {current!r}")
    return current


def _assert(result: dict[str, Any], assertion: dict[str, Any]) -> None:
    if not isinstance(assertion, dict) or not isinstance(assertion.get("op"), str):
        raise EvalContractError(f"Assertion must contain a string op: {assertion!r}")
    op = assertion["op"]
    if op in BROWSER_ASSERTIONS:
        raise EvalContractError(f"Browser assertion reached offline runner: {op}")
    if op not in COMMON_ASSERTIONS:
        raise EvalContractError(f"Unknown assertion op {op!r} for offline result")
    if not isinstance(result, dict):
        raise AssertionError(f"Offline probe returned {type(result).__name__}, expected an object")
    value = _get(result, assertion.get("path", ""))
    expected = assertion.get("value")
    if op == "equals" and value != expected:
        raise AssertionError(f"{assertion['path']}={value!r}, expected {expected!r}")
    if op == "not_equals" and value == expected:
        raise AssertionError(f"{assertion['path']} unexpectedly equals {expected!r}")
    if op == "null" and value is not None:
        raise AssertionError(f"{assertion['path']}={value!r}, expected null")
    if op == "nonempty" and not value:
        raise AssertionError(f"{assertion['path']} is empty")
    if op == "contains":
        if isinstance(value, list):
            ok = expected in value
        else:
            ok = str(expected) in str(value)
        if not ok:
            raise AssertionError(f"{assertion['path']} does not contain {expected!r}")
    if op == "not_contains" and str(expected) in str(value):
        raise AssertionError(f"{assertion['path']} contains forbidden {expected!r}")
    if op == "contains_any":
        values = assertion.get("values")
        if not isinstance(values, list) or not values:
            raise EvalContractError("contains_any requires a non-empty values list")
        if not any(str(item) in str(value) for item in values):
            raise AssertionError(f"{assertion['path']} contains none of {values!r}")


def _load_cases(selected_suites: set[str] | None, selected_cases: set[str] | None) -> list[dict[str, Any]]:
    all_cases: list[dict[str, Any]] = []
    for path in sorted(EVAL_ROOT.glob("*.json")):
        suite = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(suite, dict) or not isinstance(suite.get("suite"), str) or not isinstance(suite.get("cases"), list):
            raise EvalContractError(f"Invalid eval suite contract in {path.relative_to(ROOT)}")
        if suite.get("schema_version") != "1.0":
            raise EvalContractError(f"Unsupported eval schema in {path.relative_to(ROOT)}: {suite.get('schema_version')!r}")
        if selected_suites and suite["suite"] not in selected_suites:
            continue
        for case in suite["cases"]:
            if not isinstance(case, dict) or not isinstance(case.get("id"), str):
                raise EvalContractError(f"Invalid eval case in {path.relative_to(ROOT)}")
            mode = case.get("mode", "offline")
            probe = case.get("probe")
            expected_probes = {"offline": OFFLINE_PROBES, "live": LIVE_PROBES}.get(mode)
            if expected_probes is None:
                raise EvalContractError(f"Case {case['id']!r} has unknown mode {mode!r}")
            if probe not in expected_probes:
                raise EvalContractError(f"Case {case['id']!r} has unknown {mode} probe {probe!r}")
            assertions = case.get("assertions", [])
            if not isinstance(assertions, list):
                raise EvalContractError(f"Case {case['id']!r} assertions must be a list")
            for assertion in assertions:
                if not isinstance(assertion, dict) or not isinstance(assertion.get("op"), str):
                    raise EvalContractError(f"Case {case['id']!r} has an assertion without a string op")
                op = assertion["op"]
                allowed = COMMON_ASSERTIONS | (BROWSER_ASSERTIONS if mode == "browser" else set())
                if op not in allowed:
                    raise EvalContractError(f"Case {case['id']!r} has unknown assertion op {op!r}")
                if op in COMMON_ASSERTIONS and not isinstance(assertion.get("path"), str):
                    raise EvalContractError(f"Case {case['id']!r} assertion {op!r} requires a string path")
            case = {**case, "suite": suite["suite"], "suite_file": str(path.relative_to(ROOT))}
            if selected_cases and case["id"] not in selected_cases:
                continue
            all_cases.append(case)
    if selected_suites:
        known_suites = {
            json.loads(path.read_text(encoding="utf-8"))["suite"]
            for path in EVAL_ROOT.glob("*.json")
        }
        unknown_suites = sorted(selected_suites - known_suites)
        if unknown_suites:
            raise EvalContractError(f"Unknown suite filter(s): {', '.join(unknown_suites)}")
    if selected_cases:
        known_cases = {
            case["id"]
            for path in sorted(EVAL_ROOT.glob("*.json"))
            for case in json.loads(path.read_text(encoding="utf-8")).get("cases", [])
        }
        unknown_cases = sorted(selected_cases - known_cases)
        if unknown_cases:
            raise EvalContractError(f"Unknown case filter(s): {', '.join(unknown_cases)}")
    if (selected_suites or selected_cases) and not all_cases:
        labels = []
        if selected_suites:
            labels.append(f"suite={sorted(selected_suites)!r}")
        if selected_cases:
            labels.append(f"case={sorted(selected_cases)!r}")
        raise EvalContractError(f"Eval filters matched no cases ({', '.join(labels)})")
    return all_cases


def _run_offline(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reports = []
    for case in cases:
        report = {"suite": case["suite"], "id": case["id"], "mode": case.get("mode", "offline")}
        try:
            result = _offline_probe(case)
            for assertion in case.get("assertions", []):
                _assert(result, assertion)
            report.update({"status": "passed", "result": result})
        except Exception as exc:  # noqa: BLE001
            report.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        reports.append(report)
    return reports


def _run_live(cases: list[dict[str, Any]], enabled: bool) -> list[dict[str, Any]]:
    if not cases:
        return []
    reason = (
        "Live provider evaluation is opt-in but no approved model/reference adapter is configured."
        if enabled else
        "Live provider evaluation is disabled by default; use a reviewed provider adapter and corpus."
    )
    return [{"suite": case["suite"], "id": case["id"], "mode": "live", "status": "skipped", "reason": reason} for case in cases]


def _offline_probe(case: dict[str, Any]) -> dict[str, Any]:
    data = case.get("input", {})
    if case["probe"] == "response_guidance":
        from app.services.response_style_service import prompt_guidance
        preferences, guidance = prompt_guidance(data["message"], data.get("fallback_locale", data.get("locale", "vi")))
        return {"locale": preferences.locale, "language_requested": preferences.language_requested,
                "detail": preferences.detail, "tone": preferences.tone, "format_hint": preferences.format_hint,
                "guidance": guidance}
    if case["probe"] == "sse_event_contract":
        from app.core.streaming import encode_sse_event
        events = data.get("events", [])
        return {"events": events, "encoded": "".join(encode_sse_event(name, {"text": "fixture"}) for name in events)}
    if case["probe"] == "inference_prompt":
        from app.domain.schema.inference_schema import ChatRequest
        from app.services.inference_service import prepare_chat
        messages, _options = prepare_chat(ChatRequest(**data))
        return {"roles": [m["role"] for m in messages], "last_user_content": messages[-1]["content"],
                "history_count": len(messages)-2}
    raise EvalContractError(f"Unknown generation probe {case['probe']!r}")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", action="append")
    parser.add_argument("--case", action="append")
    parser.add_argument("--mode", choices=("offline", "live", "all"), default="offline")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--require-live", action="store_true")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if args.require_live and args.mode not in {"live", "all"}:
        parser.error("--require-live needs --mode live or --mode all.")
    try:
        cases = _load_cases(set(args.suite or []), set(args.case or []))
        reports = []
        for mode in ("offline", "live"):
            if args.mode not in {mode, "all"}:
                continue
            selected = [case for case in cases if case.get("mode", "offline") == mode]
            if (args.suite or args.case) and not selected:
                reports.append({"mode": mode, "status": "failed", "error": "Selected filters matched no cases in this mode."})
            elif mode == "offline":
                reports.extend(_run_offline(selected))
            else:
                reports.extend(_run_live(selected, args.live))
    except EvalContractError as exc:
        reports = [{"mode": args.mode, "status": "failed", "error": str(exc)}]
    summary = {status: sum(item["status"] == status for item in reports) for status in ("passed", "failed", "skipped")}
    output = {"schema_version": "1.0", "command": "run_evals.py", "summary": summary, "cases": reports}
    text = json.dumps(output, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)
    return int(summary["failed"] > 0 or (args.require_live and any(item.get("mode") == "live" and item["status"] == "skipped" for item in reports)))


if __name__ == "__main__":
    raise SystemExit(main())
