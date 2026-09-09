"""Stdlib-only offline inspection commands."""

from __future__ import annotations

import json
import sys

from icgs.observability.inspect import inspect_run, trace_run


def _human_inspect(result: dict) -> None:
    print(f"status: {result['status']}")
    print(f"logs_complete: {result['logs_complete']}")
    counts = result["counts"]
    print(f"events: {counts['events']}  metrics: {counts['metrics']}  errors: {counts['errors']}")
    if result["missing_evidence"]:
        print("missing: " + ", ".join(result["missing_evidence"]))
    for warning in result["warnings"]:
        print(f"warning: {warning}", file=sys.stderr)


def _human_trace(result: dict) -> None:
    for span in result["spans"]:
        end = span["end"]
        duration = None if end is None else end.get("fields", {}).get("duration_ns")
        state = "unfinished" if span["unfinished"] else f"{duration}ns"
        print(f"{span['span_id']}: {state}")
    for event in result["events"]:
        print(f"{event.get('timestamp_utc')} {event.get('component')} {event.get('event')}")
    for warning in result["warnings"]:
        print(f"warning: {warning}", file=sys.stderr)


def run_inspect(args) -> None:
    result = inspect_run(args.run_dir)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        _human_inspect(result)


def run_trace(args) -> None:
    result = trace_run(args.run_dir, episode_id=args.episode,
                       decision_id=args.decision, candidate_id=args.candidate)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        _human_trace(result)


__all__ = ["run_inspect", "run_trace"]
