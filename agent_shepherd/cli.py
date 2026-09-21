"""Command line interface for agent-shepherd."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .core.config import ShepherdConfig
from .core.ledger import Ledger
from .core.server import create_daemon
from .core.types import Agent


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="shepherd", description="Supervise AI agents (QwenPaw, Claude Code, Codex)")
    p.add_argument("--config", type=Path, help="path to config.yaml (default: ~/.shepherd/config.yaml)")
    sub = p.add_subparsers(dest="command", required=True)

    install = sub.add_parser("install", help="install adapter hooks for an agent")
    install.add_argument("agent", choices=["qwenpaw", "claude", "codex"])
    install.add_argument("--dry-run", action="store_true")

    sub.add_parser("start", help="start the supervisor daemon")
    sub.add_parser("stop", help="stop the supervisor daemon (placeholder)")

    tail = sub.add_parser("tail", help="follow the audit ledger for a session")
    tail.add_argument("agent", choices=["qwenpaw", "claude", "codex"])
    tail.add_argument("session_id")

    replay = sub.add_parser("replay", help="replay the audit ledger for a session")
    replay.add_argument("agent", choices=["qwenpaw", "claude", "codex"])
    replay.add_argument("session_id")

    stats = sub.add_parser("stats", help="summarize what the supervisor did in one session")
    stats.add_argument("agent", choices=["qwenpaw", "claude", "codex"])
    stats.add_argument("session_id")

    ev = sub.add_parser(
        "eval",
        help="run the offline counterfactual benchmark (fault injection with known onsets)",
    )
    ev.add_argument("--seed", type=int, default=7)
    ev.add_argument("--sessions", type=int, default=8, help="sessions per fault kind")
    ev.add_argument("--steps", type=int, default=60, help="events per session")
    ev.add_argument("--json", dest="json_out", help="write the full report to this path")
    ev.add_argument(
        "--fail-under-f1",
        type=float,
        dest="fail_under_f1",
        help="exit non-zero if aggregate F1 is below this (CI gate)",
    )
    return p


def _agent(name: str) -> Agent:
    return {
        "qwenpaw": Agent.QWENPAW,
        "claude": Agent.CLAUDE,
        "codex": Agent.CODEX,
    }[name]


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cfg = ShepherdConfig.load(args.config)

    if args.command == "start":
        cfg.ensure_defaults()
        daemon = create_daemon(cfg)
        try:
            daemon.serve_forever()
        except KeyboardInterrupt:
            daemon.stop()
        return 0

    if args.command == "install":
        print(f"installing {args.agent} adapter (dry-run={args.dry_run})")
        from .adapters.backup import UnparseableConfig

        try:
            if args.agent == "qwenpaw":
                from .adapters.qwenpaw.install import install_qwenpaw

                install_qwenpaw(cfg, dry_run=args.dry_run)
            elif args.agent == "claude":
                from .adapters.claude.install import install_claude

                install_claude(cfg, dry_run=args.dry_run)
            elif args.agent == "codex":
                from .adapters.codex.install import install_codex

                install_codex(cfg, dry_run=args.dry_run)
        except UnparseableConfig as exc:
            print(
                f"refusing to write: {exc}\n"
                "the existing config does not parse, and overwriting it would destroy "
                "hand-edited settings. Fix the file (or delete it) and re-run.",
                file=sys.stderr,
            )
            return 2
        return 0

    if args.command == "stats":
        ledger = Ledger()
        agent = _agent(args.agent)
        records = ledger.iter_records(agent, args.session_id)
        verdicts = [r for r in records if r.get("type") == "verdict"]
        nudges = [r for r in verdicts if r.get("action") == "nudge"]
        by_detector: dict[str, int] = {}
        for r in nudges:
            by_detector[str(r.get("detector"))] = by_detector.get(str(r.get("detector")), 0) + 1
        last_cost = next((r.get("context") for r in reversed(verdicts) if r.get("context")), {})
        print(
            json.dumps(
                {
                    "agent": args.agent,
                    "session_id": args.session_id,
                    "records": len(records),
                    "verdicts": len(verdicts),
                    "nudges": len(nudges),
                    "nudges_by_detector": by_detector,
                    "suppressed": sum(1 for r in verdicts if "suppressed" in str(r.get("reason"))),
                    "cost": last_cost,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "eval":
        from .eval.harness import main as eval_main

        return eval_main(
            [
                "--seed",
                str(args.seed),
                "--sessions",
                str(args.sessions),
                "--steps",
                str(args.steps),
                *(["--json", args.json_out] if args.json_out else []),
                *(["--fail-under-f1", str(args.fail_under_f1)] if args.fail_under_f1 else []),
            ]
        )

    if args.command == "tail":
        ledger = Ledger()
        agent = _agent(args.agent)
        print(f"tailing {agent.value}/{args.session_id} (Ctrl-C to stop)")
        seen = 0
        while True:
            records = ledger.iter_records(agent, args.session_id)
            for rec in records[seen:]:
                print(json.dumps(rec, ensure_ascii=False))
                seen += 1
            time.sleep(0.5)

    if args.command == "replay":
        ledger = Ledger()
        for rec in ledger.iter_records(_agent(args.agent), args.session_id):
            print(json.dumps(rec, ensure_ascii=False))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())