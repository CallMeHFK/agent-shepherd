"""Command line interface for agent-shepherd."""

from __future__ import annotations

import argparse
import json
import os
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

    start = sub.add_parser("start", help="start the supervisor daemon in the foreground")
    start.add_argument("--port", type=int, help="override config port (or SHEPHERD_PORT)")
    start_bg = sub.add_parser("start-bg", help="start the daemon detached, writing a pidfile")
    start_bg.add_argument("--port", type=int)
    sub.add_parser("stop", help="stop a daemon started with `shepherd start-bg`")
    sub.add_parser("status", help="report whether the daemon is up and what it has decided")
    risk = sub.add_parser("risk", help="show the risk-controlled admission thresholds")
    risk.add_argument("--fit-from-eval", action="store_true", help=argparse.SUPPRESS)

    tail = sub.add_parser("tail", help="follow the audit ledger for a session")
    tail.add_argument("agent", choices=["qwenpaw", "claude", "codex"])
    tail.add_argument("session_id")

    replay = sub.add_parser("replay", help="replay the audit ledger for a session")
    replay.add_argument("agent", choices=["qwenpaw", "claude", "codex"])
    replay.add_argument("session_id")

    stats = sub.add_parser("stats", help="summarize what the supervisor did in one session")
    stats.add_argument("agent", choices=["qwenpaw", "claude", "codex"])
    stats.add_argument("session_id")

    cfgcmd = sub.add_parser("config", help="show and edit ~/.shepherd/config.yaml")
    cfgsub = cfgcmd.add_subparsers(dest="subcommand", required=True)
    cfgsub.add_parser("show", help="every effective setting and where it came from")
    setter = cfgsub.add_parser("set", help="set one setting, e.g. judge.model step-3")
    setter.add_argument("key")
    setter.add_argument("value")
    cfgsub.add_parser("merge", help="add settings this file predates, keeping your values")
    cfgsub.add_parser("prune", help="delete settings this file no longer configures anything")
    cfgsub.add_parser("judge", help="list model names the configured judge endpoint accepts")

    sub.add_parser("doctor", help="what is configured, what is missing, what to do next")
    ev = sub.add_parser(
        "eval",
        help="run the offline counterfactual benchmark (fault injection with known onsets)",
    )
    ev.add_argument("--seed", type=int, default=7)
    ev.add_argument("--sessions", type=int, default=8, help="sessions per fault kind")
    ev.add_argument("--steps", type=int, default=60, help="events per session")
    ev.add_argument("--json", dest="json_out", help="write the full report to this path")
    ev.add_argument("--fault", action="append", help="only run these faults (repeatable)")
    ev.add_argument("--judge", action="store_true", help="enable Tier 1 (needs a reachable judge)")
    ev.add_argument(
        "--fit-risk",
        metavar="OUT",
        default=None,
        help="fit admission thresholds from the corpus labels into OUT "
        "(use `all` for the file the daemon reads)",
    )
    ev.add_argument(
        "--fail-under-f1",
        type=float,
        dest="fail_under_f1",
        help="exit non-zero if aggregate F1 is below this (CI gate)",
    )
    return p


def _pidfile() -> Path:
    from .core.ledger import default_root

    return default_root() / "daemon.pid"


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
        port = getattr(args, "port", None) or os.environ.get("SHEPHERD_PORT")
        if port:
            cfg.port = int(port)
        daemon = create_daemon(cfg)
        try:
            daemon.serve_forever()
        except KeyboardInterrupt:
            daemon.stop()
        return 0

    if args.command == "start-bg":
        import subprocess

        cfg.ensure_defaults()
        if args.port:
            cfg.port = int(args.port)
        _pidfile().parent.mkdir(parents=True, exist_ok=True)
        log = _pidfile().with_suffix(".log")
        with open(log, "ab") as fh:
            proc = subprocess.Popen(
                [sys.executable, "-m", "agent_shepherd.cli", "start"]
                + ([ "--port", str(args.port)] if args.port else []),
                stdout=fh,
                stderr=fh,
                start_new_session=True,
                env={
                    **dict(os.environ),
                    # Unbuffered, or the "listening" line and any traceback sit in
                    # the child's block-buffered stdout until it is killed and the
                    # log file is left empty.
                    "PYTHONUNBUFFERED": "1",
                    **({"SHEPHERD_PORT": str(args.port)} if args.port else {}),
                },
            )
        _pidfile().write_text(str(proc.pid))
        # Returning here would report success for a process that is not yet
        # listening: the drift detector's Monte-Carlo fit runs before the socket
        # is bound, so for several seconds every adapter POST is refused and the
        # hook answers {} -- indistinguishable from "healthy, nothing to say".
        # That is exactly how a real "detector does not fire" report turned out to
        # be a readiness bug, so wait for /health before claiming anything.
        url = f"http://127.0.0.1:{cfg.port}"
        import httpx

        deadline = time.time() + 90.0
        while time.time() < deadline:
            if proc.poll() is not None:
                print(f"daemon exited rc={proc.returncode}; see {log}", file=sys.stderr)
                _pidfile().unlink(missing_ok=True)
                return 1
            try:
                if httpx.get(f"{url}/health", timeout=2.0, trust_env=False).json().get("ok"):
                    print(f"daemon ready pid={proc.pid} on {url} (log={log})")
                    return 0
            except Exception:  # noqa: BLE001 - still starting
                time.sleep(0.4)
        print(f"daemon pid={proc.pid} never became healthy on {url}; see {log}", file=sys.stderr)
        return 1

    if args.command == "stop":
        import signal

        path = _pidfile()
        if not path.exists():
            print("no pidfile: the daemon was not started with `shepherd start-bg`", file=sys.stderr)
            return 1
        pid = int(path.read_text().strip() or 0)
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"sent SIGTERM to {pid}")
        except ProcessLookupError:
            print(f"pid {pid} is not running; removed stale pidfile")
        path.unlink(missing_ok=True)
        return 0

    if args.command == "status":
        import httpx

        url = os.environ.get("SHEPHERD_DAEMON_URL", "http://127.0.0.1:4890").rstrip("/")
        # Print which copy of the package this command actually loaded. Two
        # checkouts of the same project can both be pip-installed, and the stale
        # one wins depending on the working directory -- which makes "the new
        # feature is not working" look like a code bug when it is a provenance
        # problem. One line here settles it before anyone spends time on that.
        print(f"code loaded from: {Path(__file__).resolve().parent}")
        try:
            ok = httpx.get(f"{url}/health", timeout=2.0, trust_env=False).json().get("ok")
            print(f"daemon {url}: {'up' if ok else 'unexpected response'}")
        except Exception as exc:  # noqa: BLE001
            print(f"daemon {url}: DOWN ({type(exc).__name__}) — supervision is failing open")
            return 1
        return 0

    if args.command == "risk":
        from .core.judge.risk import RiskModel

        model = RiskModel.load(
            defaults={"judge": cfg.policy.nudge_threshold, "judge_block": cfg.policy.block_threshold},
            target_fpr=cfg.policy.risk_target_fpr,
            min_samples=cfg.policy.risk_min_samples,
        )
        print("source of truth: " + str(model.path))
        for key in ("judge", "judge_block"):
            evidence = model.samples.get(key)
            n = (evidence.n_negative if evidence else 0) + (len(evidence.drift_scores) if evidence else 0)
            state = "calibrated" if key in model.thresholds else f"prior (needs >= {model.min_samples} labeled)"
            print(f"  {key}: {model.threshold_for(key):.2f}  [{state}, {n} labeled]")
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

    if args.command == "config":
        from .core import config as cfgmod

        path = args.config or (Path.home() / ".shepherd" / "config.yaml")
        if args.subcommand == "show":
            print(f"config file: {path}  ({'exists' if Path(path).exists() else 'absent — defaults only'})")
            print(f"{'key':38s} {'value':22s} source")
            for row in cfgmod.effective_settings(path):
                print(f"{row['key']:38s} {str(row['value'])[:22]:22s} {row['source']}")
            stale = cfgmod.stale_keys(path)
            if stale:
                print("\nSTALE keys in the file (they configure nothing any more): " + ", ".join(stale))
                print("run `shepherd config prune` to drop them, `shepherd config merge` to add settings the file predates.")
            return 0

        if args.subcommand == "merge":
            added, written = cfgmod.merge_missing_keys(path)
            print(f"added {len(added)} setting(s): " + (", ".join(added) if added else "none — file is current"))
            if written:
                print(f"wrote {written}")
            return 0

        if args.subcommand == "prune":
            removed, written = cfgmod.prune_stale_keys(path)
            print(f"removed {len(removed)} stale key(s): " + (", ".join(removed) if removed else "none — file is clean"))
            if written:
                print(f"wrote {written}")
            return 0

        if args.subcommand == "set":
            import yaml

            key, value = args.key, args.value
            if "." not in key:
                print("usage: shepherd config set <section>.<key> <value>", file=sys.stderr)
                return 2
            section, name = key.split(".", 1)
            data = {}
            if Path(path).exists():
                data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
            try:
                parsed = yaml.safe_load(value)
            except yaml.YAMLError:
                parsed = value
            if section == "agents" and "." in name:
                agent, opt = name.split(".", 1)
                data.setdefault("agents", {}).setdefault(agent, {})[opt] = parsed
            else:
                data.setdefault(section, {})[name] = parsed
            from .adapters.backup import (
                write_json,  # noqa: F401  (backup helper lives with adapters)
            )

            saved = None
            if Path(path).exists():

                stamp = time.strftime("%Y%m%d-%H%M%S")
                saved = Path(path).with_name(f"{Path(path).name}.bak-{stamp}")
                import shutil

                shutil.copy2(path, saved)
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
            print(f"set {key} = {parsed!r} in {path}")
            if saved:
                print(f"previous config preserved at {saved}")
            return 0

        if args.subcommand == "judge":
            import httpx

            url = cfg.judge.base_url.rstrip("/")
            try:
                resp = httpx.get(f"{url}/models", timeout=8.0, trust_env=False)
                resp.raise_for_status()
                payload = resp.json()
            except Exception as exc:  # noqa: BLE001
                print(f"cannot reach {url}/models: {type(exc).__name__}: {exc}", file=sys.stderr)
                return 1
            models = payload.get("data") or payload.get("models") or []
            names = []
            for entry in models:
                for field in ("id", "name", "model", "display_name"):
                    if isinstance(entry, dict) and entry.get(field):
                        names.append(str(entry[field]))
                        break
            print(f"{url} accepts {len(names)} model name(s):")
            for n in names[:25]:
                print(f"  {n}")
            if names:
                print(f"\nset one with:  shepherd config set judge.model '{names[0]}'")
            return 0

    if args.command == "doctor":
        from .core import config as cfgmod

        path = args.config or (Path.home() / ".shepherd" / "config.yaml")
        problems = 0
        print(f"code        {Path(__file__).resolve().parent}")
        print(f"config file {path} {'(exists)' if Path(path).exists() else '(ABSENT — running on defaults)'}")
        stale = cfgmod.stale_keys(path)
        if stale:
            problems += 1
            print(f"STALE keys   {', '.join(stale)} -> run `shepherd config prune` (merge only adds)")
        if not cfg.judge.model:
            problems += 1
            print("judge.model  UNSET -> `shepherd config judge` lists accepted names")
        print(
            f"judge        {cfg.judge.base_url} model={cfg.judge.model or '<unset>'} "
            f"key={'set' if cfg.judge.api_key else 'none (no auth header will be sent)'}"
        )
        if True:  # loopback endpoints can be up and misconfigured too
            import httpx

            try:
                httpx.get(f"{cfg.judge.base_url.rstrip('/')}/models", timeout=6.0, trust_env=False).raise_for_status()
                print("judge reachability  ok")
            except Exception as exc:  # noqa: BLE001
                problems += 1
                print(f"judge reachability  FAIL ({type(exc).__name__}) -> Tier 1 stays asleep, Tier 0 still runs")
        url = os.environ.get("SHEPHERD_DAEMON_URL", f"http://127.0.0.1:{cfg.port}").rstrip("/")
        import httpx

        try:
            httpx.get(f"{url}/health", timeout=3.0, trust_env=False).raise_for_status()
            print(f"daemon       up on {url}")
        except Exception:  # noqa: BLE001
            problems += 1
            print(f"daemon       DOWN on {url} -> `systemctl --user enable --now shepherd` or `shepherd start-bg`")
        for agent in ("claude", "codex", "qwenpaw"):
            installed = {
                "claude": Path.home() / ".claude" / "settings.json",
                "codex": Path.home() / ".codex" / "hooks.json",
                "qwenpaw": Path.home() / ".qwenpaw" / "plugins" / "agent-shepherd",
            }[agent]
            marker = installed.exists() and (
                "shepherd" in installed.read_text(encoding="utf-8", errors="ignore")
                if installed.is_file()
                else True
            )
            note = ""
            if not marker and agent == "claude":
                note = "  (a provider switcher may own this file; re-run `shepherd install claude` after switching)"
            print(f"adapter {agent:8s} {'installed' if marker else 'NOT installed'} -> {installed}{note}")
            problems += 0 if marker else 1
        from .core.judge.risk import RiskModel

        model = RiskModel.load(
            {"judge": cfg.policy.nudge_threshold, "judge_block": cfg.policy.block_threshold},
            target_fpr=cfg.policy.risk_target_fpr,
        )
        for key in ("judge", "judge_block"):
            n = len(model.samples.get(key).clean_scores if model.samples.get(key) else [])
            print(f"threshold {key:12s} {model.threshold_for(key):.2f} [{'calibrated' if key in model.thresholds else 'prior'}, {n} clean labels]")
        print(f"\n{problems} thing(s) needing attention." if problems else "\nnothing outstanding.")
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
                *[x for f in (args.fault or []) for x in ("--fault", f)],
                *(["--judge"] if args.judge else []),
                *(["--fit-risk", args.fit_risk] if args.fit_risk else []),
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