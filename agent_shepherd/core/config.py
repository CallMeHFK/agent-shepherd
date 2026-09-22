"""Configuration loading for the supervisor daemon and adapters.

The supervisor reads a single YAML file at ``~/.shepherd/config.yaml``. It is
deliberately small and human-editable. Unknown keys are ignored so a newer
supervisor can still start with an older config file.

Two backend modes are supported for the judge model:

1. ``agnes`` — the Sapiens AI model family, served at the Agnes AI Hub
   (OpenAI-compatible API). This is the default because it is the only
   reachable model endpoint in the local environment today.
2. ``openai`` / ``vllm`` — any OpenAI-compatible endpoint (local vLLM,
   cc-switch, etc.) with a user-supplied base URL and model name.

Never store real API keys in this file. Keys come from environment variables
or from the shell snapshots of the agent being supervised.
"""

from __future__ import annotations

import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_ENV_REF = re.compile(r"\$\{(\w+)\}")


def expand_env(value: str) -> str:
    """Resolve ``${VAR}`` references written into the config file.

    The starter config ships ``api_key: ${SHEPHERD_JUDGE_API_KEY}`` as a literal
    string. Without expansion that text is sent as the bearer token -- which also
    defeats "no key means no auth header", because the value is not empty -- so an
    unset variable resolves to empty, meaning unconfigured, everywhere.
    """
    return _ENV_REF.sub(lambda m: os.environ.get(m.group(1), ""), value or "")


@dataclass
class JudgeConfig:
    """How the supervisor reaches its LLM judge."""

    backend: str = "agnes"
    base_url: str = "https://apihub.agnes-ai.com/v1"
    model: str = "agnes-3.0-flash"
    api_key: str = ""
    timeout: float = 30.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> JudgeConfig:
        data = data or {}
        cfg = cls(
            backend=expand_env(str(data.get("backend", "agnes"))),
            base_url=expand_env(str(data.get("base_url", "https://apihub.agnes-ai.com/v1"))),
            model=expand_env(str(data.get("model", "agnes-3.0-flash"))),
            api_key=expand_env(str(data.get("api_key", ""))),
            timeout=float(data.get("timeout", 30.0)),
        )
        # Env vars override the file. This lets a user keep the config file
        # clean while still pointing at a different judge at runtime.
        cfg.backend = os.environ.get("SHEPHERD_JUDGE_BACKEND", cfg.backend)
        cfg.base_url = os.environ.get("SHEPHERD_JUDGE_BASE_URL", cfg.base_url)
        cfg.model = os.environ.get("SHEPHERD_JUDGE_MODEL", cfg.model)
        cfg.api_key = os.environ.get("SHEPHERD_JUDGE_API_KEY", cfg.api_key)
        cfg.timeout = float(os.environ.get("SHEPHERD_JUDGE_TIMEOUT", cfg.timeout))
        return cfg


@dataclass
class PolicyConfig:
    """When the supervisor wakes up, and how aggressive it is.

    The judge (Tier 1) wakes at natural checkpoints — prompt start, iteration
    end, session stop — and, mid-iteration, only when the cheap CUSUM drift
    statistic climbs toward its alarm line. This onset-based gating avoids the
    "state-saturation trap": a fixed threshold on cumulative state fires on a
    large constant fraction of actions, so the judge's cost ends up exceeding
    the agent's.

    ``nudge_threshold`` / ``block_threshold`` / ``max_guidance_tokens`` are
    enforced by the engine. The two score thresholds are *priors*: once enough
    labeled outcomes accumulate they are replaced by conformal-risk-control
    calibrated values (see :mod:`agent_shepherd.core.judge.risk`), because a
    hand-picked cut on a verbalized confidence is not a control (arXiv
    2606.21399, 2412.14737).
    """

    nudge_threshold: float = 0.60
    block_threshold: float = 0.85
    max_guidance_tokens: int = 500
    fail_open: bool = True
    # Hysteresis: repeat NUDGEs from the same detector (or the judge) for the
    # same session are suppressed within this window, so the agent is not
    # re-nagged with a verdict it already received. BLOCK/ESCALATE are never
    # suppressed.
    nudge_cooldown_seconds: float = 300.0
    # CUSUM drift detector (Tier 0). Its alarm threshold is calibrated by
    # Monte-Carlo simulation so the false-alarm rate over ``drift_horizon``
    # *re-checks* stays at or below ``drift_target_fpr`` — an overlapping-window
    # budget, which is what the agent actually experiences (arXiv 2607.17336).
    drift_enabled: bool = True
    drift_target_fpr: float = 0.05
    drift_horizon: int = 120
    # Adopt the session's own pre-alarm failure rate as the CUSUM reference once
    # enough results have been seen (self-starting control chart).
    drift_adaptive_baseline: bool = True
    # How much an unobservable outcome (empty / lost response) counts toward the
    # drift statistic, relative to a definite failure at 1.0.
    drift_unknown_weight: float = 0.75
    # Soft trigger: when the CUSUM statistic reaches this fraction of its
    # alarm line on a tool result, wake the LLM judge early. 1.0 disables the
    # soft trigger (the judge wakes only at checkpoints).
    drift_watch_fraction: float = 0.6
    # Review at every iteration boundary even when the iteration was clean. The
    # offline benchmark measured this as ~23% of all events waking the judge on
    # *healthy* sessions, so the default is to wake on evidence: prompt start and
    # session stop are always reviewed, an iteration boundary only when something
    # in it did not come back clean.
    review_clean_iterations: bool = False
    # Binding drift: acting on a near-miss sibling of the entity that was
    # resolved (arXiv 2607.18316). ``binding_similarity`` is the token-Jaccard
    # above which two paths are considered confusable.
    binding_enabled: bool = True
    binding_similarity: float = 0.6
    # Delegation contract for the off-spec detector: an edit is admissible when
    # the path is named in the goal, matched by an allow-glob, or already
    # observed this session. Deny-globs are checked first and BLOCK.
    # Nudge on edits to a path nothing in the session named, allowed or read.
    # Off by default: measured at ~1 false nudge per healthy session, and
    # writing a brand-new test file trips it.
    scope_nudge_unobserved: bool = False
    scope_allow_globs: list[str] = field(default_factory=list)
    scope_deny_globs: list[str] = field(default_factory=list)
    # Conformal risk control: the per-source false-intervention budget the
    # calibrated thresholds are fitted to, and the evidence needed before a
    # fitted value is trusted over the configured prior.
    risk_target_fpr: float = 0.05
    risk_min_samples: int = 20
    # Promote repeatedly-followed guidance into a header the agent sees up
    # front instead of re-nagging it mid-run (arXiv 2509.03990).
    rulebook_enabled: bool = True
    rulebook_budget_tokens: int = 160

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> PolicyConfig:
        data = data or {}

        def _globs(key: str) -> list[str]:
            raw = data.get(key) or []
            return [str(item) for item in raw] if isinstance(raw, (list, tuple)) else []

        return cls(
            nudge_threshold=float(data.get("nudge_threshold", 0.60)),
            block_threshold=float(data.get("block_threshold", 0.85)),
            max_guidance_tokens=int(data.get("max_guidance_tokens", 500)),
            fail_open=bool(data.get("fail_open", True)),
            nudge_cooldown_seconds=float(data.get("nudge_cooldown_seconds", 300.0)),
            drift_enabled=bool(data.get("drift_enabled", True)),
            drift_target_fpr=float(data.get("drift_target_fpr", 0.05)),
            drift_horizon=int(data.get("drift_horizon", 120)),
            drift_adaptive_baseline=bool(data.get("drift_adaptive_baseline", True)),
            drift_unknown_weight=float(data.get("drift_unknown_weight", 0.75)),
            drift_watch_fraction=float(data.get("drift_watch_fraction", 0.6)),
            review_clean_iterations=bool(data.get("review_clean_iterations", False)),
            binding_enabled=bool(data.get("binding_enabled", True)),
            binding_similarity=float(data.get("binding_similarity", 0.6)),
            scope_nudge_unobserved=bool(data.get("scope_nudge_unobserved", False)),
            scope_allow_globs=_globs("scope_allow_globs"),
            scope_deny_globs=_globs("scope_deny_globs"),
            risk_target_fpr=float(data.get("risk_target_fpr", 0.05)),
            risk_min_samples=int(data.get("risk_min_samples", 20)),
            rulebook_enabled=bool(data.get("rulebook_enabled", True)),
            rulebook_budget_tokens=int(data.get("rulebook_budget_tokens", 160)),
        )


@dataclass
class AgentConfig:
    """Per-agent enable/disable and adapter-specific options."""

    enabled: bool = True
    block_enabled: bool = False
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> AgentConfig:
        data = data or {}
        return cls(
            enabled=bool(data.get("enabled", True)),
            block_enabled=bool(data.get("block_enabled", False)),
            options=dict(data.get("options", {})),
        )


@dataclass
class ShepherdConfig:
    judge: JudgeConfig
    policy: PolicyConfig
    agents: dict[str, AgentConfig]
    port: int = 4890

    @classmethod
    def load(cls, path: Path | str | None = None) -> ShepherdConfig:
        """Load config from ``~/.shepherd/config.yaml`` or a given path."""
        path = Path(path) if path else Path.home() / ".shepherd" / "config.yaml"
        data: dict[str, Any] = {}
        if path.exists():
            with open(path, encoding="utf-8") as fh:
                loaded = yaml.safe_load(fh)
                data = loaded if isinstance(loaded, dict) else {}

        agents_raw = data.get("agents", {})
        agents = {
            name: AgentConfig.from_dict(raw)
            for name, raw in agents_raw.items()
        } if isinstance(agents_raw, dict) else {}

        return cls(
            judge=JudgeConfig.from_dict(data.get("judge")),
            policy=PolicyConfig.from_dict(data.get("policy")),
            agents=agents,
            port=int(data.get("port", 4890)),
        )

    def agent_config(self, agent: str) -> AgentConfig:
        return self.agents.get(agent, AgentConfig())

    def ensure_defaults(self) -> None:
        """Create a starter config file if none exists (idempotent)."""
        path = Path.home() / ".shepherd" / "config.yaml"
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        starter = {
            "judge": {
                "backend": "agnes",
                "base_url": "https://apihub.agnes-ai.com/v1",
                "model": "agnes-3.0-flash",
                "api_key": "${SHEPHERD_JUDGE_API_KEY}",
                "timeout": 30,
            },
            "policy": {
                "nudge_threshold": 0.6,
                "block_threshold": 0.85,
                "max_guidance_tokens": 500,
                "fail_open": True,
                "nudge_cooldown_seconds": 300,
                "drift_enabled": True,
                "drift_target_fpr": 0.05,
                "drift_horizon": 120,
                "drift_adaptive_baseline": True,
                "drift_unknown_weight": 0.75,
                "drift_watch_fraction": 0.6,
                "review_clean_iterations": False,
                "binding_enabled": True,
                "binding_similarity": 0.6,
                "scope_nudge_unobserved": False,
                "scope_allow_globs": [],
                "scope_deny_globs": ["*.env", "*.pem", ".git/*", "*.secret*"],
                "risk_target_fpr": 0.05,
                "risk_min_samples": 20,
                "rulebook_enabled": True,
                "rulebook_budget_tokens": 160,
            },
            "agents": {
                "qwenpaw": {"enabled": True, "block_enabled": False},
                "claude": {"enabled": True, "block_enabled": False},
                "codex": {"enabled": True, "block_enabled": False},
            },
            "port": 4890,
        }
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(starter, fh, sort_keys=False, allow_unicode=True)

# Env vars that override judge settings, in the same precedence order the loader
# uses. Kept next to the code that reads them so the two cannot drift apart.
JUDGE_ENV = {
    "backend": "SHEPHERD_JUDGE_BACKEND",
    "base_url": "SHEPHERD_JUDGE_BASE_URL",
    "model": "SHEPHERD_JUDGE_MODEL",
    "api_key": "SHEPHERD_JUDGE_API_KEY",
    "timeout": "SHEPHERD_JUDGE_TIMEOUT",
}

SECRET_FIELDS = {"api_key"}


def _defaults(obj: object) -> dict:
    from dataclasses import fields

    return {f.name: getattr(obj, f.name) for f in fields(obj)}


def effective_settings(path: Path | str | None = None) -> list[dict]:
    """Every setting the daemon will use, and where that value came from.

    A supervisor configured across a YAML file, five environment variables and
    dataclass defaults is not inspectable by reading any one of them, and
    "is the model set?" turns into guessing. This answers it per key: `file`,
    `env:NAME` or `default`.
    """
    path = Path(path) if path else Path.home() / ".shepherd" / "config.yaml"
    raw: dict = {}
    if path.exists():
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            raw = loaded if isinstance(loaded, dict) else {}
        except (OSError, yaml.YAMLError):
            raw = {}
    file_policy = raw.get("policy") or {}
    file_judge = raw.get("judge") or {}
    rows: list[dict] = []

    def add(section: str, name: str, value: object, default: object, env: str | None, in_file: bool) -> None:
        source = f"env:{env}" if env and os.environ.get(env) else ("file" if in_file else "default")
        shown = "***" if name in SECRET_FIELDS and value else ("<unset>" if not value else value)
        rows.append({"key": f"{section}.{name}", "value": shown, "source": source, "default": default})

    jd = JudgeConfig.from_dict(raw.get("judge"))
    jd_default = JudgeConfig()
    for name, value in _defaults(jd).items():
        add("judge", name, value, _defaults(jd_default)[name], JUDGE_ENV.get(name), name in file_judge)

    pd = PolicyConfig.from_dict(raw.get("policy"))
    pd_default = PolicyConfig()
    for name, value in _defaults(pd).items():
        add("policy", name, value, _defaults(pd_default)[name], None, name in file_policy)

    add("root", "port", int(raw.get("port", 4890)), 4890, None, "port" in raw)
    for agent, opts in sorted((raw.get("agents") or {}).items()):
        cfg = AgentConfig.from_dict(opts)
        add("agents", f"{agent}.enabled", cfg.enabled, True, None, True)
        add("agents", f"{agent}.block_enabled", cfg.block_enabled, False, None, True)
    return rows


def stale_keys(path: Path | str | None = None) -> list[str]:
    """Keys present in the file that no longer configure anything.

    Without this, a retired key sits in a config file looking like a live
    setting, and someone wastes time tuning it.
    """
    path = Path(path) if path else Path.home() / ".shepherd" / "config.yaml"
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return []
    out: list[str] = []
    for section, keys in (("policy", raw.get("policy")), ("judge", raw.get("judge"))):
        allowed = {f.name for f in fields_of(PolicyConfig if section == "policy" else JudgeConfig)}
        for key in (keys or {}):
            if key not in allowed:
                out.append(f"{section}.{key}")
    return sorted(out)


def fields_of(cls):
    from dataclasses import fields

    return fields(cls)


def _backup_and_write(path: Path, raw: dict) -> None:
    """Write ``raw`` to ``path``, preserving the previous file first.

    Local copy of the adapters' safety helper: core must not import from adapters.
    """
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    aside = path.with_name(f"{path.name}.bak-{stamp}")
    shutil.copy2(path, aside)
    print(f"previous config preserved at {aside}")
    path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")


def merge_missing_keys(path: Path | str | None = None, *, dry_run: bool = False) -> tuple[list[str], Path | None]:
    """Add any setting the file does not mention, keeping the user's values.

    `ensure_defaults` used to write a starter only when the file was absent, so a
    config written for an older release never learned about a new control and no
    command said so. Existing keys are left byte-for-byte alone and the previous
    file is backed up first.
    """
    path = Path(path) if path else Path.home() / ".shepherd" / "config.yaml"
    if not path.exists():
        return [], None
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    added: list[str] = []
    sections = {
        "policy": {k: v for k, v in _defaults(PolicyConfig()).items()},
        "judge": {k: v for k, v in _defaults(JudgeConfig()).items() if k != "api_key"},
    }
    for section, defaults in sections.items():
        block = raw.setdefault(section, {})
        if not isinstance(block, dict):
            continue
        for key, value in defaults.items():
            if key not in block:
                block[key] = value
                added.append(f"{section}.{key}")
    if added and not dry_run:
        _backup_and_write(path, raw)
    return added, path


def prune_stale_keys(path: Path | str | None = None, *, dry_run: bool = False) -> tuple[list[str], Path | None]:
    """Drop keys the current release no longer reads.

    ``merge`` only ever adds, so a retired setting stayed behind wearing the
    look of a live control. This is the other half of that, with the same backup.
    """
    path = Path(path) if path else Path.home() / ".shepherd" / "config.yaml"
    if not path.exists():
        return [], None
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    removed: list[str] = []
    for section in ("policy", "judge"):
        block = raw.get(section)
        if not isinstance(block, dict):
            continue
        allowed = {f.name for f in fields_of(PolicyConfig if section == "policy" else JudgeConfig)}
        for key in list(block):
            if key not in allowed:
                del block[key]
                removed.append(f"{section}.{key}")
    if removed and not dry_run:
        _backup_and_write(path, raw)
    return removed, path
