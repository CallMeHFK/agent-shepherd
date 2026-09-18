"""Shared ``shepherd-hook`` console entry point.

Dispatches to the Claude Code or Codex thin-shell based on the first argument:

    shepherd-hook claude
    shepherd-hook codex
"""

from __future__ import annotations

import sys

from .claude.hook import main as claude_main
from .codex.hook import main as codex_main


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in {"claude", "codex"}:
        print("usage: shepherd-hook <claude|codex>", file=sys.stderr)
        return 2
    return claude_main() if sys.argv[1] == "claude" else codex_main()


if __name__ == "__main__":
    raise SystemExit(main())