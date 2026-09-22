# agent-shepherd — QwenPaw supervisor plugin

An in-process observer of QwenPaw's reasoning loop that asks a supervisor for a
verdict at each iteration boundary and injects corrective guidance when the
agent drifts. This directory is the entire plugin: `plugin.json` is the manifest
QwenPaw reads, `backend/main.py` is its `entry.backend`.

## Read this first: a bundle is not a supervisor

The plugin only POSTs events to the shepherd daemon on `127.0.0.1:4890`. When
nothing listens there it **fails open** — it loads, registers its middleware and
stop handler, and supervises nothing, which looks exactly like a working
install. So the daemon has to be there:

```bash
# the package is not on PyPI
uv pip install "git+https://github.com/CallMeHFK/agent-shepherd.git@v0.2.0"
shepherd start-bg
shepherd status        # "up", and check the `code loaded from:` line it prints
```

QwenPaw `>=2.2.0,<2.3.0` (`qwenpaw_version` in `plugin.json`).

## Install

```bash
# from a checkout of this repository
shepherd install qwenpaw

# from the release asset — same bytes, no checkout needed
qwenpaw plugin install https://github.com/CallMeHFK/agent-shepherd/releases/latest/download/agent-shepherd.zip

# or a local archive
qwenpaw plugin install agent-shepherd.zip
```

While QwenPaw is running these hot-load the plugin; while it is stopped the tree
is written to `~/.qwenpaw/plugins/agent-shepherd/` and loaded on the next start.

Not every `.zip` on the release page is this bundle: GitHub's auto-generated
source archive (`.../archive/...`) has no `plugin.json` where the installer looks
for it, and is rejected.

## Check that it took

- `qwenpaw plugin install` prints the loaded plugin name; the app's plugin list
  shows `agent-shepherd` with the version from `plugin.json`.
- `ls ~/.qwenpaw/plugins/ | grep agent-shepherd` must list **exactly one**
  entry. QwenPaw treats every subdirectory of `plugins/` as a plugin and each
  copy carries the same manifest id, so a backup left in there is a second
  plugin that can win the load race — the app then runs last release's middleware
  while every path you check points at the new one. `shepherd install qwenpaw`
  moves asides to `~/.qwenpaw/plugin-archive/` for this reason.
- Run an agent session, then `shepherd stats qwenpaw <session-id>` — a session
  with zero events means the plugin is not the one being loaded.

## Uninstall

```bash
printf 'y\n' | qwenpaw plugin uninstall agent-shepherd   # the CLI prompt has no --yes flag
```

## Configuration

Nothing here is configured in `plugin.json`. The judge backend, admission
thresholds, per-agent opt-ins and scope globs live in `~/.shepherd/config.yaml`,
written on the daemon's first start:

```bash
shepherd doctor                 # what is set, what is missing, what to do next
shepherd config show            # every effective value and where it came from
```

Guidance is advisory by default: an agent only ever receives a `block` if
`block_enabled` was set for it in `agents.<name>`.

The daemon protocol, the detectors, and the benchmark that measures them are
documented in the repository README and `RESEARCH_NOTES.md` at
<https://github.com/CallMeHFK/agent-shepherd>.
