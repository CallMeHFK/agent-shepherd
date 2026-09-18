"""Agent-specific adapters.

Each adapter does two things:
1. normalize the agent's native events into the supervisor's canonical
   ``AgentEvent`` and POST them to the daemon, and
2. translate the daemon's ``Verdict`` back into the agent's native injection
   mechanism (hooks, stop handlers, approval REST calls, etc.).
"""