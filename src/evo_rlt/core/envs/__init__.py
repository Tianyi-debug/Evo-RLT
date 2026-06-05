from __future__ import annotations

from evo_rlt.core.collector import Environment
from evo_rlt.core.envs.reaching import ReachingEnvironment


def make_env(env_type: str, **kwargs) -> Environment:
    """Factory for creating environments by name."""
    if env_type == "reaching":
        return ReachingEnvironment(**kwargs)
    if env_type == "dummy":
        from evo_rlt.core.collector import DummyEnvironment
        return DummyEnvironment(**kwargs)
    raise ValueError(f"Unknown env_type: {env_type}")


__all__ = ["ReachingEnvironment", "make_env"]
