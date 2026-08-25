"""Tests for lane-based model routing and conversation budget."""

from pathlib import Path

from devin_fleet_harness.config import (
    Config,
    ConversationBudget,
    DevinConfig,
    HarnessConfig,
    PersonaConfig,
    RoutingConfig,
    Secrets,
)
from devin_fleet_harness.engine.router import ModelRouter


def _make_config(routing: RoutingConfig | None = None) -> Config:
    return Config(
        devin=DevinConfig(bin="/bin/echo", model="swe-1-7"),
        persona=PersonaConfig(
            name="test-pilot",
            profile_root=Path(__file__).parent / "fixtures" / "test-pilot",
        ),
        harness=HarnessConfig(
            routing=routing or RoutingConfig(),
            sessions_root=Path("/tmp/sessions"),
            session_store_path=Path("/tmp/sessions.jsonl"),
        ),
        secrets=Secrets(WINDSURF_API_KEY="test-key"),
    )


def test_disabled_router_uses_default_model() -> None:
    router = ModelRouter(_make_config())
    route = router.resolve("refactor this code")
    assert route.model == "swe-1-7"
    assert route.notice is None
    assert not route.budget_exceeded


def test_lane_routing_by_keywords() -> None:
    cfg = _make_config(
        RoutingConfig(
            enabled=True,
            lanes={
                "mechanical": "glm-5-2",
                "compliance": "dual-arm",
                "conflict": "arbiter",
            },
            lane_keywords={
                "mechanical": ["refactor", "test", "fix"],
                "compliance": ["audit", "review", "policy"],
                "conflict": ["disagree", "conflict", "merge"],
            },
        )
    )
    router = ModelRouter(cfg)

    assert router.resolve("refactor the auth layer").model == "glm-5-2"
    assert router.resolve("audit the policy file").model == "dual-arm"
    assert router.resolve("we have a merge conflict").model == "arbiter"
    assert router.resolve("hello").model == "swe-1-7"


def test_budget_warning_notice() -> None:
    cfg = _make_config(
        RoutingConfig(
            enabled=True,
            budget=ConversationBudget(enabled=True, max_total_tokens=1000, warning_threshold=0.8),
        )
    )
    router = ModelRouter(cfg)

    route = router.resolve("hello", cumulative_tokens=900)
    assert route.notice is not None
    assert "Approaching conversation token budget" in route.notice
    assert not route.budget_exceeded


def test_budget_exceeded_hard_cap() -> None:
    cfg = _make_config(
        RoutingConfig(
            enabled=True,
            budget=ConversationBudget(enabled=True, max_total_tokens=1000, hard_cap=True),
        )
    )
    router = ModelRouter(cfg)

    route = router.resolve("hello", cumulative_tokens=1001)
    assert route.budget_exceeded
    assert "reached its token budget" in (route.notice or "")
    assert route.model == "swe-1-7"


def test_budget_exceeded_soft_cap_uses_fallback() -> None:
    cfg = _make_config(
        RoutingConfig(
            enabled=True,
            fallback_model="fallback-model",
            budget=ConversationBudget(enabled=True, max_total_tokens=1000, hard_cap=False),
        )
    )
    router = ModelRouter(cfg)

    route = router.resolve("hello", cumulative_tokens=1001)
    assert route.budget_exceeded
    assert route.model == "fallback-model"


def test_remaining_budget() -> None:
    cfg = _make_config(
        RoutingConfig(
            enabled=True,
            budget=ConversationBudget(enabled=True, max_total_tokens=1000),
        )
    )
    router = ModelRouter(cfg)
    assert router.remaining_budget(300) == 700
    assert router.remaining_budget(1200) == 0


def test_budget_status() -> None:
    cfg = _make_config(
        RoutingConfig(
            enabled=True,
            budget=ConversationBudget(enabled=True, max_total_tokens=1000),
        )
    )
    router = ModelRouter(cfg)
    status = router.budget_status(500)
    assert status["enabled"] is True
    assert status["used_tokens"] == 500
    assert status["remaining_tokens"] == 500
    assert not status["exceeded"]
