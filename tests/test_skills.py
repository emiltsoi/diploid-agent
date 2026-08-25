"""Tests for the skill loader and manager."""

from pathlib import Path

from devin_fleet_harness.skills import SkillManager


def _write_skill(root: Path, name: str, content: str) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_dir.joinpath("SKILL.md").write_text(content, encoding="utf-8")
    return skill_dir


def test_skill_manager_loads_skill(tmp_path: Path) -> None:
    skill_dir = tmp_path / ".devin" / "skills" / "review"
    skill_dir.mkdir(parents=True)
    skill_dir.joinpath("SKILL.md").write_text(
        "---\n"
        "name: review\n"
        "description: Review staged changes\n"
        "allowed-tools:\n  - read\n  - grep\n---\n"
        "\n"
        "Run git diff --staged and review for issues.\n",
        encoding="utf-8",
    )

    manager = SkillManager(
        personas_root=tmp_path,
        shared_root=tmp_path,
        chat_cwd_root=tmp_path,
    )
    skills = manager.list_skills()
    assert len(skills) == 1
    assert skills[0].name == "review"
    assert "git diff --staged" in skills[0].content


def test_skill_manager_loads_from_shared_and_persona(tmp_path: Path) -> None:
    shared = tmp_path / "personas" / "shared"
    persona = tmp_path / "personas" / "test-pilot"
    shared.mkdir(parents=True)
    persona.mkdir(parents=True)

    _write_skill(
        shared / "skills",
        "health-check",
        "---\nname: health-check\n---\n\nRun the test suite.\n",
    )
    _write_skill(
        persona / "skills",
        "review",
        "---\nname: review\n---\n\nReview staged changes.\n",
    )

    manager = SkillManager(
        personas_root=tmp_path / "personas",
        shared_root=shared,
        chat_cwd_root=tmp_path,
    )
    names = {s.name for s in manager.list_skills()}
    assert names == {"health-check", "review"}


def test_skill_manager_syncs_to_chat(tmp_path: Path) -> None:
    shared = tmp_path / "personas" / "shared"
    shared.mkdir(parents=True)
    _write_skill(
        shared / "skills",
        "review",
        "---\nname: review\n---\n\nReview staged changes.\n",
    )

    manager = SkillManager(
        personas_root=tmp_path / "personas",
        shared_root=shared,
        chat_cwd_root=tmp_path / "sessions",
    )
    cwd = tmp_path / "sessions" / "chat-1"
    cwd.mkdir(parents=True)
    manager.sync_to_chat("chat-1", cwd)
    assert (cwd / ".devin" / "skills" / "review" / "SKILL.md").exists()


def test_skill_manager_creates_chat_skill(tmp_path: Path) -> None:
    manager = SkillManager(
        personas_root=tmp_path,
        shared_root=tmp_path,
        chat_cwd_root=tmp_path / "sessions",
    )
    manager.create_chat_skill("chat-2", "memo", "---\nname: memo\n---\n\nRemember this.\n")
    path = tmp_path / "sessions" / "chat-2" / ".devin" / "skills" / "memo" / "SKILL.md"
    assert path.exists()


def test_skill_manager_loads_curriculum_skill() -> None:
    repo_root = Path(__file__).parent.parent
    shared = repo_root / "personas" / "shared"
    manager = SkillManager(
        personas_root=repo_root / "personas",
        shared_root=shared,
        chat_cwd_root=None,
    )
    curriculum = manager.skill("curriculum")
    assert curriculum is not None
    assert curriculum.name == "curriculum"
    assert "curriculum_add_word" in curriculum.allowed_tools


def test_memory_skill_is_available_in_shared_root(tmp_path: Path) -> None:
    from devin_fleet_harness.config import Config, SkillsConfig
    from devin_fleet_harness.skills import SkillManager

    config = Config(
        persona={"name": "test-pilot", "profile_root": "/tmp"},
        harness={
            "sessions_root": str(tmp_path),
            "skills": SkillsConfig(
                shared_root=Path("personas/shared"),
                default_enabled=["memory"],
            ),
        },
    )
    manager = SkillManager(
        personas_root=Path("personas"),
        shared_root=Path("personas/shared"),
        chat_cwd_root=tmp_path,
    )
    assert "memory" in config.harness.skills.default_enabled
    skills = manager.list_skills()
    assert any(s.name == "memory" for s in skills)


def test_skill_manager_matches_triggers(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    shared.mkdir(parents=True)
    _write_skill(
        shared / "skills",
        "review",
        "---\nname: review\ntriggers:\n  - review staged\n---\n\nReview staged changes.\n",
    )
    _write_skill(
        shared / "skills",
        "model-review",
        "---\nname: model-review\ntriggers:\n  - model review\n---\n\nRun model review.\n",
    )
    _write_skill(
        shared / "skills",
        "memory",
        "---\nname: memory\ntriggers:\n  - user\n---\n\nRecall facts.\n",
    )

    manager = SkillManager(
        personas_root=tmp_path,
        shared_root=shared,
        chat_cwd_root=None,
    )
    assert manager.match_skills("please model review this repo") == {"model-review"}
    assert manager.match_skills("review staged changes") == {"review"}
    assert manager.match_skills("remember this fact") == set()  # "user" is ignored


def test_skill_manager_builds_index_and_active_text(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    shared.mkdir(parents=True)
    _write_skill(
        shared / "skills",
        "review",
        "---\nname: review\ndescription: Review staged changes\ntriggers:\n  - review staged\nallowed-tools:\n  - exec\n---\n\nRun git diff --staged.\n",
    )

    manager = SkillManager(
        personas_root=tmp_path,
        shared_root=shared,
        chat_cwd_root=None,
    )
    index = manager.skill_index_text(None, active={"review"})
    assert index is not None
    assert "review" in index
    assert "Review staged changes" in index
    assert "(active)" in index

    active = manager.active_skills_text({"review"}, None)
    assert active is not None
    assert "Run git diff --staged." in active
    assert "exec" in active


def test_skill_manager_matches_slash_command(tmp_path: Path) -> None:
    """A slash command like /review should activate the matching skill."""
    shared = tmp_path / "shared"
    shared.mkdir(parents=True)
    _write_skill(
        shared / "skills",
        "review",
        "---\nname: review\ntriggers:\n  - review staged\n---\n\nReview staged changes.\n",
    )

    manager = SkillManager(
        personas_root=tmp_path,
        shared_root=shared,
        chat_cwd_root=None,
    )
    assert manager.match_skills("/review") == {"review"}
    assert manager.match_skills("please review staged changes") == {"review"}
    assert manager.match_skills("please review") == set()


def test_skill_manager_sync_only_active_skills(tmp_path: Path) -> None:
    """sync_to_chat copies only the requested active skills, not every skill."""
    shared = tmp_path / "personas" / "shared"
    shared.mkdir(parents=True)
    _write_skill(
        shared / "skills",
        "review",
        "---\nname: review\n---\n\nReview staged changes.\n",
    )
    _write_skill(
        shared / "skills",
        "model-review",
        "---\nname: model-review\n---\n\nRun model review.\n",
    )

    manager = SkillManager(
        personas_root=tmp_path / "personas",
        shared_root=shared,
        chat_cwd_root=tmp_path / "sessions",
    )
    cwd = tmp_path / "sessions" / "chat-1"
    cwd.mkdir(parents=True)
    manager.sync_to_chat("chat-1", cwd, enabled={"review"})

    assert (cwd / ".devin" / "skills" / "review" / "SKILL.md").exists()
    assert not (cwd / ".devin" / "skills" / "model-review" / "SKILL.md").exists()


def test_skill_manager_match_skills_respects_enabled_set(tmp_path: Path) -> None:
    """match_skills only considers skills in the provided enabled set."""
    shared = tmp_path / "shared"
    shared.mkdir(parents=True)
    _write_skill(
        shared / "skills",
        "review",
        "---\nname: review\ntriggers:\n  - review staged\n---\n\nReview staged changes.\n",
    )
    _write_skill(
        shared / "skills",
        "model-review",
        "---\nname: model-review\ntriggers:\n  - model review\n---\n\nRun model review.\n",
    )

    manager = SkillManager(
        personas_root=tmp_path,
        shared_root=shared,
        chat_cwd_root=None,
    )
    assert manager.match_skills("please model review this repo", enabled={"review"}) == set()
    assert manager.match_skills("please model review this repo") == {"model-review"}
