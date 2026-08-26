"""声音档案默认引擎与具体模型的回归测试。"""

from types import SimpleNamespace

from sqlalchemy import create_engine, inspect, text

from backend.database.migrations import run_migrations
from backend.models import GenerationRequest, VoiceProfileCreate
from backend.routes.generations import _resolve_generation_model_size
from backend.services.profiles import _validate_profile_fields


def test_new_profile_payload_accepts_cosyvoice_rl_and_defaults_to_chinese():
    profile = VoiceProfileCreate(
        name="河南话样本",
        default_engine="cosyvoice",
        default_model_size="rl",
    )

    assert profile.language == "zh"
    assert profile.default_engine == "cosyvoice"
    assert profile.default_model_size == "rl"


def test_profile_default_model_must_match_cosyvoice_engine():
    error = _validate_profile_fields(
        voice_type="cloned",
        preset_engine=None,
        preset_voice_id=None,
        design_prompt=None,
        default_engine="qwen",
        default_model_size="rl",
    )

    assert error is not None


def test_generation_inherits_profile_model_unless_request_overrides_it():
    profile = SimpleNamespace(default_engine="cosyvoice", default_model_size="base")
    inherited = GenerationRequest(
        profile_id="voice-id",
        text="测试",
        engine=None,
    )
    overridden = GenerationRequest(
        profile_id="voice-id",
        text="测试",
        engine="cosyvoice",
        model_size="rl",
    )

    assert _resolve_generation_model_size(inherited, profile, "cosyvoice") == "base"
    assert _resolve_generation_model_size(overridden, profile, "cosyvoice") == "rl"


def test_profile_default_model_is_not_reused_by_an_overridden_engine():
    profile = SimpleNamespace(default_engine="cosyvoice", default_model_size="base")
    request = GenerationRequest(
        profile_id="voice-id",
        text="测试",
        engine="qwen",
    )

    assert _resolve_generation_model_size(request, profile, "qwen") == "1.7B"


def test_migration_adds_nullable_default_model_size_to_existing_profiles(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'profiles.db'}")
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE profiles (
                    id VARCHAR PRIMARY KEY,
                    name VARCHAR NOT NULL,
                    language VARCHAR,
                    default_engine VARCHAR
                )
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO profiles (id, name, language, default_engine)
                VALUES ('existing', '旧声音', 'zh', 'cosyvoice')
                """
            )
        )

    run_migrations(engine)

    columns = {column["name"] for column in inspect(engine).get_columns("profiles")}
    assert "default_model_size" in columns
    with engine.connect() as connection:
        value = connection.execute(
            text("SELECT default_model_size FROM profiles WHERE id = 'existing'")
        ).scalar_one()
    assert value is None

