"""The user document preserves both semantic and presentation settings."""

from pathlib import Path

import pytest
import yaml

from pstrain.lib.config import resolve_config
from pstrain.lib.config.user import PstrainUserConfig, get_user_config


def test_user_preferences_and_semantics_share_one_document(tmp_path: Path) -> None:
    path = tmp_path / "user.yaml"
    path.write_text(
        "config_version: 1\nfeatures:\n  alpha: 0.9\n"
        "training:\n  tied:\n    max_iterations: 7\njson_output:\n  indent: 4\n"
    )
    user = PstrainUserConfig.load(path)
    user.cache_dir = tmp_path / "cache"
    assert user.json_output.indent == 4
    before = resolve_config(tmp_path, user_config_path=path)
    assert before.profile.features.alpha == 0.9
    assert before.profile.training.tied.max_iterations == 7
    assert before.fields["features.alpha"].winner.source_kind == "user"
    user.json_output.indent = 3
    user.save(path)
    after = resolve_config(tmp_path, user_config_path=path)
    assert after.profile == before.profile
    saved = yaml.safe_load(path.read_text())
    assert saved["config_version"] == 1
    assert saved["json_output"]["indent"] == 3
    assert saved["features"] == {"alpha": 0.9}
    assert saved["training"] == {"tied": {"max_iterations": 7}}
    assert "alignment" not in saved  # Saving preferences must not pin absent overrides.


def test_atomic_user_save_preserves_old_document_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "user.yaml"
    old = "config_version: 1\nfeatures:\n  alpha: 0.9\n"
    path.write_text(old)
    user = PstrainUserConfig.load(path)
    user.cache_dir = tmp_path / "cache"
    user.json_output.indent = 4

    def fail_replace(self: Path, target: Path) -> Path:
        raise OSError("replacement unavailable")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="replacement unavailable"):
        user.save(path)
    assert path.read_text() == old
    assert list(tmp_path.iterdir()) == [path]


def test_user_load_refuses_unknown_versioned_fields(tmp_path: Path) -> None:
    path = tmp_path / "user.yaml"
    path.write_text("config_version: 1\ntrainng: {}\n")
    with pytest.raises(ValueError, match="trainng"):
        PstrainUserConfig.load(path)
    with pytest.raises(ValueError, match="trainng"):
        resolve_config(tmp_path, user_config_path=path)


def test_legacy_user_save_preserves_effective_overrides(tmp_path: Path) -> None:
    path = tmp_path / "user.yaml"
    path.write_text(
        "features:\n  num_ceps: 17\nparallel:\n  n_jobs: 2\n"
        "defaults:\n  n_iterations: 9\njson_output:\n  indent: 4\n"
    )
    with pytest.warns(FutureWarning):
        before = resolve_config(tmp_path, user_config_path=path)
    user = PstrainUserConfig.load(path)
    user.cache_dir = tmp_path / "cache"
    user.save(path)
    after = resolve_config(tmp_path, user_config_path=path)
    assert after.profile == before.profile
    assert after.profile.features.ncep == 17
    assert after.profile.runner.jobs == 2
    assert user.defaults.n_iterations == 9
    assert user.json_output.indent == 4


def test_user_path_override_does_not_poison_default_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    default = tmp_path / "default.yaml"
    custom = tmp_path / "custom.yaml"
    default.write_text("config_version: 1\njson_output:\n  indent: 3\n")
    custom.write_text("config_version: 1\njson_output:\n  indent: 4\n")
    monkeypatch.setenv("PSTRAIN_USER_CONFIG", str(default))
    assert get_user_config(force_reload=True).json_output.indent == 3
    assert get_user_config(custom).json_output.indent == 4
    assert get_user_config().json_output.indent == 3


def test_project_overlay_still_rejects_presentation_preferences(tmp_path: Path) -> None:
    (tmp_path / "etc").mkdir()
    (tmp_path / "etc" / "config.yaml").write_text("config_version: 1\njson_output: {}\n")
    with pytest.raises(ValueError, match="json_output"):
        resolve_config(tmp_path, user_config_path=tmp_path / "absent")
