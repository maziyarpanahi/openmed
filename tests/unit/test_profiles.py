"""Unit tests for configuration profiles functionality."""

import os
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from openmed.core.config import (
    OpenMedConfig,
    PROFILE_PRESETS,
    PROFILES_DIR,
    list_profiles,
    get_profile,
    save_profile,
    delete_profile,
    load_config_with_profile,
)


class TestProfilePresets:
    """Tests for built-in profile presets."""

    def test_dev_profile_exists(self):
        """Test dev profile is defined."""
        assert "dev" in PROFILE_PRESETS
        assert PROFILE_PRESETS["dev"]["log_level"] == "DEBUG"

    def test_prod_profile_exists(self):
        """Test prod profile is defined."""
        assert "prod" in PROFILE_PRESETS
        assert PROFILE_PRESETS["prod"]["log_level"] == "WARNING"

    def test_test_profile_exists(self):
        """Test test profile is defined."""
        assert "test" in PROFILE_PRESETS
        assert PROFILE_PRESETS["test"]["use_medical_tokenizer"] is False

    def test_fast_profile_exists(self):
        """Test fast profile is defined."""
        assert "fast" in PROFILE_PRESETS
        assert PROFILE_PRESETS["fast"]["timeout"] == 120


class TestOpenMedConfigProfiles:
    """Tests for OpenMedConfig profile methods."""

    def test_from_profile_dev(self):
        """Test creating config from dev profile."""
        config = OpenMedConfig.from_profile("dev")

        assert config.log_level == "DEBUG"
        assert config.timeout == 600
        assert config.profile == "dev"

    def test_from_profile_prod(self):
        """Test creating config from prod profile."""
        config = OpenMedConfig.from_profile("prod")

        assert config.log_level == "WARNING"
        assert config.timeout == 300
        assert config.profile == "prod"

    def test_from_profile_with_overrides(self):
        """Test creating config from profile with overrides."""
        config = OpenMedConfig.from_profile("dev", timeout=1000)

        assert config.log_level == "DEBUG"  # From profile
        assert config.timeout == 1000  # Override

    def test_from_profile_unknown(self):
        """Test error on unknown profile."""
        with pytest.raises(ValueError, match="Unknown profile"):
            OpenMedConfig.from_profile("nonexistent")

    def test_with_profile(self):
        """Test applying profile to existing config."""
        config = OpenMedConfig(default_org="CustomOrg", timeout=999)
        new_config = config.with_profile("dev")

        assert new_config.default_org == "CustomOrg"  # Preserved
        assert new_config.log_level == "DEBUG"  # From profile
        assert new_config.timeout == 600  # From profile (overrides)
        assert new_config.profile == "dev"

    def test_with_profile_unknown(self):
        """Test error when applying unknown profile."""
        config = OpenMedConfig()

        with pytest.raises(ValueError, match="Unknown profile"):
            config.with_profile("nonexistent")

    def test_profile_in_to_dict(self):
        """Test profile is included in to_dict."""
        config = OpenMedConfig.from_profile("test")
        data = config.to_dict()

        assert data["profile"] == "test"

    def test_from_dict_with_profile(self):
        """Test from_dict preserves profile."""
        config = OpenMedConfig.from_dict({"profile": "prod", "timeout": 500})

        assert config.profile == "prod"
        assert config.timeout == 500


class TestProfileEnvVar:
    """Tests for OPENMED_PROFILE environment variable."""

    def test_env_profile_applied(self):
        """Test profile from environment variable."""
        with patch.dict(os.environ, {"OPENMED_PROFILE": "dev"}):
            config = OpenMedConfig()
            assert config.profile == "dev"

    def test_explicit_profile_overrides_env(self):
        """Test explicit profile overrides env var."""
        with patch.dict(os.environ, {"OPENMED_PROFILE": "dev"}):
            config = OpenMedConfig(profile="prod")
            assert config.profile == "prod"


class TestListProfiles:
    """Tests for list_profiles function."""

    def test_list_builtin_profiles(self):
        """Test listing built-in profiles."""
        profiles = list_profiles()

        assert "dev" in profiles
        assert "prod" in profiles
        assert "test" in profiles
        assert "fast" in profiles

    def test_list_profiles_sorted(self):
        """Test profiles are sorted."""
        profiles = list_profiles()
        assert profiles == sorted(profiles)

    def test_list_includes_custom_profiles(self):
        """Test listing includes custom profiles."""
        with TemporaryDirectory() as tmp_dir:
            profiles_dir = Path(tmp_dir) / "profiles"
            profiles_dir.mkdir()

            # Create a custom profile
            custom_profile = profiles_dir / "custom.toml"
            custom_profile.write_text("log_level = DEBUG", encoding="utf-8")

            with patch("openmed.core.config.PROFILES_DIR", profiles_dir):
                profiles = list_profiles()

            assert "custom" in profiles


class TestGetProfile:
    """Tests for get_profile function."""

    def test_get_builtin_profile(self):
        """Test getting built-in profile."""
        settings = get_profile("dev")

        assert settings["log_level"] == "DEBUG"
        assert settings["timeout"] == 600

    def test_get_unknown_profile(self):
        """Test error on unknown profile."""
        with pytest.raises(ValueError, match="Unknown profile"):
            get_profile("nonexistent")

    def test_get_custom_profile(self):
        """Test getting custom profile from file."""
        with TemporaryDirectory() as tmp_dir:
            profiles_dir = Path(tmp_dir) / "profiles"
            profiles_dir.mkdir()

            # Create custom profile
            custom_profile = profiles_dir / "myprofile.toml"
            custom_profile.write_text("timeout = 999\nlog_level = \"ERROR\"", encoding="utf-8")

            with patch("openmed.core.config.PROFILES_DIR", profiles_dir):
                settings = get_profile("myprofile")

            assert settings["timeout"] == 999
            assert settings["log_level"] == "ERROR"


class TestSaveProfile:
    """Tests for save_profile function."""

    def test_save_profile(self):
        """Test saving a custom profile."""
        with TemporaryDirectory() as tmp_dir:
            profiles_dir = Path(tmp_dir) / "profiles"

            with patch("openmed.core.config.PROFILES_DIR", profiles_dir):
                settings = {"log_level": "DEBUG", "timeout": 1000}
                saved_path = save_profile("custom", settings)

            assert saved_path.exists()
            content = saved_path.read_text(encoding="utf-8")
            assert "log_level" in content
            assert "timeout" in content

    def test_save_profile_creates_directory(self):
        """Test save_profile creates profiles directory."""
        with TemporaryDirectory() as tmp_dir:
            profiles_dir = Path(tmp_dir) / "profiles"
            assert not profiles_dir.exists()

            with patch("openmed.core.config.PROFILES_DIR", profiles_dir):
                save_profile("test", {"log_level": "INFO"})

            assert profiles_dir.exists()


class TestDeleteProfile:
    """Tests for delete_profile function."""

    def test_delete_custom_profile(self):
        """Test deleting a custom profile."""
        with TemporaryDirectory() as tmp_dir:
            profiles_dir = Path(tmp_dir) / "profiles"
            profiles_dir.mkdir()

            # Create custom profile
            custom_profile = profiles_dir / "myprofile.toml"
            custom_profile.write_text("log_level = DEBUG", encoding="utf-8")
            assert custom_profile.exists()

            with patch("openmed.core.config.PROFILES_DIR", profiles_dir):
                result = delete_profile("myprofile")

            assert result is True
            assert not custom_profile.exists()

    def test_delete_nonexistent_profile(self):
        """Test deleting nonexistent profile returns False."""
        with TemporaryDirectory() as tmp_dir:
            profiles_dir = Path(tmp_dir) / "profiles"
            profiles_dir.mkdir()

            with patch("openmed.core.config.PROFILES_DIR", profiles_dir):
                result = delete_profile("nonexistent")

            assert result is False

    def test_delete_builtin_profile_raises(self):
        """Test deleting built-in profile raises error."""
        with pytest.raises(ValueError, match="Cannot delete built-in profile"):
            delete_profile("dev")


class TestLoadConfigWithProfile:
    """Tests for load_config_with_profile function."""

    def test_load_with_explicit_profile(self):
        """Test loading config with explicit profile."""
        config = load_config_with_profile(profile_name="dev")

        assert config.log_level == "DEBUG"
        assert config.profile == "dev"

    def test_load_with_env_profile(self):
        """Test loading config with env profile."""
        with patch.dict(os.environ, {"OPENMED_PROFILE": "prod"}):
            config = load_config_with_profile()

        assert config.log_level == "WARNING"
        assert config.profile == "prod"

    def test_explicit_profile_overrides_env(self):
        """Test explicit profile overrides env var."""
        with patch.dict(os.environ, {"OPENMED_PROFILE": "prod"}):
            config = load_config_with_profile(profile_name="dev")

        assert config.log_level == "DEBUG"
        assert config.profile == "dev"

    def test_load_without_profile(self):
        """Test loading config without profile."""
        with patch.dict(os.environ, {}, clear=True):
            # Clear any existing OPENMED_PROFILE
            os.environ.pop("OPENMED_PROFILE", None)
            config = load_config_with_profile()

        # Should return defaults without profile
        assert config.log_level == "INFO"
