"""
Tests for enhanced configuration with user workspaces.
"""

from app.config import Config, DEFAULT_USER_ID


class TestConfigUserWorkspace:
    """Tests for user workspace functionality in Config"""

    def test_default_user_id(self):
        """Should have default user ID"""
        config = Config()
        assert config.user_id == DEFAULT_USER_ID
        assert config.user_id == "default"

    def test_user_persist_path_property(self):
        """Should generate user-specific persist path"""
        config = Config(persist_path="vectorstore", user_id="alice")
        assert config.user_persist_path == "vectorstore/alice"

        config.user_id = "bob"
        assert config.user_persist_path == "vectorstore/bob"

    def test_from_env_loads_user_id(self, monkeypatch):
        """Should load user ID from environment variable"""
        monkeypatch.setenv("ORION_USER_ID", "test_user")

        config = Config.from_env()
        assert config.user_id == "test_user"
        assert config.user_persist_path == f"{config.persist_path}/test_user"

    def test_multiple_users_different_paths(self):
        """Should generate different paths for different users"""
        config1 = Config(user_id="alice", persist_path="vectorstore")
        config2 = Config(user_id="bob", persist_path="vectorstore")

        assert config1.user_persist_path != config2.user_persist_path
        assert config1.user_persist_path == "vectorstore/alice"
        assert config2.user_persist_path == "vectorstore/bob"

    def test_user_workspace_isolation(self):
        """Should ensure user workspace paths are isolated"""
        users = ["alice", "bob", "charlie"]
        configs = [Config(user_id=user, persist_path="vectorstore") for user in users]

        paths = [config.user_persist_path for config in configs]

        # All paths should be unique
        assert len(set(paths)) == len(paths)

        # All should contain the base path and user ID
        for i, path in enumerate(paths):
            assert "vectorstore" in path
            assert users[i] in path
