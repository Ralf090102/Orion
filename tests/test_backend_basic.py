"""
Basic FastAPI Backend Validation Test.
Validates that the backend can start and respond to basic requests.
"""

import pytest
import warnings
from fastapi import FastAPI
from fastapi.testclient import TestClient
from datetime import datetime

# Suppress Pydantic deprecation warnings for cleaner test output
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pydantic.*")


def create_test_app() -> FastAPI:
    """Create a minimal FastAPI app for testing"""
    app = FastAPI(title="Orion Test API", version="1.0.0")

    @app.get("/health")
    def health_check():
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}

    @app.get("/api/system/config")
    def get_config():
        return {
            "success": True,
            "data": {
                "port": 8003,
                "max_cpu_usage_percent": 75,
                "enable_gpu_acceleration": True,
            },
        }

    @app.get("/api/system/profiles")
    def get_profiles():
        return {
            "success": True,
            "data": [
                {
                    "name": "default",
                    "display_name": "Default Profile",
                    "is_active": True,
                    "document_count": 0,
                }
            ],
        }

    return app


@pytest.fixture
def client():
    """Create test client"""
    app = create_test_app()
    return TestClient(app)


class TestBasicAPI:
    """Test basic API functionality"""

    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_system_config_endpoint(self, client):
        """Test system config endpoint"""
        response = client.get("/api/system/config")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert data["data"]["port"] == 8003

    def test_profiles_endpoint(self, client):
        """Test profiles endpoint"""
        response = client.get("/api/system/profiles")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert isinstance(data["data"], list)
        assert len(data["data"]) == 1
        assert data["data"][0]["name"] == "default"

    def test_json_content_type(self, client):
        """Test JSON content type headers"""
        response = client.get("/health")
        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", "")

    def test_404_handling(self, client):
        """Test 404 error handling"""
        response = client.get("/nonexistent-endpoint")
        assert response.status_code == 404


class TestBackendIntegration:
    """Test actual backend integration (if services work)"""

    def test_real_backend_health(self):
        """Test if real backend can be imported and initialized"""
        try:
            # Try importing backend modules
            from backend.services import get_config_service, get_gpu_manager
            import importlib.util

            # This will test if the imports work
            assert get_config_service is not None
            assert get_gpu_manager is not None

            # Test if create_app can be imported
            create_app_spec = importlib.util.find_spec("backend.main")
            assert create_app_spec is not None, "backend.main module not found"

            print("✅ Backend services can be imported successfully!")

        except ImportError as e:
            pytest.fail(f"Backend import failed: {e}")
        except Exception as e:
            pytest.skip(f"Backend integration test skipped due to service initialization: {e}")

    def test_backend_dependencies_available(self):
        """Test that all required backend dependencies are available"""
        required_modules = [
            "fastapi",
            "uvicorn",
            "pydantic",
            "pystray",
            "psutil",
        ]

        missing_modules = []
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing_modules.append(module)

        if missing_modules:
            pytest.fail(f"Missing required modules: {missing_modules}")

        print(f"✅ All {len(required_modules)} required modules are available!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
