import os
import json
import joblib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from app.core.config import settings
from app.core.logging import logger


class ModelRegistry:
    """Manages serialization, versioning, loading, and metadata tracking of ML models."""

    def __init__(self, registry_dir: str = None):
        self.registry_dir = Path(registry_dir or settings.MODEL_REGISTRY_PATH)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.registry_dir / "registry_metadata.json"
        self._init_metadata()

    def _init_metadata(self):
        if not self.metadata_file.exists():
            with open(self.metadata_file, "w") as f:
                json.dump({"models": {}}, f, indent=2)

    def _read_metadata(self) -> Dict[str, Any]:
        with open(self.metadata_file, "r") as f:
            return json.load(f)

    def _save_metadata(self, data: Dict[str, Any]):
        with open(self.metadata_file, "w") as f:
            json.dump(data, f, indent=2)

    def register_model(
        self,
        model_name: str,
        model_obj: Any,
        model_type: str,
        features: list,
        metrics: Dict[str, float],
        parameters: Dict[str, Any] = None,
        version: str = "v1.0.0",
    ) -> str:
        """Saves model artifact and updates central metadata registry."""
        model_filename = f"{model_name}_{version}.joblib"
        model_path = self.registry_dir / model_filename

        joblib.dump(model_obj, model_path)

        meta = self._read_metadata()
        meta["models"][model_name] = {
            "model_name": model_name,
            "version": version,
            "model_type": model_type,
            "artifact_file": model_filename,
            "artifact_path": str(model_path),
            "features": features,
            "metrics": metrics,
            "parameters": parameters or {},
            "registered_at": datetime.utcnow().isoformat(),
        }
        self._save_metadata(meta)
        logger.info(f"Successfully registered model '{model_name}' ({version}) at {model_path}")
        return str(model_path)

    def get_model(self, model_name: str) -> Optional[Any]:
        """Loads a registered model artifact by name."""
        meta = self._read_metadata()
        if model_name not in meta["models"]:
            logger.warning(f"Model '{model_name}' not found in registry.")
            return None
        
        info = meta["models"][model_name]
        path = Path(info["artifact_path"])
        if not path.exists():
            logger.error(f"Model artifact file missing at: {path}")
            return None

        return joblib.load(path)

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Returns metadata for a specific model."""
        meta = self._read_metadata()
        return meta["models"].get(model_name)

    def list_all_models(self) -> Dict[str, Any]:
        """Returns all registered models and their performance metrics."""
        return self._read_metadata().get("models", {})


registry = ModelRegistry()
