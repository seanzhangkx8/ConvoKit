import os
from pathlib import Path
import yaml
from typing import Optional

class GenAIConfigManager:
    def __init__(self, path: Optional[str] = None):
        if path is None:
            path = os.path.expanduser("~/.convokit/config.yml")
        self.path = Path(path)
        self._data = {}
        self._load()

    def _load(self):
        if self.path.exists():
            self._data = yaml.safe_load(self.path.read_text()) or {}
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._data = {}
            self._save()
            
    def _save(self):
        self.path.write_text(yaml.safe_dump(self._data))

    def set_api_key(self, provider: str, key: str):
        self._data.setdefault('api_keys', {})[provider] = key
        self._save()

    def get_api_key(self, provider: str) -> Optional[str]:
        env = os.getenv(f"{provider.upper()}_API_KEY")
        if env:
            return env
        return self._data.get('api_keys', {}).get(provider)
