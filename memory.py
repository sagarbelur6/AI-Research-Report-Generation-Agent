# memory.py
import os
import json
from typing import Dict, Any
import time

# Simple file-based episodic and long-term memory

class EpisodicMemory:
    def __init__(self, path="memory/episodic.json"):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path, "w") as f:
                json.dump({}, f)

    def read(self, episode_id: str):
        with open(self.path, "r") as f:
            db = json.load(f)
        return db.get(episode_id)

    def write(self, episode_id: str, data: Dict[str, Any]):
        with open(self.path, "r") as f:
            db = json.load(f)
        db[episode_id] = {"created_at": time.time(), "data": data}
        with open(self.path, "w") as f:
            json.dump(db, f, indent=2)

class LongTermMemory:
    def __init__(self, path="memory/longterm.json"):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path, "w") as f:
                json.dump({}, f)

    def read(self, key: str):
        with open(self.path, "r") as f:
            db = json.load(f)
        return db.get(key)

    def write(self, key: str, data):
        with open(self.path, "r") as f:
            db = json.load(f)
        db[key] = {"updated_at": time.time(), "value": data}
        with open(self.path, "w") as f:
            json.dump(db, f, indent=2)
