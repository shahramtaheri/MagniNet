import json, os, time

class JsonlLogger:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path

    def log(self, record: dict):
        record = dict(record)
        record["time"] = time.time()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
