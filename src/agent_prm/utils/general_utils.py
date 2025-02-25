import json

def load_json(fp: str):
    with open(fp, "r") as f:
        return json.load(f)
    
def save_json(fp: str, data: dict):
    with open(fp, "w") as f:
        json.dump(data, f, indent=4)