from __future__ import annotations

import json

import requests


BASE_URL = "http://127.0.0.1:8010"


def main() -> None:
    info = requests.get(f"{BASE_URL}/model-info", timeout=10).json()
    features = info["features"]

    payload = {
        "features": {
            feature: 0 for feature in features
        }
    }

    resp = requests.post(
        f"{BASE_URL}/predict",
        json=payload,
        timeout=10,
    )

    print("status:", resp.status_code)
    print(json.dumps(resp.json(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
