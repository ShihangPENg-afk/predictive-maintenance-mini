from __future__ import annotations

import json

import requests


BASE_URL = "http://127.0.0.1:8010"


def main() -> None:
    try:
        info_resp = requests.get(f"{BASE_URL}/model-info", timeout=10)
        info_resp.raise_for_status()
        info = info_resp.json()
    except requests.RequestException as exc:
        print(f"error: cannot reach {BASE_URL}/model-info: {exc}")
        raise SystemExit(1) from exc

    if "features" not in info:
        print("error: /model-info response missing 'features'")
        raise SystemExit(1)

    features = info["features"]

    payload = {
        "features": {
            feature: 0 for feature in features
        }
    }

    try:
        resp = requests.post(
            f"{BASE_URL}/predict",
            json=payload,
            timeout=10,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"error: cannot reach {BASE_URL}/predict: {exc}")
        raise SystemExit(1) from exc

    print("status:", resp.status_code)
    print(json.dumps(resp.json(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
