"""Generate simulated manufacturing quality data for EDA and modeling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "manufacturing_quality.csv"

N_ROWS = 500
RANDOM_SEED = 42


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def defect_probability(
    temperature: np.ndarray,
    pressure: np.ndarray,
    vibration: np.ndarray,
) -> np.ndarray:
    """Higher temperature, pressure, and vibration increase defect risk."""
    risk_score = (
        0.10 * (temperature - 72.0)
        + 1.50 * (pressure - 5.2)
        + 1.20 * (vibration - 2.0)
    )
    return sigmoid(risk_score - 0.8)


def generate_dataset(n_rows: int = N_ROWS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    temperature = rng.normal(72.0, 5.0, n_rows).clip(55.0, 90.0)
    pressure = rng.normal(5.2, 0.6, n_rows).clip(3.0, 7.5)
    vibration = rng.normal(2.0, 0.8, n_rows).clip(0.2, 4.5)
    speed = rng.normal(120.0, 15.0, n_rows).clip(80.0, 160.0)
    humidity = rng.normal(48.0, 8.0, n_rows).clip(25.0, 70.0)

    defect_prob = defect_probability(temperature, pressure, vibration)
    is_defect = rng.random(n_rows) < defect_prob
    target = np.where(is_defect, "defect", "normal")

    return pd.DataFrame(
        {
            "temperature": np.round(temperature, 2),
            "pressure": np.round(pressure, 2),
            "vibration": np.round(vibration, 2),
            "speed": np.round(speed, 2),
            "humidity": np.round(humidity, 2),
            "target": target,
        }
    )


def main() -> None:
    df = generate_dataset()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    defect_rate = (df["target"] == "defect").mean() * 100
    print(f"✅ 已生成 {len(df)} 行数据：{OUTPUT_PATH}")
    print(f"   defect 占比：{defect_rate:.1f}%")


if __name__ == "__main__":
    main()
