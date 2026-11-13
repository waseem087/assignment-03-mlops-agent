"""Generate a reproducible synthetic classification dataset."""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data/train.csv")
    a = p.parse_args()

    X, y = make_classification(
        n_samples=a.rows, n_features=10, n_informative=6,
        n_redundant=2, class_sep=1.0, random_state=a.seed,
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
    df["label"] = y
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {df.shape}")


if __name__ == "__main__":
    main()
