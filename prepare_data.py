"""Prepare the three datasets for the odd variant of assignment 1."""

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
RANDOM_STATE = 42


@dataclass(frozen=True)
class Dataset:
    filename: str
    target: str
    numeric: tuple[str, ...]
    categorical: tuple[str, ...] = ()
    binary: tuple[str, ...] = ()
    labels: dict[int | str, int] | None = None
    separator: str = ","


DATASETS = {
    "diabetes": Dataset(
        filename="diabetes_012_health_indicators_BRFSS2015.csv",
        target="Diabetes_012",
        numeric=("BMI", "MentHlth", "PhysHlth"),
        categorical=("GenHlth", "Age", "Education", "Income"),
        binary=(
            "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
            "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
            "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex",
        ),
        labels={0: 0, 1: 1, 2: 1},
    ),
    "body_performance": Dataset(
        filename="bodyPerformance.csv",
        target="class",
        numeric=(
            "age", "height_cm", "weight_kg", "body fat_%", "diastolic", "systolic",
            "gripForce", "sit and bend forward_cm", "sit-ups counts", "broad jump_cm",
        ),
        categorical=("gender",),
        labels={"A": 0, "B": 1, "C": 2, "D": 3},
    ),
    "wine_quality": Dataset(
        filename="winequality-white.csv",
        target="quality",
        numeric=(
            "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
            "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
            "pH", "sulphates", "alcohol",
        ),
        separator=";",
    ),
}


def clean_data(raw: pd.DataFrame, dataset: Dataset) -> pd.DataFrame:
    features = list(dataset.numeric + dataset.categorical + dataset.binary)
    data = raw.loc[:, [*features, dataset.target]].copy()
    if dataset.labels is not None:
        data[dataset.target] = data[dataset.target].map(dataset.labels)
    if data[dataset.target].isna().any():
        raise ValueError(f"Missing or unknown target values in {dataset.filename}")

    if dataset.target == "class":
        pressure = ["diastolic", "systolic"]
        invalid = (data[pressure] <= 0).any(axis=1) | (data["diastolic"] > data["systolic"])
        data.loc[invalid, pressure] = np.nan

    return data.drop_duplicates()


def split_data(
    features: pd.DataFrame, target: pd.Series, stratify: bool
) -> dict[str, pd.Index]:
    # Keep identical inputs together even if their targets disagree.
    groups = features.groupby(list(features.columns), sort=False, dropna=False).ngroup()
    profiles = groups.drop_duplicates().to_numpy()
    strata = None
    if stratify:
        strata = target.groupby(groups).agg(lambda labels: ",".join(map(str, sorted(labels))))

    train, holdout = train_test_split(
        profiles, test_size=0.30, random_state=RANDOM_STATE,
        stratify=strata.loc[profiles] if strata is not None else None,
    )
    validation, test = train_test_split(
        holdout, test_size=0.50, random_state=RANDOM_STATE,
        stratify=strata.loc[holdout] if strata is not None else None,
    )
    return {
        name: features.index[groups.isin(profile_ids)]
        for name, profile_ids in (("train", train), ("val", validation), ("test", test))
    }


def make_preprocessor(dataset: Dataset) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
             list(dataset.numeric)),
            ("categorical", make_pipeline(
                SimpleImputer(strategy="most_frequent"),
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ), list(dataset.categorical)),
            ("binary", SimpleImputer(strategy="most_frequent"), list(dataset.binary)),
        ],
        verbose_feature_names_out=False,
    )


def prepare_dataset(raw: pd.DataFrame, dataset: Dataset) -> dict[str, np.ndarray]:
    data = clean_data(raw, dataset)
    target = data[dataset.target]
    features = data.drop(columns=dataset.target)
    classification = dataset.labels is not None
    splits = split_data(features, target, stratify=classification)
    preprocessor = make_preprocessor(dataset)
    preprocessor.fit(features.loc[splits["train"]])

    result = {"feature_names": np.asarray(preprocessor.get_feature_names_out(), dtype=str)}
    if classification:
        result["class_names"] = np.asarray(
            ["A", "B", "C", "D"] if dataset.target == "class" else ["0", "1"]
        )
    for name, index in splits.items():
        result[f"X_{name}"] = preprocessor.transform(features.loc[index]).astype(np.float32)
        result[f"y_{name}"] = target.loc[index].to_numpy(
            dtype=np.int64 if classification else np.float32
        )
        result[f"rows_{name}"] = index.to_numpy(dtype=np.int64)
    return result


def main() -> None:
    output = ROOT / "data" / "processed"
    output.mkdir(parents=True, exist_ok=True)
    # Create staging files directly in output so Windows gives them its inherited ACL.
    run_id = uuid4().hex
    staging = {name: output / f".{name}-{run_id}.npz" for name in DATASETS}
    try:
        for name, dataset in DATASETS.items():
            raw = pd.read_csv(ROOT / "data" / "raw" / dataset.filename, sep=dataset.separator)
            prepared = prepare_dataset(raw, dataset)
            with staging[name].open("xb") as archive:
                np.savez_compressed(archive, **prepared)
            print(name, flush=True)
            for split in ("train", "val", "test"):
                y = prepared[f"y_{split}"]
                values, counts = np.unique(y, return_counts=True)
                distribution = dict(zip(values.tolist(), counts.tolist(), strict=True))
                print(f"  {split}: {prepared[f'X_{split}'].shape}, target counts {distribution}", flush=True)
        for name, archive in staging.items():
            archive.replace(output / f"{name}.npz")
    finally:
        for archive in staging.values():
            archive.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
