"""Download the two experiment datasets (cached in data/)."""
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# Public mirrors of the original files (Udacity "Analyze A/B Test Results"; DataCamp / Kaggle "Cookie Cats").
SOURCES = {
    "ab_data.csv": "https://raw.githubusercontent.com/Hindhuja-Gutha/Analyze-AB-TestResults/master/ab_data.csv",
    "countries.csv": "https://raw.githubusercontent.com/Hindhuja-Gutha/Analyze-AB-TestResults/master/countries.csv",
    "cookie_cats.csv": "https://raw.githubusercontent.com/robguilarr/ab_testing_cookie_cats/main/datasets/cookie_cats.csv",
}


def fetch(name: str) -> Path:
    path = DATA_DIR / name
    if not path.exists():
        DATA_DIR.mkdir(exist_ok=True)
        print(f"Downloading {name}")
        urlretrieve(SOURCES[name], path)
    return path


def landing_page_raw() -> pd.DataFrame:
    """Landing-page test: one row per user exposure (user_id, timestamp, group, landing_page, converted)."""
    df = pd.read_csv(fetch("ab_data.csv"), parse_dates=["timestamp"])
    countries = pd.read_csv(fetch("countries.csv"))
    return df.merge(countries, on="user_id", how="left")


def cookie_cats() -> pd.DataFrame:
    """Cookie Cats test: one row per player (userid, version, sum_gamerounds, retention_1, retention_7)."""
    return pd.read_csv(fetch("cookie_cats.csv"))
