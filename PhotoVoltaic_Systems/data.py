"""
Data loading, preprocessing, and the cross-site split.

When you switch to real data, you should only need to (a) point config.DATA_CSV
at your file and (b) make sure the column names in config.py match.
"""
import pandas as pd

import config


def load_data():
    if not config.DATA_CSV.exists():
        print(f"[data] {config.DATA_CSV.name} not found -> generating synthetic data.")
        import make_synthetic_data
        df = make_synthetic_data.generate()
        df.to_csv(config.DATA_CSV, index=False)
    else:
        df = pd.read_csv(config.DATA_CSV)
    return df


def feature_columns(use_weather):
    cols = list(config.ELECTRICAL_FEATURES)
    if use_weather:
        cols += list(config.WEATHER_FEATURES)
    return cols


def get_xy(df, task, use_weather):
    feats = feature_columns(use_weather)
    X = df[feats].astype(float).copy()
    X = X.fillna(X.median(numeric_only=True))
    if task == "power":
        y = df[config.POWER_COL].astype(float).values
    elif task == "fault":
        y = df[config.FAULT_COL].astype(int).values
    else:
        raise ValueError("task must be 'power' or 'fault'")
    return X, y


def site_frame(df, sites):
    return df[df[config.SITE_COL].isin(sites)].reset_index(drop=True)


def within_site_split(df, site, task, use_weather):
    from sklearn.model_selection import train_test_split
    sub = site_frame(df, [site])
    X, y = get_xy(sub, task, use_weather)
    stratify = y if task == "fault" else None
    return train_test_split(
        X, y, test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE, stratify=stratify,
    )


def cross_site_split(df, train_sites, test_sites, task, use_weather):
    train = site_frame(df, train_sites)
    test = site_frame(df, test_sites)
    X_tr, y_tr = get_xy(train, task, use_weather)
    X_te, y_te = get_xy(test, task, use_weather)
    return X_tr, X_te, y_tr, y_te
