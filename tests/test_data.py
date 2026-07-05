import pandas as pd

from src.data import TARGET_COL, build_preprocessor, prepare_train_test


def test_prepare_train_test():
    df = pd.DataFrame(
        {
            "age": [50, 60, 45, 70],
            "chol": [200, 240, 180, 260],
            TARGET_COL: [0, 1, 0, 1],
        }
    )
    X_train, X_test, y_train, y_test = prepare_train_test(df, test_size=0.5)
    assert len(X_train) == 2
    assert len(X_test) == 2
    assert TARGET_COL not in X_train.columns


def test_build_preprocessor():
    df = pd.DataFrame(
        {
            "age": [50, 60],
            "sex": [1, 0],
            "cp": ["typical", "asymptomatic"],
            TARGET_COL: [0, 1],
        }
    )
    preprocessor = build_preprocessor(df)
    assert preprocessor is not None
