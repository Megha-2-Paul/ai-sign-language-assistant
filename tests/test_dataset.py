import pandas as pd
import pytest
from src.data.dataset import load_landmark_csv

def frame():
    data = {f"f{i}": [0.0, 0.1, 0.2, 0.3] for i in range(63)}
    data.update(label=["A", "A", "B", "B"], handedness=["Left", "Right", "Left", "Right"], hand_confidence=[.9, .8, .95, .85])
    return pd.DataFrame(data)

def write_csv(tmp_path, data):
    path = tmp_path / "landmarks.csv"
    data.to_csv(path, index=False)
    return path

def test_loader_excludes_metadata(tmp_path):
    x, y = load_landmark_csv(write_csv(tmp_path, frame()))
    assert x.shape == (4, 63)
    assert all(c.startswith("f") for c in x.columns)
    assert y.tolist() == ["A", "A", "B", "B"]

def test_missing_label_rejected(tmp_path):
    with pytest.raises(ValueError, match="Missing label"):
        load_landmark_csv(write_csv(tmp_path, pd.DataFrame({"f0": [1]})))

def test_empty_data_rejected(tmp_path):
    with pytest.raises(ValueError, match="no samples"):
        load_landmark_csv(write_csv(tmp_path, frame().iloc[:0]))

def test_nonnumeric_feature_rejected(tmp_path):
    data = frame()
    data["f5"] = data["f5"].astype(object)
    data.loc[0, "f5"] = "bad"
    with pytest.raises(ValueError, match="non-numeric"):
        load_landmark_csv(write_csv(tmp_path, data))

def test_wrong_feature_count_rejected(tmp_path):
    with pytest.raises(ValueError, match="exactly 63"):
        load_landmark_csv(write_csv(tmp_path, frame().drop(columns=["f62"])))
