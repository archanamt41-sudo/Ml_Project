# tests/test_train.py
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

def test_model_training():
    data = load_iris()
    X, y = data.data, data.target
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)
    assert len(model.classes_) == 3

    