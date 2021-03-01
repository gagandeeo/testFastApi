import joblib
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np

BASE_DIR = Path(__file__).resolve(strict=True).parent


def train():
    iris = load_iris()
    X = iris.data[:, :2]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123)
    svc = SVC()
    svc.fit(X_train, y_train)
    # print(svc.score(X_test, y_test))
    joblib.dump(svc, "./model.joblib", compress=True)


def predict(input_data):
    input_data = np.array(input_data).reshape(1, 2)
    model_file = Path(BASE_DIR).joinpath("./model.joblib")
    if not model_file.exists():
        return False
    model = joblib.load(model_file)
    return model.predict(input_data)[0]


# train()
# print(predict([6.1, 2.8]))
