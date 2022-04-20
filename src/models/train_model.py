from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import  train_test_split
from sklearn.metrics import classification_report
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from pathlib import Path
import logging

from src.features.build_features import create_training_data


def train_test_cls(trainX, testX, trainY, testY, classifier, le):
    model = classifier #KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
    model.fit(trainX, trainY)
    return classification_report(testY, model.predict(testX), target_names=le.classes_)


def preprocess_data(data, labels):
    data = np.array(data)
    lables = np.array(labels)
    le = LabelEncoder()
    lables = le.fit_transform(labels)
    # myset = set(lables)
    dataset_size = data.shape[0]
    data = data.reshape(dataset_size, -1)
    # print(data.shape)
    # print(lables.shape)
    # print(dataset_size)
    trainX, testX, trainY, testY = train_test_split(data, lables, test_size=0.25, random_state=42)
    return trainX, testX, trainY, testY, le

def multiple_cls_test(image_path, if_circle):
    data, labels = create_training_data(image_path, if_circle)
    trainX, testX, trainY, testY, le = preprocess_data(data, labels)
    cls = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
    results = train_test_cls(trainX, testX, trainY, testY, cls, le)
    print('The results of KNeighborsClassifier', results)

    cls = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
     max_depth=1, random_state=0)
    results = train_test_cls(trainX, testX, trainY, testY, cls, le)
    print('The results of GradientBoostingClassifier', results)

    cls = DecisionTreeClassifier(random_state=0)
    results = train_test_cls(trainX, testX, trainY, testY, cls, le)
    print('The results of DecisionTreeClassifier', results)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    project_dir = Path(__file__).resolve().parents[2]

    image_path = Path(str(project_dir) + '/data/raw/').resolve()
    # Extract pixels from the whole image
    multiple_cls_test(image_path, False)
    # Extract pixels from the circle of an image.
    multiple_cls_test(image_path, True)