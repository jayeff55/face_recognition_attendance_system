import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib
import cv2
import os


def train_base_model(face_database_path, model_filepath):
    faces = []
    labels = []
    user_list = os.listdir(face_database_path)

    # map face to labels/user's name
    for user in user_list:
        for img_name in os.listdir(f"static/faces/{user}"):
            img = cv2.imread(f"static/faces/{user}/{img_name}")
            resized_img = cv2.resize(img, (50, 50))
            faces.append(resized_img.ravel())
            labels.append(user)

    # create and train model
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, model_filepath)
