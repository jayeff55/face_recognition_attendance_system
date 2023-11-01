import csv
import os
import cv2
from datetime import date, datetime
import joblib
import numpy as np
import pandas as pd
import re


def initialise_tracking(date_today: str):
    # Check + create relevant directories and files
    if not os.path.isdir("Attendance"):
        os.mkdir("Attendance")
    if not os.path.isdir("static"):
        os.mkdir("static")
    if not os.path.isdir("static/faces"):
        os.mkdir("static/faces")

    # check if attendance file for today exist
    attendance_today_filename = f"Attendance-{date_today}.csv"
    if attendance_today_filename not in os.listdir("Attendance"):
        with open(f"Attendance/{attendance_today_filename}", 'w') as f:
            f.write('Name,UserID,Time')


# extract face from image
def extract_faces(img: np.ndarray):
    # Convert colour space of image
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Initialise VideoCapture object to access Webcam
    face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # Return boundary rectangles for face (using frontal face cascade)
    face = face_detector.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))

    return face


# identify the face
def identify_face(face_array: np.ndarray, model_filepath: str):
    model = joblib.load(model_filepath)
    return model.predict(face_array)


# add attendance to csv
def add_attendance(name: str, date_today: str):
    username = name.split("_")[0]
    userid = name.split("_")[1]
    time_now = datetime.now().strftime("%H:%M:%S")

    # check if user attendance already logged/exist
    attendance_today_filename = f"Attendance-{date_today}.csv"
    df = pd.read_csv(f"Attendance/{attendance_today_filename}")
    if username not in list(df["Name"]):
        with open(f"Attendance/{attendance_today_filename}", 'a') as f:
            f.write(f'\n{username},{userid},{time_now}')


# extract summary of attendance
def get_attendance(date_today: str):
    attendance_today_filename = f"Attendance-{date_today}.csv"
    df = pd.read_csv(f"Attendance/{attendance_today_filename}")
    names = df["Name"]
    user_ids = df["UserID"]
    times = df["Time"]
    total = len(names)

    return names, user_ids, times, total


# get total registered students
def get_total_users():
    return len(os.listdir("static/faces"))


def check_username_format(username: str):
    pattern = "^[a-zA-Z]+_{1}[0-9]+$"
    if re.match(pattern, username):
        return True
    else:
        return False


def check_user_exist(username: str):
    if username in os.listdir("static/faces"):
        return True
    else:
        return False
