import os.path

from flask import Flask, request, render_template
from attendance_tracker import *
from face_recognition_model import *
from pathlib import Path


# Defining Flask App
app = Flask(__name__)

# Configs
nimgs = 10
model_filename = "face_recognition_model.pkl"
date_today = date.today().strftime("%d_%m_%Y")

initialise_tracking(date_today)


# Routing functions
@app.route("/")
def homepage():
    names, userids, times, total = get_attendance(date_today)
    return render_template("home.html", names=names, rolls=userids, times=times, l=total)


@app.route("/start", methods=["GET"])
def start():
    if model_filename not in os.listdir("static"):
        return render_template("home.html", totalreg=get_total_users(), mess="No trained model in static folder. Please add images of user faces to continue")

    ret = True
    cap = cv2.VideoCapture(0)
    while ret:
        ret, frame = cap.read()
        if len(extract_faces(frame)) > 0:
            (x, y, w, h) = extract_faces(frame)[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (86, 32, 251), 1)
            cv2.rectangle(frame, (x, y), (x + w, y - 40), (86, 32, 251), -1)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_user = identify_face(face.reshape(1, -1), Path("static", model_filename).as_posix())[0]
            add_attendance(identified_user, date_today)
            cv2.putText(frame, f'{identified_user}', (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Attendance", frame)
        if cv2.waitKey(1) == 27:
            break  # or ret = False?
    cap.release()
    cv2.destroyAllWindows()
    names, userids, times, total = get_attendance(date_today)
    return render_template("home.html", names=names, rolls=userids, times=times, l=total, totalreg=get_total_users())


@app.route("/add", methods=["GET", "POST"])
def add():
    new_username = request.form["newusername"]
    new_userid = request.form["newuserid"]
    format_ok = check_username_format(f"{new_username}_{new_userid}")
    if not format_ok:
        return render_template("home.html", totalreg=get_total_users(),
                               mess="Username does not have correct format")
    user_exist = check_user_exist(f"{new_username}")
    if user_exist:
        return render_template("home.html", totalreg=get_total_users(),
                               mess="Username already registered/exist")

    user_img_folder = f"static/faces/{new_username}_{new_userid}"
    if not os.path.isdir(user_img_folder):
        os.mkdir(user_img_folder)

    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    add_user = True
    while add_user:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if j % 5 == 0:
                img_filename = f"{new_username}_{i}.jpg"
                cv2.imwrite(Path(user_img_folder, img_filename).as_posix(), frame[y:y+h, x:x+w])
                i += 1
            j += 1

        if j == nimgs*5:
            add_user = False
        cv2.imshow("Adding new user", frame)
        if cv2.waitKey(1) == 27:
            add_user = False
    cap.release()
    cv2.destroyAllWindows()
    print("Training model")
    train_base_model(face_database_path="static/faces", model_filepath=Path("static", model_filename).as_posix())
    names, userids, times, total = get_attendance(date_today)
    return render_template("home.html", names=names, rolls=userids, times=times, l=total, totalreg=get_total_users())


if __name__ == "__main__":
    app.run(debug=True)

