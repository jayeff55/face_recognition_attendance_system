# Facial recognition attendance system
An easy-to-use browser-based attendance system that requires only an in-built camera!

## Overview
Easily launched in any browser, this system allows:
- New users to be added for attendance taking
- For existing user attendance to be taken using facial recognition verification

As part of adding the new user, a pre-defined number of images of the new user would be collected using an in-built camera to train a k-nearest neighbors classifier.

Once a new user has been added, or if in the case of an existing user, attendance can be taken by one simple click to process the face recognition verification and logged accordingly.

![image]([https://github.com/jayeff55/face_recognition_attendance_system/assets/74664129/40076134-7c0b-4ceb-99e4-38aff075a347](https://raw.githubusercontent.com/jayeff55/face_recognition_attendance_system/main/app.png))

## Running the app
Dependencies: `python3`, `opencv-python`, `flask`

Ensure that you have all dependencies downloaded, then simply run `flask run` in the project terminal

### Technologies
- Language: Python, HTML, CSS
- Libraries: flask, openCV, sklearn
