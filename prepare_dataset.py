import cv2
import os

classes = ["happy", "sad"]

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

for class_name in classes:

    source = class_name
    target = class_name + "_faces"

    os.makedirs(target, exist_ok=True)

    files = [
        f for f in os.listdir(source)
        if f.lower().endswith(".jpg")
    ]

    saved = 0

    for file in files:

        path = os.path.join(source, file)

        image = cv2.imread(path)

        if image is None:
            continue

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        if len(faces) == 0:
            print("No face:", path)
            continue

        # Select largest detected face
        x, y, w, h = max(
            faces,
            key=lambda f: f[2] * f[3]
        )

        face = image[y:y+h, x:x+w]

        face = cv2.resize(
            face,
            (128, 128)
        )

        output = os.path.join(
            target,
            file
        )

        cv2.imwrite(output, face)

        saved += 1

    print(
        f"{class_name}: {saved} face images created"
    )

print("Dataset preparation completed!")