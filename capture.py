import cv2
import os

print("Choose class:")
print("1 = Happy")
print("2 = Sad")

choice = input("Enter 1 or 2: ")

if choice == "1":
    folder = "happy"
    prefix = "happy"
elif choice == "2":
    folder = "sad"
    prefix = "sad"
else:
    print("Invalid choice")
    exit()

os.makedirs(folder, exist_ok=True)

# Find existing photos
existing_files = [
    f for f in os.listdir(folder)
    if f.lower().endswith(".jpg")
]

count = len(existing_files)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera not found")
    exit()

print(f"Existing photos: {count}")
print("Press S to save photo")
print("Press Q to quit")

while True:
    ret, frame = camera.read()

    if not ret:
        print("Cannot read camera")
        break

    cv2.putText(
        frame,
        "S = Save | Q = Quit",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"{prefix}: {count}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("Dataset Capture", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):

        count += 1

        filename = os.path.join(
            folder,
            f"{prefix}{count}.jpg"
        )

        cv2.imwrite(filename, frame)

        print("Saved:", filename)

    elif key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print(f"Total {prefix} photos: {count}")