import tensorflow as tf
from tensorflow.keras import layers, models

# Load face-only dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    ".",
    labels="inferred",
    label_mode="binary",
    class_names=["happy_faces", "sad_faces"],
    image_size=(128, 128),
    batch_size=8,
    validation_split=0.2,
    subset="training",
    seed=123
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    ".",
    labels="inferred",
    label_mode="binary",
    class_names=["happy_faces", "sad_faces"],
    image_size=(128, 128),
    batch_size=8,
    validation_split=0.2,
    subset="validation",
    seed=123
)

print("Classes:", train_ds.class_names)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.shuffle(100).prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# CNN
model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),

    layers.Rescaling(1.0 / 255),

    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20
)

model.save("happy_sad_model.keras")

print("================================")
print("FACE MODEL TRAINING COMPLETED!")
print("Model saved successfully!")
print("================================")