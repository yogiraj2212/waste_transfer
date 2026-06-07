import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization

# Load Pretrained Model
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model
base_model.trainable = False

# Build Model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation="relu"),
    Dense(128, activation='relu'),
    Dense(3, activation="softmax")
])

# Compile Model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Data Generator
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=30,
    zoom_range=0.2,
    horizontal_flip=True
)

# Training Data
train = datagen.flow_from_directory(
    "D:\\yogiraj\\extra_work\\waste_transfer\\dataset",
    target_size=(224, 224),
    class_mode="categorical",
    subset="training"
)

print(train.class_indices)

# Validation Data
val = datagen.flow_from_directory(
    "D:\\yogiraj\\extra_work\\waste_transfer\\dataset",
    target_size=(224, 224),
    class_mode="categorical",
    subset="validation"
)

# Train Model
model.fit(
    train,
    validation_data=val,
    epochs=1
)
# Evaluate model on validation data
loss, accuracy = model.evaluate(val)

print("Final Validation Accuracy:", accuracy)
print("Final Validation Loss:", loss)

# Save Model
model.save("model.h5")
