import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

df = pd.read_csv("data/figures.csv")

df["ratio_ap"] = df["area"] / df["perimeter"]

encoder = LabelEncoder()
df["label_enc"] = encoder.fit_transform(df["label"])
num_classes = df["label_enc"].nunique()


X = df[["area", "perimeter", "corners", "ratio_ap"]]
y = df["label_enc"]

model = keras.Sequential([
    layers.Dense(16, activation="relu", input_shape=(X.shape[1],)),
    layers.Dense(8, activation="relu"),
    layers.Dense(num_classes, activation="softmax")])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history = model.fit(X, y, epochs = 500, verbose = 0)

plt.plot(history.history['loss'], label = "Втрати (loss)")
plt.plot(history.history['accuracy'], label = "Точність (accuracy)")
plt.xlabel("Епоха")
plt.ylabel("Значення")
plt.title("Процес навчання")
plt.legend()
plt.show()
plt.savefig("statistik.png")

final_accuracy = history.history['accuracy'][-1]
final_loss = history.history['loss'][-1]
