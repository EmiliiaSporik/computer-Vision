import pandas as pd #робта з csv (таблицями)
import numpy as np #математичні операції
import tensorflow as tf #для нейронок
from tensorflow import keras # бібліотека для тс
from tensorflow.keras import layers #для шарів
from sklearn.preprocessing import LabelEncoder #текстові мітки в числа
import matplotlib.pyplot as plt #

df = pd.read_csv("data/figures.csv") #робота з
# print(df.head())

encoder = LabelEncoder()
df["label_enc"] = encoder.fit_transform(df["label"])

#3 обираємо елементи для навчання
X = df[["area", "perimeter", "corners"]]
y = df["label_enc"]

#4 створення моделі
model = keras.Sequential([layers.Dense(8, activation = "relu", input_shape = (3,)),
                          layers.Dense(8, activation = "relu"),
                          layers.Dense(8, activation = "softmax")]) #шари розташовані послідовно

#5 навчання моделі
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
history = model.fit(X, y, epochs = 200, verbose = 0)
#6 візуалізація навчання
plt.plot(history.history['loss'], label = "Втрати (loss)")
plt.plot(history.history['accuracy'], label = "Точність (accuracy)")
plt.xlabel("Епоха")
plt.ylabel("Значення")
plt.title("Процес навчання")
plt.legend()
plt.show()

test = np.array([18, 16, 0])

pred = model.predict(test)
print(f"Імовірність по кожному класу: {pred}")
print(f"модель визначила: {encoder.inverse_transform([np.argmax(pred)])}")