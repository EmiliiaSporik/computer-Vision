import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from tensorflow.keras.preprocessing import image

#downloading files
train_ds = tf.keras.preprocessing.image_dataset_from_directory("data/train", image_size=(128, 128), batch_size=30, label_mode="categorical")
test_ds = tf.keras.preprocessing.image_dataset_from_directory("data/test", image_size=(128, 128), batch_size=30, label_mode="categorical")

#нормалізація зображень
normalization_layer = layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

model = models.Sequential()

#прості ознаки
model.add(layers.Conv2D(
    filters = 32, #кількість фільрів
    kernel_size = (3, 3), #розмір фільтрів
    activation='relu', #функція активації
    input_shape=(128, 128, 3) #форма вхідного зображення
))
model.add(layers.MaxPooling2D(pool_size = (2, 2))) #зменшуємо карту ознако у 2 рази

model.add(layers.Conv2D(filters = 64, kernel_size = (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size = (2, 2)))

model.add(layers.Conv2D(filters = 128, kernel_size = (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size = (2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(64, activation = "relu"))
model.add(layers.Dense(3, activation = "softmax"))

#компіляція
model.compile(
    optimizer = "adam",
    loss = "categorical_crossentropy",
    metrics = ["accuracy"]
)

history = model.fit(
    train_ds,
    epochs = 15,
    validation_data = test_ds
)

test_loss, test_acc = model.evaluate(test_ds)
print(f"Якісь {test_acc}")

class_name = ["cars", "cats", "dogs"]
img = image.load_img("image/", target_size=(128, 128))

image_array = image.img_to_array(img)
image_array = image_array/255.0
image_array = np.expand_dims(image_array, axis=0)
predictions = model.predict(image_array)
predict_index = np.argmax(predictions[0])

print(f'Імовірність по класам: {predictions[0]}')
print(f'Модель визначила: {class_name[predict_index]}')