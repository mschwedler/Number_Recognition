import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from PIL import Image

# MNIST-Datensatz laden und normalisieren
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Modell definieren
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# Modell kompilieren
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Modell trainieren
history = model.fit(train_images, train_labels, epochs=50, batch_size=32, validation_split=0.2)

# Modell lokal speichern
model.save('mnist_model.h5')

# Trainingshistorie speichern
with open('training_history.pkl', 'wb') as file:
    pickle.dump(history.history, file)

print("Modell und Trainingshistorie wurden gespeichert.")


def load_custom_image(file_path):
    with Image.open(file_path) as img: # type: ignore
        # Konvertiere das Bild zu Graustufen
        img = img.convert('L')
        
        # Skaliere das Bild auf 28x28 Pixel
        img = img.resize((28, 28))
        
        # Konvertiere das Bild zu einem numpy-Array und normalisiere es
        img_array = np.array(img).astype('float32') / 255.0
        
        # Reshape für das Modell
        return img_array.reshape(1, 28, 28)

# Eigenes Bild testen
custom_image_path = '1.png'  # Pfad zu Ihrem eigenen Bild
if os.path.exists(custom_image_path):
    try:
        custom_image = load_custom_image(custom_image_path)
        prediction = model.predict(custom_image)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        # Bild anzeigen
        plt.imshow(custom_image.reshape(28, 28), cmap='gray')
        plt.title(f"Vorhergesagte Ziffer: {predicted_digit}\nGenauigkeit: {confidence:.2f}%")
        plt.axis('off')
        plt.show()

        print(f"Vorhergesagte Ziffer: {predicted_digit}")
        print(f"Genauigkeit: {confidence:.2f}%")
    except Exception as e:
        print(f"Fehler beim Laden oder Verarbeiten des Bildes: {e}")
else:
    print(f"Die Datei {custom_image_path} wurde nicht gefunden.")