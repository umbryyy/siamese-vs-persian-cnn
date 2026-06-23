import tensorflow as tf
from tensorflow.keras import layers, models

#rete neurale

def create_model():
    model = models.Sequential([
        #immagini 128x128 con 3 canali (RGB)
        layers.Input(shape=(128, 128, 3)),
        # Primo blocco convoluzionale
        # Estrae feature base (bordi, linee, texture)
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),

        # Secondo blocco convoluzionale
        # Estrae caratteristiche più complesse (forme, parti del corpo del gatto)
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),

        # Terzo blocco convoluzionale
        # Riconosce pattern ancora più astratti (muso, occhi, struttura)
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model