from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# 1. carica modello
model = load_model("gatti_model.keras")

# 2. carica immagine
img = image.load_img("test.jpg", target_size=(128, 128))

# 3. converti immagine in array
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# 4. predizione
prediction = model.predict(img_array)

# 5. interpretazione
print("Valore output:", prediction)

if prediction > 0.5:
    print("👉 Persiano")
else:
    print("👉 Siamese")