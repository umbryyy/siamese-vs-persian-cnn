from model import create_model
from data import load_data

train_data, val_data = load_data()

model = create_model()

model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

model.save("gatti_model.keras")
print(train_data.class_indices)
