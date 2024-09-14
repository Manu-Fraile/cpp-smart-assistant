# train_model.py
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Sample data (task description, hour, priority)
data = {
    "description": [
        "Buy groceries",
        "Finish report",
        "Call mom",
        "Prepare presentation",
    ],
    "hour": [14, 9, 18, 11],
    "priority": [2, 1, 3, 2],
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Tokenize and pad sequences for text input
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(df["description"])
sequences = tokenizer.texts_to_sequences(df["description"])
padded_sequences = pad_sequences(sequences, maxlen=10)

# Prepare input features and labels
X = np.hstack((padded_sequences, np.array(df["hour"]).reshape(-1, 1)))
y = np.array(df["priority"])

# Define the model
model = Sequential(
    [
        Embedding(1000, 8, input_length=X.shape[1] - 1),
        Flatten(),
        Dense(10, activation="relu"),
        Dense(1, activation="linear"),
    ]
)

model.compile(optimizer="adam", loss="mean_squared_error")

# Train the model
model.fit(X, y, epochs=10)

# Save the model
model.save("models/task_priority_model.h5")

# Save the tokenizer
joblib.dump(tokenizer, "models/tokenizer.pkl")
