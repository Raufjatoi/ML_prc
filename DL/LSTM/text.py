from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dense
import numpy as np

# Sample text data
text = "Hello world! This is a simple LSTM text generation example."

# Initialize and fit tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
sequence_data = tokenizer.texts_to_sequences([text])[0]

# Prepare input and output sequences
X = []
y = []
for i in range(1, len(sequence_data)):
    X.append(sequence_data[:-i])
    y.append(sequence_data[i])

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Building LSTM model for text generation
model = Sequential()
# Add Embedding layer - vocabulary size plus 1, output dim 10
model.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=10, input_length=X.shape[1]))
# Add LSTM layer with 50 units
model.add(LSTM(50))
# Add Dense layer with softmax activation for word prediction
model.add(Dense(len(tokenizer.word_index)+1, activation='softmax'))

# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(X, y, epochs=100, verbose=1)

# Function to generate new text
def generate_text(seed_text, next_words):
    for _ in range(next_words):
        # Convert seed text to sequence
        tokenized_text = tokenizer.texts_to_sequences([seed_text])[0]
        tokenized_text = np.array([tokenized_text])
        # Predict next word
        predicted_word_index = model.predict(tokenized_text, verbose=0).argmax()
        # Add predicted word to seed text
        seed_text += " " + tokenizer.index_word[predicted_word_index]
    return seed_text

# Generate new text starting with "Hello"
generated_text = generate_text("Hello", 5)
print(generated_text)