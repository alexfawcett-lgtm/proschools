import tensorflow as tf 
import numpy as np 

# Expanded to 5 items
user_inputs = ["hello", "how are you", "fine thank you" , "What is your name?" , "Who made you?"]
responses = ["Hey there!", "I am great!", "See you later!"  , "I am Mr.Pro!" , "I was created by ProSchools AI team!"]

# Build vocab
vec = tf.keras.layers.TextVectorization(max_tokens=50, output_sequence_length=10)
vec.adapt(np.array(user_inputs))

X = vec(np.array(user_inputs))

# FIXED: Match the exact number of elements (Now 5 indices for your 5 input sentences)
y = np.array([0, 1, 2, 3, 4]) 

# Model architecture layout
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(10,)),
    tf.keras.layers.Embedding(50, 8),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8)),
    tf.keras.layers.Dense(len(responses), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(X, y, epochs=300, verbose=0)
print("Training complete!")

# Test evaluation
test = vec(np.array(["hello"]))
scores = model.predict(test, verbose=0)[0]
best = np.argmax(scores)

print(f"\nInput: 'hello'")
print(f"Response: {responses[best]}")
print(f"Confidence: {scores[best]:.2f}")
