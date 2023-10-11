import pandas as pd
from sklearn.model_selection import train_test_split

# Read the CSV file
df = pd.read_csv('preprocessed_dataset.csv')

# Splitting the dataset into 80% training and 20% testing
X = df['text']
y = df['emotion']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dense, Bidirectional
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Tokenize and pad sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_train_pad = pad_sequences(X_train_seq, maxlen=100)
X_test_pad = pad_sequences(X_test_seq, maxlen=100)

# Convert labels to categorical
le = LabelEncoder()
y_train_cat = to_categorical(le.fit_transform(y_train))
y_test_cat = to_categorical(le.transform(y_test))

# LSTM Model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=50, input_length=100))
#model.add(LSTM(50))
model.add(Bidirectional(LSTM(50)))
num_classes = y_train_cat.shape[1]
model.add(Dense(num_classes, activation='softmax'))
#model.add(Dense(2, activation='softmax'))  # Assuming only 'frustrated' and 'engaged' emotions
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# For Bi-LSTM, replace the LSTM layer with:
# model.add(Bidirectional(LSTM(50)))

# Train the model
history = model.fit(X_train_pad, y_train_cat, epochs=10, validation_data=(X_test_pad, y_test_cat))


loss, accuracy = model.evaluate(X_test_pad, y_test_cat)
print(f'Testing Accuracy: {accuracy*100:.2f}%')


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Plot Training vs Testing
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Confusion Matrix
#y_pred = model.predict_classes(X_test_pad)
import numpy as np
y_pred = np.argmax(model.predict(X_test_pad), axis=-1)
cm = confusion_matrix(le.transform(y_test), y_pred)
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

def predict_emotion(sentence):
    sequence = tokenizer.texts_to_sequences([sentence])
    padded_sequence = pad_sequences(sequence, maxlen=100)
    prediction = np.argmax(model.predict(padded_sequence), axis=-1)
    return le.inverse_transform(prediction)[0]

# Test
test_sentence = "feeling program finally work priceless"
predicted_emotion = predict_emotion(test_sentence)
print('Input your text\n')
print(f"The sentence '{test_sentence}' is predicted to express: {predicted_emotion}.")
