import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, Embedding,LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler # type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential # type: ignore
import pickle



# Paths for saving
MODEL_PATH = "chatbot_model.h5"
LABEL_ENCODER_PATH = "label_encoder.npy"
INTENTS_FILE = "intents.json"
PERFORMANCE_LOG_PATH = "model_performance.json"


# load intent.json
def load_intents(intents_file):
    with open(intents_file) as file:
        data = json.load(file)
    return data['intents']


def preprocess_data(intents):
    training_sentences = []
    training_labels = []
    classes = []

    
    for intent in intents:
        for pattern in intent['patterns']:
            training_sentences.append(pattern)
            training_labels.append(intent['tag'])
    
    classes = sorted(set(training_labels))

    tokenizer = Tokenizer()  #tokenizer
    tokenizer.fit_on_texts(training_sentences)

    words = list(tokenizer.word_index.keys())

    # Saving tokenizer for reuse
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

    le = LabelEncoder()
    training_labels = le.fit_transform(training_labels)
    np.save(LABEL_ENCODER_PATH, le.classes_) #saving label encoder for reuse

    sequences = tokenizer.texts_to_sequences(training_sentences)

    max_sequence_length = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

    return padded_sequences, training_labels, tokenizer, classes, words, max_sequence_length


#training and saving the model
def train_and_save_model(padded_sequences, training_labels, max_sequence_length, classes, words):
    X_train, X_val, y_train, y_val = train_test_split(
        np.array(padded_sequences), 
        np.array(training_labels), 
        test_size=0.2, 
        random_state=42
    )
    model = Sequential([
    Embedding(input_dim=len(words) + 1, output_dim=64, input_length=max_sequence_length),
    Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.3)),
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(len(classes), activation='softmax')
])

    model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

#Learning rate scheduler 
    scheduler = LearningRateScheduler(lambda epoch: 1e-4 * 10 ** (epoch / 20))

#Early stopping 
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


    history = model.fit(
    X_train, y_train, 
    epochs=100, 
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, scheduler], 
    verbose=1
)

    
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

   
    performance = {
        "training_loss": history.history['loss'],
        "training_accuracy": history.history['accuracy'],
        "validation_loss": history.history['val_loss'],
        "validation_accuracy": history.history['val_accuracy']
    }
    with open(PERFORMANCE_LOG_PATH, 'w') as perf_file:
        json.dump(performance, perf_file, indent=4)
    print(f"Model performance saved to {PERFORMANCE_LOG_PATH}")


if __name__ == "__main__":
    intents = load_intents(INTENTS_FILE)
    padded_sequences, training_labels, tokenizer, classes, words, max_sequence_length = preprocess_data(intents)
    
    train_and_save_model(padded_sequences, training_labels, max_sequence_length, classes, words)
    print("Training completed!")
