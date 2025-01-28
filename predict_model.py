import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from sklearn.preprocessing import LabelEncoder


MODEL_PATH = "saved_files/chatbot_model.h5"
LABEL_ENCODER_PATH = "saved_files/label_encoder.npy"
TOKENIZER_PATH = "saved_files/tokenizer.pkl"

#trained model
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

# tokenizer
def load_tokenizer():
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer

#label encoder
def load_label_encoder():
    le = np.load(LABEL_ENCODER_PATH, allow_pickle=True)
    return le

# Preprocessing
def preprocess_input(sentence, tokenizer, max_sequence_length):
   
    sequence = tokenizer.texts_to_sequences([sentence])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length, padding='post')
    return padded_sequence

# Prediction
def predict_class(sentence):

    model = load_model()
    tokenizer = load_tokenizer()
    le = load_label_encoder()


    max_sequence_length = model.input_shape[1] 
    preprocessed_input = preprocess_input(sentence, tokenizer, max_sequence_length)

    prediction = model.predict(preprocessed_input)

    predicted_class_index = np.argmax(prediction)  #index of the predicted class
    predicted_class_label = le[predicted_class_index] # class
    
    return predicted_class_label


