import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Dense, Dropout, LSTM, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.backend import ctc_batch_cost
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Convert labels to integer sequences
def text_to_seq(text):
    return [ctoi[char] for char in text]

# Load images and labels
def load_data(df, path, target_shape=None):
    images = []
    labels = []

    for index, row in df.iterrows():
        try:
            img = img_to_array(load_img(f"{path}/{row['FILENAME']}"))
            
            # If target_shape is provided, resize the image
            if target_shape:
                img = cv2.resize(img, target_shape[:2])
            
            images.append(img)
            label_seq = text_to_seq(row['IDENTITY'])
            labels.append(label_seq)
        except Exception as e:
            print(f"Error loading {row['FILENAME']}: {e}")

    labels_padded = pad_sequences(labels, maxlen=max_label_len, padding='post')
    return np.array(images), np.array(labels_padded)

# Load the data
train_df = pd.read_csv('written_name_train_v2.csv')
train_df = train_df[train_df['IDENTITY'].apply(lambda x: isinstance(x, str))]  # Filter out non-string entries
characters = set(char for label in train_df['IDENTITY'] for char in label)
ctoi = {char: i for i, char in enumerate(characters)}
itoc = {i: char for char, i in ctoi.items()}
max_label_len = max(train_df['IDENTITY'].apply(len))

train_images, train_labels = load_data(train_df, 'train_v2/train', target_shape=(50, 284, 3))
validation_df = pd.read_csv('written_name_validation_v2.csv')
validation_images, validation_labels = load_data(validation_df, 'validation_v2/validation')

# Build the model
input_shape = train_images[0].shape
inputs = Input(shape=input_shape, name='input_image')

# CNN layers
x = Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
x = MaxPooling2D((2,2))(x)
x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = MaxPooling2D((2,2))(x)
x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
x = MaxPooling2D((2,2))(x)

# RNN layers
rnn_input_shape = (input_shape[0] // 8, (input_shape[1] // 8) * 128)
x = Reshape(target_shape=rnn_input_shape)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.25)(x)
x = LSTM(128, return_sequences=True)(x)
x = Dropout(0.25)(x)

# Final layer with linear activation
x = Dense(len(characters) + 1, activation='linear', name='logits')(x)

# Calculate CTC loss
labels = Input(name='labels', shape=[max_label_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')

def compute_ctc_loss(args):
    y_pred, labels, input_length, label_length = args
    return ctc_batch_cost(labels, y_pred, input_length, label_length)

ctc_loss = Lambda(compute_ctc_loss, output_shape=(1,), name='ctc')([x, labels, input_length, label_length])

model = Model(inputs=[inputs, labels, input_length, label_length], outputs=ctc_loss)
model.compile(optimizer='adam', loss={'ctc': lambda y_true, y_pred: y_pred})

# Prepare CTC inputs
train_input_lengths = np.ones((train_images.shape[0], 1)) * rnn_input_shape[0]
train_label_lengths = np.array([len(label) for label in train_labels])
validation_input_lengths = np.ones((validation_images.shape[0], 1)) * rnn_input_shape[0]
validation_label_lengths = np.array([len(label) for label in validation_labels])

# Train the model
model.fit([train_images, train_labels, train_input_lengths, train_label_lengths], np.zeros(len(train_images)),
          validation_data=([validation_images, validation_labels, validation_input_lengths, validation_label_lengths], np.zeros(len(validation_images))),
          epochs=10, batch_size=32)

# Save the model
model.save('htr_model_ctc.h5')
