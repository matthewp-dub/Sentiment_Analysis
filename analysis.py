import pandas as pd
import numpy as np
import pickle
import keras
from keras import Input, Model
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense
from keras.optimizers import RMSprop
import nltk
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def get_model():
    """
    this function will either load or create a model to predict the sentiment of tweets with -1 for
    negative, 0 for neutral, and 1 for positive
    """
    pass


def train_model(max_words=140):
    """
    trains a model to identify tweets from scratch
    """
    # imports the training data
    X_vector, y_vector = load_training_data()

    # Does the actual machine learning. based off of:
    # https://learnremote.medium.com/sentiment-analysis-using-1d-convolutional-neural-networks-part-1-f8b6316489a2

    text_input_layer = Input(shape=(max_words,))
    embedding_layer = Embedding(max_words, 50)(text_input_layer)
    text_layer = Conv1D(256, 3, activation='relu')(embedding_layer)
    text_layer = MaxPooling1D(3)(text_layer)
    text_layer = Conv1D(256, 3, activation='relu')(text_layer)
    text_layer = MaxPooling1D(3)(text_layer)
    text_layer = Conv1D(256, 3, activation='relu')(text_layer)
    text_layer = MaxPooling1D(3)(text_layer)
    text_layer = Conv1D(256, 3, activation='relu')(text_layer)
    text_layer = MaxPooling1D(3)(text_layer)
    text_layer = Conv1D(256, 3, activation='relu')(text_layer)
    text_layer = MaxPooling1D(3)(text_layer)
    text_layer = GlobalMaxPooling1D()(text_layer)
    text_layer = Dense(256, activation='relu')(text_layer)
    output_layer = Dense(1, activation='sigmoid')(text_layer)
    model = Model(text_input_layer, output_layer)
    model.summary()
    model.compile(optimizer=RMSprop(lr=0.001),
                  loss='binary_crossentropy',
                  metrics=['acc'])

    # callbacks
    callback_list = [
        keras.callbacks.EarlyStopping(
            patience=1,
            monitor='acc',
        ),

        keras.callbacks.TensorBoard(
            log_dir='log_dir_m1',
            histogram_freq=1,
            embeddings_freq=1,
        ),

        # keras.callbacks.ModelCheckpoint(
        #     monitor='val_loss',
        #     save_best_only=True,
        #     filepath='model/movie_sentiment_m1.h5',
        # ),

        keras.callbacks.ReduceLROnPlateau(
            patience=1,
            factor=0.1,
        )
    ]

    # fits the model to training data

    history = model.fit(X_vector, y_vector, epochs=50, batch_size=128, callbacks=callback_list)


def load_training_data(training_data_path='Twitter_training_data.csv', max_words=140):
    """
    loads the training data from training_data_path and returns a tuple (X_vector, y_vector)
    :param training_data_path:
    """

    if os.path.exists('cache/trainingdata.pickle'):
        with open('cache/trainingdata.pickle', 'rb') as pickle_file:
            return pickle.load(pickle_file)

    # imports training data as csv dataframe, initializes x and y vectors, removes stopwords in X vector, and assigns
    # sentiment in y vector
    csv_dataframe = pd.read_csv(training_data_path)
    raw_text = csv_dataframe['text']
    raw_sentiment = csv_dataframe['target']

    # nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    assert len(raw_text) == len(raw_sentiment)
    vector_length = len(raw_text)
    X_matrix = np.zeros((vector_length, max_words), dtype=object)
    y_vector = np.zeros(vector_length)

    for text_index, text_val in enumerate(raw_text):
        token_text = word_tokenize(text_val)
        filtered_text = [word for word in token_text if word not in stop_words]
        capped_text = filtered_text[:max_words]
        for token_index, token in enumerate(capped_text):
            X_matrix[text_index, token_index] = token

    for sentiment_index, sentiment_val in enumerate(raw_sentiment):
        assert sentiment_val == 0 or sentiment_val == 2 or sentiment_val == 4
        y_vector[sentiment_index] = (sentiment_val - 2) / 2

    # Saves pickle
    with open('cache/trainingdata.pickle', 'wb') as pickle_file:
        pickle.dump((X_matrix, y_vector), pickle_file)

    return (X_matrix, y_vector)
