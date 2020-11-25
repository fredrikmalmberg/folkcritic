import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import matplotlib.pyplot as plt
import importlib


def create_model(input_size):
    # Setting up a very basic network
    inputs = keras.Input(shape=(input_size), name="states")
    x = layers.Dense(1, activation="relu", name="dense_2")(inputs)
    outputs = layers.Dense(1, activation="sigmoid", name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    model.compile(
        loss=loss,
        optimizer=keras.optimizers.Adam(learning_rate = 0.0001),
        metrics=["accuracy"],
    )
    return model


if __name__ == '__main__':
    # Specifies how much of the concatenated state we want to use
    input_size = int(1536)
    model = create_model(input_size)
    batch_size = 100
    x_real = pickle.load(open('real_c_new', 'rb'))
    x_real = x_real[:, :input_size]
    x_generated = pickle.load(open('generated_c', 'rb'))
    x_generated = x_generated[:, :input_size]
    y_real = np.ones(x_real.shape[0])
    y_generated = np.zeros(x_generated.shape[0])
    x_train = np.append(x_real, x_generated, axis=0)
    y_train = np.append(y_real, y_generated, axis=0)
    history = model.fit(x_train, y_train, batch_size=batch_size, validation_split=0.2, epochs=20)

    fg = importlib.import_module('folkcritic-generator')
    generator = fg.Generator()
    generator.load_pretrained_generator('metadata/folkrnn_v2.pkl')
    data_path = 'data/data_v2'

    # Running some tests
    out = model.predict(x_generated)
    _ = plt.hist(out, bins=100)
    plt.show()
    print('mean and var, max, min', np.mean(out), np.var(out), np.max(out), np.min(out))

    incorrect_generated = []
    for n, val in enumerate(out):
        if val>0.5:
           incorrect_generated.append(n)
    print('\nGenerated songs correctly classified: ', 1-len(incorrect_generated)/len(out))
    print("Prediction stats \nMean: {:01f}, Var: {:01f}, Max: {:01f}, Min: {:01f}".format(np.mean(out), np.var(out),
                                                                                           np.max(out), np.min(out)))

    generated_tunes = pickle.load(open('generated_tunes', 'rb'))
    print('\nExamples of generated songs incorrectly classified: ')
    for n in incorrect_generated:
        print('Prediction: ', out[n])
        for line in generated_tunes[n]:
            print(line)

    out = model.predict(x_real)
    _ = plt.hist(out, bins=100)
    plt.show()

    incorrect_real = []
    for n, val in enumerate(out):
        if val<0.5:
           incorrect_real.append(n)
    print('\nReal songs correctly classified: ', 1-len(incorrect_real)/len(out))
    print("Prediction stats \nMean: {:01f}, Var: {:01f}, Max: {:01f}, Min: {:01f}".format(np.mean(out), np.var(out),
                                                                                           np.max(out), np.min(out)))

    real_idx = pickle.load(open('real_idx', 'rb'))
    idxs = real_idx
    print('\nExamples of real songs incorrectly classified: ')
    for n in incorrect_real:
        tune = generator.get_tune_from_idx(data_path, idxs[n])
        print('Prediction: ', out[n])
        print(tune)