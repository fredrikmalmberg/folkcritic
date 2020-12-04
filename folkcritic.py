import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import matplotlib.pyplot as plt
import importlib

class Critic:
    def __init__(self):
        self.model = self.create_model(1536)

    def create_model(self, input_size):
        # Setting up a very basic network
        inputs = keras.Input(shape=(input_size), name="states")
        #x = layers.Dense(64, activation="relu", name="dense_2")(inputs)
        x = inputs
        outputs = layers.Dense(1, activation="sigmoid", name="predictions")(x)
        model = keras.Model(inputs=inputs, outputs=outputs)

        model.summary()

        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        model.compile(
            loss=loss,
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            metrics=["accuracy"],
        )
        return model

    def preprocess_data(self, data_path):
        cwd = data_path
        b = 0
        generated_h = pickle.load(open(cwd + 'generated_h_reset_' + str(b), 'rb'))
        generated_tunes = pickle.load(open(cwd + 'generated_tunes_' + str(b), 'rb'))
        while True:
            try:
                b += 1
                generated_h = np.append(np.copy(generated_h),
                                                    pickle.load(open(cwd + 'generated_h_reset_' + str(b), 'rb')),
                                                    axis=0)
                generated_tunes = np.append(np.copy(generated_tunes), pickle.load(open(cwd + 'generated_tunes_' + str(b), 'rb')),
                                       axis=0)
            except:
                break
        x_generated = generated_h

        b = 0
        real_h = pickle.load(open(cwd + 'real_h_' + str(b), 'rb'))
        real_tunes = pickle.load(open(cwd + 'real_tunes_' + str(b), 'rb'))
        while True:
            try:
                b += 1
                real_h = np.append(np.copy(real_h), pickle.load(open(cwd + 'real_h_' + str(b), 'rb')),
                                         axis=0)
                real_tunes = np.append(np.copy(real_tunes), pickle.load(open(cwd + 'real_tunes_' + str(b), 'rb')),
                                       axis=0)
            except:
                break
        x_real = real_h

        return x_real, x_generated, real_tunes, generated_tunes

    def pretrain_model(self, data_path):
        x_real, x_generated, real_tunes, generated_tunes = self.preprocess_data(data_path)
        print('training with {} real and {} generated tunes'.format(x_real.shape[0], x_generated.shape[0]))
        y_real = np.ones(x_real.shape[0])
        y_generated = np.zeros(x_generated.shape[0])
        x_train = np.append(x_real, x_generated, axis=0)
        y_train = np.append(y_real, y_generated, axis=0)
        history = self.model.fit(x_train, y_train, batch_size=100, validation_split=0.2, epochs=20)

    def train_single(self, hidden_state):
        x_train = hidden_state
        y_train = np.zeros(x_train.shape[0])
        history = self.model.fit(x_train, y_train, batch_size=1, validation_split=0, epochs=1)

    def predict(self, input):
        return self.model.predict(input)

    def load_model(self, path):
        self.model = keras.models.load_model(path)

if __name__ == '__main__':

    pretrain_and_save = False
    load_and_test = True
    data_path = 'state_data/'
    model_path = 'saved_models/'

    if pretrain_and_save:
        critic = Critic()
        critic.pretrain_model(data_path)
        critic.model.save(model_path)

    if load_and_test:
        critic = Critic()
        critic.load_model('saved_models/')
        fg = importlib.import_module('folkcritic-generator')
        generator = fg.Generator()
        generator.load_pretrained_generator('metadata/folkrnn_v2.pkl')
        x_real, x_generated, real_tunes, generated_tunes = critic.preprocess_data(data_path)

        out = critic.model.predict(x_generated)
        _ = plt.hist(out, bins=50, label='Generated', alpha = 0.5)
        print('mean: {:.2f}, var: {:.2f}, max: {:.2f}, min: {:.2f}'.format(np.mean(out), np.var(out), np.max(out),
                                                                           np.min(out)))
        i = None
        worst = 0
        incorrect_generated = []
        for n, val in enumerate(out):
            if val > 0.5:
                incorrect_generated.append(n)
            if val > worst:
                worst = val
                i = n
        print('Generated songs correctly classified: {:.2f}%'.format(100 * (1 - len(incorrect_generated) / len(out))))


        out = critic.model.predict(x_real)
        _ = plt.hist(out, bins=50, label='Real', alpha = 0.5)
        plt.title('Prediction histogram (23000 tunes of each class)')
        plt.legend()
        plt.show()
        print('mean: {:.2f}, var: {:.2f}, max: {:.2f}, min: {:.2f}'.format(np.mean(out), np.var(out), np.max(out),
                                                                           np.min(out)))
        i = None
        worst = 2
        incorrect_real = []
        for n, val in enumerate(out):
            if val < 0.5:
                incorrect_real.append(n)
            if val < worst:
                worst = val
                i = n
        print('Real songs correctly classified: {:.2f}%'.format(100 * (1 - len(incorrect_real) / len(out))))

