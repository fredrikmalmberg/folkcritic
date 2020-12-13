import pickle
import numpy as np
import time
from tqdm import tqdm

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x, T):
    expx = np.exp(x / T)
    sumexpx = np.sum(expx)
    if sumexpx == 0:
        maxpos = x.argmax()
        x = np.zeros(x.shape, dtype=x.dtype)
        x[0][maxpos] = 1
    else:
        x = expx / sumexpx
    return x


class Generator:
    def __init__(self):
        self.LSTM_Wxi = []
        self.LSTM_Wxf = []
        self.LSTM_Wxc = []
        self.LSTM_Wxo = []
        self.LSTM_Whi = []
        self.LSTM_Whf = []
        self.LSTM_Whc = []
        self.LSTM_Who = []
        self.LSTM_bi = []
        self.LSTM_bf = []
        self.LSTM_bc = []
        self.LSTM_bo = []
        self.LSTM_cell_init = []
        self.LSTM_hid_init = []
        self.htm1 = []
        self.ctm1 = []
        self.idx2token = []
        self.start_idx = None
        self.end_idx = None
        self.rng = np.random.RandomState(42)
        self.vocab_idxs = None
        self.token2idx = None
        self.FC_output_W = None
        self.FC_output_b = None
        self.hidden_state = []
        self.cell_state = []
        self.numlayers = None
        self.sizeofx = None
        self.experiment_id = None

    def load_pretrained_generator(self, metadata_path):
        """
        Sets up the network based on pretrained weights
        :param metadata_path: Path to a trained model
        """
        f = open(metadata_path, 'rb')
        metadata = pickle.load(f, encoding="latin1")
        self.experiment_id = metadata['experiment_id']
        self.token2idx = metadata['token2idx']
        self.idx2token = dict((v, k) for k, v in self.token2idx.items())
        self.vocab_size = len(self.token2idx)
        self.start_idx, self.end_idx = self.token2idx['<s>'], self.token2idx['</s>']
        self.vocab_idxs = np.arange(self.vocab_size)

        self.numlayers = 3  # hard coded for now, but this should be saved in the model pickle
        for jj in range(self.numlayers):
            self.LSTM_Wxi.append(metadata['param_values'][2 + jj * 14 - 1])
            self.LSTM_Whi.append(metadata['param_values'][3 + jj * 14 - 1])
            self.LSTM_bi.append(metadata['param_values'][4 + jj * 14 - 1])
            self.LSTM_Wxf.append(metadata['param_values'][5 + jj * 14 - 1])
            self.LSTM_Whf.append(metadata['param_values'][6 + jj * 14 - 1])
            self.LSTM_bf.append(metadata['param_values'][7 + jj * 14 - 1])
            self.LSTM_Wxc.append(metadata['param_values'][8 + jj * 14 - 1])
            self.LSTM_Whc.append(metadata['param_values'][9 + jj * 14 - 1])
            self.LSTM_bc.append(metadata['param_values'][10 + jj * 14 - 1])
            self.LSTM_Wxo.append(metadata['param_values'][11 + jj * 14 - 1])
            self.LSTM_Who.append(metadata['param_values'][12 + jj * 14 - 1])
            self.LSTM_bo.append(metadata['param_values'][13 + jj * 14 - 1])
            self.LSTM_cell_init.append(metadata['param_values'][14 + jj * 14 - 1])
            self.LSTM_hid_init.append(metadata['param_values'][15 + jj * 14 - 1])
            self.htm1.append(self.LSTM_hid_init[jj])
            self.ctm1.append(self.LSTM_cell_init[jj])

        self.FC_output_W = metadata['param_values'][43]
        self.FC_output_b = metadata['param_values'][44]
        self.sizeofx = self.LSTM_Wxi[0].shape[0]

    def generate_tunes(self, ntunes, rng_seed = 42, temperature=1.0):
        """
        Generates a specified number of tunes and returns network states and tunes
        :param ntunes: Number of tunes to generate
        :param rng_seed: Seed for the random generation
        :param temperature: The temp used when generating
        :return: Tunes, hidden states and cell states of the network for each tune
        """
        self.rng = np.random.RandomState(rng_seed)
        all_hidden_states, all_cell_states = None, None
        all_tunes = []
        n = 0
        for i in tqdm(range(ntunes)):
            output = []
            for jj in range(self.numlayers):
                self.htm1[jj] = self.LSTM_hid_init[jj]
                self.ctm1[jj] = self.LSTM_cell_init[jj]
            sequence = [self.start_idx]
            while sequence[-1] != self.end_idx:
                x = np.zeros(self.sizeofx, dtype=np.int8)
                x[sequence[-1]] = 1
                for jj in range(self.numlayers):
                    it = sigmoid(np.dot(x, self.LSTM_Wxi[jj]) + np.dot(self.htm1[jj], self.LSTM_Whi[jj]) + self.LSTM_bi[jj])
                    ft = sigmoid(np.dot(x, self.LSTM_Wxf[jj]) + np.dot(self.htm1[jj], self.LSTM_Whf[jj]) + self.LSTM_bf[jj])
                    ct = np.multiply(ft, self.ctm1[jj]) + np.multiply(it, np.tanh(
                        np.dot(x, self.LSTM_Wxc[jj]) + np.dot(self.htm1[jj], self.LSTM_Whc[jj]) + self.LSTM_bc[jj]))
                    ot = sigmoid(np.dot(x, self.LSTM_Wxo[jj]) + np.dot(self.htm1[jj], self.LSTM_Who[jj]) + self.LSTM_bo[jj])
                    ht = np.multiply(ot, np.tanh(ct))
                    x = ht
                    self.ctm1[jj] = ct
                    self.htm1[jj] = ht
                output.append(softmax(np.dot(x, self.FC_output_W) + self.FC_output_b, temperature))
                next_itoken = self.rng.choice(self.vocab_idxs, p=output[-1].squeeze())
                if next_itoken != self.end_idx: # saving state before end of song token
                    self.hidden_state = self.htm1
                    self.cell_state = self.ctm1
                sequence.append(next_itoken)
                if len(sequence) > 1000: break
            abc_tune = [self.idx2token[j] for j in sequence[1:-1]]
            if len(abc_tune) > 2:
                tune = ['X:' + repr(i), abc_tune[0], abc_tune[1], ' '.join(abc_tune[2:])]
                all_tunes.append(tune)
                h_state_to_store = np.expand_dims(np.concatenate(self.hidden_state).ravel(), axis = 0)
                c_state_to_store = np.expand_dims(np.concatenate(self.cell_state).ravel(), axis = 0)
                if i == 0:
                    all_hidden_states = h_state_to_store
                    all_cell_states = c_state_to_store
                else:
                    all_hidden_states = np.append(np.copy(all_hidden_states), h_state_to_store, axis=0)
                    all_cell_states = np.append(np.copy(all_cell_states), c_state_to_store, axis=0)
            else:
                print('Failed generation - skipping song')

        return all_tunes, all_hidden_states, all_cell_states

    def set_state_from_seed(self, seed, modify=None):
        """
        Sets the network states based on a single seed
        :param seed: Input seed to be sent through the network
        :param modify: If specified we replace the ending of the song with random tokens
        """
        # Converting the seed passed as an argument into a list of idx
        seed_sequence = [self.start_idx]

        for token in seed.split(' '):
            seed_sequence.append(self.token2idx[token])

        # For destroying the beginning or end of the data
        if modify is not None:
            if modify > 0:
                r = np.min([len(seed_sequence)-2, modify])
                for i in range(r):
                    seed_sequence[-i] = self.rng.choice(self.vocab_idxs)
            else:
                r = np.min([len(seed_sequence) - 2, -modify])
                for i in range(r):
                    seed_sequence[i] = self.rng.choice(self.vocab_idxs)

        # initialise network by resetting to init values
        for jj in range(self.numlayers):
            self.htm1[jj] = self.LSTM_hid_init[jj]
            self.ctm1[jj] = self.LSTM_cell_init[jj]
        for tok in seed_sequence[:-1]:
            x = np.zeros(self.sizeofx, dtype=np.int8)
            x[tok] = 1
            # Resetting seed before starting
            for jj in range(self.numlayers):
                it = sigmoid(np.dot(x, self.LSTM_Wxi[jj]) + np.dot(self.htm1[jj], self.LSTM_Whi[jj]) + self.LSTM_bi[jj])
                ft = sigmoid(np.dot(x, self.LSTM_Wxf[jj]) + np.dot(self.htm1[jj], self.LSTM_Whf[jj]) + self.LSTM_bf[jj])
                ct = np.multiply(ft, self.ctm1[jj]) + np.multiply(it, np.tanh(
                    np.dot(x, self.LSTM_Wxc[jj]) + np.dot(self.htm1[jj], self.LSTM_Whc[jj]) + self.LSTM_bc[jj]))
                ot = sigmoid(np.dot(x, self.LSTM_Wxo[jj]) + np.dot(self.htm1[jj], self.LSTM_Who[jj]) + self.LSTM_bo[jj])
                ht = np.multiply(ot, np.tanh(ct))
                x = ht
                self.ctm1[jj] = ct
                self.htm1[jj] = ht
        abc_tune = [self.idx2token[j] for j in seed_sequence[1:-1]]
        new_tune = [abc_tune[0], abc_tune[1], ''.join(abc_tune[2:])]
        return new_tune

    def get_states_from_data(self, data_path, start_at = 0, no_of_songs=None, modify=None):
        """
        Function to get all the cell states and hidden states from real tune data
        :param data_path: path to where the data is stored
        :param start_at: index to start at when reading tunes
        :param no_of_songs: If we want to specify a specific number of songs and not use all the data
        :param modify: If specified we replace the ending of the song with random tokens
        :return: all hidden states and all cell states for the network and the given songs
        """
        with open(data_path, 'r') as f:
            data = f.read()
        tunes = data.split('\n\n')
        del data
        all_hidden_states = None
        all_cell_states = None
        all_tunes = []
        print('Total number of tunes found ', len(tunes))
        if no_of_songs is None:
            no_of_songs = len(tunes)
            start_at = 0
        for idx in tqdm(range(start_at, start_at + no_of_songs)):
            tune = tunes[idx]
            try:
                seed = tune.replace('\n', ' ')
                seed_sequence = self.set_state_from_seed(seed, modify=modify)
                h_state_to_store = np.expand_dims(np.concatenate(self.htm1).ravel(), axis=0)
                c_state_to_store = np.expand_dims(np.concatenate(self.ctm1).ravel(), axis=0)
                if idx == start_at:
                    all_hidden_states = h_state_to_store
                    all_cell_states = c_state_to_store

                else:
                    all_hidden_states = np.append(np.copy(all_hidden_states), h_state_to_store, axis=0)
                    all_cell_states = np.append(np.copy(all_cell_states), c_state_to_store, axis=0)
                all_tunes.append(seed_sequence)
            except:
                print('Could not parse song ' + str(idx) + ' from data')
                pass
        return all_hidden_states, all_cell_states, all_tunes

if __name__ == '__main__':
    generator = Generator()
    # Loading weights
    generator.load_pretrained_generator('metadata/folkrnn_v2.pkl')

    # Different ways of generating states
    states_from_seed = False
    states_from_seed_mod = False
    states_from_generated = False

    number_of_songs = 23000

    if states_from_seed:
        cwd = 'state_data/'
        data_path = 'data/data_v2'
        batch_size = 1000
        for b in range(number_of_songs // batch_size):
            print('---Running batch {} of {} ---'.format(b, number_of_songs // batch_size))
            print('Seeding with {} songs'.format(batch_size))
            real_h_states, real_c_states, real_tunes = \
                    generator.get_states_from_data(data_path, start_at = b*batch_size, no_of_songs=batch_size)
            #pickle.dump(real_tunes, open(cwd + "real_tunes_" + str(b), "wb"))
            #pickle.dump(real_h_states, open(cwd + "real_h_" + str(b), "wb"))
            #pickle.dump(real_c_states, open(cwd + "real_c_" + str(b), "wb"))

    if states_from_seed_mod:
        data_path = 'data/data_v2'
        #real_idx = generator.get_parsable_idx(data_path)
        #pickle.dump(real_idx, open("real_idx", "wb"))
        real_hidden_states, real_cell_states, real_tunes = [], [], []

        mod_range = 100
        for l in range(mod_range):
            real_hidden_states_tmp, real_cell_states_tmp, real_tunes_tmp = \
                generator.get_states_from_data(data_path, no_of_songs=number_of_songs, modify=-l)
            real_hidden_states.append(real_hidden_states_tmp)
            real_cell_states.append(real_cell_states_tmp)
            real_tunes.append(real_tunes_tmp)
        pickle.dump(real_tunes, open("state_data/real_tunes_", "wb"))
        pickle.dump(real_hidden_states, open("state_data/real_h_mod_b", "wb"))
        pickle.dump(real_cell_states, open("state_data/real_c_mod_b", "wb"))

    if states_from_generated:
        if states_from_seed:
            number_of_songs = len(real_hidden_states)
        else:
            number_of_songs = 23000
        batch_size = 1000
        cwd = 'state_data/'
        for b in range(number_of_songs//batch_size):
            print('---Running batch {} of {} ---'.format(b, number_of_songs//batch_size))
            print('Generating {} songs'.format(batch_size))
            generated_tunes, generated_hidden_states, generated_cell_states = \
                generator.generate_tunes(batch_size, temperature=1.0)
            pickle.dump(generated_tunes, open(cwd + "generated_tunes_" + str(b), "wb"))
            pickle.dump(generated_hidden_states, open(cwd + "generated_h_" + str(b), "wb"))
            pickle.dump(generated_cell_states, open(cwd + "generated_c_" + str(b), "wb"))
            h_states = None
            c_states = None
            print('Seeding with {} songs'.format(batch_size))
            for idx in tqdm(range(len(generated_tunes))):
                tune = generated_tunes[idx]
                abc_tune = ' '.join([tune[1], tune[2], tune[3]])
                _ = generator.set_state_from_seed(abc_tune)
                h_state_to_store = np.expand_dims(np.concatenate(generator.htm1).ravel(), axis=0)
                c_state_to_store = np.expand_dims(np.concatenate(generator.ctm1).ravel(), axis=0)
                if idx == 0:
                    h_states = h_state_to_store
                    c_states = c_state_to_store
                else:
                    h_states = np.append(np.copy(h_states), h_state_to_store, axis=0)
                    c_states = np.append(np.copy(c_states), c_state_to_store, axis=0)
            pickle.dump(h_states, open(cwd + "generated_h_reset_" + str(b), "wb"))
            pickle.dump(c_states, open(cwd + "generated_c_reset_" + str(b), "wb"))





