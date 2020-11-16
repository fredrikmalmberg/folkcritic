import pickle
import numpy as np
import time

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
        self.rng = None
        self.vocab_idxs = None
        self.token2idx = None
        self.FC_output_W = None
        self.FC_output_b = None
        self.hidden_state = []
        self.numlayers = None
        self.sizeofx = None
        self.experiment_id = None

    def load_pretrained_generator(self, metadata_path):
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

    def set_state_from_seed(self, seed):
        # Converting the seed passed as an argument into a list of idx
        seed_sequence = [self.start_idx]

        for token in seed.split(' '):
            seed_sequence.append(self.token2idx[token])

        # initialise network
        for tok in seed_sequence[:-1]:
            x = np.zeros(self.sizeofx, dtype=np.int8)
            x[tok] = 1
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

        for jj in range(self.numlayers):
            self.LSTM_hid_init[jj] = self.htm1[jj]
            self.LSTM_cell_init[jj] = self.ctm1[jj]

        # Keeping separate 'hidden states' as they are different from htm1 when generating
        self.hidden_state = self.htm1

    def generate_tunes(self, ntunes, rng_seed = 42, temperature=1.0, display=False, save_to_disk=False):
        self.rng = np.random.RandomState(rng_seed)
        target_path = "samples/%s-s%d-%.2f-%s.txt" % (
            self.experiment_id, rng_seed, temperature, time.strftime("%Y%m%d-%H%M%S", time.localtime()))
        all_hidden_states = []
        all_tunes = []
        for i in range(ntunes):
            # initialise network
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
                if next_itoken != self.end_idx: # not saving state when at end of song token
                    self.hidden_state = self.htm1
                sequence.append(next_itoken)
                if len(sequence) > 1000: break
            abc_tune = [self.idx2token[j] for j in sequence[1:-1]]
            if save_to_disk:
                print('X:' + repr(i))
                f = open(target_path, 'a+')
                f.write('X:' + repr(i) + '\n')
                f.write(abc_tune[0] + '\n')
                f.write(abc_tune[1] + '\n')
                f.write(' '.join(abc_tune[2:]) + '\n\n')
                f.close()
            if display:
                print('X:' + repr(i))
                print(abc_tune[0])
                print(abc_tune[1])
                print(''.join(abc_tune[2:]) + '\n')
            tune = ['X:' + repr(i), abc_tune[0], abc_tune[1], ''.join(abc_tune[2:])]
            all_tunes.append(tune)
            all_hidden_states.append(self.hidden_state)

        return all_tunes, all_hidden_states

    def get_hidden_states_from_data(self, data_path, no_of_songs = None):
        with open(data_path, 'r') as f:
            data = f.read()
        tunes = data.split('\n\n')
        del data
        all_hidden_states = []
        for idx, tune in enumerate(tunes):
            seed = tunes[0].replace('\n', ' ')
            self.set_state_from_seed(seed)
            all_hidden_states.append(self.hidden_state)
            print("song ", idx)
            if no_of_songs is not None and idx+2 > no_of_songs:
                return all_hidden_states
        return all_hidden_states

if __name__ == '__main__':
    generator = Generator()
    generator.load_pretrained_generator('metadata/folkrnn_v2.pkl')
    tunes, states = generator.generate_tunes(2)
    data_path = 'data/data_v2'
    all_hidden_states = generator.get_hidden_states_from_data(data_path, 3)


