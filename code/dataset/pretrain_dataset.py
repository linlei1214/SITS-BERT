from torch.utils.data import Dataset
import torch
import numpy as np
import random

class PretrainDataset(Dataset):
    def __init__(self, file_path, feature_num, seq_len):
        """
        :param file_path: pre-training file path
        :param feature_num: number of input features
        :param seq_len: padded sequence length
        """

        self.seq_len = seq_len
        self.dimension = feature_num

        with open(file_path, 'r') as ifile:
            self.Data = ifile.readlines()
            print("loading data successful")
            self.TS_num = len(self.Data)

    def __len__(self):
        return self.TS_num

    def __getitem__(self, item):
        line = self.Data[item]

        # line[-1] == '\n' should be discarded
        line_data = line[:-1].split(',')
        line_data = list(map(float, line_data))
        line_data = np.array(line_data, dtype=float)

        ts = np.reshape(line_data, (self.dimension + 1, -1)).T
        ts_length = ts.shape[0]

        bert_mask = np.zeros((self.seq_len,), dtype=int)
        bert_mask[:ts_length] = 1

        # BOA reflectances
        ts_origin = np.zeros((self.seq_len, self.dimension))
        ts_origin[:ts_length, :] = ts[:, :-1] / 10000.0

        # day of year
        doy = np.zeros((self.seq_len,), dtype=int)
        doy[:ts_length] = np.squeeze(ts[:, -1])

        # randomly add noise to some observations
        ts_masking, mask = self.random_masking(ts_origin, ts_length)

        output = {"bert_input": ts_masking,
                  "bert_target": ts_origin,
                  "bert_mask": bert_mask,
                  "loss_mask": mask,
                  "time": doy,
                  }

        return {key: torch.from_numpy(value) for key, value in output.items()}

    def random_masking(self, ts, ts_length):

        ts_masking = ts.copy()
        mask = np.zeros((self.seq_len,), dtype=int)

        for i in range(ts_length):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                mask[i] = 1

                if prob < 0.5:
                    ts_masking[i, :] += np.random.uniform(low=-0.5, high=0, size=(self.dimension,))

                else:
                    ts_masking[i, :] += np.random.uniform(low=0, high=0.5, size=(self.dimension,))

        return ts_masking, mask
