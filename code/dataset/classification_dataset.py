from torch.utils.data import Dataset
import torch
import numpy as np

class ClassificationDataset(Dataset):
    def __init__(self, file_path, feature_num, seq_len):
        """
        :param file_path: fine-tuning file path
        :param feature_num: number of input features
        :param seq_len: padded sequence length
        """

        self.seq_len = seq_len
        self.dimension = feature_num

        with open(file_path, 'r') as ifile:
            self.Data = ifile.readlines()
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

        output = {"bert_input": ts_origin,
                  "bert_mask": bert_mask,
                  "time": doy,
                  }

        return {key: torch.from_numpy(value) for key, value in output.items()}



