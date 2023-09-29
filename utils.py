import os
import pickle

import numpy as np
import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def xvfl_to_feature(arr):
    # xvfl means "position (x) and velocity (v) of the follower (f) and the leader (l)"
    # that is, the columns stand for: 1. position of the follower; 2. velocity of the follower
    #                                 3. position of the leader;   4. velocity of the leader


    # dx
    dx = arr[:, 2] - arr[:, 0]
    dx = dx.reshape(-1, 1)
    dx = dx[:-1, :]

    # dv
    dv = arr[:, 3] - arr[:, 1]
    dv = dv.reshape(-1, 1)
    dv = dv[:-1, :]

    # vf and a
    vf = arr[:, 1]
    a = np.diff(vf) * 10
    vf = vf.reshape(-1, 1)
    vf = vf[:-1, :]
    a = a.reshape(-1, 1)
    return np.hstack([dx, dv, vf, a])


class TrajectoryDataset(Dataset):
    def __init__(self, data, obs_len=1, pred_len=1, train_data_size=0.7, train_val_test='train'):
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = self.obs_len + self.pred_len - 1
        seq_list = []
        for feature_a in data:
            num_sequences = feature_a.shape[0] - self.seq_len + 1
            for idx in range(num_sequences):
                curr_seq_data = feature_a[idx:idx+self.seq_len]  # (seq_len, 4)
                seq_list.append(curr_seq_data)

        seq_list = np.array(seq_list)  # (N, seq_len, 4)
        no_norm_features = seq_list[:, :, :3]
        no_norm_features = no_norm_features.astype(np.float32)
        # normalization
        f_mean = np.mean(no_norm_features, axis=0)
        f_std = np.std(no_norm_features, axis=0)
        features = (no_norm_features - f_mean) / f_std
        # tmp = features.squeeze()
        labels = seq_list[:, :, [3]]
        labels = labels.astype(np.float32)

        total_num = features.shape[0]

        permutation = np.random.permutation(total_num)
        features, no_norm_features, labels = features[permutation], no_norm_features[permutation], labels[permutation]

        train_idx_list = list(np.arange(0, int(total_num*train_data_size)))
        val_idx_list = list(np.arange(int(total_num*0.7), int(total_num*0.8)))
        test_idx_list = list(np.arange(int(total_num*0.8), total_num))
        if train_val_test.lower() == 'train':
            self.features = features[train_idx_list]
            self.no_norm_features = no_norm_features[train_idx_list]
            self.labels = labels[train_idx_list]
        elif train_val_test.lower() == 'val':
            self.features = features[val_idx_list]
            self.no_norm_features = no_norm_features[val_idx_list]
            self.labels = labels[val_idx_list]
        elif train_val_test.lower() == 'test':
            self.features = features[test_idx_list]
            self.no_norm_features = no_norm_features[test_idx_list]
            self.labels = labels[test_idx_list]
        else:
            raise Exception("Wrong train_val_test input!")
        self.obs = self.features[:, :self.obs_len, :]
        self.no_norm_obs = self.no_norm_features[:, :self.obs_len, :]
        self.a_pred = self.labels[:, self.obs_len-1:, :]
        # print(1)
    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.obs[idx], self.no_norm_obs[idx], self.a_pred[idx]


if __name__ == "__main__":
    file_name = 'real_data_lane3_f2l2.pickle'
    with open(os.path.join('data', file_name), 'rb') as f:
        if file_name.find('real') != -1:
            xvfl = pickle.load(f)
            USE_GA = True
        else:
            data_pickle = pickle.load(f)
            xvfl = data_pickle['idm_data']  # x, v of leading and following

    # state to feature
    feature_a = list(map(xvfl_to_feature, xvfl[:-1 * 10]))
    feature_a_in_one = np.vstack(feature_a)  # (N, 4)

    train_feeder = TrajectoryDataset(feature_a_in_one, train_val_test='train')
    for i in range(len(train_feeder)):
        f, l = train_feeder[i]

    train_loader = torch.utils.data.DataLoader(dataset=train_feeder, batch_size=128, shuffle=False)
    for data in train_loader:
        f, l = data

    print(1)