import os
import pickle
import random
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from GA.GA import GA
from argument_parser import args

from model.nn_model import *
from model.physics import IDM
from utils import *

if not torch.cuda.is_available() or args.device == 'cpu':
    args.device = torch.device('cpu')
else:
    args.device = 'cuda:0'

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

# Training parameter
ALPHA = args.alpha
EPOCH = args.num_epochs
BATCH_SIZE = args.batch_size
obs_seq_len = args.obs_seq_len
pred_seq_len = args.pred_seq_len
train_data_size = args.train_data_size

INPUT_DIM = 3
N_HIDDEN = 3
HIDDEN_DIM = 60
OUTPUT_DIM = 1

device = args.device

nn_args = (INPUT_DIM, OUTPUT_DIM, N_HIDDEN, HIDDEN_DIM)
nn_kwargs = {"activation_type": "sigmoid",
             "last_activation_type": "none",
             "device": device}

params_trainable = {
    "v0": False,
    "T": False,
    "s0": False,
    "a": False,
    "b": False
}

optimizer_kwargs = {
    "lr": 0.001
}

optimizer_physics_kwargs = {
    "lr": 0.1
}

args_GA = {
    'sol_per_pop': 10,
    'num_parents_mating': 5,
    'num_mutations': 1,  # set 1 to mutate all the parameters
    'mutations_extend': 0.1,
    'num_generations': 10,

    'delta_t': 0.1,
    'mse': 'position',
    'RMSPE_alpha_X': 0.5,
    'RMSPE_alpha_V': 0.5,
    'lb': [10, 0, 0, 0, 0],
    'ub': [40, 10, 10, 5, 5]
}

checkpoint_dir = './checkpoint/{}/{}/'.format(args.model, args.tag)
log_file = os.path.join(checkpoint_dir, 'log_test.txt')

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

with open(checkpoint_dir + 'args.pkl', 'wb') as fp:
    pickle.dump(args, fp)


def my_print(content):
    with open(log_file, 'a') as writer:
        print(content)
        writer.write(content + '\n')


def compute_Loss(pred, GT, error_order=2):
    loss_t = torch.sum(torch.abs(pred - GT) ** error_order, dim=-1)  # (batch_size, T, 1) -> (batch_size, T)
    loss_all = loss_t.sum(dim=-1)
    loss_avg = torch.mean(loss_all)
    return loss_avg


def main():
    my_print('-----------------------')
    my_print('| TRAINING PARAMETERS |')
    my_print('-----------------------')
    my_print('| model: %s' % args.model.upper())
    my_print('| num_epochs: %d' % args.num_epochs)
    my_print('| batch_size: %d' % args.batch_size)
    my_print('| device: %s' % args.device)
    my_print('| obs sequence length: %s' % args.obs_seq_len)
    my_print('| pred sequence length: %s' % args.pred_seq_len)
    my_print('| NN learning rate: %f' % args.lr_nn)
    my_print('| training data size: %f' % args.train_data_size)
    my_print('| physics model learning rate: %f' % args.lr_phy)
    my_print('| weighting for physics-informed loss: %f' % (1 - args.alpha))
    my_print('| tag: %s' % args.tag)
    my_print('-----------------------')

    data_path = os.path.join(args.data_dir, args.data_dict)
    with open(data_path, 'rb') as f:
        if data_path.find('real') != -1:
            xvfl = pickle.load(f)
            USE_GA = True
        else:
            data_pickle = pickle.load(f)
            xvfl = data_pickle['idm_data']  # x, v of leading and following

    if USE_GA is True:
        ga = GA(args_GA)
        # para, mse, duration = ga.executeGA(xvfl)
        para = np.array([30.51004085, 1.08624567, 7.24089548, 1.80457837, 3.37961181])
        para = {"v0": para[0],
                "T": para[1],
                "s0": para[2],
                "a": para[3],
                "b": para[4]}
    else:
        para = data_pickle['para']

    my_print("Parameters of IDM model are: "+str(para))

    # state to feature
    feature_a = list(map(xvfl_to_feature, xvfl))

    train_feeder = TrajectoryDataset(feature_a, obs_len=obs_seq_len, pred_len=pred_seq_len,
                                     train_data_size=train_data_size, train_val_test='train')
    train_loader = DataLoader(dataset=train_feeder, batch_size=BATCH_SIZE,
                              shuffle=True)
    val_feeder = TrajectoryDataset(feature_a, obs_len=obs_seq_len, pred_len=pred_seq_len,
                                   train_val_test='val')
    val_loader = DataLoader(dataset=val_feeder, batch_size=BATCH_SIZE,
                            shuffle=False)
    my_print("Length of train set: %d ..." % len(train_feeder))
    my_print("Length of train batch: %d ..." % len(train_loader))
    my_print("Length of val set: %d ..." % len(val_feeder))
    my_print("Length of val batch: %d ..." % len(val_loader))

    if args.model == 'mlp':
        net = MLP(input_size=args.input_size * args.obs_seq_len,
                  output_size=args.output_size * args.pred_seq_len).to(device)
    elif args.model == 'lstm':
        net = Vanilla_LSTM(input_size=args.input_size,
                           output_size=args.output_size).to(device)
    elif args.model == 'att_lstm':
        net = Att_LSTM(input_size=args.input_size,
                       output_size=args.output_size).to(device)
    elif args.model == 'cnn':
        net = TCN(input_size=args.input_size,
                  output_size=args.output_size).to(device)
    else:
        raise Exception("Wrong model selected!")

    physics = IDM(para, params_trainable, device=device)

    optimizer = torch.optim.Adam(
        [p for p in net.parameters() if p.requires_grad is True]
        , **optimizer_kwargs)

    if sum(list(params_trainable.values())) > 0:
        optimizer_physics = torch.optim.Adam(
            [p for p in physics.torch_params.values() if p.requires_grad is True]
            , **optimizer_physics_kwargs)
    else:
        optimizer_physics = None

    train_total_loss = []
    val_total_loss = []
    best_val_loss = np.float32('inf')
    best_epoch = 0

    for epoch in range(EPOCH):
        net.train()
        running_loss = 0
        for i, data in enumerate(train_loader):
            train_feature, no_norm_train_feature, train_label = data  # (batch_size, T, C) -> (1024, 5, 3)  (1024, 1, 1)
            X_train = train_feature[:int(BATCH_SIZE * 0.8), :].to(device)
            a_train = train_label[:int(BATCH_SIZE * 0.8), :].to(device)
            X_aux_nn = train_feature[:, :, :].to(device)
            X_aux_phy = no_norm_train_feature[:, -1, :].to(device)

            a_pred = net(X_train)  # (batch_size, T, 1)
            a_pred_aux_nn = net(X_aux_nn)
            a_pred_aux_phy = physics(X_aux_phy)  # (1024, )

            loss_obs = compute_Loss(a_pred, a_train)
            loss_aux = compute_Loss(torch.flatten(a_pred_aux_nn), a_pred_aux_phy)

            loss = ALPHA * loss_obs + (1 - ALPHA) * loss_aux
            running_loss += loss.item()

            optimizer.zero_grad()
            if sum(list(params_trainable.values())) > 0:
                optimizer_physics.zero_grad()

            loss.backward()

            optimizer.step()
            if sum(list(params_trainable.values())) > 0:
                optimizer_physics.step()

        train_total_loss.append(running_loss)

        net.eval()
        validaton_loss = 0
        for i, data in enumerate(val_loader):
            val_feature, _, val_label = data  # [1024, 3] [1024, 1]
            val_feature = val_feature.to(device)
            val_label = val_label.to(device)

            a_pred_val = net(val_feature)
            loss_val = compute_Loss(val_label, a_pred_val)
            validaton_loss += loss_val.item()
        val_total_loss.append(validaton_loss)

        if validaton_loss < best_val_loss:
            best_epoch = epoch
            best_val_loss = validaton_loss
            torch.save(net.state_dict(), checkpoint_dir + 'val_best.pth')

        lr = [param_group['lr'] for param_group in optimizer.param_groups][0]
        my_print('|{}|Epoch:{:>5}| loss_train: {:.2f}| loss_val:{:.2f}| lr: {}| best_epoch: {}'.format(datetime.now(),
                                                                                                       epoch,
                                                                                                       running_loss,
                                                                                                       validaton_loss,
                                                                                                       lr,
                                                                                                       best_epoch))

    my_print('============Test')
    test_feeder = TrajectoryDataset(feature_a, obs_len=obs_seq_len, pred_len=pred_seq_len,
                                    train_val_test='test')
    test_loader = DataLoader(dataset=test_feeder, batch_size=1,
                             shuffle=False)
    net.load_state_dict(torch.load(checkpoint_dir + 'val_best.pth'))
    test_total_loss_nn = []
    test_total_loss_phy = []
    progress_bar = tqdm(total=len(test_loader))
    for i, data in enumerate(test_loader):
        progress_bar.update(1)
        test_feature, no_norm_test_feature, test_label = data  # [1024, 3] [1024, 1]
        test_feature = test_feature.to(device)
        no_norm_test_feature = no_norm_test_feature.to(device)
        test_label = test_label.to(device)

        a_pred_nn = net(test_feature)
        loss_nn = compute_Loss(test_label, a_pred_nn).cpu().detach().numpy()

        a_pred_phy = physics(no_norm_test_feature[:, -1, :])  # (1024, )
        loss_phy = compute_Loss(torch.flatten(test_label), a_pred_phy)

        test_total_loss_nn.append(loss_nn.item())
        test_total_loss_phy.append(loss_phy.item())

    progress_bar.close()
    my_print('|{}|best epoch:{:>5}| loss_nn: {:.2f}| loss_phy:{:.2f}'.format(datetime.now(),
                                                                                best_epoch,
                                                                                np.mean(test_total_loss_nn),
                                                                                np.mean(test_total_loss_phy)
                                                                                ))


if __name__ == '__main__':
    main()
