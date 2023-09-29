import argparse

parser = argparse.ArgumentParser()

#Model specific parameters
parser.add_argument('--input_size', type=int, default=3)
parser.add_argument('--output_size', type=int, default=1)
parser.add_argument('--model', type=str, default='cnn', help='mlp, lstm, att_lstm, cnn')


#Data specifc paremeters
parser.add_argument('--obs_seq_len', type=int, default=5)
parser.add_argument('--pred_seq_len', type=int, default=1)
parser.add_argument('--device',
                    help='what device to perform training on',
                    type=str,
                    default='cuda:0')
parser.add_argument("--data_dir",
                    help="what dir to look in for data",
                    type=str,
                    default='./data')
parser.add_argument("-data_dict",
                    help="what file to load for training data",
                    type=str,
                    default='real_data_lane3_f2l2.pickle')
parser.add_argument("-train_data_size",
                    help="training data size, should be less than 0.7",
                    type=float,
                    default=0.7)

# Training specifc parameters
parser.add_argument('--batch_size', type=int, default=1024,
                    help='minibatch size')
parser.add_argument('--num_epochs', type=int, default=3,
                    help='number of epochs')
parser.add_argument('--lr_nn', type=float, default=0.001,
                    help='learning rate for NN')
parser.add_argument('--lr_phy', type=float, default=0.1,
                    help='learning rate for physics mode')
parser.add_argument('--alpha', type=float, default=1.,
                    help='weighting for physics-informed loss')
parser.add_argument('--tag', default='tag',
                    help='personal tag for the model ')
parser.add_argument('--seed',
                    help='manual seed to use, default is 123',
                    type=int,
                    default=1234)


args = parser.parse_args()
