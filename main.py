from __future__ import print_function

import argparse
from torch.utils.data import DataLoader
from ESPCN.solver import SubPixelTrainer
from dataset.data import get_training_set, get_test_set

# ===========================================================
# Training settings
# ===========================================================
# FSRCNN batchSize:8  Epoch:50 Inital learning ration:0.01 upscale factor:4  ReLU   PSNR:  loss:
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
# hyper-parameters
parser.add_argument('--batchSize', type=int, default=4, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=4, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.01')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--upscale_factor', '-uf',  type=int, default=4, help="super resolution upscale factor")
args = parser.parse_args()

def main():
    print('===> Loading datasets')
    train_set = get_training_set(args.upscale_factor)
    test_set = get_test_set(args.upscale_factor)
    training_data_loader = DataLoader(dataset=train_set, batch_size=args.batchSize, shuffle=True)
    testing_data_loader = DataLoader(dataset=test_set, batch_size=args.testBatchSize, shuffle=False)
    model = SubPixelTrainer(args, training_data_loader, testing_data_loader)
    model.run()

if __name__ == '__main__':
    main()
