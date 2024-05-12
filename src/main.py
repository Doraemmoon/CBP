import torch
import numpy as np
import argparse
import logging
from trainer import training
from data_loader import dataLoader
import os
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# seed
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


if __name__ == '__main__':
    same_seeds(4321)
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='seq_split')
    parser.add_argument('--dataset', type=str, default='Retail')
    parser.add_argument('--h1', type=int, default=5)
    parser.add_argument('--h2', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--l2', type=float, default=1e-2)
    parser.add_argument('--decay', type=float, default=0.6)
    parser.add_argument('--delta', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--opt', type=str, default='Adam')
    parser.add_argument('--dim', type=int, default=32)
    parser.add_argument('--isTrain', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--evalEpoch', type=int, default=1)
    parser.add_argument('--m', type=float, default=1)
    parser.add_argument('--n', type=float, default=0.002)
    parser.add_argument('--testOrder', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--beta', type=float, default=0.9)

    config = parser.parse_args()

    if config.mode == 'time_split':
        resultFileName = 'time_split'
    else:
        resultFileName = 'seq_split'

    config.saveFile = config.model + '_' + config.mode + '_' + str(config.decay) + '_' + \
                      config.dataset + '_' + str(config.batch_size) + '_' + str(config.dim) + '_' + \
                      config.opt + '_' + str(config.lr) + '_' + str(config.l2) + '_' + \
                      str(config.alpha) + '_' + str(config.beta)

    config.fileRoot = resultFileName + '/' + config.dataset

    if config.isTrain == 1:
        config.rRoot = os.path.join('all_valid_results', config.fileRoot)
        writer = SummaryWriter(log_dir=config.rRoot + '/log_dir_500', comment='scalars')
    else:
        config.rRoot = os.path.join('all_results', config.fileRoot)


        writer = SummaryWriter(log_dir=config.rRoot + '/log_dir_500', comment='scalars')

    logName = os.path.join(config.rRoot, config.saveFile)

    if os.path.exists(logName):
        print(logName +'done!')
        quit()

    logging.basicConfig(filename=logName, level=logging.DEBUG)
    logging.info('This is info message')
    logger = logging.getLogger(__name__)

    logger.info(config)
    print(config)


    dataset = dataLoader(config)
    if config.isTrain:
        config.padIdx = dataset.numItemsTrain
    else:
        config.padIdx = dataset.numItemsTest

    print('start training')
    training(dataset, config, logger, device, writer)
