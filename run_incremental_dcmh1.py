import torch
import os
import argparse
import incremental2
from loguru import logger
from data.data_loader import load_data
import numpy as np



def run():

    bits = [16, 32, 64]
    oldb_path = ['2021-04-04 21_23_58.532220_MIRFlickr_21_16bit_old-B.t','2021-04-05 13_24_27.373714_MIRFlickr_21_32bit_old-B.t', '2021-04-05 13_25_37.959615_MIRFlickr_21_64bit_old-B.t']
    args = load_config()

    for i in range(len(bits)):
        args.code_length = bits[i]
        path = os.path.join('./checkpoints', oldb_path[i])
        print('============')
        print('bits:')
        print(args.code_length)
        print('path:')
        print(path)
        print('============')
        logger.add('logs/DICH/{time}.log', rotation='500 MB', level='INFO')
        logger.info(args)
        torch.backends.cudnn.benchmark = True

        # Load dataset
        query_dataloader, seen_dataloader, unseen_dataloader, retrieval_dataloader = load_data(
            args.dataset,
            args.root,
            args.num_seen,
            args.batch_size,
            args.num_workers,
        )

        # # Increment learning
        B = torch.load(path)

        # B = np.load(os.path.join('./checkpoints', '1625901420.14_MIRFlickr_21_16bit_old-B-x.npy'))
        # B = np.load(os.path.join('./checkpoints', '1625950811.89_NUSWIDE_18_32bit_old-B-x.npy'))

        mAP_i2t, mAP_t2i = incremental2.increment(
            query_dataloader,
            seen_dataloader,
            unseen_dataloader,
            retrieval_dataloader,
            B,
            args.code_length,
            args.device,
            args.lr_img,
            args.lr_txt,
            args.max_iter,
            args.max_epoch,
            args.num_samples,
            args.batch_size,
            args.root,
            args.dataset,
            args.incremental_lamda,
            args.incremental_mu,
            args.incremental_theta,
            args.incremental_gamma,
            args.incremental_eta,
            args.topk,
            args.num_seen,
            args.PRETRAIN_MODEL_PATH
        )
        logger.info('[incremental][mapi2t:{:.4f}][mapt2i:{:.4f}]'.format(mAP_i2t, mAP_t2i))


def load_config():
    """
    Load configuration.

    Args
        None

    Returns
        args(argparse.ArgumentParser): Configuration.
    """
    # mirflikcr
    parser = argparse.ArgumentParser(description='incremental_ADCH_PyTorch')
    parser.add_argument('--dataset', default='MIRFlickr',
                        help='Dataset name.')
    parser.add_argument('--root', default='./dataset/MIRFlickr/',
                        help='Path of dataset')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='Batch size.(default: 64)')
    parser.add_argument('--lr_txt', default=1e-2, type=float,
                        help='Learning rate.(default: 1e-2)')
    parser.add_argument('--lr_img', default=1e-4, type=float,
                        help='Learning rate.(default: 1e-4)')
    parser.add_argument('--code-length', default=16, type=int,
                        help='Binary hash code length.(default: 16)')
    parser.add_argument('--max-iter', default=5, type=int,
                        help='Number of iterations.(default: 30)')
    parser.add_argument('--max-epoch', default=3, type=int,
                        help='Number of epochs.(default: 3)')
    parser.add_argument('--num-seen', default=21, type=int,
                        help='Number of seen classes.(default: 21)')
    parser.add_argument('--num-samples', default=2000, type=int,
                        help='Number of sampling data points.(default: 2000)')
    parser.add_argument('--num-workers', default=8, type=int,
                        help='Number of loading data threads.(default: 8)')
    parser.add_argument('--topk', default=-1, type=int,
                        help='Calculate map of top k.(default: all)')
    parser.add_argument('--gpu', default=7, type=int,
                        help='Using gpu.(dfault: False)')
    parser.add_argument('--incremental_lamda', default=1, type=float,
                        help='Hyper-parameter.(default: 1)')
    parser.add_argument('--incremental_mu', default=1e-3, type=float,
                        help='Hyper-parameter.(default: 1e-3)')
    parser.add_argument('--incremental_theta', default=0.01, type=float,
                        help='Hyper-parameter.(default: 0.01)')
    parser.add_argument('--incremental_gamma', default=100, type=float,
                        help='Hyper-parameter.(default: 100)')
    parser.add_argument('--incremental_eta', default=10, type=float,
                        help='Hyper-parameter.(default: 10)')
    parser.add_argument('--PRETRAIN_MODEL_PATH',default='./models/imagenet-vgg-f.mat',
                            help = 'path of the pretrain_model')
    args = parser.parse_args()



    # GPU
    if args.gpu is None:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda:%d" % args.gpu)


    return args


if __name__ == '__main__':
    run()
