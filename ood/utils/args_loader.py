import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples in neural networks')

    parser.add_argument('--in-dataset', default="kvasir", type=str, help='kvasir dataset')
    parser.add_argument('--num_classes', default=3, type=int, help='number of classes' )

    parser.add_argument('--model-arch', default="resnet18", type=str, help='model architecture available: [resnet18, deit]')
    parser.add_argument('--weights', help='model weights to load')
    
    parser.add_argument('--seed', default=42, type=int, help='seed')
    parser.add_argument('-b', '--batch-size', default=32, type=int, help='mini-batch size')

    parser.add_argument('--base-dir', type=str, help='result directory')
    parser.add_argument('--id_path_train', help='path to id train dataset')
    parser.add_argument('--id_path_valid', help='path to id valid dataset')
    parser.add_argument('--ood_path', help='path to ood dataset')
    
    parser.set_defaults(argument=True)
    args = parser.parse_args()
    return args