import logging
import argparse
import numpy as np
from search import NetworkMix
from utils import count_parameters_in_MB

def get_params(args):
    cifar_classes = len(np.array(args.class_labels))
    total_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
                     'frog', 'horse', 'ship', 'truck']
    classes = []
    for i in np.array(args.class_labels):
        classes.append(total_classes[i])
    # intialize model
    channels = args.channels
    layers = args.layers
    ops = args.ops
    kernels = args.kernels
    logging.info("Model Channels (width) %s \nModel Layers (depth) %s", channels, layers)
    logging.info("Model Ops %s \nModel Kernels %s", ops, kernels)
    model = NetworkMix(channels, cifar_classes, layers, ops, kernels)
    count = count_parameters_in_MB(model)
    logging.info("Parameter Count (MB) =  {}\n".format(count))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("params")
    parser.add_argument('--channels', type=int, default=88, help=' number of channels(width)')
    parser.add_argument('--layers', type=int, default=14, help=' number of layers(depth)')
    parser.add_argument('--class_labels', type=int, nargs='*', default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                        help='num classes', required=False)
    parser.add_argument('--ops', type=int, nargs='*', default=[0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], help='operations')
    parser.add_argument('--kernels', type=int, nargs='*', default=[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], help='kernels')
    # create args object
    args = parser.parse_args()
    # set logging
    log_format = '%(message)s'
    logging.basicConfig(filename='params.txt', level=logging.INFO, format=log_format, filemode='a')
    get_params(args)
