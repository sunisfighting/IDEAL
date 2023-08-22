import argparse
import yaml
from yaml import SafeLoader

def parameter_parser():
    """
    A method to parse up command line parameters.
    """


    parser = argparse.ArgumentParser(description = "Run .")

    parser.add_argument("--path",
                        type= str,
                        default = './datasets',
                        help = "the directory for loading datasets.")

    parser.add_argument("--dataset",
                        type = str,
                        default = "ppi",
                        help = "the name of dataset for use")

    parser.add_argument("--mode",
                        type= int,
                        default = 4)

    parser.add_argument("--loss",
                        type= str,
                        default = 'lsc',
                        help="ips; css; lsc.")

    parser.add_argument("--drop_edge_rate",
                        type= float,
                        default = 0.4,
                        help="the probability for removing edges in view 1.")

    parser.add_argument("--drop_feature_rate",
                        type= float,
                        default= 0.1,
                        help="the probability for masking features in view 2.")

    parser.add_argument("--dim_h",
                        type= int,
                        default= 512,
                        help="the dimension of hidden layer of GCN.")

    parser.add_argument("--dim_p",
                        type= int,
                        default= 128,
                        help="the dimension of projection layer for contrast.")

    parser.add_argument("--base_layer",
                        type=str,
                        default='GCNConv',
                        help="the basic encoder layer.")

    parser.add_argument("--activation",
                        type=str,
                        default= 'prelu',
                        help="the activation function of encoder layer: ['relu','prelu'].")

    parser.add_argument("--num_layer",
                        type= int,
                        default= 2,
                        help="the number of hidden layer of GCN.")

    parser.add_argument("--tau",
                        type= float,
                        default= 1,
                        help="the temperature for contrast.")

    parser.add_argument("--learning_rate",
                        type = float,
                        default = 0.001,
                        help="the learning rate for optimizer.")

    parser.add_argument("--weight_decay",
                        type= float,
                        default= 0,
                        help="the weight decay rate for optimizer.")

    parser.add_argument("--train_epoch",
                        type= int,
                        default= 1,
                        help="the number of training epochs.")

    parser.add_argument("--le_epoch",
                        type= int,
                        default= 2000,
                        help="the number of linear evaluation training epochs.")

    parser.add_argument("--batch_size",
                        type=int,
                        default= 0,
                        help="the batch size for batch semi-loss.")

    parser.add_argument("--num_parts",
                        type=int,
                        default= 7000,
                        help="the num of partitions for metis.")

    parser.add_argument("--part_bsize",
                        type=int,
                        default=20,
                        help="the num of partitions in each cluster.")

    parser.add_argument("--infer_bsize",
                        type=int,
                        default=256,
                        help="the batch_size when inferring on large graph.")

    parser.add_argument("--metrics",
                        type=bool,
                        default=False,
                        help="whether compute metrics.")

    return parser.parse_args()