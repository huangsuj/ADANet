
import argparse

def parameter_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--device", type=str, default="1", help="Device: cuda:num or cpu")
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument("--path", type=str, default="./datasets/", help="Path of datasets")
    parser.add_argument("--dataset", type=str, default="BBCSports", help="Name of datasets")
    parser.add_argument("--seed", type=int, default=12, help="Random seed for train-test split. Default is 12.")
    parser.add_argument("--shuffle_seed", type=int, default=42, help="Random seed for train-test split. Default is 42.")
    parser.add_argument("--fix_seed", action='store_true', default=True, help="xx")

    parser.add_argument("--early-stop", type=bool, default=False, help="If early stop")
    parser.add_argument("--patience", type=int, default=20, help="Patience for early stop")
    parser.add_argument("--verbose", type=int, default=1, help="Show training details.")

    parser.add_argument("--n_repeated", type=int, default=5, help="Number of repeated times. Default is 5.")

    parser.add_argument("--knns", type=int, default=20, help="k of KNN graph. default=20")
    parser.add_argument("--common_neighbors", type=int, default=2, help="Number of common neighbors (when using pruning strategy 2)")
    parser.add_argument("--pr1", action='store_true', default=True, help="Using prunning strategy 1 or not")
    parser.add_argument("--pr2", action='store_true', default=True, help="Using prunning strategy 2 or not")


    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-6, help="Weight decay") #me
    parser.add_argument("--num_epoch", type=int, default=200, help="Number of training epochs. Default is 200.")

    parser.add_argument("--data-split-mode", type=str, default="Ratio", help="Data split mode: Number or Ratio")
    parser.add_argument("--train-ratio", type=int, default=0.1, help="Train data ratio. Default is 0.1.")
    parser.add_argument("--valid-ratio", type=int, default=0.1, help="Valid data ratio. Default is 0.1.")
    parser.add_argument("--test-ratio", type=int, default=0.8, help="Test data ratio. Default is 0.8.")

    parser.add_argument("--alpha", nargs='+', type=float, default = 0.001, help="hyperparameter")
    parser.add_argument("--beta", nargs='+', type=float, default = 0.001, help="hyperparameter")
    parser.add_argument("--gamma", nargs='+', type=float, default = 0.001, help="hyperparameter")
    parser.add_argument("--mu", nargs='+', type=float, default = 0.001, help="hyperparameter")
    parser.add_argument("--hdim", nargs='+', type=int, default=[512], help="Number of hidden dimensions")
    parser.add_argument("--layers", type=int, default=2, help="number of layer.")
    parser.add_argument("--dropout", type=float, default=0.7, help="Dropout rate.")


    args = parser.parse_args()

    return args