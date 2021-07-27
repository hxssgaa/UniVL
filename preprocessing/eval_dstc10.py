import argparse
import json
import neval


def get_args(description='Eval DSTC 10'):
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--pred_json', type=str, default='data/youcookii_singlef_train.csv', help='')
    parser.add_argument('--gold_json', type=str, default='data/youcookii_singlef_val.csv', help='')

    return parser.parse_args()


def eval(args):
    pred = json.load(open(args.pred_json))
    gold = json.load(open(args.gold_json))
    print(neval.compute_metrics(pred, [gold]))


if __name__ == '__main__':
    eval(get_args())
