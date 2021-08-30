import argparse
import json
import neval


def read_text(data_path):
    res = []
    with open(data_path) as f:
        for line in f.readlines():
            res.append(line.strip())
    return res


def get_args(description='Eval DSTC 10'):
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--pred', type=str, default='', help='')
    parser.add_argument('--gold', type=str, default='', help='')

    return parser.parse_args()


def eval(args):
    pred = read_text(args.pred)
    gold = read_text(args.gold)
    print(neval.compute_metrics(pred, [gold]))


if __name__ == '__main__':
    eval(get_args())
