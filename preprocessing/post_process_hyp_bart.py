import json
from nlgeval import compute_metrics


def main():
    hyp = json.load(open('ckpts/ckpt_charades_caption/hyp_bartbase.json'))
    hyp = [hyp[idx] for idx in range(len(hyp)) if idx % 10 == 0]
    hyp = list(map(lambda x: x[6:].strip().replace(', ', ' , ').replace('.', ' .'), hyp))
    with open('ckpts/ckpt_charades_caption/hyp_bartbase.txt', 'w') as f:
        for line in hyp:
            f.writelines(line + '\n')
    print('done')


def eval():
    for idx in range(6):
        print(idx)
        try:
            metrics_dict = compute_metrics(hypothesis='ckpts/ckpt_charades_caption_best/hyp%d.txt' % idx,
                                            references=['ckpts/ckpt_charades_caption_best/ref%d.txt' % idx])
        except:
            continue


if __name__ == "__main__":
    eval()