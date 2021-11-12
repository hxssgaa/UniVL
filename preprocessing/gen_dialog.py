import json


def read_text(data_path):
    res = []
    with open(data_path) as f:
        for line in f.readlines():
            res.append(line.strip())
    return res


def main(include_last=True):
    test_set = json.load(open('data/dstc10/test_set4DSTC10-AVSD.json'))
    hyp = read_text('ckpts/ckpt_charades_caption_test/hyp.txt')
    idx = 0
    for idx_d, d in enumerate(test_set['dialogs']):
        len_c = len(d['dialog'])
        for idx_c, c in enumerate(d['dialog']):
            if not include_last and idx_c == len_c - 1:
                idx += 1
                test_set['dialogs'][idx_d]['dialog'] = test_set['dialogs'][idx_d]['dialog'][:-1]
                continue
            test_set['dialogs'][idx_d]['dialog'][idx_c]['answer'] = hyp[idx]
            idx += 1
    json.dump(test_set, open('ckpts/ckpt_charades_caption_test/predict.json', 'w'))
    print('done')


if __name__ == '__main__':
    main(include_last=True)
