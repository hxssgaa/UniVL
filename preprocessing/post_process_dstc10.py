import pandas as pd
import json
import pickle as pkl

from pandas.io import pickle


def read_text(data_path):
    res = []
    with open(data_path) as f:
        for line in f.readlines():
            res.append(line.strip())
    return res


def remove_duplicates(line):
    line_spt = line.split()
    prev = None
    cnt = 0
    t_idx = len(line_spt)
    for idx, each in enumerate(line_spt):
        if each == prev:
            cnt += 1
        else:
            cnt = 0
        if cnt == 3:
            t_idx = idx
            break
        prev = each
    return ' '.join(line_spt[:t_idx])

def main():
    hyp = read_text('ckpts/ckpt_youcook_caption/test_hyp_univl_nonsummary_ansgen.txt')#json.load(open('eval/dstc7avsd_eval/sample/test_bart_dialoghisonly.json'))
    test_data = pkl.load(open('data/dstc10/dstc10_data.test.pickle', 'rb'))
    template = json.load(open('eval/dstc7avsd_eval/data/template.json'))
    cnt = 0
    mapping_idxes = []
    for each_test_data in list(test_data.values()):
        for c in list(each_test_data['text']):
            if c ==  '__UNDISCLOSED__':
                mapping_idxes.append(cnt)
            cnt += 1
    hyp = [hyp[idx].replace('\' ', '\'').replace(' ,', ',').replace('robot: ', '').strip() for idx in mapping_idxes]
    for idx in range(len(template['dialogs'])):
        template['dialogs'][idx]['dialog'][0]['answer'] = remove_duplicates(hyp[idx])
    json.dump(template, open('eval/dstc7avsd_eval/sample/univl_without_pred_summary_ansgen_pred.json', 'w'))
    print('done')


if __name__ == "__main__":
    main()
