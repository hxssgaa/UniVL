import pickle as pkl
import pandas as pd
import json
import numpy as np


def read_text(data_path):
    res = []
    with open(data_path) as f:
        for line in f.readlines():
            res.append(line.strip())
    return res


def main():
    gold_summary_data = pkl.load(open('data/dstc10/dstc10_data.summary.pickle', 'rb'))
    pred_summary = read_text('ckpts/ckpt_youcook_caption/test_hyp_univl_caption.txt')
    test = json.load(open('data/dstc10/test_set4DSTC7-AVSD.json'))
    test_ids = list(pd.read_csv('data/dstc10/dstc10_test.csv')['video_id'])
    features_pkl = pkl.load(open('data/dstc10/dstc10_videos_features_all.pickle', 'rb'))
    for idx_d, d in enumerate(test['dialogs']):
        image_id = d['image_id']
        gold_summary_data[image_id] = dict()
        his = []
        transcript = []
        text = []
        summary, caption = pred_summary[idx_d].split(' | ')
        for c in d['dialog']:
            c_his = ' '.join(his)
            q = 'User: ' + c['question']
            a = 'Robot: ' + c['answer']
            transcript.append((c_his + ' ' + q + ' | ' + caption + ' | '+ summary).strip())
            text.append(a.replace('Robot: ', ''))
            his.append(q)
            his.append(a)
        gold_summary_data[image_id]['text'] = np.array(text).reshape(-1)
        gold_summary_data[image_id]['transcript'] = np.array(transcript).reshape(-1)
        n = gold_summary_data[image_id]['text'].shape[0]
        gold_summary_data[image_id]['start'] = np.array([0] * n)
        gold_summary_data[image_id]['end'] = np.array([features_pkl[image_id].shape[0]] * n)
    pkl.dump(gold_summary_data, open('data/dstc10/dstc10_data.summary.pred.pickle', 'wb'))
    print('done')


if __name__ == '__main__':
    main()