import json
import torch
import numpy as np
import pickle as pkl
import os


def main():
    train_ids = os.listdir('data/atis/Audio_features_ATIS/train/train_feature')
    train_ids = list(map(lambda x: x.replace('.pkl', ''), train_ids))
    test_ids = os.listdir('data/atis/Audio_features_ATIS/test/test_feature')
    test_ids = list(map(lambda x: x.replace('.pkl', ''), test_ids))
    with open('data/atis/atis_train.csv', 'w') as f:
        for idx, img in enumerate(['X'] + train_ids):
            if idx == 0:
                f.writelines('video_id,feature_file\n')
            else:
                f.writelines('%s,%s\n' % (img, img))
    with open('data/atis/atis_test.csv', 'w') as f:
        for idx, img in enumerate(['X'] + test_ids):
            if idx == 0:
                f.writelines('video_id,feature_file\n')
            else:
                f.writelines('%s,%s\n' % (img, img))
    print('done')


def convert_to_numpy():
    path = '/home/hxssgaa/Developer/UniVL/data/atis/Audio_features_ATIS/features_all'
    output_path = '/home/hxssgaa/Developer/UniVL/data/atis/Audio_features_ATIS/features'
    for n in os.listdir(path):
        d = np.array(torch.load(open(os.path.join(path, n), 'rb')))
        np.save(output_path + n.replace('.pkl', '.npy'), d)
    print('done')


def gen_data():
    features = pkl.load(open('data/atis/atis_asr_features.pickle', 'rb'))
    train_src = json.load(open('data/atis/train_src.json'))
    train_tgt = json.load(open('data/atis/train_tgt.json'))
    train_asr = json.load(open('data/atis/train_asr_src.json'))
    test_src = json.load(open('data/atis/test_src.json'))
    test_tgt = json.load(open('data/atis/test_tgt.json'))
    test_asr = json.load(open('data/atis/test_asr_src.json'))
    train_asr = list(map(lambda x: x[x.rindex('/')+1:].replace('.pkl', ''), train_asr))
    test_asr = list(map(lambda x: x[x.rindex('/')+1:].replace('.pkl', ''), test_asr))
    all_src = train_src + test_src
    all_tgt = train_tgt + test_tgt
    all_asr = train_asr + test_asr
    res_pkl = dict()
    for idx, e in enumerate(all_asr):
        res_pkl[e] = dict()
        res_pkl[e]['text'] = np.array([all_tgt[idx]]).reshape(-1)
        res_pkl[e]['transcript'] = np.array([all_src[idx]]).reshape(-1)
        n = res_pkl[e]['text'].shape[0]
        res_pkl[e]['start'] = np.array([0] * n)
        res_pkl[e]['end'] = np.array([features[e].shape[0]] * n)
    pkl.dump(res_pkl, open('data/atis/atis_data.pickle', 'wb'))
    print('done')


if __name__ == '__main__':
    features = pkl.load(open('data/atis/atis_asr_features.pickle', 'rb'))
    gen_data()