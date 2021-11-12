import json
import numpy as np
import pickle as pkl
from collections import defaultdict


def preprocess_simm2_for_split(split, used_keys=None):
    src = json.load(open('data/simmc/processed/%s_predict.json' % split))
    tgt = json.load(open('data/simmc/processed/%s_target.json' % split))
    src_dict = defaultdict(list)
    tgt_dict = defaultdict(list)
    
    format_target = lambda x: x['blief_state'] + ' ' + x['response']
    format_src = lambda x: x['context']
    for idx, e in enumerate(src):
        image_id = e['scene_image'][e['scene_image'].rindex('/')+1:].replace('.png', '')
        if used_keys and image_id in used_keys:
            image_id = image_id + '_0'
        src_dict[image_id].append(e)
        tgt_dict[image_id].append(tgt[idx])
    res_pkl = dict()
    for k in src_dict:
        res_pkl[k] = dict()
        res_pkl[k]['text'] = np.array(list(map(format_target, tgt_dict[k]))).reshape(-1)
        res_pkl[k]['transcript'] = np.array(list(map(format_src, src_dict[k]))).reshape(-1)
        n = res_pkl[k]['text'].shape[0]
        res_pkl[k]['start'] = np.array([0] * n)
        res_pkl[k]['end'] = np.array([1] * n)
    return res_pkl


def preprocess_simmc2():
    splits = ['train', 'dev', 'devtest']
    all_res = []
    used_keys = set()
    for split in splits:
        all_res.append(preprocess_simm2_for_split(split, used_keys=used_keys))
        used_keys |= set(all_res[-1])
    train_keys = list(sorted(all_res[0]))
    dev_keys = list(sorted(all_res[1]))
    devtest_keys = list(sorted(all_res[2]))
    with open('data/simmc/train.csv', 'w') as f:
        for idx, img in enumerate(['X'] + train_keys):
            if idx == 0:
                f.writelines('video_id,feature_file\n')
            else:
                f.writelines('%s,%s\n' % (img, img))
    with open('data/simmc/dev.csv', 'w') as f:
        for idx, img in enumerate(['X'] + dev_keys):
            if idx == 0:
                f.writelines('video_id,feature_file\n')
            else:
                f.writelines('%s,%s\n' % (img, img))
    with open('data/simmc/devtest.csv', 'w') as f:
        for idx, img in enumerate(['X'] + devtest_keys):
            if idx == 0:
                f.writelines('video_id,feature_file\n')
            else:
                f.writelines('%s,%s\n' % (img, img))
    all_res_dict = dict()
    for e in all_res:
        all_res_dict.update(e)
    pkl.dump(all_res_dict, open('data/simmc/train_data.no_obj.pickle', 'wb'))
    print('done')
    return all_res_dict


if __name__ == '__main__':
    preprocess_simmc2()