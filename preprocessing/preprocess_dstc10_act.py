import pickle as pkl
import pandas as pd
import numpy as np
import json
from numpy import nan


def read_text(file_path):
    lines = []
    with open(file_path) as f:
        for l in f.readlines():
            lines.append(l.strip())
    return lines


def preprocess_dstc10_act():
    cap_data = pkl.load(open('data/dstc10/dstc10_data_all.caption.pickle', 'rb'))
    train_act_data = pd.read_csv('data/dstc10/Charades_v1_train.csv')
    test_act_data = pd.read_csv('data/dstc10/Charades_v1_test.csv')
    all_acts = list(train_act_data['actions']) + list(test_act_data['actions'])
    all_ids = list(train_act_data['id']) + list(test_act_data['id'])
    all_acts = {all_ids[idx]: e for idx, e in enumerate(all_acts) if isinstance(e, str)}
    act_classes = read_text('data/dstc10/Charades_v1_classes.txt')
    act_classes = {e.split(maxsplit=1)[0]: e.split(maxsplit=1)[1] for e in act_classes}
    act_classes_reverse_map = {v: k for k, v in act_classes.items()}
    train_act_extra = json.load(open('data/dstc10/train_set4DSTC10-AVSD.json'))
    valid_act_extra = json.load(open('data/dstc10/valid_set4DSTC10-AVSD_with_action.json'))
    for d in train_act_extra['dialogs'] + valid_act_extra['dialogs']:
        image_id = d['image_id']
        action_str = ';'.join('%s %s %s' % (act_classes_reverse_map[act['action']], 
            act['timestamp'].split(':')[0], act['timestamp'].split(':')[1]) for act in d['actions'])
        all_acts[image_id] = action_str

    for k in cap_data:
        act_text_spt = list(filter(lambda x: x, all_acts.get(k, '').split(';')))
        text = [act_classes[e.split()[0]] for e in act_text_spt]
        start = [int(float(e.split()[1])) for e in act_text_spt]
        end = [int(float(e.split()[2])) for e in act_text_spt]
        text = np.array(text).reshape(-1)
        start = np.array(start).reshape(-1)
        end = np.array(end).reshape(-1)
        cap_data[k]['text'] = text
        cap_data[k]['start'] = start
        cap_data[k]['end'] = end

    pkl.dump(cap_data, open('data/dstc10/dstc10_data_all.act.pickle', 'wb'))
    print('done')


if __name__ == '__main__':
    preprocess_dstc10_act()