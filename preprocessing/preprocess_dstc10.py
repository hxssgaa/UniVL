import pickle as pkl
import numpy as np
import json


def main(include_summary=False, is_last=False):
    caption_pkl = pkl.load(open('data/dstc10/dstc10_data.caption.pickle', 'rb'))
    train = json.load(open('data/dstc10/train_set4DSTC7-AVSD.json'))
    dev = json.load(open('data/dstc10/valid_set4DSTC7-AVSD.json'))
    for d in train['dialogs'] + dev['dialogs']:
        image_id = d['image_id']
        his = []
        transcript = []
        text = []
        for c in d['dialog']:
            c_his = ' '.join(his)
            q = 'User: ' + c['question']
            a = 'Robot: ' + c['answer']
            if include_summary:
                transcript.append((c_his + ' ' + q + ' | ' + d['caption'] + ' | '+ d['summary']).strip())
            else:
                transcript.append((c_his + ' ' + q).strip())
            text.append(a.replace('Robot: ', ''))
            his.append(q)
            his.append(a)
        if is_last:
            text = text[-1:]
            transcript = transcript[-1:]
        caption_pkl[image_id]['text'] = np.array(text).reshape(-1)
        caption_pkl[image_id]['transcript'] = np.array(transcript).reshape(-1)
        n = caption_pkl[image_id]['text'].shape[0]
        caption_pkl[image_id]['start'] = np.array(list(caption_pkl[image_id]['start']) * n)
        caption_pkl[image_id]['end'] = np.array(list(caption_pkl[image_id]['end']) * n)
    if include_summary:
        pkl.dump(caption_pkl, open('data/dstc10/dstc10_data.summary.pickle', 'wb'))
    elif is_last:
        pkl.dump(caption_pkl, open('data/dstc10/dstc10_data.last.pickle', 'wb'))
    else:
        pkl.dump(caption_pkl, open('data/dstc10/dstc10_data.pickle', 'wb'))
    print('done')


def main_test_csv():
    test = json.load(open('data/dstc10/test_set4DSTC7-AVSD.json'))
    test_imgids = []
    for d in test['dialogs']:
        image_id = d['image_id']
        test_imgids.append(image_id)
    with open('data/dstc10/dstc10_test.csv', 'w') as f:
        for idx, img in enumerate(['X'] + test_imgids):
            if idx == 0:
                f.writelines('video_id,feature_file\n')
            else:
                f.writelines('%s,%s\n' % (img, img))
    print('done')


def main_merge_test_data():
    data_pkl = pkl.load(open('data/dstc10/dstc10_data.pickle', 'rb'))
    test_src = json.load(open('data/dstc10/test_src.json'))
    test_tgt = json.load(open('data/dstc10/test_tgt.json'))
    test = json.load(open('eval/dstc7avsd_eval/sample/baseline_i3d_rgb-i3d_flow-vggish.json'))
    features_pkl = pkl.load(open('data/dstc10/dstc10_videos_features_all.pickle', 'rb'))
    res_pkl = dict()
    for d in test['dialogs']:
        image_id = d['image_id']
        his = []
        transcript = []
        text = []
        res_pkl[image_id] = dict()
        for c in d['dialog']:
            c_his = ' '.join(his)
            q = 'User: ' + c['question']
            a = 'Robot: ' + c['answer']
            transcript.append((c_his + ' ' + q).strip())
            text.append(a.replace('Robot: ', ''))
            his.append(q)
            his.append(a)
        res_pkl[image_id]['text'] = np.array(text).reshape(-1)
        res_pkl[image_id]['transcript'] = np.array(transcript).reshape(-1)
        n = res_pkl[image_id]['text'].shape[0]
        res_pkl[image_id]['start'] = np.array([0] * n)
        res_pkl[image_id]['end'] = np.array([features_pkl[image_id].shape[0]] * n)
    pkl.dump(res_pkl, open('data/dstc10/dstc10_data.test.pickle', 'wb'))
    print('done')


def merge_test_data_ground_truth():
    test_data = pkl.load(open('data/dstc10/dstc10_data.test.pickle', 'rb'))
    gold = json.load(open('eval/dstc7avsd_eval/data/test_set4DSTC7-AVSD_partial.json'))
    gold_annotation_map = {gold['images'][idx]['name'][:-2]: e['caption'] for idx, e in enumerate(gold['annotations'])}
    for k, v in gold_annotation_map.items():
        test_data[k]['text'] = np.array(list(test_data[k]['text'][:-1]) + [v]).reshape(-1)
    pkl.dump(test_data, open('data/dstc10/dstc10_data.test.gold.pickle', 'wb'))
    print('done')


def merge_to_all():
    train_data = pkl.load(open('data/dstc10/dstc10_data.pickle', 'rb'))
    test_data = pkl.load(open('data/dstc10/dstc10_data.test.gold.pickle', 'rb'))
    for k, v in test_data.items():
        train_data[k] = v
    pkl.dump(train_data, open('data/dstc10/dstc10_data.all.pickle', 'wb'))
    print('done')


if __name__ == '__main__':
    merge_to_all()