import pickle as pkl
import json

import numpy as np


def main():
    caption_pkl = pkl.load(open('data/dstc10/dstc10_data.caption.pickle', 'rb'))
    test = json.load(open('data/dstc10/test_set4DSTC7-AVSD.json'))
    features_pkl = pkl.load(open('data/dstc10/dstc10_videos_features_all.pickle', 'rb'))
    for d in test['dialogs']:
        image_id = d['image_id']
        caption_pkl[image_id] = dict()
        summary = d['summary']
        caption = d['caption']
        text = '%s | %s' % (summary, caption)
        caption_pkl[image_id]['text'] = np.array([text]).reshape(-1)
        caption_pkl[image_id]['transcript'] = np.array(['none']).reshape(-1)
        caption_pkl[image_id]['start'] = np.array([0]).reshape(-1)
        caption_pkl[image_id]['end'] = np.array([features_pkl[image_id].shape[0]]).reshape(-1)
    pkl.dump(caption_pkl, open('data/dstc10/dstc10_data_all.caption.pickle', 'wb'))
    print('done')


if __name__ == '__main__':
    main()