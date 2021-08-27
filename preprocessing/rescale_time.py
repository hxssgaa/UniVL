import json
import pickle as pkl

def main():
    times = []
    features = pkl.load(open('data/dstc10/dstc10_videos_features_all.pickle', 'rb'))
    with open('ckpts/ckpt_charades_caption_best/hyp_time.txt') as f:
        for l in f.readlines():
            times.append(list(map(float, l.strip().split(':'))))
    dialogs = json.load(open('data/dstc10/valid_set4DSTC10-AVSD_with_action.json'))
    image_ids = [d['image_id'] for d in dialogs['dialogs'] for c in d['dialog']]
    total_len = [features[e].shape[0] for e in image_ids]
    scaled_times = [[min(times[idx][0], total_len[idx]-5), min(times[idx][1], total_len[idx])] for idx in range(len(times))]
    with open('ckpts/ckpt_charades_caption_best/hyp_time_scaled.txt', 'w') as f:
        for l in scaled_times:
            f.writelines('%.1f:%.1f\n' % tuple(l))
    print('done')


if __name__ == '__main__':
    main()
