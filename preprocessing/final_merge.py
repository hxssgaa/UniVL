import json
from collections import defaultdict

def read_text(data_path):
    res = []
    with open(data_path) as f:
        for line in f.readlines():
            res.append(line.strip())
    return res


def main():
    original_test = json.load(open('data/dstc10/test_set4DSTC10-AVSD.json'))
    predict = json.load(open('ckpts/ckpt_charades_caption_test/predict.json'))
    scaled_time = read_text('ckpts/ckpt_charades_caption_test/hyp_time_scaled.txt')
    hyp_exp = read_text('ckpts/ckpt_charades_caption_exps/hyp.txt')
    all_idx = 0
    for idx_d, d in enumerate(original_test['dialogs']):
        for idx_c, c in enumerate(d['dialog']):
            if idx_c == len(d['dialog']) - 1:
                hyp_time = list(map(float, scaled_time[all_idx].split(':')))
                h_exp = hyp_exp[all_idx].strip()
                h_ans = predict['dialogs'][idx_d]['dialog'][idx_c]['answer']
                original_test['dialogs'][idx_d]['dialog'][idx_c]['answer'] = h_ans
                original_test['dialogs'][idx_d]['dialog'][idx_c]['reason'][0]['timestamp'] = hyp_time
                original_test['dialogs'][idx_d]['dialog'][idx_c]['reason'][0]['sentence'] = h_exp
            all_idx += 1
    json.dump(original_test, open('ckpts/ckpt_charades_caption_test/final_predict.json', 'w'))
    print('done')


def main2():
    original_test = json.load(open('data/dstc10/test_set4DSTC10-AVSD.json'))
    predict = read_text('ckpts/ckpt_huili/hyp_complete_results.txt')
    scaled_time = read_text('ckpts/ckpt_charades_caption_test_2/hyp_time_scaled.txt')
    hyp_exp = read_text('ckpts/ckpt_huili/hyp_complete_results_exp.txt')
    all_idx = 0
    predict = predict[1:]
    hyp_exp = hyp_exp[1:]
    predict_map = defaultdict(list)
    for e in predict:
        e_spt = e.split('\t')
        predict_map[e_spt[0].strip()].append(e_spt[2].strip())
    hyp_exp_map = {e.split('\t')[0].strip(): e.split('\t')[2].strip() for e in hyp_exp}
    print()
    for idx_d, d in enumerate(original_test['dialogs']):
        image_id = d['image_id']
        for idx_c, c in enumerate(d['dialog']):
            if idx_c == len(d['dialog']) - 1:
                hyp_time = list(map(float, scaled_time[all_idx].split(':')))
                h_exp = hyp_exp_map[image_id].strip()
                h_ans = predict_map[image_id][-1].strip()
                original_test['dialogs'][idx_d]['dialog'][idx_c]['answer'] = h_ans
                original_test['dialogs'][idx_d]['dialog'][idx_c]['reason'][0]['timestamp'] = hyp_time
                original_test['dialogs'][idx_d]['dialog'][idx_c]['reason'][0]['sentence'] = h_exp
            all_idx += 1
    json.dump(original_test, open('ckpts/ckpt_charades_caption_test/final_predict2.json', 'w'))
    print('done')


if __name__ == '__main__':
    main2()
