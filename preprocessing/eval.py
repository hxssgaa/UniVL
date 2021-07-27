import json


def write_text(lines, data_path):
    with open(data_path, 'w') as f:
        for line in lines:
            f.writelines(line.strip() + '\n')


if __name__ == '__main__':
    ref = json.load(open('ckpts/ckpt_youcook_caption/result_test_set4DSTC7-AVSD_b5_p1.0_beam_search_undisclosed1_ref.json'))
    ref_text = [e['caption'] for e in ref['annotations']]
    hyp = json.load(open('ckpts/ckpt_youcook_caption/result_test_set4DSTC7-AVSD_b5_p1.0_beam_search_undisclosed1_hyp.json'))
    hyp_text = [e['caption'] for e in hyp]
    write_text(ref_text, 'ckpts/ckpt_youcook_caption/test_ref.txt')
    write_text(hyp_text, 'ckpts/ckpt_youcook_caption/bart_pred.txt')
    print('done')