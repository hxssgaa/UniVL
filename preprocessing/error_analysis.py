import json
from collections import defaultdict


TYPE_MAP = {
    'does': 0,
    'is': 0,
    'what': 1,
    'how': 2,
    'can': 0,
    'do': 0,
    'are': 0,
    'did': 0,
    'so': 5,
    'where': 3,
    'was': 0,
    'why': 4,
}


def read_text(file_name):
    with open(file_name) as f:
        return list(map(str.strip, list(f.readlines())))

def group_by_types(l, id_map):
    return [[l[idx2] for idx2 in range(len(id_map)) if id_map[idx2] == idx] for idx in range(6)]


def error_analysis():
    val_data = json.load(open('data/dstc10/valid_set4DSTC10-AVSD_with_action.json'))
    cs = [c for d in val_data['dialogs'] for c in d['dialog']]
    type_cnt = defaultdict(int)
    id_map = []
    for e in cs:
        type_idx = TYPE_MAP.get(e['question'].split()[0], 5)
        type_cnt[e['question'].split()[0]] += 1
        id_map.append(type_idx)
    type_cnt = list(sorted(list(type_cnt.items()), key=lambda x: x[1], reverse=True))
    refs = group_by_types(read_text('ckpts/ckpt_charades_caption_best/ref.txt'), id_map)
    hyps = group_by_types(read_text('ckpts/ckpt_charades_caption_best/hyp.txt'), id_map)
    for idx in range(len(refs)):
        with open('ckpts/ckpt_charades_caption_best/ref%d.txt' % idx, 'w') as f:
            f.writelines(list(map(lambda x: x + '\n', refs[idx])))
        with open('ckpts/ckpt_charades_caption_best/hyp%d.txt' % idx, 'w') as f:
            f.writelines(list(map(lambda x: x + '\n', hyps[idx])))
        print(len(refs[idx]) / len(id_map))
    print('done')


if __name__ == '__main__':
    error_analysis()