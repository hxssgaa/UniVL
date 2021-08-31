import json


def preprocess_simm2_for_split(split):
    data = json.load(open('data/simmc2/simmc2_dials_dstc10_%s.json' % split))
    output_dict = dict()
    for d in data['dialogue_data']:
        start_ids = list(sorted(list(map(int, d['scene_ids'].keys()))))
        end_ids = start_ids[1:]
        end_ids.append(len(d['dialogue']))
        for scene_idx in range(len(start_ids)):
            start_id = start_ids[scene_idx]
            end_id = end_ids[scene_idx]
            dialogues = d['dialogue'][start_id: end_id]
            scene_id = d['scene_ids'][str(start_id)]
            print()
        print()


def preprocess_simmc2():
    splits = ['train']
    for split in splits:
        preprocess_simm2_for_split(split)


if __name__ == '__main__':
    preprocess_simmc2()