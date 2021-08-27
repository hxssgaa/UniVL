import json


def read_text(data_path):
    res = []
    with open(data_path) as f:
        for line in f.readlines():
            res.append(line.strip())
    return res



def iou(interval_1, interval_2):
    start_i, end_i = interval_1[0], interval_1[1]
    start, end = interval_2[0], interval_2[1]
    intersection = max(0, min(end, end_i) - max(start, start_i))
    union = min(max(end, end_i) - min(start, start_i), end-start + end_i-start_i)
    iou = float(intersection) / (union + 1e-8)
    return iou


def set_iou(intervals_1, intervals_2):
    def within_interval(intervals, t):
        return sum([interval[0] <= t < interval[1] for interval in intervals]) > 0
    tick = 0.1  # tick time [sec]
    intervals = intervals_1 + intervals_2
    lower = int(min([intrv[0] for intrv in intervals]) / tick)
    upper = int(max([intrv[1] for intrv in intervals]) / tick)
    intersection = 0
    union = 0
    for t in range(lower, upper):
        time = t * tick
        intersection += int(within_interval(intervals_1, time) & within_interval(intervals_2, time))
        union += int(within_interval(intervals_1, time) | within_interval(intervals_2, time))
    iou = float(intersection) / (union + 1e-8)
    return iou


def average_iou1(ref, hypo):
    ious = []
    for key, intervals in ref.items():
        if key in hypo:
            for r_int in intervals:
                ious.append(max([iou(r_int, h_int) for h_int in hypo[key]]))
        else:
            ious.extend([0.0] * len(intervals))
    if len(ious) > 0:
        return sum(ious) / len(ious)
    else:
        return 0.0

def average_iou2(ref, hypo):
    ious = []
    for key, intervals in ref.items():
        if key in hypo:
            ious.append(set_iou(intervals, hypo[key]))
        else:
            ious.append(0.0)
    if len(ious) > 0:
        return sum(ious) / len(ious)
    else:
        return 0.0


if __name__ == '__main__':
    pred = read_text('ckpts/ckpt_charades_caption_best/hyp_time_scaled.txt')
    gold = json.load(open('data/dstc10/valid_set4DSTC10-AVSD+reason.json'))
    gold = [list(map(lambda x: x['timestamp'], c['reason'])) for d in gold['dialogs'] for c in d['dialog']]
    pred = list(map(lambda x: [list(map(float, x.split(':')))], pred))
    gold = {str(idx): gold[idx] for idx in range(len(gold))}
    pred = {str(idx): pred[idx] for idx in range(len(pred))}
    print(average_iou1(gold, pred))
    print(average_iou2(gold, pred))