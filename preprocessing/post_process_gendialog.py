import itertools

def read_text(data_path):
    res = []
    with open(data_path) as f:
        for line in f.readlines():
            res.append(line.strip())
    return res


def write_text(lines, data_path):
    with open(data_path, 'w') as f:
        for line in lines:
            f.writelines(line.strip() + '\n')


def main():
    hyp = read_text('ckpts/ckpt_charades_gendialog384/hyp.txt')
    all_answers = []
    empty = 0
    for idx, d in enumerate(hyp):
        answers = d.split('question :')
        answers = [e[e.index('answer :')+9:].strip() for e in answers if 'answer :' in e]
        answers = answers[:10]
        while len(answers) < 10:
            answers.append("")
            empty += 1
        all_answers += answers
    print(empty / len(all_answers))
    write_text(all_answers, 'ckpts/ckpt_charades_gendialog384/hyp_processed.txt')
    print('done')


if __name__ == '__main__':
    main()