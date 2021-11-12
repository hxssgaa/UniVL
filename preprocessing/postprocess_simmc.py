import json


def read_text(data_path):
    res = []
    with open(data_path) as f:
        for line in f.readlines():
            res.append(line.strip())
    return res

def process_line_to_dst(line):
    left_bracket = line.index('[')
    right_bracket = line.rindex(')')
    intent = line[:left_bracket].strip().replace(' ', '').upper()
    objects = line[left_bracket: right_bracket +1].strip()
    return '=> Belief State : %s %s <  > <EOB>' % (intent, objects)

def process_line_to_ans_gen(line):
    right_bracket = line.rindex(')')
    s = line[right_bracket + 1:].strip()
    return '<EOB> %s <EOS>' % s



def main():
    hyp = read_text('ckpts/ckpt_simmc_devtest/hyp.txt')
    hyp = list(map(process_line_to_dst, hyp))
    with open('ckpts/ckpt_simmc_devtest/simmc2_dials_dstc10_devtest_predict_medcat2.txt', 'w') as f:
        for line in hyp:
            f.writelines(line + '\n')
    print('done')


if __name__ == '__main__':
    main()
