"""
Manage beam search info structure.
Heavily borrowed from OpenNMT-py.
For code in OpenNMT-py, please check the following link (maybe in oldest version):
https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/Beam.py
"""

import torch
import numpy as np
from torch._C import Size

class Constants():
    def __init__(self):
        self.PAD = 0
        self.UNK = 1
        self.BOS = 2
        self.EOS = 3
        self.QUS = 3160
        self.QUS_NEXT = 1024
        self.PAD_WORD = '[PAD]'
        self.UNK_WORD = '[UNK]'
        self.BOS_WORD = '[CLS]'
        self.EOS_WORD = '[SEP]'

    @classmethod
    def from_tokenizer(cls, tokenizer):
        instance = cls()
        instance.PAD = tokenizer.vocab[instance.PAD_WORD]
        instance.UNK = tokenizer.vocab[instance.UNK_WORD]
        instance.BOS = tokenizer.vocab[instance.BOS_WORD]
        instance.EOS = tokenizer.vocab[instance.EOS_WORD]
        return instance

class Beam():
    ''' Beam search for dialogue answer generation with given context'''

    def _prepare_dialog_start_end_positions(self):
        starts = [idx for idx in range(len(self.dialog_context)) 
            if self.dialog_context[idx] == self.constants.QUS and self.dialog_context[idx+1] == self.constants.QUS_NEXT]
        ends = [idx for idx in range(len(self.dialog_context)) 
            if self.dialog_context[idx] == self.constants.QUS and self.dialog_context[idx+1] == self.constants.QUS_NEXT][1:] + list(np.where(self.dialog_context == self.constants.PAD)[0][:1])
        return starts, ends

    def __init__(self, size, device=False, tokenizer=None, dialog_context=None):
        if tokenizer is None:
            self.constants = Constants()
        else:
            self.constants = Constants.from_tokenizer(tokenizer)
        self.tokenizer = tokenizer

        self.size = size
        self._done = False
        # The score for each interface on the beam.
        self.scores = torch.zeros((size,), dtype=torch.float, device=device)
        self.all_scores = []

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [torch.full((size,), self.constants.BOS, dtype=torch.long, device=device)]

        if dialog_context is not None:
            self.dialog_context = dialog_context
            self.dialog_starts, self.dialog_ends = self._prepare_dialog_start_end_positions()
            init_len = self.dialog_ends[0] - self.dialog_starts[0]
            for idx in range(init_len):
                self.next_ys.append(torch.full((self.size,), self.dialog_context[self.dialog_starts[0] + idx], dtype=torch.long, device=device))
                self.prev_ks.append(torch.full((self.size,), 0, dtype=torch.long, device=device))
        self.started = False
        self.dialog_index = [1] * self.size
            # self.next_ys[0] = torch.full((self.size,), self.constants.BOS, dtype=torch.long, device=device)
            # self.next_ys[0][1:] = torch.tensor(self.dialog_context[init_len], dtype=torch.long, device=device)
            # self.scores = torch.zeros((self.size,), dtype=torch.float, device=device)
            # self.prev_ks = [torch.zeros(()) for idx in range(self.size - 1)]

    def get_current_state(self):
        "Get the outputs for the current timestep."
        return self.get_tentative_hypothesis()

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    @property
    def done(self):
        return self._done

    def _check_and_renormalise_beam_words(self):
        if self.dialog_context is None:
            return
        device = self.next_ys[0].device
        update_ys = [[] for _ in range(self.size)]
        update_ks = [[] for _ in range(self.size)]
        for each_beam in range(self.size):
            if self.next_ys[-1][each_beam] == self.constants.QUS_NEXT and self.next_ys[-2][each_beam] == self.constants.QUS and self.dialog_index[each_beam] < len(self.dialog_starts):
                cur_index = self.dialog_index[each_beam]
                init_len = self.dialog_ends[cur_index] - self.dialog_starts[cur_index]
                for idx in range(init_len):
                    update_ys[each_beam].append(self.dialog_context[self.dialog_starts[cur_index] + idx])
                    update_ks[each_beam].append(0)
                self.dialog_index[each_beam] += 1
        # Remove EOS token if dialogue generation hasn't been completed
        for each_beam in range(self.size):
            if self.next_ys[-1][each_beam] == self.constants.EOS:
                self.next_ys = self.next_ys[:-1]
                self.prev_ks = self.prev_ks[:-1]
        update_ys = [e if not e else e[2:] for e in update_ys]
        update_ks = [e if not e else e[2:] for e in update_ks]
        max_len_update_ys = np.max([len(e) for e in update_ys])
        update_ys = [e if len(e) == max_len_update_ys else e + [0] * (max_len_update_ys - len(e)) for e in update_ys]
        update_ks = [e if len(e) == max_len_update_ys else e + [0] * (max_len_update_ys - len(e)) for e in update_ks]
        for idx in range(max_len_update_ys):
            self.next_ys.append(torch.tensor([update_ys[each_beam][idx] for each_beam in range(self.size)], dtype=torch.long, device=device))
            self.prev_ks.append(torch.tensor([update_ks[each_beam][idx] for each_beam in range(self.size)], dtype=torch.long, device=device))

    def advance(self, word_prob, word_length=None):

        "Update beam status and check if finished or not."
        num_words = word_prob.size(1)
        # Sum the previous scores.
        if self.started:
            beam_lk = word_prob + self.scores.unsqueeze(1).expand_as(word_prob)
        else:
            beam_lk = word_prob[0]
        flat_beam_lk = beam_lk.view(-1)
        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True) # 1st sort
        self.all_scores.append(self.scores)
        self.scores = best_scores
        # bestScoresId is flattened as a (beam x word) array,
        # so we need to calculate which word and beam each score came from
        prev_k = best_scores_id // num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append(best_scores_id - prev_k * num_words)
        # End condition is when top-of-beam is EOS.
        if self.next_ys[-1][0].item() == self.constants.EOS and self.dialog_index[0] == len(self.dialog_starts):
            self._done = True
            return True
        self._check_and_renormalise_beam_words()
        self.started = True

        return self._done

    def sort_scores(self):
        "Sort the scores."
        return torch.sort(self.scores, 0, True)

    def get_the_best_score_and_idx(self):
        "Get the score of the best in the beam."
        scores, ids = self.sort_scores()
        return scores[1], ids[1]

    def get_tentative_hypothesis(self):
        "Get the decoded sequence for the current timestep."

        if len(self.next_ys) == 1:
            dec_seq = self.next_ys[0].unsqueeze(1)
        else:
            _, keys = self.sort_scores()
            hyps = [self.get_hypothesis(k) for k in keys]
            hyps = [[self.constants.BOS] + h for h in hyps]
            dec_seq = torch.LongTensor(hyps)

        return dec_seq

    def get_hypothesis(self, k):
        """ Walk back to construct the full hypothesis. """
        hyp = []
        for j in range(len(self.prev_ks) - 1, -1, -1):
            hyp.append(self.next_ys[j+1][k])
            k = self.prev_ks[j][k]

        return list(map(lambda x: x.item(), hyp[::-1]))
