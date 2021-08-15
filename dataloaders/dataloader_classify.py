from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import pickle
import random

class Classify_DataLoader(Dataset):
    """Classification task dataset loader."""
    def __init__(
            self,
            csv,
            data_path,
            features_path,
            tokenizer,
            feature_framerate=1.0,
            max_words=30,
            max_frames=100,
    ):
        """
        Args:
        """
        self.csv = pd.read_csv(csv)
        self.data_dict = pickle.load(open(data_path, 'rb'))
        self.feature_dict = pickle.load(open(features_path, 'rb'))
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer

        # Get iterator video ids
        self.video_id_list = [itm for itm in self.csv['video_id'].values]

        # Get all candidate list
        cand_list = set()
        for video_id in self.video_id_list:
            texts = self.data_dict[video_id]['text']
            cand_list.update(list(texts))
        cand_list = list(map(str, sorted(cand_list)))
        cand_list.append('none')
        cand_map = {e: idx for idx, e in enumerate(cand_list)}
        self.cand_words = np.zeros((len(cand_list), self.max_words), dtype=np.long)
        self.cand_words_mask = np.zeros((len(cand_list), self.max_words), dtype=np.long)

        for i in range(len(cand_list)):
            words = self.tokenizer.tokenize(cand_list[i])
            words = ["[CLS]"] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + ["[SEP]"]
            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)

            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            self.cand_words[i] = np.array(input_ids)
            self.cand_words_mask[i] = np.array(input_mask)

        # Get start and end
        self.start = np.zeros((len(self.video_id_list), self.max_frames, len(cand_list)), dtype=np.float32)
        self.end = np.zeros((len(self.video_id_list), self.max_frames, len(cand_list)), dtype=np.float32)
        for idx, video_id in enumerate(self.video_id_list):
            starts = self.data_dict[video_id]['start']
            ends = self.data_dict[video_id]['end']
            starts[starts >= 58] = 58
            ends[ends >= 59] = 59
            texts = list(self.data_dict[video_id]['text'])
            if not texts:
                continue
            act_ids = [cand_map[text] for text in texts]
            for idx_act_id, act_id in enumerate(act_ids):
                self.start[idx][starts[idx_act_id]][act_id] = 1
                self.end[idx][ends[idx_act_id]][act_id] = 1

    def __len__(self):
        return len(self.video_id_list)

    def _get_video(self, idx):
        s = [1]
        video_mask = np.zeros((len(s), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(s)

        video_features = self.feature_dict[self.csv["feature_file"].values[idx]]
        video = np.zeros((len(s), self.max_frames, video_features.shape[-1]), dtype=np.float)
        for i in range(len(s)):
            video_slice = video_features[:]

            if self.max_frames < video_slice.shape[0]:
                video_slice = video_slice[:self.max_frames]

            slice_shape = video_slice.shape
            max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_shape[0] else slice_shape[0]
            if len(video_slice) < 1:
                print("video_id: {}, start: {}, end: {}".format(self.csv["video_id"].values[idx], start, end))
            else:
                video[i][:slice_shape[0]] = video_slice

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask

    def __getitem__(self, feature_idx):

        video_id = self.video_id_list[feature_idx]

        video, video_mask = self._get_video(feature_idx)

        return self.cand_words, self.cand_words_mask, video, video_mask, self.start[feature_idx], self.end[feature_idx]
