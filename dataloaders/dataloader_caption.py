from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import pickle
import re
import random
import io

class Caption_DataLoader(Dataset):
    """Caption generation dataset loader."""
    def __init__(
            self,
            csv,
            data_path,
            video_features_path,
            audio_features_path,
            tokenizer,
            feature_framerate=1.0,
            max_words=30,
            max_frames=100,
            skip_visual=False,
            skip_audio=False,
    ):
        """
        Args:
        """
        self.csv = pd.read_csv(csv)
        self.data_dict = pickle.load(open(data_path, 'rb'))
        self.video_feature_dict = pickle.load(open(video_features_path, 'rb'))
        if audio_features_path:
            self.audio_feature_dict = pickle.load(open(audio_features_path, 'rb'))
            self.audio_feature_size = self.audio_feature_dict[self.csv["feature_file"].values[0]].shape[-1]
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        self.skip_visual = skip_visual
        self.skip_audio = skip_audio

        self.video_feature_size = self.video_feature_dict[self.csv["feature_file"].values[0]].shape[-1]

        # Get iterator video ids
        video_id_list = [itm for itm in self.csv['video_id'].values]
        self.video_id2idx_dict = {video_id: id for id, video_id in enumerate(video_id_list)}
        # Get all captions
        self.iter2video_pairs_dict = {}
        iter_idx_ = 0
        for video_id in video_id_list:
            data_dict = self.data_dict[video_id]
            n_caption = len(data_dict['start'])
            for sub_id in range(n_caption):
                self.iter2video_pairs_dict[iter_idx_] = (video_id, sub_id)
                iter_idx_ += 1

    def __len__(self):
        return len(self.iter2video_pairs_dict)

    def _get_text(self, video_id, sub_id):
        data_dict = self.data_dict[video_id]
        k = 1
        r_ind = [sub_id]

        starts = np.zeros(k)
        ends = np.zeros(k)
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)
        pairs_masked_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_token_labels = np.zeros((k, self.max_words), dtype=np.long)

        pairs_input_caption_ids = np.zeros((k, self.max_words), dtype=np.long)
        pairs_output_caption_ids = np.zeros((k, self.max_words), dtype=np.long)
        pairs_decoder_mask = np.zeros((k, self.max_words), dtype=np.long)

        for i in range(k):
            ind = r_ind[i]
            start_, end_ = data_dict['start'][ind], data_dict['end'][ind]
            starts[i], ends[i] = start_, end_
            total_length_with_CLS = self.max_words - 1
            words = self.tokenizer.tokenize(data_dict['transcript'][ind])

            words = ["[CLS]"] + words
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + ["[SEP]"]

            # Mask Language Model <-----
            token_labels = []
            masked_tokens = words.copy()
            for token_id, token in enumerate(masked_tokens):
                if token_id == 0 or token_id == len(masked_tokens) - 1:
                    token_labels.append(-1)
                    continue
                prob = random.random()
                # mask token with 15% probability
                if prob < 0.15:
                    prob /= 0.15

                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        masked_tokens[token_id] = "[MASK]"

                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        masked_tokens[token_id] = random.choice(list(self.tokenizer.vocab.items()))[0]

                    # -> rest 10% randomly keep current token

                    # append current token to output (we will predict these later)
                    try:
                        token_labels.append(self.tokenizer.vocab[token])
                    except KeyError:
                        # For unknown words (should not occur with BPE vocab)
                        token_labels.append(self.tokenizer.vocab["[UNK]"])
                        # print("Cannot find token '{}' in vocab. Using [UNK] insetad".format(token))
                else:
                    # no masking token (will be ignored by loss function later)
                    token_labels.append(-1)
            # -----> Mask Language Model

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            masked_token_ids = self.tokenizer.convert_tokens_to_ids(masked_tokens)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                masked_token_ids.append(0)
                token_labels.append(-1)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words
            assert len(masked_token_ids) == self.max_words
            assert len(token_labels) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)
            pairs_masked_text[i] = np.array(masked_token_ids)
            pairs_token_labels[i] = np.array(token_labels)

            # For generate captions
            caption_words = self.tokenizer.tokenize(data_dict['text'][ind])
            if len(caption_words) > total_length_with_CLS:
                caption_words = caption_words[:total_length_with_CLS]
            input_caption_words = ["[CLS]"] + caption_words
            output_caption_words = caption_words + ["[SEP]"]

            # For generate captions
            input_caption_ids = self.tokenizer.convert_tokens_to_ids(input_caption_words)
            output_caption_ids = self.tokenizer.convert_tokens_to_ids(output_caption_words)
            decoder_mask = [1] * len(input_caption_ids)
            while len(input_caption_ids) < self.max_words:
                input_caption_ids.append(0)
                output_caption_ids.append(0)
                decoder_mask.append(0)
            assert len(input_caption_ids) == self.max_words
            assert len(output_caption_ids) == self.max_words
            assert len(decoder_mask) == self.max_words

            pairs_input_caption_ids[i] = np.array(input_caption_ids)
            pairs_output_caption_ids[i] = np.array(output_caption_ids)
            pairs_decoder_mask[i] = np.array(decoder_mask)

        return pairs_text, pairs_mask, pairs_segment, pairs_masked_text, pairs_token_labels,\
               pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, starts, ends

    def _get_video(self, idx, s, e):
        if self.skip_visual:
            return None, None, None, None
        video_mask = np.zeros((len(s), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(s)

        video_features = self.video_feature_dict[self.csv["feature_file"].values[idx]]
        video = np.zeros((len(s), self.max_frames, self.video_feature_size), dtype=np.float)
        for i in range(len(s)):
            start = int(s[i] * self.feature_framerate)
            end = int(e[i] * self.feature_framerate) + 1
            video_slice = video_features[start:end]

            if self.max_frames < video_slice.shape[0]:
                video_slice = video_slice[:self.max_frames]

            slice_shape = video_slice.shape
            max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_shape[0] else slice_shape[0]
            if len(video_slice) < 1:
                print("video_id: {}, start: {}, end: {}".format(self.csv["video_id"].values[idx], start, end))
                # pass
            else:
                video[i][:slice_shape[0]] = video_slice

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        # Mask Frame Model <-----
        video_labels_index = [[] for _ in range(len(s))]
        masked_video = video.copy()
        for i, video_pair_ in enumerate(masked_video):
            for j, _ in enumerate(video_pair_):
                if j < max_video_length[i]:
                    prob = random.random()
                    # mask token with 15% probability
                    if prob < 0.15:
                        masked_video[i][j] = [0.] * video.shape[-1]
                        video_labels_index[i].append(j)
                    else:
                        video_labels_index[i].append(-1)
                else:
                    video_labels_index[i].append(-1)
        video_labels_index = np.array(video_labels_index, dtype=np.long)
        # -----> Mask Frame Model

        return video, video_mask, masked_video, video_labels_index

    def _get_audio(self, idx, s, e):
        audio_mask = np.zeros((len(s), self.max_frames), dtype=np.long)
        max_audio_length = [0] * len(s)

        audio_features = self.audio_feature_dict[self.csv["feature_file"].values[idx]]
        audio = np.zeros((len(s), self.max_frames, self.audio_feature_size), dtype=np.float)
        for i in range(len(s)):
            start = int(s[i] * self.feature_framerate)
            end = int(e[i] * self.feature_framerate) + 1
            audio_slice = audio_features[start:end]

            if self.max_frames < audio_slice.shape[0]:
                audio_slice = audio_slice[:self.max_frames]

            slice_shape = audio_slice.shape
            max_audio_length[i] = max_audio_length[i] if max_audio_length[i] > slice_shape[0] else slice_shape[0]
            if len(audio_slice) < 1:
                print("video_id: {}, start: {}, end: {}".format(self.csv["video_id"].values[idx], start, end))
                # pass
            else:
                audio[i][:slice_shape[0]] = audio_slice

        for i, v_length in enumerate(max_audio_length):
            audio_mask[i][:v_length] = [1] * v_length

        # Mask Frame Model <-----
        audio_labels_index = [[] for _ in range(len(s))]
        masked_audio = audio.copy()
        for i, video_pair_ in enumerate(masked_audio):
            for j, _ in enumerate(video_pair_):
                if j < max_audio_length[i]:
                    prob = random.random()
                    # mask token with 15% probability
                    if prob < 0.15:
                        masked_audio[i][j] = [0.] * audio.shape[-1]
                        audio_labels_index[i].append(j)
                    else:
                        audio_labels_index[i].append(-1)
                else:
                    audio_labels_index[i].append(-1)
        audio_labels_index = np.array(audio_labels_index, dtype=np.long)
        # -----> Mask Frame Model

        return audio, audio_mask, masked_audio, audio_labels_index

    def __getitem__(self, feature_idx):

        video_id, sub_id = self.iter2video_pairs_dict[feature_idx]
        idx = self.video_id2idx_dict[video_id]

        pairs_text, pairs_mask, pairs_segment, \
        pairs_masked_text, pairs_token_labels, pairs_input_caption_ids, \
        pairs_decoder_mask, pairs_output_caption_ids, starts, ends = self._get_text(video_id, sub_id)

        if self.skip_visual:
            return pairs_text, pairs_mask, pairs_segment, \
                   pairs_masked_text, pairs_token_labels, \
                   pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids

        video, video_mask, masked_video, video_labels_index = self._get_video(idx, starts, ends)

        audio, audio_mask, masked_audio, audio_labels_index = self._get_audio(idx, starts, ends)

        return pairs_text, pairs_mask, pairs_segment, video, video_mask, audio, audio_mask, \
               pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, masked_audio, audio_labels_index, \
               pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids
