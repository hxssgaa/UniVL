# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

from modules.until_module import PreTrainedModel, LayerNorm, CrossEn, MILNCELoss, MaxMarginRankingLoss
from modules.module_bert import BertModel, BertConfig, BertOnlyMLMHead
from modules.module_visual import VisualModel, VisualConfig, VisualOnlyMLMHead
from modules.module_audio import AudioModel, AudioConfig, AudioOnlyMLMHead
from modules.module_cross import CrossModel, CrossConfig
from modules.module_decoder import DecoderModel, DecoderConfig
from modules.module_ef import MultimodalModel, MultimodalConfig

logger = logging.getLogger(__name__)


class UniVLPreTrainedModel(PreTrainedModel, nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, bert_config, visual_config, audio_config, cross_config, decoder_config, *inputs, **kwargs):
        # utilize bert config as base config
        super(UniVLPreTrainedModel, self).__init__(bert_config)
        self.bert_config = bert_config
        self.visual_config = visual_config
        self.audio_config = audio_config
        self.cross_config = cross_config
        self.decoder_config = decoder_config

        self.bert = None
        self.visual = None
        self.audio = None
        self.cross = None
        self.decoder = None
    
    @classmethod
    def rename_state_dict(cls, state_dict, visual_num_hidden_layers):
        _rename_map = {
            'bert.embeddings': 'mm_model.text_embeddings',
            'bert.encoder.layer.0.attention': 'mm_model.encoder.layer.0.text_attention',
            'bert.encoder.layer.0.intermediate': 'mm_model.encoder.layer.0.text_intermediate',
            'bert.encoder.layer.0.output': 'mm_model.encoder.layer.0.text_output',
            'bert.pooler': 'mm_model.text_pooler',
            'visual.embeddings': 'mm_model.visual_embeddings',
            'visual.encoder.layer.0.attention': 'mm_model.encoder.layer.%d.visual_attention' % (12 - visual_num_hidden_layers),
            'visual.encoder.layer.0.intermediate': 'mm_model.encoder.layer.%d.visual_intermediate' % (12 - visual_num_hidden_layers),
            'visual.encoder.layer.0.output': 'mm_model.encoder.layer.%d.visual_output' % (12 - visual_num_hidden_layers),
            'visual.pooler': 'mm_model.visual_pooler',
        }
        updates = dict()
        for k, v in _rename_map.items():
            if '.0.' in k:
                is_bert = k.startswith('bert.')
                if is_bert:
                    for idx in range(1, 13):
                        updates[k.replace('.0.', '.%d.' % idx)] = v.replace('.0.', '.%d.' % idx)
                else:
                    for idx in range(1, 7):
                        updates[k.replace('.0.', '.%d.' % idx)] = v.replace('.0.', '.%d.' % (idx + 12 - visual_num_hidden_layers))
        _rename_map.update(updates)
        new_state_dicts = dict()
        for k, v in state_dict.items():
            found = False
            for each_k in _rename_map:
                if k.startswith(each_k):
                    found = True
                    new_state_dicts[k.replace(each_k, _rename_map[each_k])] = v
                    break
            if not found:
                new_state_dicts[k] = v
        return new_state_dicts

    @classmethod
    def from_pretrained(cls, pretrained_bert_name, visual_model_name, audio_model_name, cross_model_name, decoder_model_name,
                        state_dict=None, cache_dir=None, type_vocab_size=2, *inputs, **kwargs):

        task_config = None
        if "task_config" in kwargs.keys():
            task_config = kwargs["task_config"]
            if not hasattr(task_config, "local_rank"):
                task_config.__dict__["local_rank"] = 0
            elif task_config.local_rank == -1:
                task_config.local_rank = 0

        bert_config, state_dict = BertConfig.get_config(pretrained_bert_name, cache_dir, type_vocab_size, state_dict, task_config=task_config)
        visual_config, _ = VisualConfig.get_config(visual_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)
        audio_config, _ = AudioConfig.get_config(audio_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)
        cross_config, _ = CrossConfig.get_config(cross_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)
        decoder_config, _ = DecoderConfig.get_config(decoder_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)

        model = cls(bert_config, visual_config, audio_config, cross_config, decoder_config, *inputs, **kwargs)
        model.bert = None
        model.audio = None
        model.visual = None
        delattr(model, 'bert')
        delattr(model, 'audio')
        delattr(model, 'visual')

        state_dict = cls.rename_state_dict(state_dict, visual_config.num_hidden_layers)

        if state_dict is not None:
            model = cls.init_preweight(model, state_dict, task_config=task_config)

        return model

class Normalize2DFeatures(nn.Module):
    def __init__(self, dim_size):
        super(Normalize2DFeatures, self).__init__()
        self.norm2d = LayerNorm(dim_size)

    def forward(self, video):
        video = torch.as_tensor(video).float()
        video = video.view(-1, video.shape[-2], video.shape[-1])
        video = self.norm2d(video)
        return video

def show_log(task_config, info):
    if task_config is None or task_config.local_rank == 0:
        logger.warning(info)

def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            show_log(source_config, "Set {}.{}: {}.".format(target_name,
                                                            target_attr_name, getattr(target_config, target_attr_name)))
    return target_config

def check_attr(target_name, task_config):
    return hasattr(task_config, target_name) and task_config.__dict__[target_name]

class UniVL(UniVLPreTrainedModel):
    def __init__(self, bert_config, visual_config, audio_config, cross_config, decoder_config, task_config):
        super(UniVL, self).__init__(bert_config, visual_config, audio_config, cross_config, decoder_config)
        self.task_config = task_config
        self.ignore_video_index = -1

        assert self.task_config.max_words <= bert_config.max_position_embeddings
        assert self.task_config.max_words <= decoder_config.max_target_embeddings
        assert self.task_config.max_frames <= visual_config.max_position_embeddings
        assert self.task_config.max_frames <= audio_config.max_position_embeddings
        assert self.task_config.max_words + self.task_config.max_frames <= cross_config.max_position_embeddings

        self._stage_one = True
        self._stage_two = False

        if check_attr('stage_two', self.task_config):
            self._stage_one = False
            self._stage_two = self.task_config.stage_two
        show_log(task_config, "Stage-One:{}, Stage-Two:{}".format(self._stage_one, self._stage_two))

        self.train_sim_after_cross = False
        if self._stage_one and check_attr('train_sim_after_cross', self.task_config):
            self.train_sim_after_cross = True
            show_log(task_config, "Test retrieval after cross encoder.")

        # Text Encoder ===>
        bert_config = update_attr("bert_config", bert_config, "num_hidden_layers",
                                   self.task_config, "text_num_hidden_layers")
        self.bert = BertModel(bert_config)
        bert_word_embeddings_weight = self.bert.embeddings.word_embeddings.weight
        bert_position_embeddings_weight = self.bert.embeddings.position_embeddings.weight
        # <=== End of Text Encoder

        # Video Encoder ===>
        if not task_config.skip_visual:
            visual_config = update_attr("visual_config", visual_config, "num_hidden_layers",
                                        self.task_config, "visual_num_hidden_layers")
            self.visual = VisualModel(visual_config)
            visual_word_embeddings_weight = self.visual.embeddings.word_embeddings.weight
        if not task_config.skip_audio:
            audio_config = update_attr("audio_config", audio_config, "num_hidden_layers",
                                        self.task_config, "visual_num_hidden_layers")
            self.audio = AudioModel(audio_config)
            audio_word_embeddings_weight = self.audio.embeddings.word_embeddings.weight

        mm_config = MultimodalConfig()
        self.mm_model = MultimodalModel(mm_config, self.bert.config, self.visual.config, self.audio.config)

        # <=== End of Video Encoder
        # # TODO: Modify later
        # drop_p = 0.1 
        # H = 4
        # self.bi_modal_att_text_visual = MultiheadedAttention(bert_config.hidden_size, visual_config.hidden_size, visual_config.hidden_size, H, drop_p, bert_config.hidden_size)
        # self.bi_modal_att_text_audio = MultiheadedAttention(bert_config.hidden_size, audio_config.hidden_size, audio_config.hidden_size, H, drop_p, bert_config.hidden_size)
        # self.res_layer_visual = ResidualConnection(bert_config.hidden_size, drop_p)
        # self.res_layer_audio = ResidualConnection(bert_config.hidden_size, drop_p)

        # self.bi_modal_att_visual_text = MultiheadedAttention(visual_config.hidden_size, bert_config.hidden_size, bert_config.hidden_size, H, drop_p, visual_config.hidden_size)
        # self.bi_modal_att_visual_audio = MultiheadedAttention(visual_config.hidden_size, audio_config.hidden_size, audio_config.hidden_size, H, drop_p, visual_config.hidden_size)
        # self.res_layer_visual_text = ResidualConnection(visual_config.hidden_size, drop_p)
        # self.res_layer_visual_audio = ResidualConnection(visual_config.hidden_size, drop_p)

        # self.bi_modal_att_audio_text = MultiheadedAttention(audio_config.hidden_size, bert_config.hidden_size, bert_config.hidden_size, H, drop_p, audio_config.hidden_size)
        # self.bi_modal_att_audio_visual = MultiheadedAttention(audio_config.hidden_size, visual_config.hidden_size, visual_config.hidden_size, H, drop_p, audio_config.hidden_size)
        # self.res_layer_audio_text = ResidualConnection(audio_config.hidden_size, drop_p)
        # self.res_layer_audio_visual = ResidualConnection(audio_config.hidden_size, drop_p)

        if self._stage_one is False or self.train_sim_after_cross:
            # Cross Encoder ===>
            cross_config = update_attr("cross_config", cross_config, "num_hidden_layers",
                                        self.task_config, "cross_num_hidden_layers")
            self.cross = CrossModel(cross_config)
            # <=== End of Cross Encoder

            if self.train_sim_after_cross is False:
                # Decoder ===>
                decoder_config = update_attr("decoder_config", decoder_config, "num_decoder_layers",
                                           self.task_config, "decoder_num_hidden_layers")
                self.decoder = DecoderModel(decoder_config, bert_word_embeddings_weight, bert_position_embeddings_weight)
                # <=== End of Decoder

            if self.task_config.do_pretrain and not task_config.skip_visual:
                self.cls = BertOnlyMLMHead(bert_config, bert_word_embeddings_weight)
                self.cls_visual = VisualOnlyMLMHead(visual_config, visual_word_embeddings_weight)
                self.alm_loss_fct = CrossEntropyLoss(ignore_index=-1)
                
            self.similarity_dense = nn.Linear(bert_config.hidden_size, 1)
            self.decoder_loss_fct = CrossEntropyLoss(ignore_index=-1)

        self.normalize_video_feature = Normalize2DFeatures(task_config.video_dim)
        self.normalize_audio_feature = Normalize2DFeatures(task_config.audio_dim)

        mILNCELoss = MILNCELoss(batch_size=task_config.batch_size // task_config.n_gpu, n_pair=task_config.n_pair, )
        maxMarginRankingLoss = MaxMarginRankingLoss(margin=task_config.margin,
                                                    negative_weighting=task_config.negative_weighting,
                                                    batch_size=task_config.batch_size // task_config.n_gpu,
                                                    n_pair=task_config.n_pair,
                                                    hard_negative_rate=task_config.hard_negative_rate, )

        if task_config.use_mil:
            self.loss_fct = CrossEn() if self._stage_two else mILNCELoss
            self._pretrain_sim_loss_fct = mILNCELoss
        else:
            self.loss_fct = CrossEn() if self._stage_two else maxMarginRankingLoss
            self._pretrain_sim_loss_fct = maxMarginRankingLoss

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, video, video_mask=None,
                audio=None, audio_mask=None, pairs_masked_text=None, pairs_token_labels=None, 
                masked_video=None, video_labels_index=None, masked_audio=None, audio_labels_index=None, 
                input_caption_ids=None, decoder_mask=None, output_caption_ids=None):

        input_ids = input_ids.view(-1, input_ids.shape[-1])
        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        if not self.task_config.skip_visual:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = self.normalize_video_feature(video)
        if not self.task_config.skip_audio:
            audio_mask = audio_mask.view(-1, audio_mask.shape[-1])
            audio = self.normalize_audio_feature(audio)

        if input_caption_ids is not None:
            input_caption_ids = input_caption_ids.view(-1, input_caption_ids.shape[-1])
            decoder_mask = decoder_mask.view(-1, decoder_mask.shape[-1])

        sequence_output, visual_output, audio_output = self.get_sequence_visual_audio_output(
            input_ids, token_type_ids, attention_mask, video, video_mask, audio, audio_mask, shaped=True)

        if self.training:
            loss = 0.
            if self._stage_one:
                sim_matrix = self.get_similarity_logits(sequence_output, visual_output, attention_mask,
                                                        video_mask, shaped=True)
                sim_loss = self.loss_fct(sim_matrix)
                loss += sim_loss

            if self._stage_two:
                if self.task_config.do_pretrain:
                    pairs_masked_text = pairs_masked_text.view(-1, pairs_masked_text.shape[-1])
                    pairs_token_labels = pairs_token_labels.view(-1, pairs_token_labels.shape[-1])

                    masked_video = self.normalize_video_feature(masked_video)
                    video_labels_index = video_labels_index.view(-1, video_labels_index.shape[-1])

                    sequence_output_alm, visual_output_alm = self.get_sequence_visual_audio_output(pairs_masked_text, token_type_ids,
                                                                                             attention_mask, masked_video, video_mask, shaped=True)

                    cross_output, pooled_output, concat_mask = self._get_cross_output(sequence_output_alm, visual_output_alm, attention_mask, video_mask)
                    sequence_cross_output, visual_cross_output = torch.split(cross_output, [attention_mask.size(-1), video_mask.size(-1)], dim=1)

                    alm_loss = self._calculate_mlm_loss(sequence_cross_output, pairs_token_labels)
                    loss += alm_loss

                    nce_loss = self._calculate_mfm_loss(visual_cross_output, video, video_mask, video_labels_index)
                    loss += nce_loss

                    sim_matrix = self.get_similarity_logits(sequence_output, visual_output, attention_mask, video_mask,
                                                            shaped=True, _pretrain_joint=True)
                    sim_loss_joint = self._pretrain_sim_loss_fct(sim_matrix)
                    loss += sim_loss_joint

                if (input_caption_ids is not None) and \
                        (self.task_config.do_pretrain
                         or (self.task_config.do_pretrain is False and self.task_config.task_type == "caption")):
                    if self.task_config.do_pretrain:
                        decoder_scores, res_tuples, _ = self._get_decoder_score(sequence_output_alm, visual_output_alm,
                                                                             input_ids, attention_mask, video_mask,
                                                                             input_caption_ids, decoder_mask, shaped=True)
                    elif self.task_config.task_type == "caption":
                        decoder_scores, res_tuples, _ = self._get_decoder_score(sequence_output, visual_output, audio_output,
                                                                             input_ids, attention_mask, video_mask, audio_mask,
                                                                             input_caption_ids, decoder_mask, shaped=True)
                    else:
                        raise NotImplementedError

                    output_caption_ids = output_caption_ids.view(-1, output_caption_ids.shape[-1])
                    decoder_loss = self.decoder_loss_fct(decoder_scores.view(-1, self.bert_config.vocab_size), output_caption_ids.view(-1))
                    loss += decoder_loss

                if self.task_config.do_pretrain or self.task_config.task_type == "retrieval":
                    if self.task_config.do_pretrain:
                        sim_matrix_text_visual = self.get_similarity_logits(sequence_output_alm, visual_output_alm,
                                                                            attention_mask, video_mask, shaped=True)
                    elif self.task_config.task_type == "retrieval":
                        sim_matrix_text_visual = self.get_similarity_logits(sequence_output, visual_output,
                                                                            attention_mask, video_mask, shaped=True)
                    else:
                        raise NotImplementedError

                    sim_loss_text_visual = self.loss_fct(sim_matrix_text_visual)
                    loss += sim_loss_text_visual

            return loss
        else:
            return None

    def _calculate_mlm_loss(self, sequence_output_alm, pairs_token_labels):
        alm_scores = self.cls(sequence_output_alm)
        alm_loss = self.alm_loss_fct(alm_scores.view(-1, self.bert_config.vocab_size), pairs_token_labels.view(-1))
        return alm_loss

    def _calculate_mfm_loss(self, visual_output_alm, video, video_mask, video_labels_index):
        afm_scores = self.cls_visual(visual_output_alm)
        afm_scores_tr = afm_scores.view(-1, afm_scores.shape[-1])

        video_tr = video.permute(2, 0, 1)
        video_tr = video_tr.view(video_tr.shape[0], -1)

        logits_matrix = torch.mm(afm_scores_tr, video_tr)
        video_mask_float = video_mask.to(dtype=torch.float)
        mask_matrix = torch.mm(video_mask_float.view(-1, 1), video_mask_float.view(1, -1))
        masked_logits = logits_matrix + (1. - mask_matrix) * -1e8

        logpt = F.log_softmax(masked_logits, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt

        video_labels_index_mask = (video_labels_index != self.ignore_video_index)
        nce_loss = nce_loss.masked_select(video_labels_index_mask.view(-1))
        nce_loss = nce_loss.mean()
        return nce_loss

    def get_sequence_visual_audio_output(self, input_ids, token_type_ids, attention_mask, video, video_mask, audio, audio_mask, shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            if not self.task_config.skip_visual:
                video_mask = video_mask.view(-1, video_mask.shape[-1])
                video = self.normalize_video_feature(video)
            if not self.task_config.skip_audio:
                audio_mask = audio_mask.view(-1, video_mask.shape[-1])
                audio = self.normalize_audio_feature(audio)

        encoded_layers, _, _, _ = self.mm_model(input_ids, video, video_mask, audio, audio_mask, 
            text_token_type_ids=token_type_ids, text_attention_mask=attention_mask, output_all_encoded_layers=True)
        sequence_output = encoded_layers[-1][0]
        visual_output = encoded_layers[-1][1]
        audio_output = encoded_layers[-1][2]
        return sequence_output, visual_output, audio_output
        # encoded_layers, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True)
        # sequence_output = encoded_layers[-1]

        # if not self.task_config.skip_visual:
        #     visual_layers, _ = self.visual(video, video_mask, output_all_encoded_layers=True)
        #     visual_output = visual_layers[-1]
        # else:
        #     visual_output = None
        # if not self.task_config.skip_audio:
        #     audio_layers, _ = self.audio(audio, audio_mask, output_all_encoded_layers=True)
        #     audio_output = audio_layers[-1]

        # return sequence_output, visual_output, audio_output

    def _get_cross_output(self, sequence_output, visual_output, audio_output, attention_mask, video_mask, audio_mask):
        if self.task_config.skip_visual:
            return sequence_output, None, attention_mask

        # video_mask = video_mask.unsqueeze(1)
        # audio_mask = audio_mask.unsqueeze(1)
        # attention_mask = attention_mask.unsqueeze(1)
        # def sublayer_att_text_visual(text): return self.bi_modal_att_text_visual(text, visual_output, visual_output, video_mask)
        # def sublayer_att_text_audio(text): return self.bi_modal_att_text_audio(text, audio_output, audio_output, audio_mask)

        # def sublayer_att_visual_text(visual): return self.bi_modal_att_visual_text(visual, sequence_output, sequence_output, attention_mask)
        # def sublayer_att_visual_audio(visual): return self.bi_modal_att_visual_audio(visual, audio_output, audio_output, audio_mask)

        # def sublayer_att_audio_text(audio): return self.bi_modal_att_audio_text(audio, visual_output, visual_output, video_mask)
        # def sublayer_att_audio_visual(audio): return self.bi_modal_att_audio_visual(audio, audio_output, audio_output, audio_mask)

        # visual_aware_sequence_output = self.res_layer_visual(sequence_output, sublayer_att_text_visual)
        # visual_audio_aware_sequence_output = self.res_layer_audio(visual_aware_sequence_output, sublayer_att_text_audio)

        # text_aware_visual_output = self.res_layer_visual_text(visual_output, sublayer_att_visual_text)
        # text_audio_aware_visual_output = self.res_layer_visual_audio(text_aware_visual_output, sublayer_att_visual_audio)
        
        # text_aware_audio_output = self.res_layer_audio_text(audio_output, sublayer_att_audio_text)
        # text_visual_aware_audio_output = self.res_layer_audio_visual(text_aware_audio_output, sublayer_att_audio_visual)
        
        # video_mask = video_mask.squeeze(1)
        # audio_mask = audio_mask.squeeze(1)
        # attention_mask = attention_mask.squeeze(1)
        # sequence_output = visual_audio_aware_sequence_output
        # visual_output = text_audio_aware_visual_output
        # audio_output = text_visual_aware_audio_output

        tuple_features = (sequence_output, visual_output, audio_output)
        tuple_masks = (attention_mask, video_mask, audio_mask)
        valid = tuple([tuple_features[idx] is not None for idx in range(len(tuple_features))])
        valid_features = tuple([tuple_features[idx] for idx in range(len(valid)) if valid[idx]])
        valid_masks = tuple([tuple_masks[idx] for idx in range(len(valid)) if valid[idx]])

        concat_features = torch.cat(valid_features, dim=1)  # concatnate tokens and frames
        concat_mask = torch.cat(valid_masks, dim=1)
        # text_type_ = torch.zeros_like(attention_mask)
        # video_type_ = torch.ones_like(video_mask)
        # audio_type_ = torch.ones_like(audio_mask)
        # concat_type = torch.cat((text_type_, video_type_, audio_type_), dim=1)

        return concat_features, None, concat_mask
        # cross_layers, pooled_output = self.cross(concat_features, concat_type, concat_mask, output_all_encoded_layers=True)
        # cross_output = cross_layers[-1]
        # return cross_output, pooled_output, concat_mask

    def _mean_pooling_for_similarity(self, sequence_output, visual_output, attention_mask, video_mask,):
        attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1)
        attention_mask_un[:, 0, :] = 0.
        sequence_output = sequence_output * attention_mask_un
        text_out = torch.sum(sequence_output, dim=1) / torch.sum(attention_mask_un, dim=1, dtype=torch.float)

        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum

        return text_out, video_out

    def _cross_similarity(self, sequence_output, visual_output, attention_mask, video_mask):
        b_text, s_text, h_text = sequence_output.size()
        b_visual, s_visual, h_visual = visual_output.size()

        retrieve_logits_list = []
        step_size = 5

        split_size = [step_size] * (b_text // step_size)
        release_size = b_text - sum(split_size)
        if release_size > 0:
            split_size += [release_size]

        sequence_output_splits = torch.split(sequence_output, split_size, dim=0)
        attention_mask_splits = torch.split(attention_mask, split_size, dim=0)
        for i in range(len(split_size)):
            sequence_output_row = sequence_output_splits[i]
            attention_mask_row = attention_mask_splits[i]
            sequence_output_l = sequence_output_row.unsqueeze(1).repeat(1, b_visual, 1, 1)
            sequence_output_l = sequence_output_l.view(-1, s_text, h_text)
            attention_mask_l = attention_mask_row.unsqueeze(1).repeat(1, b_visual, 1)
            attention_mask_l = attention_mask_l.view(-1, s_text)

            step_truth = sequence_output_row.size(0)
            visual_output_r = visual_output.unsqueeze(0).repeat(step_truth, 1, 1, 1)
            visual_output_r = visual_output_r.view(-1, s_visual, h_visual)
            video_mask_r = video_mask.unsqueeze(0).repeat(step_truth, 1, 1)
            video_mask_r = video_mask_r.view(-1, s_visual)

            cross_output, pooled_output, concat_mask = \
                self._get_cross_output(sequence_output_l, visual_output_r, attention_mask_l, video_mask_r)
            retrieve_logits_row = self.similarity_dense(pooled_output).squeeze(-1).view(step_truth, b_visual)

            retrieve_logits_list.append(retrieve_logits_row)
        retrieve_logits = torch.cat(retrieve_logits_list, dim=0)
        return retrieve_logits

    def get_similarity_logits(self, sequence_output, visual_output, attention_mask, video_mask, shaped=False, _pretrain_joint=False):
        if shaped is False:
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        if (self._stage_two and _pretrain_joint is False) or self.train_sim_after_cross:
            retrieve_logits = self._cross_similarity(sequence_output, visual_output, attention_mask, video_mask)
        else:
            text_out, video_out = self._mean_pooling_for_similarity(sequence_output, visual_output, attention_mask, video_mask)
            if self.task_config.use_mil is False:
                text_out = F.normalize(text_out, dim=-1)
                video_out = F.normalize(video_out, dim=-1)
            retrieve_logits = torch.matmul(text_out, video_out.t())

        return retrieve_logits

    def _get_decoder_score(self, sequence_output, visual_output, audio_output, input_ids, attention_mask, video_mask, audio_mask, input_caption_ids, decoder_mask, shaped=False):

        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            if not self.task_config.skip_visual:
                video_mask = video_mask.view(-1, video_mask.shape[-1])
            else:
                video_mask = None
            if not self.task_config.skip_audio:
                audio_mask = audio_mask.view(-1, audio_mask.shape[-1])
            else:
                audio_mask = None

            input_caption_ids = input_caption_ids.view(-1, input_caption_ids.shape[-1])
            decoder_mask = decoder_mask.view(-1, decoder_mask.shape[-1])

        res_tuples = ()
        cross_output, pooled_output, concat_mask = self._get_cross_output(sequence_output, visual_output, audio_output, attention_mask, video_mask, audio_mask)
        decoder_scores, dec_att_scores = self.decoder(input_caption_ids, encoder_outs=cross_output, answer_mask=decoder_mask, encoder_mask=concat_mask)

        return decoder_scores, res_tuples, dec_att_scores

    def decoder_caption(self, sequence_output, visual_output, audio_output, input_ids, attention_mask, video_mask, audio_mask, input_caption_ids, decoder_mask,
                        shaped=False, get_logits=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            if not self.task_config.skip_visual:
                video_mask = video_mask.view(-1, video_mask.shape[-1])
            else:
                video_mask = None
            if not self.task_config.skip_audio:
                audio_mask = audio_mask.view(-1, audio_mask.shape[-1])
            else:
                audio_mask = None

            input_caption_ids = input_caption_ids.view(-1, input_caption_ids.shape[-1])
            decoder_mask = decoder_mask.view(-1, decoder_mask.shape[-1])

        decoder_scores, _, decoder_attn_scores = self._get_decoder_score(sequence_output, visual_output, audio_output,
                                                    input_ids, attention_mask, video_mask, audio_mask,
                                                    input_caption_ids, decoder_mask, shaped=True)

        if get_logits:
            return decoder_scores, decoder_attn_scores

        _, decoder_scores_result = torch.max(decoder_scores, -1)

        return decoder_scores_result, decoder_attn_scores
