from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import json
import math
import logging
import tarfile
import tempfile
import shutil

import torch
from torch import nn
import torch.nn.functional as F
from .file_utils import cached_path
from .until_config import PretrainedConfig
from .until_module import PreTrainedModel, LayerNorm, ACT2FN
from modules.module_attn import MultiheadedAttention, ResidualConnection

logger = logging.getLogger(__name__)


class MultimodalConfig(PretrainedConfig):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self,
        initializer_range=0.02):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        self.initializer_range = initializer_range


class TextEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(TextEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class VisualEmbeddings(nn.Module):
    """Construct the embeddings from word and position embeddings.
    """
    def __init__(self, config):
        super(VisualEmbeddings, self).__init__()

        self.word_embeddings = nn.Linear(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_embeddings):
        seq_length = input_embeddings.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_embeddings.device)
        position_ids = position_ids.unsqueeze(0).expand(input_embeddings.size(0), -1)

        words_embeddings = self.word_embeddings(input_embeddings)
        # words_embeddings = self.transform_act_fn(words_embeddings)

        position_embeddings = self.position_embeddings(position_ids)
        embeddings = words_embeddings + position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class Pooler(nn.Module):
    def __init__(self, config):
        super(Pooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class SelfAttention(nn.Module):
    def __init__(self, config):
        super(SelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class SelfOutput(nn.Module):
    def __init__(self, config):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.self = SelfAttention(config)
        self.output = SelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output

class Intermediate(nn.Module):
    def __init__(self, config):
        super(Intermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class Output(nn.Module):
    def __init__(self, config):
        super(Output, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class MultimodalLayer(nn.Module):
    def __init__(self, config, text_config, visual_config, audio_config, enable_text=True, enable_visual=True, enable_audio=True, layer_number=None):
        super(MultimodalLayer, self).__init__()

        drop_p = 0.1
        H = 4

        if enable_text:
            self.text_attention = Attention(text_config)
            self.text_intermediate = Intermediate(text_config)
            self.text_output = Output(text_config)
        if enable_visual:
            self.visual_attention = Attention(visual_config)
            self.visual_intermediate = Intermediate(visual_config)
            self.visual_output = Output(visual_config)
        if enable_audio:
            self.audio_attention = Attention(audio_config)
            self.audio_intermediate = Intermediate(audio_config)
            self.audio_output = Output(audio_config)

        enable_fusion = False #layer_number > 8
        if enable_text and enable_visual and enable_fusion:
            self.bi_modal_att_text_visual = MultiheadedAttention(text_config.hidden_size, visual_config.hidden_size, visual_config.hidden_size, H, drop_p, text_config.hidden_size)
            self.bi_modal_att_visual_text = MultiheadedAttention(visual_config.hidden_size, text_config.hidden_size, text_config.hidden_size, H, drop_p, visual_config.hidden_size)
            self.res_layer_text_visual = ResidualConnection(text_config.hidden_size, drop_p)
            self.res_layer_visual_text = ResidualConnection(visual_config.hidden_size, drop_p)

        if enable_text and enable_audio and enable_fusion:
            self.bi_modal_att_text_audio = MultiheadedAttention(text_config.hidden_size, audio_config.hidden_size, audio_config.hidden_size, H, drop_p, text_config.hidden_size)
            self.bi_modal_att_audio_text = MultiheadedAttention(audio_config.hidden_size, text_config.hidden_size, text_config.hidden_size, H, drop_p, audio_config.hidden_size)
            self.res_layer_text_audio = ResidualConnection(text_config.hidden_size, drop_p)
            self.res_layer_audio_text = ResidualConnection(audio_config.hidden_size, drop_p)

        if enable_visual and enable_audio and enable_fusion:
            self.bi_modal_att_visual_audio = MultiheadedAttention(visual_config.hidden_size, audio_config.hidden_size, audio_config.hidden_size, H, drop_p, visual_config.hidden_size)
            self.bi_modal_att_audio_visual = MultiheadedAttention(audio_config.hidden_size, visual_config.hidden_size, visual_config.hidden_size, H, drop_p, audio_config.hidden_size)
            self.res_layer_visual_audio = ResidualConnection(visual_config.hidden_size, drop_p)
            self.res_layer_audio_visual = ResidualConnection(audio_config.hidden_size, drop_p)

        self.enable_text = enable_text
        self.enable_visual = enable_visual
        self.enable_audio = enable_audio
        self.layer_number = layer_number
        self.enable_fusion = enable_fusion

    def forward(self, hidden_states, attention_mask):
        # Self attention
        if self.enable_text:
            text_attention_output = self.text_attention(hidden_states[0], attention_mask[0])
        if self.enable_visual:
            visual_attention_output = self.visual_attention(hidden_states[1], attention_mask[1])
        if self.enable_audio:
            audio_attention_output = self.audio_attention(hidden_states[2], attention_mask[2])

        # Multimodal attention
        attention_mask = [1 - m.squeeze(1) / (-10000.0) for m in attention_mask if m is not None]
        def sublayer_att_text_visual(text): return self.bi_modal_att_text_visual(text, visual_attention_output, visual_attention_output, attention_mask[1])
        def sublayer_att_text_audio(text): return self.bi_modal_att_text_audio(text, audio_attention_output, audio_attention_output, attention_mask[2])

        def sublayer_att_visual_text(visual): return self.bi_modal_att_visual_text(visual, text_attention_output, text_attention_output, attention_mask[0])
        def sublayer_att_visual_audio(visual): return self.bi_modal_att_visual_audio(visual, audio_attention_output, audio_attention_output, attention_mask[2])

        def sublayer_att_audio_text(audio): return self.bi_modal_att_audio_text(audio, text_attention_output, text_attention_output, attention_mask[0])
        def sublayer_att_audio_visual(audio): return self.bi_modal_att_audio_visual(audio, visual_attention_output, visual_attention_output, attention_mask[1])
        if self.enable_text and self.enable_visual and self.enable_fusion:
            visual_aware_text_output = self.res_layer_text_visual(text_attention_output, sublayer_att_text_visual)
            text_aware_visual_output = self.res_layer_visual_text(visual_attention_output, sublayer_att_visual_text)
            text_attention_output = visual_aware_text_output
            visual_attention_output = text_aware_visual_output
        if self.enable_text and self.enable_audio and self.enable_fusion:
            audio_aware_text_output = self.res_layer_text_audio(text_attention_output, sublayer_att_text_audio)
            text_aware_audio_output = self.res_layer_audio_text(audio_attention_output, sublayer_att_audio_text)
            text_attention_output = audio_aware_text_output
            audio_attention_output = text_aware_audio_output
        if self.enable_visual and self.enable_audio and self.enable_fusion:
            audio_aware_visual_output = self.res_layer_visual_audio(visual_attention_output, sublayer_att_visual_audio)
            visual_aware_audio_output = self.res_layer_audio_visual(audio_attention_output, sublayer_att_audio_visual)
            visual_attention_output = audio_aware_visual_output
            audio_attention_output = visual_aware_audio_output
        attention_mask = [1 - m.unsqueeze(1) / (-10000.0) for m in attention_mask if m is not None]

        # Position-wise feed-forward
        layer_output = []
        if self.enable_text:
            intermediate_text_output = self.text_intermediate(text_attention_output)
            text_layer_output = self.text_output(intermediate_text_output, text_attention_output)
            layer_output.append(text_layer_output)
        else:
            layer_output.append(hidden_states[0])
        if self.enable_visual:
            intermediate_visual_output = self.visual_intermediate(visual_attention_output)
            visual_layer_output = self.visual_output(intermediate_visual_output, visual_attention_output)
            layer_output.append(visual_layer_output)
        else:
            layer_output.append(hidden_states[1])
        if self.enable_audio:
            intermediate_audio_output = self.audio_intermediate(audio_attention_output)
            audio_layer_output = self.audio_output(intermediate_audio_output, audio_attention_output)
            layer_output.append(audio_layer_output)
        else:
            layer_output.append(hidden_states[2])
        return layer_output


class MultimodalEncoder(nn.Module):
    def __init__(self, config, text_config, visual_config, audio_config):
        super(MultimodalEncoder, self).__init__()
        text_num_hidden = text_config.num_hidden_layers
        max_num_hidden = text_num_hidden
        if visual_config is not None:
            visual_num_hidden = visual_config.num_hidden_layers
            max_num_hidden = max(max_num_hidden, visual_num_hidden)
        else:
            visual_num_hidden = 0
        if audio_config is not None:
            audio_num_hidden = audio_config.num_hidden_layers
            max_num_hidden = max(max_num_hidden, audio_num_hidden)
        else:
            audio_num_hidden = 0
        
        enable_text = list(reversed([idx < text_num_hidden for idx in range(max_num_hidden)]))
        enable_visual = list(reversed([idx < visual_num_hidden for idx in range(max_num_hidden)]))
        enable_audio = list(reversed([idx < audio_num_hidden for idx in range(max_num_hidden)]))
        
        self.layer = nn.ModuleList([MultimodalLayer(config, text_config, visual_config, audio_config, 
            enable_text=enable_text[idx], enable_visual=enable_visual[idx], enable_audio=enable_audio[idx], layer_number=idx) for idx in range(max_num_hidden)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for idx_layer, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class MultimodalModel(PreTrainedModel):
    @classmethod
    def from_pretrained(cls, config, text_model, visual_model, audio_model):
        model = cls(config, text_model.config, visual_model.config, audio_model.config)
        

    def __init__(self, config, text_config, visual_config, audio_config):
        super(MultimodalModel, self).__init__(config)
        self.text_embeddings = TextEmbeddings(text_config)
        if visual_config is not None:
            self.visual_embeddings = VisualEmbeddings(visual_config)
            self.visual_pooler = Pooler(visual_config)
        if audio_config is not None:
            self.audio_embeddings = VisualEmbeddings(audio_config)
            self.audio_pooler = Pooler(audio_config)
        self.encoder = MultimodalEncoder(config, text_config, visual_config, audio_config)
        self.text_pooler = Pooler(text_config)
        self.enable_video = visual_config is not None
        self.enable_audio = audio_config is not None

        self.apply(self.init_weights)

    def _prepare_attention_mask(self, attention_mask):
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, text_input_ids, video, video_attention_mask, audio, audio_attention_mask,
        text_token_type_ids=None, text_attention_mask=None, output_all_encoded_layers=True):

        if text_attention_mask is None:
            text_attention_mask = torch.ones_like(text_input_ids)
        if self.enable_video and video_attention_mask is None:
            video_attention_mask = torch.ones(video.size(0), video.size(1))
        if self.enable_audio and audio_attention_mask is None:
            audio_attention_mask = torch.ones(audio.size(0), audio.size(1))
        if text_token_type_ids is None:
            text_token_type_ids = torch.zeros_like(text_input_ids)

        extended_text_attention_mask = self._prepare_attention_mask(text_attention_mask)
        text_embedding_output = self.text_embeddings(text_input_ids, text_token_type_ids)

        if self.enable_video:
            extended_visual_attention_mask = self._prepare_attention_mask(video_attention_mask)
            visual_embedding_output = self.visual_embeddings(video)
        else:
            extended_visual_attention_mask = None
            visual_embedding_output = None

        if self.enable_audio:
            extended_audio_attention_mask = self._prepare_attention_mask(audio_attention_mask)
            audio_embedding_output = self.audio_embeddings(audio)
        else:
            extended_audio_attention_mask = None
            audio_embedding_output = None

        encoded_layers = self.encoder([text_embedding_output, visual_embedding_output, audio_embedding_output],
                                      [extended_text_attention_mask, extended_visual_attention_mask, extended_audio_attention_mask],
                                      output_all_encoded_layers=output_all_encoded_layers)
        all_output = encoded_layers[-1]
        pooled_text_output = self.text_pooler(all_output[0])
        if self.enable_video:
            pooled_visual_output = self.visual_pooler(all_output[1])
        else:
            pooled_visual_output = None
        if self.enable_audio:
            pooled_audio_output = self.audio_pooler(all_output[2])
        else:
            pooled_audio_output = None
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_text_output, pooled_visual_output, pooled_audio_output, #pooled_text_output, pooled_visual_output, pooled_audio_output