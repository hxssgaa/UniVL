TRAIN_CSV="data/dstc10/dstc10_train.csv"
VAL_CSV="data/dstc10/dstc10_val.csv"
DATA_PATH="data/dstc10/dstc10_data.all.pickle"
VIDEO_FEATURES_PATH="data/dstc10/dstc10_videos_features_all.pickle"
AUDIO_FEATURES_PATH="data/dstc10/dstc10_audio_features_all.pickle"
INIT_MODEL="weight/univl.pretrained.bin"
# INIT_MODEL="ckpts/ckpt_youcook_caption/pytorch_model.bin.ans_gen.medcat.dialoghisonly"
OUTPUT_ROOT="ckpts"

python -m torch.distributed.launch --nproc_per_node=2 \
	main_task_caption.py \
	--do_train --num_thread_reader=8 \
	--epochs=20 --batch_size=32 \
	--n_display=100 \
	--train_csv ${TRAIN_CSV} \
	--val_csv ${VAL_CSV} \
	--data_path ${DATA_PATH} \
	--video_features_path ${VIDEO_FEATURES_PATH} \
	--audio_features_path ${AUDIO_FEATURES_PATH} \
	--output_dir ${OUTPUT_ROOT}/ckpt_charades_caption --bert_model bert-base-uncased \
	--do_lower_case --lr 3e-5 --max_words 256 --max_frames 60 \
	--batch_size_val 64 --visual_num_hidden_layers 6 \
	--decoder_num_hidden_layers 3 --video_dim 1024 --audio_dim 128 --stage_two \
	--init_model ${INIT_MODEL} --datatype default

