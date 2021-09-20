TRAIN_CSV="data/dstc10/dstc10_train.csv"
VAL_CSV="data/dstc10/dstc10_test.csv"
DATA_PATH="data/dstc10/dstc10_data.test.pickle"
VIDEO_FEATURES_PATH="data/dstc10/dstc10_test_video_features.pickle"
AUDIO_FEATURES_PATH="data/dstc10/dstc10_test_audio_features.pickle"
INIT_MODEL="ckpts/ckpt_charades_caption_2/pytorch_model.bin.5"
# INIT_MODEL="ckpts/ckpt_youcook_caption/pytorch_model.bin.ans_gen.medcat.dialoghisonly"
OUTPUT_ROOT="ckpts"

python -m torch.distributed.launch --nproc_per_node=2 \
	main_task_caption.py \
	--do_eval --num_thread_reader=8 \
	--epochs=7 --batch_size=36 \
	--n_display=100 \
	--train_csv ${TRAIN_CSV} \
	--val_csv ${VAL_CSV} \
	--data_path ${DATA_PATH} \
	--video_features_path ${VIDEO_FEATURES_PATH} \
	--audio_features_path ${AUDIO_FEATURES_PATH} \
	--output_dir ${OUTPUT_ROOT}/ckpt_charades_caption_test_2 --bert_model bert-base-uncased \
	--do_lower_case --lr 3e-5 --max_words 300 --max_frames 60 \
	--batch_size_val 64 --visual_num_hidden_layers 1 \
	--decoder_num_hidden_layers 3 --video_dim 1024 --audio_dim 128 --stage_two \
	--init_model ${INIT_MODEL} --datatype default

