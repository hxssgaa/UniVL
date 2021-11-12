TRAIN_CSV="data/simmc/train.csv"
VAL_CSV="data/simmc/devtest.csv"
DATA_PATH="data/simmc/train_data.no_obj.pickle"
# VIDEO_FEATURES_PATH="data/dstc10/dstc10_videos_features_all.pickle"
# INIT_MODEL="weight/univl.pretrained.bin"
INIT_MODEL="ckpts/ckpt_simmc/pytorch_model.bin.9"
OUTPUT_ROOT="ckpts"

python -m torch.distributed.launch --nproc_per_node=2 \
	main_task_caption.py \
	--do_eval --num_thread_reader=8 \
	--epochs=7 --batch_size=20 \
	--n_display=100 \
	--train_csv ${TRAIN_CSV} \
	--val_csv ${VAL_CSV} \
	--data_path ${DATA_PATH} \
	--output_dir ${OUTPUT_ROOT}/ckpt_simmc_devtest --bert_model bert-base-uncased \
	--do_lower_case --lr 3e-5 --max_words 512 --max_frames 60 \
	--batch_size_val 36 --visual_num_hidden_layers 3 \
	--decoder_num_hidden_layers 3 --video_dim 1024 --audio_dim 128 --stage_two --skip_visual --skip_audio \
	--init_model ${INIT_MODEL} --datatype default

