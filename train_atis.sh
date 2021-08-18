TRAIN_CSV="data/atis/atis_train.csv"
VAL_CSV="data/atis/atis_test.csv"
DATA_PATH="data/atis/atis_data.pickle"
FEATURES_PATH="data/atis/atis_asr_features.pickle"
# INIT_MODEL="weight/univl.pretrained.bin"
INIT_MODEL="ckpts/ckpt_youcook_caption/pytorch_model.bin.4"
OUTPUT_ROOT="ckpts"

python -m torch.distributed.launch --nproc_per_node=1 \
	main_task_caption.py \
	--do_eval --num_thread_reader=1 \
	--epochs=20 --batch_size=16 \
	--n_display=100 \
	--train_csv ${TRAIN_CSV} \
	--val_csv ${VAL_CSV} \
	--data_path ${DATA_PATH} \
	--features_path ${FEATURES_PATH} \
	--output_dir ${OUTPUT_ROOT}/ckpt_youcook_caption --bert_model bert-base-uncased \
	--do_lower_case --lr 3e-5 --max_words 256 --max_frames 96 \
	--batch_size_val 64 --visual_num_hidden_layers 6 \
	--decoder_num_hidden_layers 3 --video_dim 256 --stage_two \
	--init_model ${INIT_MODEL}

