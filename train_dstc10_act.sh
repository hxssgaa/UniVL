DATATYPE="charades"
TRAIN_CSV="data/dstc10/dstc10_train.csv"
VAL_CSV="data/dstc10/dstc10_val.csv"
DATA_PATH="data/dstc10/dstc10_data_all.act.pickle"
FEATURES_PATH="data/dstc10/dstc10_videos_features_all.pickle"
INIT_MODEL="weight/univl.pretrained.bin"
OUTPUT_ROOT="ckpts"

python -m torch.distributed.launch --nproc_per_node=2 \
main_task_classify.py \
--do_train --num_thread_reader=16 \
--epochs=5 --batch_size=16 \
--n_display=100 \
--train_csv ${TRAIN_CSV} \
--val_csv ${VAL_CSV} \
--data_path ${DATA_PATH} \
--features_path ${FEATURES_PATH} \
--output_dir ${OUTPUT_ROOT}/ckpt_charades_act_classify --bert_model bert-base-uncased \
--do_lower_case --lr 3e-5 --max_words 11 --max_frames 60 --num_cands 158 \
--batch_size_val 64 --visual_num_hidden_layers 6 --train_sim_after_cross \
--datatype ${DATATYPE} --init_model ${INIT_MODEL}
