# MSRVTT Model Training 
FILE_DATA_PATH='../DataSet/MSRVTT/data/file/'
VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
python -m torch.distributed.launch --nproc_per_node=4 \
main_task_retrieval.py --do_train --num_thread_reader=0 \
--epochs=5 --batch_size=256 --n_display=10 \
--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
--features_path ${VIDEO_DATA_PATH}/video_frame_input \
--output_dir ../Model/msrvtt_retrieval_looseType \
--log_dir ../Log/msrvtt_retrieval_looseType \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
--datatype msrvtt \
--expand_msrvtt_sentences  \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--pretrained_clip_name ViT-B/32

