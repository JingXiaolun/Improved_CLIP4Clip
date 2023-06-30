# CLIP4Clip Improved Model Training on MSR-VTT DataSet

###################################################################### LooseType-seqTransf ##################################################################################
# use token fusion + keep rate (0.9) + meanP video output (1)
FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
python -m torch.distributed.launch --nproc_per_node=4 --master_port='29501' \
main_task_retrieval.py --do_train --num_thread_reader=0 \
--epochs=5 --batch_size=128 --n_display=50 \
--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
--features_path ${VIDEO_DATA_PATH}/video_frame_input \
--output_dir ../Model/Fusion_Removal/Fusion_Improved/keep_rate_0.7/meanP_video_output/msrvtt_retrieval_looseType_seqTransf \
--log_dir ../Log/Fusion_Removal/Fusion_Improved/keep_rate_0.7/meanP_video_output/msrvtt_retrieval_looseType_seqTransf \
--visualize_dir ../Visualize/Log/Fusion_Removal/Fusion_Improved/keep_rate_0.7/meanP_video_output/msrvtt_retrieval_looseType_seqTransf \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
--datatype msrvtt \
--expand_msrvtt_sentences  \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d \
--sim_header seqTransf --keep_rate 0.7 \
--pretrained_clip_name ViT-B/32

## use token fusion + keep rate (0.9) + cls video output (2)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29502' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Fusion_Removal/Fusion_Improved/keep_rate_0.9/cls_video_output/msrvtt_retrieval_looseType_seqTransf \
#--log_dir ../Log/Fusion_Removal/Fusion_Improved/keep_rate_0.9/cls_video_output/msrvtt_retrieval_looseType_seqTransf \
#--visualize_dir ../Visualize/Log/Fusion_Removal/Fusion_Improved/keep_rate_0.9/cls_video_output/msrvtt_retrieval_looseType_seqTransf \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d \
#--sim_header seqTransf --keep_rate 0.9 --video_output cls \
#--pretrained_clip_name ViT-B/32

## use token removal + keep rate (0.9) + meanP video output (3)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29503' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Fusion_Removal/Removal_Improved/keep_rate_0.9/meanP_video_output/msrvtt_retrieval_looseType_seqTransf \
#--log_dir ../Log/Fusion_Removal/Removal_Improved/keep_rate_0.9/meanP_video_output/msrvtt_retrieval_looseType_seqTransf \
#--visualize_dir ../Visualize/Log/Fusion_Removal/Removal_Improved/keep_rate_0.9/meanP_video_output/msrvtt_retrieval_looseType_seqTransf \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d \
#--sim_header seqTransf --keep_rate 0.9 --fuse_token 0 \
#--pretrained_clip_name ViT-B/32

## use token removal + keep rate (0.9) + cls video output (4)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29504' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Fusion_Removal/Removal_Improved/keep_rate_0.9/cls_video_output/msrvtt_retrieval_looseType_seqTransf \
#--log_dir ../Log/Fusion_Removal/Removal_Improved/keep_rate_0.9/cls_video_output/msrvtt_retrieval_looseType_seqTransf \
#--visualize_dir ../Visualize/Log/Fusion_Removal/Removal_Improved/keep_rate_0.9/cls_video_output/msrvtt_retrieval_looseType_seqTransf \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d \
#--sim_header seqTransf --keep_rate 0.9 --fuse_token 0 --video_output cls \
#--pretrained_clip_name ViT-B/32

