# CLIP4Clip Improved Model Training on LSMDC DataSet

############################################################### LooseType-seqTransf ########################################################################################
## use token fusion + keep rate + meanP video output
#FILE_DATA_PATH='../DataSet/LSMDC/data/compressed/split_file'
#VIDEO_DATA_PATH='../DataSet/LSMDC/data/compressed/video_frame_input'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29502' \
#main_task_retrieval.py --do_train --num_thread_reader=2 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--data_path ${FILE_DATA_PATH} \
#--features_path ${VIDEO_DATA_PATH} \
#--output_dir ../Model/Fusion_Removal/Fusion_Improved/keep_rate_0.6/meanP_video_output/lsmdc_retrieval_looseType_seqTransf \
#--log_dir ../Log/Fusion_Removal/Fusion_Improved/keep_rate_0.6/meanP_video_output/lsmdc_retrieval_looseType_seqTransf \
#--visualize_dir ../Visualize/Log/Fusion_Removal/Fusion_Improved/keep_rate_0.6/meanP_video_output/lsmdc_retrieval_looseType_seqTransf \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype lsmdc \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d \
#--sim_header seqTransf --keep_rate 0.6 \
#--pretrained_clip_name ViT-B/32

## use token removal + keep rate (0.8) + meanP video output
#FILE_DATA_PATH='../DataSet/LSMDC/data/compressed/split_file'
#VIDEO_DATA_PATH='../DataSet/LSMDC/data/compressed/video_frame_input'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29503' \
#main_task_retrieval.py --do_train --num_thread_reader=2 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--data_path ${FILE_DATA_PATH} \
#--features_path ${VIDEO_DATA_PATH} \
#--output_dir ../Model/Fusion_Removal/Removal_Improved/keep_rate_0.8/meanP_video_output/lsmdc_retrieval_looseType_seqTransf \
#--log_dir ../Log/Fusion_Removal/Removal_Improved/keep_rate_0.8/meanP_video_output/lsmdc_retrieval_looseType_seqTransf \
#--visualize_dir ../Visualize/Log/Fusion_Removal/Removal_Improved/keep_rate_0.8/meanP_video_output/lsmdc_retrieval_looseType_seqTransf \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype lsmdc \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d \
#--sim_header seqTransf --keep_rate 0.8 --fuse_token 0 \
#--pretrained_clip_name ViT-B/32

# use token removal + keep rate (0.7) + meanP video output
FILE_DATA_PATH='../DataSet/LSMDC/data/compressed/split_file'
VIDEO_DATA_PATH='../DataSet/LSMDC/data/compressed/video_frame_input'
python -m torch.distributed.launch --nproc_per_node=4 --master_port='29505' \
main_task_retrieval.py --do_train --num_thread_reader=2 \
--epochs=5 --batch_size=128 --n_display=50 \
--data_path ${FILE_DATA_PATH} \
--features_path ${VIDEO_DATA_PATH} \
--output_dir ../Model/Fusion_Removal/Removal_Improved/keep_rate_0.7/meanP_video_output/lsmdc_retrieval_looseType_seqTransf \
--log_dir ../Log/Fusion_Removal/Removal_Improved/keep_rate_0.7/meanP_video_output/lsmdc_retrieval_looseType_seqTransf \
--visualize_dir ../Visualize/Log/Fusion_Removal/Removal_Improved/keep_rate_0.7/meanP_video_output/lsmdc_retrieval_looseType_seqTransf \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
--datatype lsmdc \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d \
--sim_header seqTransf --keep_rate 0.7 --fuse_token 0 \
--pretrained_clip_name ViT-B/32
