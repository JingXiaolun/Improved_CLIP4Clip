# Offical CLIP4Clip4 model training 
################################################################################ MSRVTT DataSet #######################################################################
### LooseType-seqTransf
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file/'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29501' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Fusion_Removal/Offical/msrvtt_retrieval_looseType_seqTransf \
#--log_dir ../Log/Fusion_Removal/Offical/msrvtt_retrieval_looseType_seqTransf \
#--visualize_dir ../Visualize/Log/Fusion_Removal/Offical/msrvtt_retrieval_looseType_seqTransf \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d \
#--sim_header seqTransf --keep_rate 1.0  \
#--pretrained_clip_name ViT-B/32

###################################################################### MSVD DataSet ##################################################################################
### LooseType-seqTransf
#FILE_DATA_PATH='../DataSet/MSVD/data/compressed/split_file'
#VIDEO_DATA_PATH='../DataSet/MSVD/data/compressed/video_frame_input'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29502' \
#main_task_retrieval.py --do_train --num_thread_reader=2 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--data_path ${FILE_DATA_PATH} \
#--features_path ${VIDEO_DATA_PATH} \
#--output_dir ../Model/Fusion_Removal/Offical/msvd_retrieval_looseType_seqTransf \
#--log_dir ../Log/Fusion_Removal/Offical/msvd_retrieval_looseType_seqTransf \
#--visualize_dir ../Visualize/Log/Fusion_Removal/Offical/msvd_retrieval_looseType_seqTransf \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msvd \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d \
#--sim_header seqTransf --keep_rate 1.0 \
#--pretrained_clip_name ViT-B/32

############################################ LSMDC ######################################
## LooseType-seqTransf
FILE_DATA_PATH='../DataSet/LSMDC/data/compressed/split_file'
VIDEO_DATA_PATH='../DataSet/LSMDC/data/compressed/video_frame_input'
python -m torch.distributed.launch --nproc_per_node=4 --master_port='29504' \
main_task_retrieval.py --do_train --num_thread_reader=2 \
--epochs=5 --batch_size=128 --n_display=50 \
--data_path ${FILE_DATA_PATH} \
--features_path ${VIDEO_DATA_PATH} \
--output_dir ../Model/Fusion_Removal/Offical/lsmdc_retrieval_looseType_seqTransf \
--log_dir ../Log/Fusion_Removal/Offical/lsmdc_retrieval_looseType_seqTransf \
--visualize_dir ../Visualize/Log/Fusion_Removal/Offical/lsmdc_retrieval_looseType_seqTransf \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
--datatype lsmdc \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d \
--sim_header seqTransf --keep_rate 1.0 \
--pretrained_clip_name ViT-B/32

############################################ ActivityNet ################################

############################################ DiDeMo ########################################
### LooseType-seqTransf
#FILE_DATA_PATH='../DataSet/DiDeMo/data/compressed/split_file'
#VIDEO_DATA_PATH='../DataSet/DiDeMo/data/compressed/split_video'
#python -m torch.distributed.launch --nproc_per_node=8 --master_port='29505' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=64 --n_display=50 \
#--data_path ${FILE_DATA_PATH} \
#--features_path ${VIDEO_DATA_PATH} \
#--output_dir ../Model/Fusion_Removal/Offical/didemo_retrieval_looseType_seqTransf \
#--log_dir ../Log/Fusion_Removal/Offical/didemo_retrieval_looseType_seqTransf \
#--visualize_dir ../Visualize/Log/Fusion_Removal/Offical/didemo_retrieval_looseType_seqTransf \
#--lr 1e-4 --max_words 64 --max_frames 64 --batch_size_val 16 \
#--datatype didemo \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d \
#--sim_header seqTransf --keep_rate 1.0 \
#--pretrained_clip_name ViT-B/32
