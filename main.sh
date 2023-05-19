# Offical CLIP4Clip4 model training 
################################################################################ MSRVTT DataSet #######################################################################
### LooseType-MeanP
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file/'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Offical/msrvtt_retrieval_looseType_meanP \
#--log_dir ../Log/Offical/msrvtt_retrieval_looseType_meanP \
#--visualize_dir ../Visualize/Log/Offical/msrvtt_retrieval_looseType_meanP \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header meanP \
#--pretrained_clip_name ViT-B/32

### LooseType-seqLSTM
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file/'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29502' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Offical/msrvtt_retrieval_looseType_seqLSTM \
#--log_dir ../Log/Offical/msrvtt_retrieval_looseType_seqLSTM \
#--visualize_dir ../Visualize/Log/Offical/msrvtt_retrieval_looseType_seqLSTM \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqLSTM \
#--pretrained_clip_name ViT-B/32

### LooseType-seqTransf
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file/'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Offical/msrvtt_retrieval_looseType_seqTransf \
#--log_dir ../Log/Offical/msrvtt_retrieval_looseType_seqTransf \
#--visualize_dir ../Visualize/Log/Offical/msrvtt_retrieval_looseType_seqTransf \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--pretrained_clip_name ViT-B/32

### TightType-tightTransf
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29502' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Offical/msrvtt_retrieval_tightTransf \
#--log_dir ../Log/Offical/msrvtt_retrieval_tightTransf \
#--visualize_dir ../Visualize/Log/Offical/msrvtt_retrieval_tightTransf \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--linear_patch 2d --sim_header tightTransf \
#--pretrained_clip_name ViT-B/32

###################################################################### MSVD DataSet ##################################################################################
### LooseType-MeanP
#FILE_DATA_PATH='../DataSet/MSVD/data/compressed/split_file'
#VIDEO_DATA_PATH='../DataSet/MSVD/data/compressed/video_frame_input'
#python -m torch.distributed.launch --nproc_per_node=4  --master_port='29502' \
#main_task_retrieval.py --do_train --num_thread_reader=2 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--data_path ${FILE_DATA_PATH} \
#--features_path ${VIDEO_DATA_PATH} \
#--output_dir ../Model/Offical/msvd_retrieval_looseType_meanP \
#--log_dir ../Log/Offical/msvd_retrieval_looseType_meanP \
#--visualize_dir ../Visualize/Log/Offical/msvd_retrieval_looseType_meanP \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msvd \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header meanP \
#--pretrained_clip_name ViT-B/32

## LooseType-seqLSTM
FILE_DATA_PATH='../DataSet/MSVD/data/compressed/split_file'
VIDEO_DATA_PATH='../DataSet/MSVD/data/compressed/video_frame_input'
python -m torch.distributed.launch --nproc_per_node=4 \
main_task_retrieval.py --do_train --num_thread_reader=2 \
--epochs=5 --batch_size=128 --n_display=50 \
--data_path ${FILE_DATA_PATH} \
--features_path ${VIDEO_DATA_PATH} \
--output_dir ../Model/Offical/msvd_retrieval_looseType_seqLSTM \
--log_dir ../Log/Offical/msvd_retrieval_looseType_seqLSTM \
--visualize_dir ../Visualize/Log/Offical/msvd_retrieval_looseType_seqLSTM \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
--datatype msvd \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header seqLSTM \
--pretrained_clip_name ViT-B/32

### LooseType-seqTransf
#FILE_DATA_PATH='../DataSet/MSVD/data/compressed/split_file'
#VIDEO_DATA_PATH='../DataSet/MSVD/data/compressed/video_frame_input'
#python -m torch.distributed.launch --nproc_per_node=4 \
#main_task_retrieval.py --do_train --num_thread_reader=2 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--data_path ${FILE_DATA_PATH} \
#--features_path ${VIDEO_DATA_PATH} \
#--output_dir ../Model/Offical/msvd_retrieval_looseType_seqTransf \
#--log_dir ../Log/Offical/msvd_retrieval_looseType_seqTransf \
#--visualize_dir ../Visualize/Log/Offical/msvd_retrieval_looseType_seqTransf \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msvd \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--pretrained_clip_name ViT-B/32

### TightType-tightTransf
#FILE_DATA_PATH='../DataSet/MSVD/data/compressed/split_file'
#VIDEO_DATA_PATH='../DataSet/MSVD/data/compressed/video_frame_input'
#python -m torch.distributed.launch --nproc_per_node=4 \
#main_task_retrieval.py --do_train --num_thread_reader=2 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--data_path ${FILE_DATA_PATH} \
#--features_path ${VIDEO_DATA_PATH} \
#--output_dir ../Model/Offical/msvd_retrieval_tightTransf \
#--log_dir ../Log/Offical/msvd_retrieval_tightTransf \
#--visualize_dir ../Visualize/Log/Offical/msvd_retrieval_tightTransf \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msvd \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--linear_patch 2d --sim_header tightTransf \
#--pretrained_clip_name ViT-B/32

############################################ LSMDC ######################################

############################################ ActivityNet ################################





######################################################################################################################################################################
######################################################################################################################################################################
######################################################################################################################################################################
######################################################################################################################################################################
# Improved CLIP4Clip4 model training (introduce se block)
################################################################################ MSRVTT DataSet #######################################################################
### LooseType-MeanP (add squeeze excitation and aggregation block)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file/'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Squeeze_Excitation_Aggregation_Improved/msrvtt_retrieval_looseType_meanP \
#--log_dir ../Log/Squeeze_Excitation_Aggregation_Improved/msrvtt_retrieval_looseType_meanP \
#--visualize_dir ../Visualize/Log/Squeeze_Excitation_Aggregation_Improved/msrvtt_retrieval_looseType_meanP \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 --reduction_ratio 4 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header meanP \
#--se_block  --se_type merge \
#--pretrained_clip_name ViT-B/32

### LooseType-MeanP (add expand excitation and aggregation block)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file/'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Expand_Excitation_Aggregation_Improved/msrvtt_retrieval_looseType_meanP \
#--log_dir ../Log/Expand_Excitation_Aggregation_Improved/msrvtt_retrieval_looseType_meanP \
#--visualize_dir ../Visualize/Log/Expand_Excitation_Aggregation_Improved/msrvtt_retrieval_looseType_meanP \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 --reduction_ratio 0.25 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header meanP \
#--se_block --se_type merge \
#--pretrained_clip_name ViT-B/32

### LooseType-seqLSTM (se prefix)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file/'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/SE_Improved/msrvtt_retrieval_looseType_seqLSTM_prefix \
#--log_dir ../Log/SE_Improved/msrvtt_retrieval_looseType_seqLSTM_prefix \
#--visualize_dir ../Visualize/Log/SE_Improved/msrvtt_retrieval_looseType_seqLSTM_prefix \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqLSTM \
#--se_block --se_pos prefix \
#--pretrained_clip_name ViT-B/32

### LooseType-seqLSTM (se suffix)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file/'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/SE_Improved/msrvtt_retrieval_looseType_seqLSTM_suffix \
#--log_dir ../Log/SE_Improved/msrvtt_retrieval_looseType_seqLSTM_suffix \
#--visualize_dir ../Visualize/Log/SE_Improved/msrvtt_retrieval_looseType_seqLSTM_suffix \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqLSTM \
#--se_block \
#--pretrained_clip_name ViT-B/32

### LooseType-seqTransf (se prefix)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file/'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/SE_Improved/msrvtt_retrieval_looseType_seqTransf_prefix \
#--log_dir ../Log/SE_Improved/msrvtt_retrieval_looseType_seqTransf_prefix \
#--visualize_dir ../Visualize/Log/SE_Improved/msrvtt_retrieval_looseType_seqTransf_prefix \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--se_block --se_pos prefix \
#--pretrained_clip_name ViT-B/32

### LooseType-seqTransf (se suffix)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file/'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/SE_Improved/msrvtt_retrieval_looseType_seqTransf_suffix \
#--log_dir ../Log/SE_Improved/msrvtt_retrieval_looseType_seqTransf_suffix \
#--visualize_dir ../Visualize/Log/SE_Improved/msrvtt_retrieval_looseType_seqTransf_suffix \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--se_block \
#--pretrained_clip_name ViT-B/32

### TightType-tightTransf (add se)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file/'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/SE_Improved/msrvtt_retrieval_tightTransf \
#--log_dir ../Log/SE_Improved/msrvtt_retrieval_tightTransf \
#--visualize_dir ../Visualize/Log/SE_Improved/msrvtt_retrieval_tightTransf \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--linear_patch 2d --sim_header tightTransf \
#--se_block \
#--pretrained_clip_name ViT-B/32

###################################################################### MSVD DataSet ##################################################################################
### LooseType-MeanP (add se)
#FILE_DATA_PATH='../DataSet/MSVD/data/compressed/split_file'
#VIDEO_DATA_PATH='../DataSet/MSVD/data/compressed/video_frame_input'
#python -m torch.distributed.launch --nproc_per_node=4 \
#main_task_retrieval.py --do_train --num_thread_reader=2 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--data_path ${FILE_DATA_PATH} \
#--features_path ${VIDEO_DATA_PATH} \
#--output_dir ../Model/SE_Improved/msvd_retrieval_looseType_meanP \
#--log_dir ../Log/SE_Improved/msvd_retrieval_looseType_meanP \
#--visualize_dir ../Visualize/Log/SE_Improved/msvd_retrieval_looseType_meanP \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msvd \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header meanP \
#--se_block \
#--pretrained_clip_name ViT-B/32

### LooseType-seqLSTM (se prefix)
#FILE_DATA_PATH='../DataSet/MSVD/data/compressed/split_file'
#VIDEO_DATA_PATH='../DataSet/MSVD/data/compressed/video_frame_input'
#python -m torch.distributed.launch --nproc_per_node=4 \
#main_task_retrieval.py --do_train --num_thread_reader=2 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--data_path ${FILE_DATA_PATH} \
#--features_path ${VIDEO_DATA_PATH} \
#--output_dir ../Model/SE_Improved/msvd_retrieval_looseType_seqLSTM_prefix \
#--log_dir ../Log/SE_Improved/msvd_retrieval_looseType_seqLSTM_prefix \
#--visualize_dir ../Visualize/Log/SE_Improved/msvd_retrieval_looseType_seqLSTM_prefix \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msvd \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqLSTM \
#--se_block --se_pos prefix \
#--pretrained_clip_name ViT-B/32

### LooseType-seqLSTM (se suffix)
#FILE_DATA_PATH='../DataSet/MSVD/data/compressed/split_file'
#VIDEO_DATA_PATH='../DataSet/MSVD/data/compressed/video_frame_input'
#python -m torch.distributed.launch --nproc_per_node=4 \
#main_task_retrieval.py --do_train --num_thread_reader=2 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--data_path ${FILE_DATA_PATH} \
#--features_path ${VIDEO_DATA_PATH} \
#--output_dir ../Model/SE_Improved/msvd_retrieval_looseType_seqLSTM_suffix \
#--log_dir ../Log/SE_Improved/msvd_retrieval_looseType_seqLSTM_suffix \
#--visualize_dir ../Visualize/Log/SE_Improved/msvd_retrieval_looseType_seqLSTM_suffix \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msvd \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqLSTM \
#--se_block \
#--pretrained_clip_name ViT-B/32

### LooseType-seqTransf (se prefix)
#FILE_DATA_PATH='../DataSet/MSVD/data/compressed/split_file'
#VIDEO_DATA_PATH='../DataSet/MSVD/data/compressed/video_frame_input'
#python -m torch.distributed.launch --nproc_per_node=4 \
#main_task_retrieval.py --do_train --num_thread_reader=2 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--data_path ${FILE_DATA_PATH} \
#--features_path ${VIDEO_DATA_PATH} \
#--output_dir ../Model/SE_Improved/msvd_retrieval_looseType_seqTransf_prefix \
#--log_dir ../Log/SE_Improved/msvd_retrieval_looseType_seqTransf_prefix \
#--visualize_dir ../Visualize/Log/SE_Improved/msvd_retrieval_looseType_seqTransf_prefix \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msvd \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--se_block --se_pos prefix \
#--pretrained_clip_name ViT-B/32

### LooseType-seqTransf (se suffix)
#FILE_DATA_PATH='../DataSet/MSVD/data/compressed/split_file'
#VIDEO_DATA_PATH='../DataSet/MSVD/data/compressed/video_frame_input'
#python -m torch.distributed.launch --nproc_per_node=4 \
#main_task_retrieval.py --do_train --num_thread_reader=2 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--data_path ${FILE_DATA_PATH} \
#--features_path ${VIDEO_DATA_PATH} \
#--output_dir ../Model/SE_Improved/msvd_retrieval_looseType_seqTransf_suffix \
#--log_dir ../Log/SE_Improved/msvd_retrieval_looseType_seqTransf_suffix \
#--visualize_dir ../Visualize/Log/SE_Improved/msvd_retrieval_looseType_seqTransf_suffix \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msvd \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--se_block \
#--pretrained_clip_name ViT-B/32

### TightType-tightTransf (add se)
#FILE_DATA_PATH='../DataSet/MSVD/data/compressed/split_file'
#VIDEO_DATA_PATH='../DataSet/MSVD/data/compressed/video_frame_input'
#python -m torch.distributed.launch --nproc_per_node=4 \
#main_task_retrieval.py --do_train --num_thread_reader=2 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--data_path ${FILE_DATA_PATH} \
#--features_path ${VIDEO_DATA_PATH} \
#--output_dir ../Model/SE_Improved/msvd_retrieval_tightTransf \
#--log_dir ../Log/SE_Improved/msvd_retrieval_tightTransf \
#--visualize_dir ../Visualize/Log/SE_Improved/msvd_retrieval_tightTransf \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msvd \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--linear_patch 2d --sim_header tightTransf \
#--se_block \
#--pretrained_clip_name ViT-B/32

############################################ LSMDC ######################################

############################################ ActivityNet ################################

