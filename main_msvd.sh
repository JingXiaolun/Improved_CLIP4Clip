####################################### LooseType-MeanP #####################################
## use squeeze excitation (1)
#FILE_DATA_PATH='../DataSet/MSVD/data/compressed/split_file'
#VIDEO_DATA_PATH='../DataSet/MSVD/data/compressed/video_frame_input'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29501' \
#main_task_retrieval.py --do_train --num_thread_reader=2 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--data_path ${FILE_DATA_PATH} \
#--features_path ${VIDEO_DATA_PATH} \
#--output_dir ../Model/Squeeze_Excitation_Improved/msvd_retrieval_looseType_meanP \
#--log_dir ../Log/Squeeze_Excitation_Improved/msvd_retrieval_looseType_meanP \
#--visualize_dir ../Visualize/Log/Squeeze_Excitation_Improved/msvd_retrieval_looseType_meanP \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msvd \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header meanP \
#--se_block --se_type excitation --reduction_ratio 4 \
#--pretrained_clip_name ViT-B/32

## use expand excitation (2) 
#FILE_DATA_PATH='../DataSet/MSVD/data/compressed/split_file'
#VIDEO_DATA_PATH='../DataSet/MSVD/data/compressed/video_frame_input'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29502' \
#main_task_retrieval.py --do_train --num_thread_reader=2 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--data_path ${FILE_DATA_PATH} \
#--features_path ${VIDEO_DATA_PATH} \
#--output_dir ../Model/Expand_Excitation_Improved/msvd_retrieval_looseType_meanP \
#--log_dir ../Log/Expand_Excitation_Improved/msvd_retrieval_looseType_meanP \
#--visualize_dir ../Visualize/Log/Expand_Excitation_Improved/msvd_retrieval_looseType_meanP \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msvd \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header meanP \
#--se_block --se_type excitation --reduction_ratio 0.25 \
#--pretrained_clip_name ViT-B/32

## use squeeze aggregation (3)
#FILE_DATA_PATH='../DataSet/MSVD/data/compressed/split_file'
#VIDEO_DATA_PATH='../DataSet/MSVD/data/compressed/video_frame_input'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29503' \
#main_task_retrieval.py --do_train --num_thread_reader=2 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--data_path ${FILE_DATA_PATH} \
#--features_path ${VIDEO_DATA_PATH} \
#--output_dir ../Model/Squeeze_Aggregation_Improved/msvd_retrieval_looseType_meanP \
#--log_dir ../Log/Squeeze_Aggregation_Improved/msvd_retrieval_looseType_meanP \
#--visualize_dir ../Visualize/Log/Squeeze_Aggregation_Improved/msvd_retrieval_looseType_meanP \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msvd \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header meanP \
#--se_block --se_type aggregation --reduction_ratio 4 \
#--pretrained_clip_name ViT-B/32

## use expand aggregation (4)
#FILE_DATA_PATH='../DataSet/MSVD/data/compressed/split_file'
#VIDEO_DATA_PATH='../DataSet/MSVD/data/compressed/video_frame_input'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29504' \
#main_task_retrieval.py --do_train --num_thread_reader=2 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--data_path ${FILE_DATA_PATH} \
#--features_path ${VIDEO_DATA_PATH} \
#--output_dir ../Model/Expand_Aggregation_Improved/msvd_retrieval_looseType_meanP \
#--log_dir ../Log/Expand_Aggregation_Improved/msvd_retrieval_looseType_meanP \
#--visualize_dir ../Visualize/Log/Expand_Aggregation_Improved/msvd_retrieval_looseType_meanP \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msvd \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header meanP \
#--se_block --se_type aggregation --reduction_ratio 0.25 \
#--pretrained_clip_name ViT-B/32

## use squeeze excitation and aggregation (5)
#FILE_DATA_PATH='../DataSet/MSVD/data/compressed/split_file'
#VIDEO_DATA_PATH='../DataSet/MSVD/data/compressed/video_frame_input'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29505' \
#main_task_retrieval.py --do_train --num_thread_reader=2 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--data_path ${FILE_DATA_PATH} \
#--features_path ${VIDEO_DATA_PATH} \
#--output_dir ../Model/Squeeze_Excitation_Aggregation_Improved/msvd_retrieval_looseType_meanP \
#--log_dir ../Log/Squeeze_Excitation_Aggregation_Improved/msvd_retrieval_looseType_meanP \
#--visualize_dir ../Visualize/Log/Squeeze_Excitation_Aggregation_Improved/msvd_retrieval_looseType_meanP \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msvd \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header meanP \
#--se_block --se_type excitation_aggregation --reduction_ratio 4 \
#--pretrained_clip_name ViT-B/32

## use expand excitation and aggregation (6)
#FILE_DATA_PATH='../DataSet/MSVD/data/compressed/split_file'
#VIDEO_DATA_PATH='../DataSet/MSVD/data/compressed/video_frame_input'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29506' \
#main_task_retrieval.py --do_train --num_thread_reader=2 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--data_path ${FILE_DATA_PATH} \
#--features_path ${VIDEO_DATA_PATH} \
#--output_dir ../Model/Expand_Excitation_Aggregation_Improved/msvd_retrieval_looseType_meanP \
#--log_dir ../Log/Expand_Excitation_Aggregation_Improved/msvd_retrieval_looseType_meanP \
#--visualize_dir ../Visualize/Log/Expand_Excitation_Aggregation_Improved/msvd_retrieval_looseType_meanP \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msvd \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header meanP \
#--se_block --se_type excitation_aggregation --reduction_ratio 0.25 \
#--pretrained_clip_name ViT-B/32

## use squeeze excitation and expand aggregation (7)
#FILE_DATA_PATH='../DataSet/MSVD/data/compressed/split_file'
#VIDEO_DATA_PATH='../DataSet/MSVD/data/compressed/video_frame_input'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29507' \
#main_task_retrieval.py --do_train --num_thread_reader=2 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--data_path ${FILE_DATA_PATH} \
#--features_path ${VIDEO_DATA_PATH} \
#--output_dir ../Model/Squeeze_Excitation_Expand_Aggregation_Improved/msvd_retrieval_looseType_meanP \
#--log_dir ../Log/Squeeze_Excitation_Expand_Aggregation_Improved/msvd_retrieval_looseType_meanP \
#--visualize_dir ../Visualize/Log/Squeeze_Excitation_Expand_Aggregation_Improved/msvd_retrieval_looseType_meanP \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msvd \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header meanP \
#--se_block --se_type excitation_aggregation --excitation_aggregation_type squeeze_expand --reduction_ratio 4 \
#--pretrained_clip_name ViT-B/32

## use expand excitation and squeeze aggregation (8)
#FILE_DATA_PATH='../DataSet/MSVD/data/compressed/split_file'
#VIDEO_DATA_PATH='../DataSet/MSVD/data/compressed/video_frame_input'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29508' \
#main_task_retrieval.py --do_train --num_thread_reader=2 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--data_path ${FILE_DATA_PATH} \
#--features_path ${VIDEO_DATA_PATH} \
#--output_dir ../Model/Expand_Excitation_Squeeze_Aggregation_Improved/msvd_retrieval_looseType_meanP \
#--log_dir ../Log/Expand_Excitation_Squeeze_Aggregation_Improved/msvd_retrieval_looseType_meanP \
#--visualize_dir ../Visualize/Log/Expand_Excitation_Squeeze_Aggregation_Improved/msvd_retrieval_looseType_meanP \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msvd \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header meanP \
#--se_block --se_type excitation_aggregation --excitation_aggregation_type expand_squeeze --reduction_ratio 0.25 \
#--pretrained_clip_name ViT-B/32

##################################################### LooseType-seqLSTM ######################################################## 
# use seqLSTM + squeeze aggregation (1)
FILE_DATA_PATH='../DataSet/MSVD/data/compressed/split_file'
VIDEO_DATA_PATH='../DataSet/MSVD/data/compressed/video_frame_input'
python -m torch.distributed.launch --nproc_per_node=4 --master_port='29501' \
main_task_retrieval.py --do_train --num_thread_reader=2 \
--epochs=5 --batch_size=128 --n_display=50 \
--data_path ${FILE_DATA_PATH} \
--features_path ${VIDEO_DATA_PATH} \
--output_dir ../Model/Squeeze_Aggregation_Improved/msvd_retrieval_looseType_seqLSTM \
--log_dir ../Log/Squeeze_Aggregation_Improved/msvd_retrieval_looseType_seqLSTM \
--visualize_dir ../Visualize/Log/Squeeze_Aggregation_Improved/msvd_retrieval_looseType_seqLSTM \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
--datatype msvd \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header seqLSTM \
--se_block --se_type aggregation --reduction_ratio 4 \
--pretrained_clip_name ViT-B/32

## use seqLSTM + expand aggregation (2)
#FILE_DATA_PATH='../DataSet/MSVD/data/compressed/split_file'
#VIDEO_DATA_PATH='../DataSet/MSVD/data/compressed/video_frame_input'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29502' \
#main_task_retrieval.py --do_train --num_thread_reader=2 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--data_path ${FILE_DATA_PATH} \
#--features_path ${VIDEO_DATA_PATH} \
#--output_dir ../Model/Expand_Aggregation_Improved/msvd_retrieval_looseType_seqLSTM \
#--log_dir ../Log/Expand_Aggregation_Improved/msvd_retrieval_looseType_seqLSTM \
#--visualize_dir ../Visualize/Log/Expand_Aggregation_Improved/msvd_retrieval_looseType_seqLSTM \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msvd \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqLSTM \
#--se_block --se_type aggregation --reduction_ratio 0.25 \
#--pretrained_clip_name ViT-B/32

## use seqLSTM + squeeze excitation and aggregation (3)
#FILE_DATA_PATH='../DataSet/MSVD/data/compressed/split_file'
#VIDEO_DATA_PATH='../DataSet/MSVD/data/compressed/video_frame_input'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29503' \
#main_task_retrieval.py --do_train --num_thread_reader=2 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--data_path ${FILE_DATA_PATH} \
#--features_path ${VIDEO_DATA_PATH} \
#--output_dir ../Model/Squeeze_Excitation_Aggregation_Improved/msvd_retrieval_looseType_seqLSTM \
#--log_dir ../Log/Squeeze_Excitation_Aggregation_Improved/msvd_retrieval_looseType_seqLSTM \
#--visualize_dir ../Visualize/Log/Squeeze_Excitation_Aggregation_Improved/msvd_retrieval_looseType_seqLSTM \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msvd \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqLSTM \
#--se_block --se_type excitation_aggregation --reduction_ratio 4 \
#--pretrained_clip_name ViT-B/32

## use seqLSTM + expand excitation and aggregation (4)
#FILE_DATA_PATH='../DataSet/MSVD/data/compressed/split_file'
#VIDEO_DATA_PATH='../DataSet/MSVD/data/compressed/video_frame_input'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29504' \
#main_task_retrieval.py --do_train --num_thread_reader=2 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--data_path ${FILE_DATA_PATH} \
#--features_path ${VIDEO_DATA_PATH} \
#--output_dir ../Model/Expand_Excitation_Aggregation_Improved/msvd_retrieval_looseType_seqLSTM \
#--log_dir ../Log/Expand_Excitation_Aggregation_Improved/msvd_retrieval_looseType_seqLSTM \
#--visualize_dir ../Visualize/Log/Expand_Excitation_Aggregation_Improved/msvd_retrieval_looseType_seqLSTM \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msvd \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqLSTM \
#--se_block --se_type excitation_aggregation --reduction_ratio 0.25 \
#--pretrained_clip_name ViT-B/32

## use seqLSTM + squeeze excitation and expand aggregation (5)
#FILE_DATA_PATH='../DataSet/MSVD/data/compressed/split_file'
#VIDEO_DATA_PATH='../DataSet/MSVD/data/compressed/video_frame_input'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29505' \
#main_task_retrieval.py --do_train --num_thread_reader=2 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--data_path ${FILE_DATA_PATH} \
#--features_path ${VIDEO_DATA_PATH} \
#--output_dir ../Model/Squeeze_Excitation_Expand_Aggregation_Improved/msvd_retrieval_looseType_seqLSTM \
#--log_dir ../Log/Squeeze_Excitation_Expand_Aggregation_Improved/msvd_retrieval_looseType_seqLSTM \
#--visualize_dir ../Visualize/Log/Squeeze_Excitation_Expand_Aggregation_Improved/msvd_retrieval_looseType_seqLSTM \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msvd \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqLSTM \
#--se_block --se_type excitation_aggregation --excitation_aggregation_type squeeze_expand --reduction_ratio 4 \
#--pretrained_clip_name ViT-B/32

## use seqLSTM + expand excitation and squeeze aggregation (6)
#FILE_DATA_PATH='../DataSet/MSVD/data/compressed/split_file'
#VIDEO_DATA_PATH='../DataSet/MSVD/data/compressed/video_frame_input'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29506' \
#main_task_retrieval.py --do_train --num_thread_reader=2 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--data_path ${FILE_DATA_PATH} \
#--features_path ${VIDEO_DATA_PATH} \
#--output_dir ../Model/Expand_Excitation_Squeeze_Aggregation_Improved/msvd_retrieval_looseType_seqLSTM \
#--log_dir ../Log/Expand_Excitation_Squeeze_Aggregation_Improved/msvd_retrieval_looseType_seqLSTM \
#--visualize_dir ../Visualize/Log/Expand_Excitation_Squeeze_Aggregation_Improved/msvd_retrieval_looseType_seqLSTM \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msvd \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqLSTM \
#--se_block --se_type excitation_aggregation --excitation_aggregation_type expand_squeeze --reduction_ratio 0.25 \
#--pretrained_clip_name ViT-B/32

## use squeeze excitation + seqLSTM + squeeze aggregation (7)
#FILE_DATA_PATH='../DataSet/MSVD/data/compressed/split_file'
#VIDEO_DATA_PATH='../DataSet/MSVD/data/compressed/video_frame_input'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29507' \
#main_task_retrieval.py --do_train --num_thread_reader=2 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--data_path ${FILE_DATA_PATH} \
#--features_path ${VIDEO_DATA_PATH} \
#--output_dir ../Model/Squeeze_Excitation_Seq_Aggregation_Improved/msvd_retrieval_looseType_seqLSTM \
#--log_dir ../Log/Squeeze_Excitation_Seq_Aggregation_Improved/msvd_retrieval_looseType_seqLSTM \
#--visualize_dir ../Visualize/Log/Squeeze_Excitation_Seq_Aggregation_Improved/msvd_retrieval_looseType_seqLSTM \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msvd \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqLSTM \
#--se_block --se_type excitation_seq_aggregation --reduction_ratio 4 \
#--pretrained_clip_name ViT-B/32

## use expand excitation + seqLSTM + expand aggregation (8)
#FILE_DATA_PATH='../DataSet/MSVD/data/compressed/split_file'
#VIDEO_DATA_PATH='../DataSet/MSVD/data/compressed/video_frame_input'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29508' \
#main_task_retrieval.py --do_train --num_thread_reader=2 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--data_path ${FILE_DATA_PATH} \
#--features_path ${VIDEO_DATA_PATH} \
#--output_dir ../Model/Expand_Excitation_Seq_Aggregation_Improved/msvd_retrieval_looseType_seqLSTM \
#--log_dir ../Log/Expand_Excitation_Seq_Aggregation_Improved/msvd_retrieval_looseType_seqLSTM \
#--visualize_dir ../Visualize/Log/Expand_Excitation_Seq_Aggregation_Improved/msvd_retrieval_looseType_seqLSTM \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msvd \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqLSTM \
#--se_block --se_type excitation_seq_aggregation --reduction_ratio 0.25 \
#--pretrained_clip_name ViT-B/32

## use squeeze excitation + seqLSTM + expand aggregation (9)
#FILE_DATA_PATH='../DataSet/MSVD/data/compressed/split_file'
#VIDEO_DATA_PATH='../DataSet/MSVD/data/compressed/video_frame_input'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29509' \
#main_task_retrieval.py --do_train --num_thread_reader=2 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--data_path ${FILE_DATA_PATH} \
#--features_path ${VIDEO_DATA_PATH} \
#--output_dir ../Model/Squeeze_Excitation_Seq_Expand_Aggregation_Improved/msvd_retrieval_looseType_seqLSTM \
#--log_dir ../Log/Squeeze_Excitation_Seq_Expand_Aggregation_Improved/msvd_retrieval_looseType_seqLSTM \
#--visualize_dir ../Visualize/Log/Squeeze_Excitation_Seq_Expand_Aggregation_Improved/msvd_retrieval_looseType_seqLSTM \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msvd \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqLSTM \
#--se_block --se_type excitation_seq_aggregation --excitation_seq_aggregation_type squeeze_expand --reduction_ratio 4 \
#--pretrained_clip_name ViT-B/32

## use expand excitation + seqLSTM + squeeze aggregation (10)
#FILE_DATA_PATH='../DataSet/MSVD/data/compressed/split_file'
#VIDEO_DATA_PATH='../DataSet/MSVD/data/compressed/video_frame_input'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29510' \
#main_task_retrieval.py --do_train --num_thread_reader=2 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--data_path ${FILE_DATA_PATH} \
#--features_path ${VIDEO_DATA_PATH} \
#--output_dir ../Model/Expand_Excitation_Seq_Squeeze_Aggregation_Improved/msvd_retrieval_looseType_seqLSTM \
#--log_dir ../Log/Expand_Excitation_Seq_Squeeze_Aggregation_Improved/msvd_retrieval_looseType_seqLSTM \
#--visualize_dir ../Visualize/Log/Expand_Excitation_Seq_Squeeze_Aggregation_Improved/msvd_retrieval_looseType_seqLSTM \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msvd \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqLSTM \
#--se_block --se_type excitation_seq_aggregation --excitation_seq_aggregation_type expand_squeeze --reduction_ratio 0.25 \
#--pretrained_clip_name ViT-B/32

##################################################### LooseType-seqTransf ######################################################## 
## use seqLSTM + squeeze aggregation (1)
#FILE_DATA_PATH='../DataSet/MSVD/data/compressed/split_file'
#VIDEO_DATA_PATH='../DataSet/MSVD/data/compressed/video_frame_input'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29501' \
#main_task_retrieval.py --do_train --num_thread_reader=2 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--data_path ${FILE_DATA_PATH} \
#--features_path ${VIDEO_DATA_PATH} \
#--output_dir ../Model/Squeeze_Aggregation_Improved/msvd_retrieval_looseType_seqTransf \
#--log_dir ../Log/Squeeze_Aggregation_Improved/msvd_retrieval_looseType_seqTransf \
#--visualize_dir ../Visualize/Log/Squeeze_Aggregation_Improved/msvd_retrieval_looseType_seqTransf \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msvd \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--se_block --se_type aggregation --reduction_ratio 4 \
#--pretrained_clip_name ViT-B/32

## use seqLSTM + expand aggregation (2)
#FILE_DATA_PATH='../DataSet/MSVD/data/compressed/split_file'
#VIDEO_DATA_PATH='../DataSet/MSVD/data/compressed/video_frame_input'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29502' \
#main_task_retrieval.py --do_train --num_thread_reader=2 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--data_path ${FILE_DATA_PATH} \
#--features_path ${VIDEO_DATA_PATH} \
#--output_dir ../Model/Expand_Aggregation_Improved/msvd_retrieval_looseType_seqTransf \
#--log_dir ../Log/Expand_Aggregation_Improved/msvd_retrieval_looseType_seqTransf \
#--visualize_dir ../Visualize/Log/Expand_Aggregation_Improved/msvd_retrieval_looseType_seqTransf \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msvd \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--se_block --se_type aggregation --reduction_ratio 0.25 \
#--pretrained_clip_name ViT-B/32

## use seqLSTM + squeeze excitation and aggregation (3)
#FILE_DATA_PATH='../DataSet/MSVD/data/compressed/split_file'
#VIDEO_DATA_PATH='../DataSet/MSVD/data/compressed/video_frame_input'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29503' \
#main_task_retrieval.py --do_train --num_thread_reader=2 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--data_path ${FILE_DATA_PATH} \
#--features_path ${VIDEO_DATA_PATH} \
#--output_dir ../Model/Squeeze_Excitation_Aggregation_Improved/msvd_retrieval_looseType_seqTransf \
#--log_dir ../Log/Squeeze_Excitation_Aggregation_Improved/msvd_retrieval_looseType_seqTransf \
#--visualize_dir ../Visualize/Log/Squeeze_Excitation_Aggregation_Improved/msvd_retrieval_looseType_seqTransf \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msvd \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--se_block --se_type excitation_aggregation --reduction_ratio 4 \
#--pretrained_clip_name ViT-B/32

## use seqLSTM + expand excitation and aggregation (4)
#FILE_DATA_PATH='../DataSet/MSVD/data/compressed/split_file'
#VIDEO_DATA_PATH='../DataSet/MSVD/data/compressed/video_frame_input'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29504' \
#main_task_retrieval.py --do_train --num_thread_reader=2 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--data_path ${FILE_DATA_PATH} \
#--features_path ${VIDEO_DATA_PATH} \
#--output_dir ../Model/Expand_Excitation_Aggregation_Improved/msvd_retrieval_looseType_seqTransf \
#--log_dir ../Log/Expand_Excitation_Aggregation_Improved/msvd_retrieval_looseType_seqTransf \
#--visualize_dir ../Visualize/Log/Expand_Excitation_Aggregation_Improved/msvd_retrieval_looseType_seqTransf \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msvd \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--se_block --se_type excitation_aggregation --reduction_ratio 0.25 \
#--pretrained_clip_name ViT-B/32

## use seqLSTM + squeeze excitation and expand aggregation (5)
#FILE_DATA_PATH='../DataSet/MSVD/data/compressed/split_file'
#VIDEO_DATA_PATH='../DataSet/MSVD/data/compressed/video_frame_input'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29505' \
#main_task_retrieval.py --do_train --num_thread_reader=2 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--data_path ${FILE_DATA_PATH} \
#--features_path ${VIDEO_DATA_PATH} \
#--output_dir ../Model/Squeeze_Excitation_Expand_Aggregation_Improved/msvd_retrieval_looseType_seqTransf \
#--log_dir ../Log/Squeeze_Excitation_Expand_Aggregation_Improved/msvd_retrieval_looseType_seqTransf \
#--visualize_dir ../Visualize/Log/Squeeze_Excitation_Expand_Aggregation_Improved/msvd_retrieval_looseType_seqTransf \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msvd \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--se_block --se_type excitation_aggregation --excitation_aggregation_type squeeze_expand --reduction_ratio 4 \
#--pretrained_clip_name ViT-B/32

## use seqLSTM + expand excitation and squeeze aggregation (6)
#FILE_DATA_PATH='../DataSet/MSVD/data/compressed/split_file'
#VIDEO_DATA_PATH='../DataSet/MSVD/data/compressed/video_frame_input'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29506' \
#main_task_retrieval.py --do_train --num_thread_reader=2 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--data_path ${FILE_DATA_PATH} \
#--features_path ${VIDEO_DATA_PATH} \
#--output_dir ../Model/Expand_Excitation_Squeeze_Aggregation_Improved/msvd_retrieval_looseType_seqTransf \
#--log_dir ../Log/Expand_Excitation_Squeeze_Aggregation_Improved/msvd_retrieval_looseType_seqTransf \
#--visualize_dir ../Visualize/Log/Expand_Excitation_Squeeze_Aggregation_Improved/msvd_retrieval_looseType_seqTransf \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msvd \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--se_block --se_type excitation_aggregation --excitation_aggregation_type expand_squeeze --reduction_ratio 0.25 \
#--pretrained_clip_name ViT-B/32

## use squeeze excitation + seqLSTM + squeeze aggregation (7)
#FILE_DATA_PATH='../DataSet/MSVD/data/compressed/split_file'
#VIDEO_DATA_PATH='../DataSet/MSVD/data/compressed/video_frame_input'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29507' \
#main_task_retrieval.py --do_train --num_thread_reader=2 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--data_path ${FILE_DATA_PATH} \
#--features_path ${VIDEO_DATA_PATH} \
#--output_dir ../Model/Squeeze_Excitation_Seq_Aggregation_Improved/msvd_retrieval_looseType_seqTransf \
#--log_dir ../Log/Squeeze_Excitation_Seq_Aggregation_Improved/msvd_retrieval_looseType_seqTransf \
#--visualize_dir ../Visualize/Log/Squeeze_Excitation_Seq_Aggregation_Improved/msvd_retrieval_looseType_seqTransf \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msvd \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--se_block --se_type excitation_seq_aggregation --reduction_ratio 4 \
#--pretrained_clip_name ViT-B/32

## use expand excitation + seqLSTM + expand aggregation (8)
#FILE_DATA_PATH='../DataSet/MSVD/data/compressed/split_file'
#VIDEO_DATA_PATH='../DataSet/MSVD/data/compressed/video_frame_input'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29508' \
#main_task_retrieval.py --do_train --num_thread_reader=2 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--data_path ${FILE_DATA_PATH} \
#--features_path ${VIDEO_DATA_PATH} \
#--output_dir ../Model/Expand_Excitation_Seq_Aggregation_Improved/msvd_retrieval_looseType_seqTransf \
#--log_dir ../Log/Expand_Excitation_Seq_Aggregation_Improved/msvd_retrieval_looseType_seqTransf \
#--visualize_dir ../Visualize/Log/Expand_Excitation_Seq_Aggregation_Improved/msvd_retrieval_looseType_seqTransf \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msvd \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--se_block --se_type excitation_seq_aggregation --reduction_ratio 0.25 \
#--pretrained_clip_name ViT-B/32

## use squeeze excitation + seqLSTM + expand aggregation (9)
#FILE_DATA_PATH='../DataSet/MSVD/data/compressed/split_file'
#VIDEO_DATA_PATH='../DataSet/MSVD/data/compressed/video_frame_input'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29509' \
#main_task_retrieval.py --do_train --num_thread_reader=2 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--data_path ${FILE_DATA_PATH} \
#--features_path ${VIDEO_DATA_PATH} \
#--output_dir ../Model/Squeeze_Excitation_Seq_Expand_Aggregation_Improved/msvd_retrieval_looseType_seqTransf \
#--log_dir ../Log/Squeeze_Excitation_Seq_Expand_Aggregation_Improved/msvd_retrieval_looseType_seqTransf \
#--visualize_dir ../Visualize/Log/Squeeze_Excitation_Seq_Expand_Aggregation_Improved/msvd_retrieval_looseType_seqTransf \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msvd \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--se_block --se_type excitation_seq_aggregation --excitation_seq_aggregation_type squeeze_expand --reduction_ratio 4 \
#--pretrained_clip_name ViT-B/32

## use expand excitation + seqLSTM + squeeze aggregation (10)
#FILE_DATA_PATH='../DataSet/MSVD/data/compressed/split_file'
#VIDEO_DATA_PATH='../DataSet/MSVD/data/compressed/video_frame_input'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29510' \
#main_task_retrieval.py --do_train --num_thread_reader=2 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--data_path ${FILE_DATA_PATH} \
#--features_path ${VIDEO_DATA_PATH} \
#--output_dir ../Model/Expand_Excitation_Seq_Squeeze_Aggregation_Improved/msvd_retrieval_looseType_seqTransf \
#--log_dir ../Log/Expand_Excitation_Seq_Squeeze_Aggregation_Improved/msvd_retrieval_looseType_seqTransf \
#--visualize_dir ../Visualize/Log/Expand_Excitation_Seq_Squeeze_Aggregation_Improved/msvd_retrieval_looseType_seqTransf \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msvd \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--se_block --se_type excitation_seq_aggregation --excitation_seq_aggregation_type expand_squeeze --reduction_ratio 0.25 \
#--pretrained_clip_name ViT-B/32
 
####################################################### TightType-tightTransf ##########################################################
## use squeeze excitation + tightTransf (1)
#FILE_DATA_PATH='../DataSet/MSVD/data/compressed/split_file'
#VIDEO_DATA_PATH='../DataSet/MSVD/data/compressed/video_frame_input'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29501' \
#main_task_retrieval.py --do_train --num_thread_reader=2 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--data_path ${FILE_DATA_PATH} \
#--features_path ${VIDEO_DATA_PATH} \
#--output_dir ../Model/Squeeze_Excitation_Improved/msvd_retrieval_tightTransf \
#--log_dir ../Log/Squeeze_Excitation_Improved/msvd_retrieval_tightTransf \
#--visualize_dir ../Visualize/Log/Squeeze_Excitation_Improved/msvd_retrieval_tightTransf \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msvd \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--linear_patch 2d --sim_header tightTransf \
#--se_block --reduction_ratio 4 
#--pretrained_clip_name ViT-B/32

## use expand excitation + tightTransf (2)
#FILE_DATA_PATH='../DataSet/MSVD/data/compressed/split_file'
#VIDEO_DATA_PATH='../DataSet/MSVD/data/compressed/video_frame_input'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29502' \
#main_task_retrieval.py --do_train --num_thread_reader=2 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--data_path ${FILE_DATA_PATH} \
#--features_path ${VIDEO_DATA_PATH} \
#--output_dir ../Model/Expand_Excitation_Improved/msvd_retrieval_tightTransf \
#--log_dir ../Log/Expand_Excitation_Improved/msvd_retrieval_tightTransf \
#--visualize_dir ../Visualize/Log/Expand_Excitation_Improved/msvd_retrieval_tightTransf \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msvd \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--linear_patch 2d --sim_header tightTransf \
#--se_block --reduction_ratio 0.25 
#--pretrained_clip_name ViT-B/32
