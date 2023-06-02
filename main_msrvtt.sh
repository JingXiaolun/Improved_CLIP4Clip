# CLIP4Clip Improved Model Training on MSR-VTT DataSet
###################################################################### LooseType-MeanP ##################################################################################
## use squeeze excitation (1)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29501' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Squeeze_Excitation_Improved/msrvtt_retrieval_looseType_meanP \
#--log_dir ../Log/Squeeze_Excitation_Improved/msrvtt_retrieval_looseType_meanP \
#--visualize_dir ../Visualize/Log/Squeeze_Excitation_Improved/msrvtt_retrieval_looseType_meanP \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header meanP \
#--se_block --se_type excitation --reduction_ratio 4 \
#--pretrained_clip_name ViT-B/32

## use expand excitation (2)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29502' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Expand_Excitation_Improved/msrvtt_retrieval_looseType_meanP \
#--log_dir ../Log/Expand_Excitation_Improved/msrvtt_retrieval_looseType_meanP \
#--visualize_dir ../Visualize/Log/Expand_Excitation_Improved/msrvtt_retrieval_looseType_meanP \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header meanP \
#--se_block --se_type excitation --reduction_ratio 0.25 \
#--pretrained_clip_name ViT-B/32

## use squeeze aggregation (3)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29503' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Squeeze_Aggregation_Improved/msrvtt_retrieval_looseType_meanP \
#--log_dir ../Log/Squeeze_Aggregation_Improved/msrvtt_retrieval_looseType_meanP \
#--visualize_dir ../Visualize/Log/Squeeze_Aggregation_Improved/msrvtt_retrieval_looseType_meanP \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header meanP \
#--se_block --se_type aggregation --reduction_ratio 4 \
#--pretrained_clip_name ViT-B/32

## use expand aggregation (4)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29504' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Expand_Aggregation_Improved/msrvtt_retrieval_looseType_meanP \
#--log_dir ../Log/Expand_Aggregation_Improved/msrvtt_retrieval_looseType_meanP \
#--visualize_dir ../Visualize/Log/Expand_Aggregation_Improved/msrvtt_retrieval_looseType_meanP \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header meanP \
#--se_block --se_type aggregation --reduction_ratio 0.25 \
#--pretrained_clip_name ViT-B/32

## use squeeze excitation and aggregation (5)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29505' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Squeeze_Excitation_Aggregation_Improved/msrvtt_retrieval_looseType_meanP \
#--log_dir ../Log/Squeeze_Excitation_Aggregation_Improved/msrvtt_retrieval_looseType_meanP \
#--visualize_dir ../Visualize/Log/Squeeze_Excitation_Aggregation_Improved/msrvtt_retrieval_looseType_meanP \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header meanP \
#--se_block --se_type excitation_aggregation --reduction_ratio 4 \
#--pretrained_clip_name ViT-B/32

## use squeeze excitation and aggregation --reduction_ratio=2 (5)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29505' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Squeeze_Excitation_Aggregation_Improved/msrvtt_retrieval_looseType_meanP_2 \
#--log_dir ../Log/Squeeze_Excitation_Aggregation_Improved/msrvtt_retrieval_looseType_meanP_2 \
#--visualize_dir ../Visualize/Log/Squeeze_Excitation_Aggregation_Improved/msrvtt_retrieval_looseType_meanP_2 \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header meanP \
#--se_block --se_type excitation_aggregation --reduction_ratio 2 \
#--pretrained_clip_name ViT-B/32

## use squeeze excitation and aggregation --reduction_ratio=3 (5)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29505' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Squeeze_Excitation_Aggregation_Improved/msrvtt_retrieval_looseType_meanP_3 \
#--log_dir ../Log/Squeeze_Excitation_Aggregation_Improved/msrvtt_retrieval_looseType_meanP_3 \
#--visualize_dir ../Visualize/Log/Squeeze_Excitation_Aggregation_Improved/msrvtt_retrieval_looseType_meanP_3 \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header meanP \
#--se_block --se_type excitation_aggregation --reduction_ratio 3 \
#--pretrained_clip_name ViT-B/32

## use squeeze excitation and aggregation --reduction_ratio=6 (5)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29505' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Squeeze_Excitation_Aggregation_Improved/msrvtt_retrieval_looseType_meanP_6 \
#--log_dir ../Log/Squeeze_Excitation_Aggregation_Improved/msrvtt_retrieval_looseType_meanP_6 \
#--visualize_dir ../Visualize/Log/Squeeze_Excitation_Aggregation_Improved/msrvtt_retrieval_looseType_meanP_6 \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header meanP \
#--se_block --se_type excitation_aggregation --reduction_ratio 6 \
#--pretrained_clip_name ViT-B/32

## use expand excitation and aggregation (6)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29506' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Expand_Excitation_Aggregation_Improved/msrvtt_retrieval_looseType_meanP \
#--log_dir ../Log/Expand_Excitation_Aggregation_Improved/msrvtt_retrieval_looseType_meanP \
#--visualize_dir ../Visualize/Log/Expand_Excitation_Aggregation_Improved/msrvtt_retrieval_looseType_meanP \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header meanP \
#--se_block --se_type excitation_aggregation --reduction_ratio 0.25 \
#--pretrained_clip_name ViT-B/32

## use squeeze excitation and expand aggregation (7)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29507' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Squeeze_Excitation_Expand_Aggregation_Improved/msrvtt_retrieval_looseType_meanP \
#--log_dir ../Log/Squeeze_Excitation_Expand_Aggregation_Improved/msrvtt_retrieval_looseType_meanP \
#--visualize_dir ../Visualize/Log/Squeeze_Excitation_Expand_Aggregation_Improved/msrvtt_retrieval_looseType_meanP \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header meanP \
#--se_block --se_type excitation_aggregation  --excitation_aggregation_type squeeze_expand --reduction_ratio 4 \
#--pretrained_clip_name ViT-B/32

## use expand excitation and squeeze aggregation (8)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29508' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Expand_Excitation_Squeeze_Aggregation_Improved/msrvtt_retrieval_looseType_meanP \
#--log_dir ../Log/Expand_Excitation_Squeeze_Aggregation_Improved/msrvtt_retrieval_looseType_meanP \
#--visualize_dir ../Visualize/Log/Expand_Excitation_Squeeze_Aggregation_Improved/msrvtt_retrieval_looseType_meanP \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header meanP \
#--se_block --se_type excitation_aggregation  --excitation_aggregation_type expand_squeeze --reduction_ratio 0.25 \
#--pretrained_clip_name ViT-B/32

###################################################################### LooseType-seqLSTM ##################################################################################
## use squeeze excitation (1)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29501' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Squeeze_Excitation_Improved/msrvtt_retrieval_looseType_seqLSTM \
#--log_dir ../Log/Squeeze_Excitation_Improved/msrvtt_retrieval_looseType_seqLSTM \
#--visualize_dir ../Visualize/Log/Squeeze_Excitation_Improved/msrvtt_retrieval_looseType_seqLSTM \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqLSTM \
#--se_block --se_type excitation --reduction_ratio 4 \
#--pretrained_clip_name ViT-B/32

## use expand excitation (2)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29502' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Expand_Excitation_Improved/msrvtt_retrieval_looseType_seqLSTM \
#--log_dir ../Log/Expand_Excitation_Improved/msrvtt_retrieval_looseType_seqLSTM \
#--visualize_dir ../Visualize/Log/Expand_Excitation_Improved/msrvtt_retrieval_looseType_seqLSTM \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqLSTM \
#--se_block --se_type excitation --reduction_ratio 0.25 \
#--pretrained_clip_name ViT-B/32

## use squeeze aggregation (3)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29503' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Squeeze_Aggregation_Improved/msrvtt_retrieval_looseType_seqLSTM \
#--log_dir ../Log/Squeeze_Aggregation_Improved/msrvtt_retrieval_looseType_seqLSTM \
#--visualize_dir ../Visualize/Log/Squeeze_Aggregation_Improved/msrvtt_retrieval_looseType_seqLSTM \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqLSTM \
#--se_block --se_type aggregation --reduction_ratio 4 \
#--pretrained_clip_name ViT-B/32

## use expand aggregation (4)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29504' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Expand_Aggregation_Improved/msrvtt_retrieval_looseType_seqLSTM \
#--log_dir ../Log/Expand_Aggregation_Improved/msrvtt_retrieval_looseType_seqLSTM \
#--visualize_dir ../Visualize/Log/Expand_Aggregation_Improved/msrvtt_retrieval_looseType_seqLSTM \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqLSTM \
#--se_block --se_type aggregation --reduction_ratio 0.25 \
#--pretrained_clip_name ViT-B/32

## use expand aggregation (4-2)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29504' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Expand_Aggregation_Improved/msrvtt_retrieval_looseType_seqLSTM_2 \
#--log_dir ../Log/Expand_Aggregation_Improved/msrvtt_retrieval_looseType_seqLSTM_2 \
#--visualize_dir ../Visualize/Log/Expand_Aggregation_Improved/msrvtt_retrieval_looseType_seqLSTM_2 \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqLSTM \
#--se_block --se_type aggregation --reduction_ratio 0.5 \
#--pretrained_clip_name ViT-B/32

## use expand aggregation (4-3)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29505' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Expand_Aggregation_Improved/msrvtt_retrieval_looseType_seqLSTM_3 \
#--log_dir ../Log/Expand_Aggregation_Improved/msrvtt_retrieval_looseType_seqLSTM_3 \
#--visualize_dir ../Visualize/Log/Expand_Aggregation_Improved/msrvtt_retrieval_looseType_seqLSTM_3 \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqLSTM \
#--se_block --se_type aggregation \
#--pretrained_clip_name ViT-B/32

## use expand aggregation (4-6)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29502' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Expand_Aggregation_Improved/msrvtt_retrieval_looseType_seqLSTM_6 \
#--log_dir ../Log/Expand_Aggregation_Improved/msrvtt_retrieval_looseType_seqLSTM_6 \
#--visualize_dir ../Visualize/Log/Expand_Aggregation_Improved/msrvtt_retrieval_looseType_seqLSTM_6 \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqLSTM \
#--se_block --se_type aggregation \
#--pretrained_clip_name ViT-B/32

## use squeeze excitation and aggregation (5)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29505' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Squeeze_Excitation_Aggregation_Improved/msrvtt_retrieval_looseType_seqLSTM \
#--log_dir ../Log/Squeeze_Excitation_Aggregation_Improved/msrvtt_retrieval_looseType_seqLSTM \
#--visualize_dir ../Visualize/Log/Squeeze_Excitation_Aggregation_Improved/msrvtt_retrieval_looseType_seqLSTM \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqLSTM \
#--se_block --se_type excitation_aggregation --reduction_ratio 4 \
#--pretrained_clip_name ViT-B/32

## use expand excitation and aggregation (6)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29506' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Expand_Excitation_Aggregation_Improved/msrvtt_retrieval_looseType_seqLSTM \
#--log_dir ../Log/Expand_Excitation_Aggregation_Improved/msrvtt_retrieval_looseType_seqLSTM \
#--visualize_dir ../Visualize/Log/Expand_Excitation_Aggregation_Improved/msrvtt_retrieval_looseType_seqLSTM \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqLSTM \
#--se_block --se_type excitation_aggregation --reduction_ratio 0.25 \
#--pretrained_clip_name ViT-B/32

## use squeeze excitation and expand aggregation (7)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29507' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Squeeze_Excitation_Expand_Aggregation_Improved/msrvtt_retrieval_looseType_seqLSTM \
#--log_dir ../Log/Squeeze_Excitation_Expand_Aggregation_Improved/msrvtt_retrieval_looseType_seqLSTM \
#--visualize_dir ../Visualize/Log/Squeeze_Excitation_Expand_Aggregation_Improved/msrvtt_retrieval_looseType_seqLSTM \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqLSTM \
#--se_block --se_type excitation_aggregation --excitation_aggregation_type squeeze_expand --reduction_ratio 4 \
#--pretrained_clip_name ViT-B/32

## use expand excitation and squeeze aggregation (8)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29508' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Expand_Excitation_Squeeze_Aggregation_Improved/msrvtt_retrieval_looseType_seqLSTM \
#--log_dir ../Log/Expand_Excitation_Squeeze_Aggregation_Improved/msrvtt_retrieval_looseType_seqLSTM \
#--visualize_dir ../Visualize/Log/Expand_Excitation_Squeeze_Aggregation_Improved/msrvtt_retrieval_looseType_seqLSTM \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqLSTM \
#--se_block --se_type excitation_aggregation --excitation_aggregation_type expand_squeeze --reduction_ratio 0.25 \
#--pretrained_clip_name ViT-B/32

## use squeeze excitation + seq + squeeze aggregation (9)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29509' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Squeeze_Excitation_Seq_Aggregation_Improved/msrvtt_retrieval_looseType_seqLSTM \
#--log_dir ../Log/Squeeze_Excitation_Seq_Aggregation_Improved/msrvtt_retrieval_looseType_seqLSTM \
#--visualize_dir ../Visualize/Log/Squeeze_Excitation_Seq_Aggregation_Improved/msrvtt_retrieval_looseType_seqLSTM \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqLSTM \
#--se_block --se_type excitation_seq_aggregation --reduction_ratio 4 \
#--pretrained_clip_name ViT-B/32

## use expand excitation + seq + expand aggregation (10)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29510' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Expand_Excitation_Seq_Aggregation_Improved/msrvtt_retrieval_looseType_seqLSTM \
#--log_dir ../Log/Expand_Excitation_Seq_Aggregation_Improved/msrvtt_retrieval_looseType_seqLSTM \
#--visualize_dir ../Visualize/Log/Expand_Excitation_Seq_Aggregation_Improved/msrvtt_retrieval_looseType_seqLSTM \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqLSTM \
#--se_block --se_type excitation_seq_aggregation --reduction_ratio 0.25 \
#--pretrained_clip_name ViT-B/32

## use squeeze excitation + seq + expand aggregation (11)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29511' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Squeeze_Excitation_Seq_Expand_Aggregation_Improved/msrvtt_retrieval_looseType_seqLSTM \
#--log_dir ../Log/Squeeze_Excitation_Seq_Expand_Aggregation_Improved/msrvtt_retrieval_looseType_seqLSTM \
#--visualize_dir ../Visualize/Log/Squeeze_Excitation_Seq_Expand_Aggregation_Improved/msrvtt_retrieval_looseType_seqLSTM \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqLSTM \
#--se_block --se_type excitation_seq_aggregation --excitation_seq_aggregation_type squeeze_expand --reduction_ratio 4 \
#--pretrained_clip_name ViT-B/32

## use expand excitation + seq + squeeze aggregation (12)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29512' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Expand_Excitation_Seq_Squeeze_Aggregation_Improved/msrvtt_retrieval_looseType_seqLSTM \
#--log_dir ../Log/Expand_Excitation_Seq_Squeeze_Aggregation_Improved/msrvtt_retrieval_looseType_seqLSTM \
#--visualize_dir ../Visualize/Log/Expand_Excitation_Seq_Squeeze_Aggregation_Improved/msrvtt_retrieval_looseType_seqLSTM \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqLSTM \
#--se_block --se_type excitation_seq_aggregation --excitation_seq_aggregation_type expand_squeeze --reduction_ratio 0.25 \
#--pretrained_clip_name ViT-B/32

## use squeeze excitation + seq + meanP (13)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29513' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Squeeze_Excitation_Seq_Improved/msrvtt_retrieval_looseType_seqLSTM \
#--log_dir ../Log/Squeeze_Excitation_Seq_Improved/msrvtt_retrieval_looseType_seqLSTM \
#--visualize_dir ../Visualize/Log/Squeeze_Excitation_Seq_Improved/msrvtt_retrieval_looseType_seqLSTM \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqLSTM \
#--se_block --se_type excitation_seq --reduction_ratio 4 \
#--pretrained_clip_name ViT-B/32

## use expand excitation + seq + meanP (14)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29514' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Expand_Excitation_Seq_Improved/msrvtt_retrieval_looseType_seqLSTM \
#--log_dir ../Log/Expand_Excitation_Seq_Improved/msrvtt_retrieval_looseType_seqLSTM \
#--visualize_dir ../Visualize/Log/Expand_Excitation_Seq_Improved/msrvtt_retrieval_looseType_seqLSTM \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqLSTM \
#--se_block --se_type excitation_seq --reduction_ratio 0.25 \
#--pretrained_clip_name ViT-B/32

###################################################################### LooseType-seqTransf ##################################################################################
## use squeeze excitation (1)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29501' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Squeeze_Excitation_Improved/msrvtt_retrieval_looseType_seqTransf \
#--log_dir ../Log/Squeeze_Excitation_Improved/msrvtt_retrieval_looseType_seqTransf \
#--visualize_dir ../Visualize/Log/Squeeze_Excitation_Improved/msrvtt_retrieval_looseType_seqTransf \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--se_block --se_type excitation --reduction_ratio 4 \
#--pretrained_clip_name ViT-B/32

## use expand excitation (2)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29502' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Expand_Excitation_Improved/msrvtt_retrieval_looseType_seqTransf \
#--log_dir ../Log/Expand_Excitation_Improved/msrvtt_retrieval_looseType_seqTransf \
#--visualize_dir ../Visualize/Log/Expand_Excitation_Improved/msrvtt_retrieval_looseType_seqTransf \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--se_block --se_type excitation --reduction_ratio 0.25 \
#--pretrained_clip_name ViT-B/32

## use squeeze aggregation (3)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29503' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Squeeze_Aggregation_Improved/msrvtt_retrieval_looseType_seqTransf \
#--log_dir ../Log/Squeeze_Aggregation_Improved/msrvtt_retrieval_looseType_seqTransf \
#--visualize_dir ../Visualize/Log/Squeeze_Aggregation_Improved/msrvtt_retrieval_looseType_seqTransf \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--se_block --se_type aggregation --reduction_ratio 4 \
#--pretrained_clip_name ViT-B/32

## use expand aggregation (4)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29504' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Expand_Aggregation_Improved/msrvtt_retrieval_looseType_seqTransf \
#--log_dir ../Log/Expand_Aggregation_Improved/msrvtt_retrieval_looseType_seqTransf \
#--visualize_dir ../Visualize/Log/Expand_Aggregation_Improved/msrvtt_retrieval_looseType_seqTransf \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--se_block --se_type aggregation --reduction_ratio 0.25 \
#--pretrained_clip_name ViT-B/32

## use expand aggregation (4-2)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29504' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Expand_Aggregation_Improved/msrvtt_retrieval_looseType_seqTransf_2 \
#--log_dir ../Log/Expand_Aggregation_Improved/msrvtt_retrieval_looseType_seqTransf_2 \
#--visualize_dir ../Visualize/Log/Expand_Aggregation_Improved/msrvtt_retrieval_looseType_seqTransf_2 \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--se_block --se_type aggregation --reduction_ratio 0.5 \
#--pretrained_clip_name ViT-B/32

## use expand aggregation (4-3)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29503' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Expand_Aggregation_Improved/msrvtt_retrieval_looseType_seqTransf_3 \
#--log_dir ../Log/Expand_Aggregation_Improved/msrvtt_retrieval_looseType_seqTransf_3 \
#--visualize_dir ../Visualize/Log/Expand_Aggregation_Improved/msrvtt_retrieval_looseType_seqTransf_3 \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--se_block --se_type aggregation \
#--pretrained_clip_name ViT-B/32

## use expand aggregation (4-6)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29502' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Expand_Aggregation_Improved/msrvtt_retrieval_looseType_seqTransf_6 \
#--log_dir ../Log/Expand_Aggregation_Improved/msrvtt_retrieval_looseType_seqTransf_6 \
#--visualize_dir ../Visualize/Log/Expand_Aggregation_Improved/msrvtt_retrieval_looseType_seqTransf_6 \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--se_block --se_type aggregation \
#--pretrained_clip_name ViT-B/32

## use squeeze excitation and aggregation (5)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29505' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Squeeze_Excitation_Aggregation_Improved/msrvtt_retrieval_looseType_seqTransf \
#--log_dir ../Log/Squeeze_Excitation_Aggregation_Improved/msrvtt_retrieval_looseType_seqTransf \
#--visualize_dir ../Visualize/Log/Squeeze_Excitation_Aggregation_Improved/msrvtt_retrieval_looseType_seqTransf \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--se_block --se_type excitation_aggregation --reduction_ratio 4 \
#--pretrained_clip_name ViT-B/32

## use expand excitation and aggregation (6)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29506' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Expand_Excitation_Aggregation_Improved/msrvtt_retrieval_looseType_seqTransf \
#--log_dir ../Log/Expand_Excitation_Aggregation_Improved/msrvtt_retrieval_looseType_seqTransf \
#--visualize_dir ../Visualize/Log/Expand_Excitation_Aggregation_Improved/msrvtt_retrieval_looseType_seqTransf \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--se_block --se_type excitation_aggregation --reduction_ratio 0.25 \
#--pretrained_clip_name ViT-B/32

## use squeeze excitation and expand aggregation (7)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29507' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Squeeze_Excitation_Expand_Aggregation_Improved/msrvtt_retrieval_looseType_seqTransf \
#--log_dir ../Log/Squeeze_Excitation_Expand_Aggregation_Improved/msrvtt_retrieval_looseType_seqTransf \
#--visualize_dir ../Visualize/Log/Squeeze_Excitation_Expand_Aggregation_Improved/msrvtt_retrieval_looseType_seqTransf \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--se_block --se_type excitation_aggregation --excitation_aggregation_type squeeze_expand --reduction_ratio 4 \
#--pretrained_clip_name ViT-B/32

## use expand excitation and squeeze aggregation (8)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29508' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Expand_Excitation_Squeeze_Aggregation_Improved/msrvtt_retrieval_looseType_seqTransf \
#--log_dir ../Log/Expand_Excitation_Squeeze_Aggregation_Improved/msrvtt_retrieval_looseType_seqTransf \
#--visualize_dir ../Visualize/Log/Expand_Excitation_Squeeze_Aggregation_Improved/msrvtt_retrieval_looseType_seqTransf \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--se_block --se_type excitation_aggregation --excitation_aggregation_type expand_squeeze --reduction_ratio 0.25 \
#--pretrained_clip_name ViT-B/32

## use squeeze excitation + seq + squeeze aggregation (9)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29509' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Squeeze_Excitation_Seq_Aggregation_Improved/msrvtt_retrieval_looseType_seqTransf \
#--log_dir ../Log/Squeeze_Excitation_Seq_Aggregation_Improved/msrvtt_retrieval_looseType_seqTransf \
#--visualize_dir ../Visualize/Log/Squeeze_Excitation_Seq_Aggregation_Improved/msrvtt_retrieval_looseType_seqTransf \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--se_block --se_type excitation_seq_aggregation --reduction_ratio 4 \
#--pretrained_clip_name ViT-B/32

## use expand excitation + seq + expand aggregation (10)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29510' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Expand_Excitation_Seq_Aggregation_Improved/msrvtt_retrieval_looseType_seqTransf \
#--log_dir ../Log/Expand_Excitation_Seq_Aggregation_Improved/msrvtt_retrieval_looseType_seqTransf \
#--visualize_dir ../Visualize/Log/Expand_Excitation_Seq_Aggregation_Improved/msrvtt_retrieval_looseType_seqTransf \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--se_block --se_type excitation_seq_aggregation --reduction_ratio 0.25 \
#--pretrained_clip_name ViT-B/32

## use squeeze excitation + seq + expand aggregation (11)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29511' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Squeeze_Excitation_Seq_Expand_Aggregation_Improved/msrvtt_retrieval_looseType_seqTransf \
#--log_dir ../Log/Squeeze_Excitation_Seq_Expand_Aggregation_Improved/msrvtt_retrieval_looseType_seqTransf \
#--visualize_dir ../Visualize/Log/Squeeze_Excitation_Seq_Expand_Aggregation_Improved/msrvtt_retrieval_looseType_seqTransf \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--se_block --se_type excitation_seq_aggregation --excitation_seq_aggregation_type squeeze_expand --reduction_ratio 4 \
#--pretrained_clip_name ViT-B/32

## use expand excitation + seq + squeeze aggregation (12)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29512' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Expand_Excitation_Seq_Squeeze_Aggregation_Improved/msrvtt_retrieval_looseType_seqTransf \
#--log_dir ../Log/Expand_Excitation_Seq_Squeeze_Aggregation_Improved/msrvtt_retrieval_looseType_seqTransf \
#--visualize_dir ../Visualize/Log/Expand_Excitation_Seq_Squeeze_Aggregation_Improved/msrvtt_retrieval_looseType_seqTransf \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--se_block --se_type excitation_seq_aggregation --excitation_seq_aggregation_type expand_squeeze --reduction_ratio 0.25 \
#--pretrained_clip_name ViT-B/32

## use squeeze excitation + seq + meanP (13)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29513' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Squeeze_Excitation_Seq_Improved/msrvtt_retrieval_looseType_seqTransf \
#--log_dir ../Log/Squeeze_Excitation_Seq_Improved/msrvtt_retrieval_looseType_seqTransf \
#--visualize_dir ../Visualize/Log/Squeeze_Excitation_Seq_Improved/msrvtt_retrieval_looseType_seqTransf \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--se_block --se_type excitation_seq --reduction_ratio 4 \
#--pretrained_clip_name ViT-B/32

## use expand excitation + seq + meanP (14)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29514' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Expand_Excitation_Seq_Improved/msrvtt_retrieval_looseType_seqTransf \
#--log_dir ../Log/Expand_Excitation_Seq_Improved/msrvtt_retrieval_looseType_seqTransf \
#--visualize_dir ../Visualize/Log/Expand_Excitation_Seq_Improved/msrvtt_retrieval_looseType_seqTransf \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--se_block --se_type excitation_seq --reduction_ratio 0.25 \
#--pretrained_clip_name ViT-B/32

###################################################################### TightType-tightTransf ##################################################################################
## squeeze excitation (1)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29501' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Squeeze_Excitation_Improved/msrvtt_retrieval_tightTransf \
#--log_dir ../Log/Squeeze_Excitation_Improved/msrvtt_retrieval_tightTransf \
#--visualize_dir ../Visualize/Log/Squeeze_Excitation_Improved/msrvtt_retrieval_tightTransf \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--linear_patch 2d --sim_header tightTransf \
#--se_block --reduction_ratio 4 
#--pretrained_clip_name ViT-B/32

### expand excitation (2)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29502' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Expand_Excitation_Improved/msrvtt_retrieval_tightTransf \
#--log_dir ../Log/Expand_Excitation_Improved/msrvtt_retrieval_tightTransf \
#--visualize_dir ../Visualize/Log/Expand_Excitation_Improved/msrvtt_retrieval_tightTransf \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--linear_patch 2d --sim_header tightTransf \
#--se_block --reduction_ratio 0.25 
#--pretrained_clip_name ViT-B/32

### expand excitation (2-2)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29503' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Expand_Excitation_Improved/msrvtt_retrieval_tightTransf_2 \
#--log_dir ../Log/Expand_Excitation_Improved/msrvtt_retrieval_tightTransf_2 \
#--visualize_dir ../Visualize/Log/Expand_Excitation_Improved/msrvtt_retrieval_tightTransf_2 \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--linear_patch 2d --sim_header tightTransf \
#--se_block --reduction_ratio 0.5
#--pretrained_clip_name ViT-B/32

### expand excitation (2-3)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29504' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Expand_Excitation_Improved/msrvtt_retrieval_tightTransf_3 \
#--log_dir ../Log/Expand_Excitation_Improved/msrvtt_retrieval_tightTransf_3 \
#--visualize_dir ../Visualize/Log/Expand_Excitation_Improved/msrvtt_retrieval_tightTransf_3 \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--linear_patch 2d --sim_header tightTransf \
#--se_block \
#--pretrained_clip_name ViT-B/32

### expand excitation (2-6)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29501' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Expand_Excitation_Improved/msrvtt_retrieval_tightTransf_6 \
#--log_dir ../Log/Expand_Excitation_Improved/msrvtt_retrieval_tightTransf_6 \
#--visualize_dir ../Visualize/Log/Expand_Excitation_Improved/msrvtt_retrieval_tightTransf_6 \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--linear_patch 2d --sim_header tightTransf \
#--se_block \
#--pretrained_clip_name ViT-B/32

#####################################################################################################################################################################################
#####################################################################################################################################################################################
#####################################################################################################################################################################################
#####################################################################################################################################################################################
#####################################################################################################################################################################################
#####################################################################################################################################################################################

# CLIP4Clip Improved Model Inference on MSR-VTT DataSet

###################################################################### LooseType-MeanP #############################################################################################
## use squeeze excitation (1)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python main_task_inference.py --do_eval --num_thread_reader=0  \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--load_dir ../Model/Squeeze_Excitation_Improved/msrvtt_retrieval_looseType_meanP \
#--max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--feature_framerate 1 --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header meanP \
#--se_block --se_type excitation --reduction_ratio 4 \
#--pretrained_clip_name ViT-B/32

## use expand excitation (2)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python main_task_inference.py --do_eval --num_thread_reader=0  \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--load_dir ../Model/Expand_Excitation_Improved/msrvtt_retrieval_looseType_meanP \
#--max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--feature_framerate 1 --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header meanP \
#--se_block --se_type excitation --reduction_ratio 0.25 \
#--pretrained_clip_name ViT-B/32

## use squeeze aggregation (3)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python main_task_inference.py --do_eval --num_thread_reader=0  \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--load_dir ../Model/Squeeze_Aggregation_Improved/msrvtt_retrieval_looseType_meanP \
#--max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--feature_framerate 1 --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header meanP \
#--se_block --se_type aggregation --reduction_ratio 4 \
#--pretrained_clip_name ViT-B/32

## use expand aggregation (4)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python main_task_inference.py --do_eval --num_thread_reader=0  \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--load_dir ../Model/Expand_Aggregation_Improved/msrvtt_retrieval_looseType_meanP \
#--max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--feature_framerate 1 --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header meanP \
#--se_block --se_type aggregation --reduction_ratio 0.25 \
#--pretrained_clip_name ViT-B/32

## use squeeze excitation and aggregation (5)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python main_task_inference.py --do_eval --num_thread_reader=0  \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--load_dir ../Model/Squeeze_Excitation_Aggregation_Improved/msrvtt_retrieval_looseType_meanP \
#--max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--feature_framerate 1 --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header meanP \
#--se_block --se_type excitation_aggregation --reduction_ratio 4 \
#--pretrained_clip_name ViT-B/32

## use squeeze excitation and aggregation (5-2)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python main_task_inference.py --do_eval --num_thread_reader=0  \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--load_dir ../Model/Squeeze_Excitation_Aggregation_Improved/msrvtt_retrieval_looseType_meanP_2 \
#--max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--feature_framerate 1 --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header meanP \
#--se_block --se_type excitation_aggregation --reduction_ratio 2 \
#--pretrained_clip_name ViT-B/32

## use squeeze excitation and aggregation (5-3)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python main_task_inference.py --do_eval --num_thread_reader=0  \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--load_dir ../Model/Squeeze_Excitation_Aggregation_Improved/msrvtt_retrieval_looseType_meanP_3 \
#--max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--feature_framerate 1 --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header meanP \
#--se_block --se_type excitation_aggregation --reduction_ratio 3 \
#--pretrained_clip_name ViT-B/32

## use squeeze excitation and aggregation (5-6)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python main_task_inference.py --do_eval --num_thread_reader=0  \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--load_dir ../Model/Squeeze_Excitation_Aggregation_Improved/msrvtt_retrieval_looseType_meanP_6 \
#--max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--feature_framerate 1 --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header meanP \
#--se_block --se_type excitation_aggregation --reduction_ratio 6 \
#--pretrained_clip_name ViT-B/32

## use expand excitation and aggregation (6)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python main_task_inference.py --do_eval --num_thread_reader=0  \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--load_dir ../Model/Expand_Excitation_Aggregation_Improved/msrvtt_retrieval_looseType_meanP \
#--max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--feature_framerate 1 --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header meanP \
#--se_block --se_type excitation_aggregation --reduction_ratio 0.25 \
#--pretrained_clip_name ViT-B/32

## use squeeze excitation and expand aggregation (7)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python main_task_inference.py --do_eval --num_thread_reader=0  \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--load_dir ../Model/Squeeze_Excitation_Expand_Aggregation_Improved/msrvtt_retrieval_looseType_meanP \
#--max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--feature_framerate 1 --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header meanP \
#--se_block --se_type excitation_aggregation  --excitation_aggregation_type squeeze_expand --reduction_ratio 4 \
#--pretrained_clip_name ViT-B/32

## use expand excitation and squeeze aggregation (8)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python main_task_inference.py --do_eval --num_thread_reader=0  \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--load_dir ../Model/Expand_Excitation_Squeeze_Aggregation_Improved/msrvtt_retrieval_looseType_meanP \
#--max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--feature_framerate 1 --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header meanP \
#--se_block --se_type excitation_aggregation  --excitation_aggregation_type expand_squeeze --reduction_ratio 0.25 \
#--pretrained_clip_name ViT-B/32

###################################################################### LooseType-seqLSTM ##################################################################################
## use squeeze excitation (1)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python main_task_inference.py --do_eval --num_thread_reader=0  \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--load_dir ../Model/Squeeze_Excitation_Improved/msrvtt_retrieval_looseType_seqLSTM \
#--max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--feature_framerate 1 --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqLSTM \
#--se_block --se_type excitation --reduction_ratio 4 \
#--pretrained_clip_name ViT-B/32

## use expand excitation (2)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python main_task_inference.py --do_eval --num_thread_reader=0  \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--load_dir ../Model/Expand_Excitation_Improved/msrvtt_retrieval_looseType_seqLSTM \
#--max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--feature_framerate 1 --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqLSTM \
#--se_block --se_type excitation --reduction_ratio 0.25 \
#--pretrained_clip_name ViT-B/32

## use squeeze excitation + seq + meanP (3)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python main_task_inference.py --do_eval --num_thread_reader=0  \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--load_dir ../Model/Squeeze_Excitation_Seq_Improved/msrvtt_retrieval_looseType_seqLSTM \
#--max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--feature_framerate 1 --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqLSTM \
#--se_block --se_type excitation_seq --reduction_ratio 4 \
#--pretrained_clip_name ViT-B/32

## use expand excitation + seq + meanP (4)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python main_task_inference.py --do_eval --num_thread_reader=0  \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--load_dir ../Model/Expand_Excitation_Seq_Improved/msrvtt_retrieval_looseType_seqLSTM \
#--max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--feature_framerate 1 --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqLSTM \
#--se_block --se_type excitation_seq --reduction_ratio 0.25 \
#--pretrained_clip_name ViT-B/32

## use squeeze aggregation (5)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python main_task_inference.py --do_eval --num_thread_reader=0  \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--load_dir ../Model/Squeeze_Aggregation_Improved/msrvtt_retrieval_looseType_seqLSTM \
#--max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--feature_framerate 1 --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqLSTM \
#--se_block --se_type aggregation --reduction_ratio 4 \
#--pretrained_clip_name ViT-B/32

## use expand aggregation (6)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python main_task_inference.py --do_eval --num_thread_reader=0  \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--load_dir ../Model/Expand_Aggregation_Improved/msrvtt_retrieval_looseType_seqLSTM \
#--max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--feature_framerate 1 --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqLSTM \
#--se_block --se_type aggregation --reduction_ratio 0.25 \
#--pretrained_clip_name ViT-B/32

## use expand aggregation of ratio 2 (6-2)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python main_task_inference.py --do_eval --num_thread_reader=0  \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--load_dir ../Model/Expand_Aggregation_Improved/msrvtt_retrieval_looseType_seqLSTM_2 \
#--max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--feature_framerate 1 --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqLSTM \
#--se_block --se_type aggregation --reduction_ratio 0.5 \
#--pretrained_clip_name ViT-B/32

## use expand aggregation of ratio 3 (6-3)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python main_task_inference.py --do_eval --num_thread_reader=0  \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--load_dir ../Model/Expand_Aggregation_Improved/msrvtt_retrieval_looseType_seqLSTM_3 \
#--max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--feature_framerate 1 --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqLSTM \
#--se_block --se_type aggregation \
#--pretrained_clip_name ViT-B/32

## use expand aggregation of ratio 6 (6-6)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python main_task_inference.py --do_eval --num_thread_reader=0  \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--load_dir ../Model/Expand_Aggregation_Improved/msrvtt_retrieval_looseType_seqLSTM_6 \
#--max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--feature_framerate 1 --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqLSTM \
#--se_block --se_type aggregation \
#--pretrained_clip_name ViT-B/32

## use squeeze excitation and aggregation (7)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python main_task_inference.py --do_eval --num_thread_reader=0  \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--load_dir ../Model/Squeeze_Excitation_Aggregation_Improved/msrvtt_retrieval_looseType_seqLSTM \
#--max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--feature_framerate 1 --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqLSTM \
#--se_block --se_type excitation_aggregation --reduction_ratio 4 \
#--pretrained_clip_name ViT-B/32

## use expand excitation and aggregation (8)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python main_task_inference.py --do_eval --num_thread_reader=0  \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--load_dir ../Model/Expand_Excitation_Aggregation_Improved/msrvtt_retrieval_looseType_seqLSTM \
#--max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--feature_framerate 1 --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqLSTM \
#--se_block --se_type excitation_aggregation --reduction_ratio 0.25 \
#--pretrained_clip_name ViT-B/32

## use squeeze excitation and expand aggregation (9)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python main_task_inference.py --do_eval --num_thread_reader=0  \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--load_dir ../Model/Squeeze_Excitation_Expand_Aggregation_Improved/msrvtt_retrieval_looseType_seqLSTM \
#--max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--feature_framerate 1 --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqLSTM \
#--se_block --se_type excitation_aggregation --excitation_aggregation_type squeeze_expand --reduction_ratio 4 \
#--pretrained_clip_name ViT-B/32

## use expand excitation and aggregation (10)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python main_task_inference.py --do_eval --num_thread_reader=0  \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--load_dir ../Model/Expand_Excitation_Squeeze_Aggregation_Improved/msrvtt_retrieval_looseType_seqLSTM \
#--max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--feature_framerate 1 --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqLSTM \
#--se_block --se_type excitation_aggregation --excitation_aggregation_type expand_squeeze --reduction_ratio 0.25 \
#--pretrained_clip_name ViT-B/32

###################################################################### LooseType-seqTransf ##################################################################################
## use squeeze excitation (1)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python main_task_inference.py --do_eval --num_thread_reader=0  \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--load_dir ../Model/Squeeze_Excitation_Improved/msrvtt_retrieval_looseType_seqTransf \
#--max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--feature_framerate 1 --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--se_block --se_type excitation --reduction_ratio 4 \
#--pretrained_clip_name ViT-B/32

## use expand excitation (2)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python main_task_inference.py --do_eval --num_thread_reader=0  \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--load_dir ../Model/Expand_Excitation_Improved/msrvtt_retrieval_looseType_seqTransf \
#--max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--feature_framerate 1 --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--se_block --se_type excitation --reduction_ratio 0.25 \
#--pretrained_clip_name ViT-B/32

## use squeeze excitation + seq + meanP (3)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python main_task_inference.py --do_eval --num_thread_reader=0  \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--load_dir ../Model/Squeeze_Excitation_Seq_Improved/msrvtt_retrieval_looseType_seqTransf \
#--max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--feature_framerate 1 --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--se_block --se_type excitation --reduction_ratio 4 \
#--pretrained_clip_name ViT-B/32

## use expand excitation + seq + meanP (4)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python main_task_inference.py --do_eval --num_thread_reader=0  \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--load_dir ../Model/Expand_Excitation_Seq_Improved/msrvtt_retrieval_looseType_seqTransf \
#--max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--feature_framerate 1 --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--se_block --se_type excitation --reduction_ratio 0.25 \
#--pretrained_clip_name ViT-B/32

## use squeeze aggregation (5)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python main_task_inference.py --do_eval --num_thread_reader=0  \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--load_dir ../Model/Squeeze_Aggregation_Improved/msrvtt_retrieval_looseType_seqTransf \
#--max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--feature_framerate 1 --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--se_block --se_type aggregation --reduction_ratio 4 \
#--pretrained_clip_name ViT-B/32

## use expand excitation (6)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python main_task_inference.py --do_eval --num_thread_reader=0  \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--load_dir ../Model/Expand_Aggregation_Improved/msrvtt_retrieval_looseType_seqTransf \
#--max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--feature_framerate 1 --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--se_block --se_type aggregation --reduction_ratio 0.25 \
#--pretrained_clip_name ViT-B/32

## use expand aggregation of ratio 2 (6-2)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python main_task_inference.py --do_eval --num_thread_reader=0  \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--load_dir ../Model/Expand_Aggregation_Improved/msrvtt_retrieval_looseType_seqTransf_2 \
#--max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--feature_framerate 1 --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--se_block --se_type aggregation --reduction_ratio 0.5 \
#--pretrained_clip_name ViT-B/32

## use expand aggregation of ratio 3 (6-3)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python main_task_inference.py --do_eval --num_thread_reader=0  \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--load_dir ../Model/Expand_Aggregation_Improved/msrvtt_retrieval_looseType_seqTransf_3 \
#--max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--feature_framerate 1 --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--se_block --se_type aggregation \
#--pretrained_clip_name ViT-B/32

## use expand aggregation of ratio 6 (6-6)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python main_task_inference.py --do_eval --num_thread_reader=0  \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--load_dir ../Model/Expand_Aggregation_Improved/msrvtt_retrieval_looseType_seqTransf_6 \
#--max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--feature_framerate 1 --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--se_block --se_type aggregation \
#--pretrained_clip_name ViT-B/32

## use squeeze excitation and aggregation (7)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python main_task_inference.py --do_eval --num_thread_reader=0  \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--load_dir ../Model/Squeeze_Excitation_Aggregation_Improved/msrvtt_retrieval_looseType_seqTransf \
#--max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--feature_framerate 1 --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--se_block --se_type excitation_aggregation --reduction_ratio 4 \
#--pretrained_clip_name ViT-B/32

## use expand excitation and aggregation (8)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python main_task_inference.py --do_eval --num_thread_reader=0  \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--load_dir ../Model/Expand_Excitation_Aggregation_Improved/msrvtt_retrieval_looseType_seqTransf \
#--max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--feature_framerate 1 --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--se_block --se_type excitation_aggregation --reduction_ratio 0.25 \
#--pretrained_clip_name ViT-B/32

## use squeeze excitation and expand aggregation (9)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python main_task_inference.py --do_eval --num_thread_reader=0  \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--load_dir ../Model/Squeeze_Excitation_Expand_Aggregation_Improved/msrvtt_retrieval_looseType_seqTransf \
#--max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--feature_framerate 1 --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--se_block --se_type excitation_aggregation --excitation_aggregation_type squeeze_expand --reduction_ratio 4 \
#--pretrained_clip_name ViT-B/32

## use expand excitation and squeeze aggregation (10)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python main_task_inference.py --do_eval --num_thread_reader=0  \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--load_dir ../Model/Expand_Excitation_Squeeze_Aggregation_Improved/msrvtt_retrieval_looseType_seqTransf \
#--max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--feature_framerate 1 --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--se_block --se_type excitation_aggregation --excitation_aggregation_type expand_squeeze --reduction_ratio 0.25 \
#--pretrained_clip_name ViT-B/32

###################################################################### TightType-tightTransf ##################################################################################
## squeeze excitation (1)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python main_task_inference.py --do_eval --num_thread_reader=0  \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--load_dir ../Model/Squeeze_Excitation_Improved/msrvtt_retrieval_tightTransf \
#--max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--feature_framerate 1 --slice_framepos 2 \
#--linear_patch 2d --sim_header tightTransf \
#--se_block --reduction_ratio 4 
#--pretrained_clip_name ViT-B/32

### expand excitation (2)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python main_task_inference.py --do_eval --num_thread_reader=0  \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--load_dir ../Model/Expand_Excitation_Improved/msrvtt_retrieval_tightTransf \
#--max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--feature_framerate 1 --slice_framepos 2 \
#--linear_patch 2d --sim_header tightTransf \
#--se_block --reduction_ratio 0.25 
#--pretrained_clip_name ViT-B/32

### expand excitation of ratio 2 (2-2)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python main_task_inference.py --do_eval --num_thread_reader=0  \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--load_dir ../Model/Expand_Excitation_Improved/msrvtt_retrieval_tightTransf_2 \
#--max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--feature_framerate 1 --slice_framepos 2 \
#--linear_patch 2d --sim_header tightTransf \
#--se_block --reduction_ratio 0.5 \
#--pretrained_clip_name ViT-B/32

### expand excitation of ratio 3 (2-3)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python main_task_inference.py --do_eval --num_thread_reader=0  \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--load_dir ../Model/Expand_Excitation_Improved/msrvtt_retrieval_tightTransf_3 \
#--max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--feature_framerate 1 --slice_framepos 2 \
#--linear_patch 2d --sim_header tightTransf \
#--se_block \
#--pretrained_clip_name ViT-B/32

### expand excitation of ratio 6 (2-6)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python main_task_inference.py --do_eval --num_thread_reader=0  \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--load_dir ../Model/Expand_Excitation_Improved/msrvtt_retrieval_tightTransf_6 \
#--max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--feature_framerate 1 --slice_framepos 2 \
#--linear_patch 2d --sim_header tightTransf \
#--se_block \
#--pretrained_clip_name ViT-B/32
