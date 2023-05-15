## LooseType-MeanP (use squeeze excitation)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29502' \
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

## LooseType-MeanP (use expand excitation)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29503' \
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

## LooseType-MeanP (use squeeze excitation and aggregation)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
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
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header meanP \
#--se_block --se_type excitation_aggregation --reduction_ratio 4 \
#--pretrained_clip_name ViT-B/32

## LooseType-MeanP (use expand excitation and aggregation)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29501' \
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

## LooseType-MeanP (use squeeze excitation and expand aggregation)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29501' \
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

## LooseType-MeanP (use expand excitation and squeeze aggregation)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29502' \
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

## LooseType-seqLSTM (use squeeze excitation and aggregation)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29501' \
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

# LooseType-seqTransf (use squeeze excitation and aggregation)
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29501' \
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

