############################################################### CLIP4Clip Improved Model Training on MSR-VTT DataSet #######################################################
############################################################### CLIP4Clip Improved Model Training on MSR-VTT DataSet #######################################################
############################################################### CLIP4Clip Improved Model Training on MSR-VTT DataSet #######################################################
############################################################### CLIP4Clip Improved Model Training on MSR-VTT DataSet #######################################################

###################################################################### LooseType-meanP ##################################################################################
# kmeans cluster 
FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
python -m torch.distributed.launch --nproc_per_node=4 --master_port='29501' \
main_task_retrieval.py --do_train --num_thread_reader=0 \
--epochs=5 --batch_size=128 --n_display=50 \
--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
--features_path ${VIDEO_DATA_PATH}/video_frame_input \
--output_dir ../Model/Frame_Cluster/Kmeans_Improved_new/msrvtt_retrieval_looseType_meanP \
--log_dir ../Log/Frame_Cluster/Kmeans_Improved_new/msrvtt_retrieval_looseType_meanP \
--visualize_dir ../Visualize/Log/Frame_Cluster/Kmeans_Improved_new/msrvtt_retrieval_looseType_meanP \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
--datatype msrvtt \
--expand_msrvtt_sentences  \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--cluster_method kmeans --max_cluster_num 10 \
--pretrained_clip_name ViT-B/32

###################################################################### LooseType-seqLSTM ##################################################################################
## kmeans cluster 
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29502' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Frame_Cluster/Kmeans_Improved/msrvtt_retrieval_looseType_seqLSTM \
#--log_dir ../Log/Frame_Cluster/Kmeans_Improved/msrvtt_retrieval_looseType_seqLSTM \
#--visualize_dir ../Visualize/Log/Frame_Cluster/Kmeans_Improved/msrvtt_retrieval_looseType_seqLSTM \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqLSTM \
#--cluster_method kmeans --cluster_num 4 \
#--pretrained_clip_name ViT-B/32

###################################################################### LooseType-seqTransf ##################################################################################
## kmeans cluster 
#FILE_DATA_PATH='../DataSet/MSRVTT/data/file'
#VIDEO_DATA_PATH='../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='29503' \
#main_task_retrieval.py --do_train --num_thread_reader=0 \
#--epochs=5 --batch_size=128 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/video_frame_input \
#--output_dir ../Model/Frame_Cluster/Kmeans_Improved/msrvtt_retrieval_looseType_seqTransf \
#--log_dir ../Log/Frame_Cluster/Kmeans_Improved/msrvtt_retrieval_looseType_seqTransf \
#--visualize_dir ../Visualize/Log/Frame_Cluster/Kmeans_Improved/msrvtt_retrieval_looseType_seqTransf \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
#--datatype msrvtt \
#--expand_msrvtt_sentences  \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--cluster_method kmeans --cluster_num 4 \
#--pretrained_clip_name ViT-B/32
