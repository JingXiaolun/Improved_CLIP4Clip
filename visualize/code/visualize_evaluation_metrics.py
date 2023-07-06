# author: jingxiaolun
# date: 2023.05.09
# description: draw curve of loss, R1, R5 and R10  

import os
import numpy as np
import matplotlib.pyplot as plt

################################### User Defined Function ###################################
def Metrics_Visualization(log_1_path, log_2_path, visualize_save_dir):
    # step1: load log_1 and log_2
    f_1 = open(log_1_path, 'r')
    lines_1 = [line.strip() for line in f_1.readlines()]

    f_2 = open(log_2_path, 'r')
    lines_2 = [line.strip() for line in f_2.readlines()]

    # step3: select relevant metrics
    ## select training loss vs epoch
    training_loss_1 = sorted([line for line in lines_1 if 'Train Loss' in line], key=lambda x: int(x.split('Epoch')[1][1]))
    training_loss_1 = [float(line.split('Train Loss: ')[1]) for line in training_loss_1]
        
    training_loss_2 = sorted([line for line in lines_2 if 'Train Loss' in line], key=lambda x: int(x.split('Epoch')[1][1]))
    training_loss_2 = [float(line.split('Train Loss: ')[1]) for line in training_loss_2]

    ## select Text2Video vs epoch
    eval_t2v_1 = sorted([line for line in lines_1 if '>>>  R@1' in line])
    ### R1
    eval_t2v_1_R1 = [float(line.split('>>> ')[1].split('-')[0].split(': ')[1]) for line in eval_t2v_1]
    ### R5
    eval_t2v_1_R5 = [float(line.split('>>> ')[1].split('-')[1].split(': ')[1]) for line in eval_t2v_1]
    ### R10
    eval_t2v_1_R10 = [float(line.split('>>> ')[1].split('-')[2].split(': ')[1]) for line in eval_t2v_1]
    ### MedianR
    eval_t2v_1_MedianR = [float(line.split('>>> ')[1].split('-')[3].split(': ')[1]) for line in eval_t2v_1]
    ### MeanR
    eval_t2v_1_MeanR = [float(line.split('>>> ')[1].split('-')[4].split(': ')[1]) for line in eval_t2v_1]

    eval_t2v_2 = sorted([line for line in lines_2 if '>>>  R@1' in line])
    ### R1
    eval_t2v_2_R1 = [float(line.split('>>> ')[1].split('-')[0].split(': ')[1]) for line in eval_t2v_2]
    ### R5
    eval_t2v_2_R5 = [float(line.split('>>> ')[1].split('-')[1].split(': ')[1]) for line in eval_t2v_2]
    ### R10
    eval_t2v_2_R10 = [float(line.split('>>> ')[1].split('-')[2].split(': ')[1]) for line in eval_t2v_2]
    ### MedianR
    eval_t2v_2_MedianR = [float(line.split('>>> ')[1].split('-')[3].split(': ')[1]) for line in eval_t2v_2]
    ### MeanR
    eval_t2v_2_MeanR = [float(line.split('>>> ')[1].split('-')[4].split(': ')[1]) for line in eval_t2v_2]

    ## select Video2Text vs epoch
    eval_v2t_1 = sorted([line for line in lines_1 if '>>>  V2T$R@1' in line])
    ### R1
    eval_v2t_1_R1 = [float(line.split('>>> ')[1].split('-')[0].split(': ')[1]) for line in eval_v2t_1]
    ### R5
    eval_v2t_1_R5 = [float(line.split('>>> ')[1].split('-')[1].split(': ')[1]) for line in eval_v2t_1]
    ### R10
    eval_v2t_1_R10 = [float(line.split('>>> ')[1].split('-')[2].split(': ')[1]) for line in eval_v2t_1]
    ### MedianR
    eval_v2t_1_MedianR = [float(line.split('>>> ')[1].split('-')[3].split(': ')[1]) for line in eval_v2t_1]
    ### MeanR
    eval_v2t_1_MeanR = [float(line.split('>>> ')[1].split('-')[4].split(': ')[1]) for line in eval_v2t_1]

    eval_v2t_2 = sorted([line for line in lines_2 if '>>>  V2T$R@1' in line])
    ### R1
    eval_v2t_2_R1 = [float(line.split('>>> ')[1].split('-')[0].split(': ')[1]) for line in eval_v2t_2]
    ### R5
    eval_v2t_2_R5 = [float(line.split('>>> ')[1].split('-')[1].split(': ')[1]) for line in eval_v2t_2]
    ### R10
    eval_v2t_2_R10 = [float(line.split('>>> ')[1].split('-')[2].split(': ')[1]) for line in eval_v2t_2]
    ### MedianR
    eval_v2t_2_MedianR = [float(line.split('>>> ')[1].split('-')[3].split(': ')[1]) for line in eval_v2t_2]
    ### MeanR
    eval_v2t_2_MeanR = [float(line.split('>>> ')[1].split('-')[4].split(': ')[1]) for line in eval_v2t_2]
        
    # step4: draw curve
    ## a. drawing training loss curve
    fig, ax = plt.subplots() 
    training_loss_figure_name = 'se_meanP_training_loss.png'
    training_loss_figure_save_path = visualize_save_dir + '/' + training_loss_figure_name

    x_axis = np.arange(1, len(training_loss_1) + 1)
    y_axis_1 = np.array(training_loss_1)
    y_axis_2 = np.array(training_loss_2)

    ax.plot(x_axis, y_axis_1, label='meanP', marker='s')
    ax.plot(x_axis, y_axis_2, label='se+meanP', marker='o')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss vs Training Epoch')
    ax.legend()

    plt.savefig(training_loss_figure_save_path)
    plt.close()

    ## b. draw eval metrics curve
    ### Text-to-Video
    fig, ax = plt.subplots(2, 3)
    text2video_metrics_figure_name = 'text2video_metrics.png'
    text2video_metrics_figure_save_path = visualize_save_dir + '/' + text2video_metrics_figure_name

    x_axis = np.arange(1, len(training_loss_1) + 1)
    y_axis_1 = [np.array(eval_t2v_1_R1), np.array(eval_t2v_2_R1)]
    y_axis_2 = [np.array(eval_t2v_1_R5), np.array(eval_t2v_2_R5)]
    y_axis_3 = [np.array(eval_t2v_1_R10), np.array(eval_t2v_2_R10)]
    y_axis_4 = [np.array(eval_t2v_1_MedianR), np.array(eval_t2v_2_MedianR)]
    y_axis_5 = [np.array(eval_t2v_1_MeanR), np.array(eval_t2v_2_MeanR)]

    ax[0, 0].plot(x_axis, y_axis_1[0], label='meanP', marker='o')
    ax[0, 0].plot(x_axis, y_axis_1[1], label='se+meanP', marker='s')
    ax[0, 0].set_xlabel('Epoch')
    ax[0, 0].set_ylabel('R@1', fontsize=10)

    ax[0, 1].plot(x_axis, y_axis_2[0], marker='o')
    ax[0, 1].plot(x_axis, y_axis_2[1], marker='s')
    ax[0, 1].set_xlabel('Epoch')
    ax[0, 1].set_ylabel('R@5', fontsize=10)

    ax[0, 2].plot(x_axis, y_axis_3[0], marker='o')
    ax[0, 2].plot(x_axis, y_axis_3[1], marker='s')
    ax[0, 2].set_xlabel('Epoch')
    ax[0, 2].set_ylabel('R@10', fontsize=10)

    ax[1, 0].plot(x_axis, y_axis_4[0], marker='o')
    ax[1, 0].plot(x_axis, y_axis_4[1], marker='s')
    ax[1, 0].set_xlabel('Epoch')
    ax[1, 0].set_ylabel('MR', fontsize=10)
    
    ax[1, 1].plot(x_axis, y_axis_4[0], marker='o')
    ax[1, 1].plot(x_axis, y_axis_4[1], marker='s')
    ax[1, 1].set_xlabel('Epoch')
    ax[1, 1].set_ylabel('MdR', fontsize=10)

    ax[1, 2].plot(x_axis, y_axis_5[0], marker='o')
    ax[1, 2].plot(x_axis, y_axis_5[1], marker='s')
    ax[1, 2].set_xlabel('Epoch')
    ax[1, 2].set_ylabel('MnR', fontsize=10)

    fig.legend(frameon=True, loc='upper right', fontsize='small')
    fig.suptitle('Text-to-Video Metrics')
    fig.tight_layout()
    plt.savefig(text2video_metrics_figure_save_path)
    plt.close()

    ### Video-to-Text
    fig, ax = plt.subplots(2, 3)
    video2text_metrics_figure_name = 'video2text_metrics.png'
    video2text_metrics_figure_save_path = visualize_save_dir + '/' + video2text_metrics_figure_name

    x_axis = np.arange(1, len(training_loss_1) + 1)
    y_axis_1 = [np.array(eval_v2t_1_R1), np.array(eval_v2t_2_R1)]
    y_axis_2 = [np.array(eval_v2t_1_R5), np.array(eval_v2t_2_R5)]
    y_axis_3 = [np.array(eval_v2t_1_R10), np.array(eval_v2t_2_R10)]
    y_axis_4 = [np.array(eval_v2t_1_MedianR), np.array(eval_v2t_2_MedianR)]
    y_axis_5 = [np.array(eval_v2t_1_MeanR), np.array(eval_v2t_2_MeanR)]

    ax[0, 0].plot(x_axis, y_axis_1[0], label='meanP', marker='o')
    ax[0, 0].plot(x_axis, y_axis_1[1], label='se+meanP', marker='s')
    ax[0, 0].set_xlabel('Epoch')
    ax[0, 0].set_ylabel('R@1', fontsize=10)

    ax[0, 1].plot(x_axis, y_axis_2[0], marker='o')
    ax[0, 1].plot(x_axis, y_axis_2[1], marker='s')
    ax[0, 1].set_xlabel('Epoch')
    ax[0, 1].set_ylabel('R@5', fontsize=10)

    ax[0, 2].plot(x_axis, y_axis_3[0], marker='o')
    ax[0, 2].plot(x_axis, y_axis_3[1], marker='s')
    ax[0, 2].set_xlabel('Epoch')
    ax[0, 2].set_ylabel('R@10', fontsize=10)

    ax[1, 0].plot(x_axis, y_axis_4[0], marker='o')
    ax[1, 0].plot(x_axis, y_axis_4[1], marker='s')
    ax[1, 0].set_xlabel('Epoch')
    ax[1, 0].set_ylabel('MR', fontsize=10)
    
    ax[1, 1].plot(x_axis, y_axis_4[0], marker='o')
    ax[1, 1].plot(x_axis, y_axis_4[1], marker='s')
    ax[1, 1].set_xlabel('Epoch')
    ax[1, 1].set_ylabel('MdR', fontsize=10)

    ax[1, 2].plot(x_axis, y_axis_5[0], marker='o')
    ax[1, 2].plot(x_axis, y_axis_5[1], marker='s')
    ax[1, 2].set_xlabel('Epoch')
    ax[1, 2].set_ylabel('MnR', fontsize=10)

    fig.legend(frameon=True, loc='upper right', fontsize='small')
    fig.suptitle('Video-to-Text Metrics')
    fig.tight_layout()
    plt.savefig(video2text_metrics_figure_save_path)
    plt.close()

############################################################### Main Function #######################################################################################################
if __name__ == '__main__':
    # step1: define log_1_path and log_2_path
    log_1_path = '../../Log/Offical/msrvtt_retrieval_looseType_meanP/2023-05-08-21-18-54-log.txt'
    log_2_path = '../../Log/SE_Improved/msrvtt_retrieval_looseType_meanP/2023-05-08-12-05-24-log.txt'

    visualize_save_dir = '../../Visualize/Figure'
    if not os.path.exists(visualize_save_dir):
        os.makedirs(visualize_save_dir)

    # step2: implement visualization
    Metrics_Visualization(log_1_path, log_2_path, visualize_save_dir)



