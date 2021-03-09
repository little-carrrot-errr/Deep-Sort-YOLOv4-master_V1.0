#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import cv2
import argparse
import numpy as np
from PIL import Image
from yolo4.yolo import YOLO4

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections

def set_gpus(gpu_index):
    if type(gpu_index) == list:
        gpu_index = ','.join(str(_) for _ in gpu_index)
    if type(gpu_index) ==int:
        gpu_index = str(gpu_index)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index

set_gpus(0)
# 1）视频文件进行目标检测 执行参数
# python detect_video_tracker.py --video test_folder.mp4 --min_score 0.6 --model_yolo model_data/yolov4.h5 --model_feature model_data/market1501.pb
# 2）摄像头实时数据目标检测 执行参数
# python detect_video_tracker.py --video 0 --min_score 0.6 --model_yolo model_data/yolov4.h5 --model_feature model_data/market1501.pb

# 外部参数配置
'''
video          输入视频文件、或摄像头的设备号（比如默认电脑的是 0 ）
min_score      设置置信度过滤，低于0.6置信度不会显示在图片中，能过滤低置信度目标物体；这个参数根据项目需求来设定
model_yolo     权重文件转为模型H5文件
model_feature  目标检测特征模型文件，如果是检测小物体的建议使用mars-small128.pb，如果是中大物体建议使用 market1501.pb
'''
parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, default='test_folder.mp4', help='data mp4 file.')
parser.add_argument('--min_score', type=float, default=0.6, help='displays the lowest tracking score.')
parser.add_argument('--model_yolo', type=str, default='model_data/yolo4.h5', help='Object detection model file.')
parser.add_argument('--model_feature', type=str, default='model_data/market1501.pb', help='target tracking model file.')
ARGS = parser.parse_args()

box_size = 2        # 边框大小
font_scale = 0.45    # 字体比例大小
from memory_profiler import profile


video_name = "test5_Trim3_50"
@profile(stream=open("outputData/logs/"+video_name+".log", 'a+'))
def run_encoder(encoder,frame, boxes):
    return encoder(frame, boxes)


# logger对象配置
import logging
from logging import handlers

def make_dir(make_dir_path):
    """如果文件夹不存在就创建"""
    path = make_dir_path.strip()
    if not os.path.exists(path):
        os.makedirs(path)
    return path


logger = logging.getLogger("outputData/logs/"+video_name+".log")
level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }
format_str = logging.Formatter('%(asctime)s - %(pathname)s - [line:%(lineno)d] - %(levelname)s > %(message)s')
logger.setLevel(level_relations.get("info"))
sh = logging.StreamHandler()
sh.setFormatter(format_str)
log_file_folder = os.path.abspath(os.path.join(os.path.dirname(__file__))) + os.sep + "logs" + os.sep
make_dir(log_file_folder)
log_file_str = log_file_folder + os.sep + "text.log"
th = handlers.TimedRotatingFileHandler(filename=log_file_str, when='H', encoding='utf-8')
th.setFormatter(format_str)
logger.addHandler(sh)
logger.addHandler(th)


if __name__ == '__main__':



    # Deep SORT 跟踪器
    encoder = generate_detections.create_box_encoder(ARGS.model_feature, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", ARGS.min_score, None)
    tracker = Tracker(metric)

    # 载入模型
    yolo = YOLO4(ARGS.model_yolo, ARGS.min_score)

    # 读取摄像头实时图像数据、或视频文件
    try:
        video = cv2.VideoCapture(int(ARGS.video))
    except:
        video = cv2.VideoCapture(ARGS.video)

    # 输出保存视频
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = video.get(cv2.CAP_PROP_FPS)
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    video_out = cv2.VideoWriter("outputData/"+video_name+".mp4", fourcc, fps, size)

    logger.info("视频高宽："+str(video.get(3))+":"+str(video.get(4)))
    logger.info("视频的帧数、帧速："+str(video.get(7))+":"+str(video.get(5)))
    n=0
    # 视频是否可以打开，进行逐帧识别绘制
    av_time = 0
    ccnt = 0
    while video.isOpened:

        # 视频读取图片帧
        ret, frame = video.read()
        if n% 10 != 0:
            n+=1
            continue

        if ret != True:
            # 任务完成后释放所有内容
            video.release()
            video_out.release()
            cv2.destroyAllWindows()
            logger.info("输入视频或摄像头数据无效！请检测文件名和路径、或摄像头是否正常工作。")
            break

        prev_time = time.time()

        start1_yolo = time.time()
        # 图片转换识别
        image = Image.fromarray(frame[...,::-1])  # bgr to rgb
        boxes, scores, classes, colors = yolo.detect_image(image)
        print("yolo v4 运行时间：",str(time.time()-start1_yolo))



        start = time.time()
        # 特征提取和检测对象列表
        features = run_encoder(encoder,frame, boxes)
        end = time.time() - start
        logger.info("结束时间："+str(end))
        # if av_time ==0:
        #     print("第一次：",str(end))
        #     av_time = 1
        #     continue
        # else:
        #     av_time +=end
        #     ccnt +=1
        #     continue


        detections = []
        for bbox, score, classe, color, feature in zip(boxes, scores, classes, colors, features):
            detections.append(Detection(bbox, score, classe, color, feature))

        # 运行非最大值抑制
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.score for d in detections])
        indices = preprocessing.non_max_suppression(boxes, 1.0, scores)
        detections = [detections[i] for i in indices]


        start_tr = time.time()
        # 追踪器刷新
        tracker.predict()
        tracker.update(detections)
        print("tracker 更新:",float(time.time() - start_tr))

        # 遍历绘制跟踪信息
        track_count = 0
        track_total = 0
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1: continue
            y1, x1, y2, x2 = np.array(track.to_tlbr(), dtype=np.int32)
            # cv2.rectangle(frame, (y1, x1), (y2, x2), (255, 255, 255), box_size//4)
            cv2.putText(
                frame, 
                "No. " + str(track.track_id),
                (y1-10, x1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (50, 0, 255),
                box_size//2,
                lineType=cv2.LINE_AA
            )
            if track.track_id > track_total: track_total = track.track_id
            track_count += 1

        # 遍历绘制检测对象信息
        totalCount = {}
        for det in detections:
            y1, x1, y2, x2 = np.array(det.to_tlbr(), dtype=np.int32)
            caption = '{} {:.2f}'.format(det.classe, det.score) if det.classe else det.score
            cv2.rectangle(frame, (y1, x1), (y2, x2), det.color, box_size)
            # 填充文字区
            text_size = cv2.getTextSize(caption, 0, font_scale, thickness=box_size)[0]
            cv2.rectangle(frame, (y1, x1), (y1 + text_size[0], x1 + text_size[1] + 8), det.color, -1)
            cv2.putText(
                frame,
                caption,
                (y1, x1 + text_size[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0),
                box_size//2,
                lineType=cv2.LINE_AA
            )
            # 统计物体数
            if det.classe not in totalCount: totalCount[det.classe] = 0
            totalCount[det.classe] += 1

        # 跟踪统计
        trackTotalStr = 'Track Total: %s' % str(track_total)
        cv2.putText(frame, trackTotalStr, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 0, 255), 2, cv2.LINE_AA)

        # 跟踪数量
        trackCountStr = 'Track Count: %s' % str(track_count)
        cv2.putText(frame, trackCountStr, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 0, 255), 2, cv2.LINE_AA)

        # 识别类数统计
        totalStr = ""
        for k in totalCount.keys(): totalStr += '%s: %d    ' % (k, totalCount[k])
        cv2.putText(frame, totalStr, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 0, 255), 2, cv2.LINE_AA)

        # 绘制时间
        curr_time = time.time()
        exec_time = curr_time - prev_time
        logger.info("识别耗时: %.2f s" %(exec_time))

        # 视频输出保存
        video_out.write(frame)
        # 绘制视频显示窗 命令行执行屏蔽呀
        cv2.namedWindow("video_reult", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("video_reult", frame)
        path = "outputData/test5_new2"
        if os.path.exists(path) is False:
            os.mkdir(path)
        cv2.imwrite(path+"/"+str(n)+".jpg", frame)
        n+=1
        # 退出窗口
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # av_time -= 1
    # 任务完成后释放所有内容
    video.release()
    video_out.release()
    cv2.destroyAllWindows()
