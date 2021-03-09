#! /usr/bin/env python
# -*- coding: utf-8 -*-

"目标检测主代码  使用YOLOv4 检测图片中的不同目标物体，默认权重支持识别80种目标"

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

# 执行命令  python detect_image.py --picture test_folder.jpg --min_score 0.6 --model_yolo model_data/yolov4.h5 --model_feature model_data/market1501.pb

# 外部参数配置
'''
picture        输入图片的名称( .jpg 或 .png 等格式的图片)
min_score      设置置信度过滤，低于0.6置信度不会显示在图片中，能过滤低置信度目标物体；这个参数根据项目需求来设定
model_yolo     权重文件转为模型H5文件
model_feature  目标检测特征模型文件，如果是检测小物体的建议使用mars-small128.pb，如果是中大物体建议使用market1501.pb
'''
parser = argparse.ArgumentParser()
parser.add_argument('--picture', type=str, default='test_folder.jpg', help='picture file.')
parser.add_argument('--min_score', type=float, default=0.6, help='Below this score (confidence level) is not displayed.')
parser.add_argument('--model_yolo', type=str, default='model_data/yolo4.h5', help='Object detection model file.')
parser.add_argument('--model_feature', type=str, default='model_data/market1501.pb', help='target tracking model file.')
ARGS = parser.parse_args()

box_size = 2        # 边框大小
font_scale = 0.45    # 字体比例大小

if __name__ == '__main__':
    # Deep SORT 跟踪器
    encoder = generate_detections.create_box_encoder(ARGS.model_feature, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", ARGS.min_score, None)
    tracker = Tracker(metric)

    # 载入模型
    yolo = YOLO4(ARGS.model_yolo, ARGS.min_score)

    # 读取图片
    frame = cv2.imread(ARGS.picture)

    # 图片转换识别
    image = Image.fromarray(frame)  # bgr to rgb
    boxes, scores, classes, colors = yolo.detect_image(image)

    # 特征提取和检测对象列表
    features = encoder(frame, boxes)
    detections = []
    for bbox, score, classe, color, feature in zip(boxes, scores, classes, colors, features):
        detections.append(Detection(bbox, score, classe, color, feature))

    # 运行非最大值抑制
    boxes = np.array([d.tlwh for d in detections])
    scores = np.array([d.score for d in detections])
    indices = preprocessing.non_max_suppression(boxes, 1.0, scores)
    detections = [detections[i] for i in indices]

    # 追踪器刷新
    tracker.predict()
    tracker.update(detections)

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
            (y1, x1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (220,20,60),
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
            font_scale, (50, 50, 50),
            box_size//2,
            lineType=cv2.LINE_AA
        )
        # 统计物体数
        if det.classe not in totalCount: totalCount[det.classe] = 0
        totalCount[det.classe] += 1

    # 跟踪统计
    trackTotalStr = 'Track Total: %s' % str(track_total)
    cv2.putText(frame, trackTotalStr, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 0, 255), 1, cv2.LINE_AA)

    # 跟踪数量
    trackCountStr = 'Track Count: %s' % str(track_count)
    cv2.putText(frame, trackCountStr, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 0, 255), 1, cv2.LINE_AA)

    # 识别类数统计
    totalStr = ""
    for k in totalCount.keys(): totalStr += '%s: %d    ' % (k, totalCount[k])
    cv2.putText(frame, totalStr, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 0, 255), 1, cv2.LINE_AA)

    # 保存检测后图片
    filename = "output" + ARGS.picture
    cv2.imwrite(filename, frame)

    # 显示目标检测效果
    cv2.namedWindow("video_reult", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("video_reult", frame)

    # waitkey代表读取键盘的输入，括号里的数字代表等待多长时间，单位ms。 0代表一直等待
    k = cv2.waitKey(0)
    if k == 27 & 0xFF == ord('q'):  # 键盘上Esc键的键值 或输入q键
        # 任务完成后释放所有内容
        cv2.destroyAllWindows()
