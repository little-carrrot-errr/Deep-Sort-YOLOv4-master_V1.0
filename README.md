# Deep SORT —— YOLO v4 目标检测跟踪

## 介绍

项目采用 `YOLO v4` 算法模型进行目标检测，使用 `Deep SORT` 目标跟踪算法。

支持系统：Windows系统、Ubuntu系统

**运行环境**

- Keras==2.4.3
- tensorflow-gpu==2.3.1
- opencv-python==4.4.0.44
- image==1.5.33
- NVIDIA GPU CUDA

## 目录结构

```text
deep-sort-yolov4

┌── deep_sort                        DeepSort目标跟踪算法
│   ├── detection.py
│   ├── generate_detections.py
│   ├── iou_matching.py
│   ├── kalman_filter.py
│   ├── linear_assignment.py
│   ├── nn_matching.py
│   ├── preprocessing.py
│   ├── track.py
│   └── tracker.py
├── model_data                       模型文件数据
│   ├── market1501.pb
│   ├── mars-small128.pb
│   ├── yolov4.h5
│   ├── yolov4.weights
│   └── README.md
├── test_picture                     目标检测测试图片
├── yolo4                            YOLOV4目标检测
│   ├── model.py
│   └── yolo.py
│─── convertToH5.py                  权重转换
│─── detect_image.py                 图片目标检测
│─── detect_video_tracker.py         视频文件、摄像头实时目标检测
│─── requirements.txt                运行环境依赖库的具体版本
│─── test.jpg                        
└─── README.md
```

## 执行

模型的权重文件需要转出模型H5文件，yolov4.weights 是的权重文件，由它生成yolov4.h5。

```shell
# 安装依赖 

【适合Ubuntu系统】
pip install -r requirements.txt
详细安装过程请参考：https://guo-pu.blog.csdn.net/article/details/109533526

【适合Windows系统】
详细安装过程请参考：https://guo-pu.blog.csdn.net/article/details/108807165


# 模型权重 `yolov4.weights` 转 `yolo4.h5`
python convertToH5.py --input_size 608 --min_score 0.3 --iou 0.5 --model_path model_data/yolov4.h5 --weights_path model_data/yolov4.weights

# 执行图片目标检测跟踪
python detect_image.py --video test.jpg --min_score 0.6 --model_yolo model_data/yolov4.h5 --model_feature model_data/mars-small128.pb

# 执行视频目标检测跟踪
python detect_video_tracker.py --video test.mp4 --min_score 0.6 --model_yolo model_data/yolov4.h5 --model_feature model_data/mars-small128.pb

# 执行摄像头目标检测跟踪
python detect_video_tracker.py --video 0 --min_score 0.6 --model_yolo model_data/yolov4.h5 --model_feature model_data/mars-small128.pb

```

