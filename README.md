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
## 搭建开发环境
#### Ubuntu系统
详细安装过程请参考：https://guo-pu.blog.csdn.net/article/details/109533526
```shell
# 安装依赖 
pip install -r requirements.txt
```
#### Windows系统
详细安装过程请参考：https://guo-pu.blog.csdn.net/article/details/108807165

## 执行

模型的权重文件需要转出模型H5文件，yolov4.weights 是的权重文件，由它生成yolov4.h5。

```shell
# 模型权重 `yolov4.weights` 转 `yolo4.h5`
python convertToH5.py --input_size 608 --min_score 0.3 --iou 0.5 --model_path model_data/yolov4.h5 --weights_path model_data/yolov4.weights

# 执行图片目标检测跟踪
python detect_image.py --video test.jpg --min_score 0.6 --model_yolo model_data/yolov4.h5 --model_feature model_data/mars-small128.pb

# 执行视频目标检测跟踪
python detect_video_tracker.py --video test.mp4 --min_score 0.6 --model_yolo model_data/yolov4.h5 --model_feature model_data/mars-small128.pb

# 执行摄像头目标检测跟踪
python detect_video_tracker.py --video 0 --min_score 0.6 --model_yolo model_data/yolov4.h5 --model_feature model_data/mars-small128.pb

```

### 补充说明
```text
min_score      设置置信度过滤，低于0.6置信度不会显示在图片中，能过滤低置信度目标物体；这个参数根据项目需求来设定
model_yolo     权重文件转为模型H5文件
model_feature  目标检测特征模型文件，如果是检测小物体的建议使用mars-small128.pb，如果是中大物体建议使用market1501.pb
```

### YOLOv4目标检测效果：
```text
在复杂的十字路口，有许多行人、车辆、自行车、交通灯被检测出来了：
  大家可以看到大部分的行人、汽车是被检测出来了，存在小部分没有被检测出来；哈哈看到广告牌上的汽车，也被识别为car 汽车（83%的把握）。如果是做特定场景的目标检测，建议大家后续采购特定场景的数据，重新训练网络，生成稳定且高精度的模型，保存权重文件，便于后续使用。
```
<img src="https://github.com/guo-pu/Deep-Sort-YOLOv4-master_V1.0/blob/main/test_picture/output_street.png" /><br/>
