## 说明文档
本文档简要说明数据处理、模型构建、训练与测试方法。所有文件运行路径为`./`。

### 运行环境
Ubuntu1804LTS   
3090*2  
CUDA==11.0   
cuDNN==8.0.5   

```shell
(Anaconda)
conda create --name pytorch python==3.7
pip install -r requirements.txt
```
即可完成环境配置。

### 检测模型
我们使用了基于`yolov5`的改进版检测模型，其主要改进内容在于anchors，适配改动后的模型cfg文件存放为`./models/yolov5x.cfg`。

### 数据处理
1. 数据清洗
   经过我们人工清洗不合格的标注和大量重复的图片，得到新的xml文件于`./data/xml/`
2. 筛去小目标并转voc-xml为yolo-txt
   ```bash 
   python ./data/voc2yolo.py
   ```
3. 加噪声
   我们使用五种方式为图片添加随机噪声
   ```bash 
   python ./data/dataRefinforce.py
   ```
4. 
   ```bash
   cp ./data/clearImgs/* ./data/images/train/
   cp ./data/clearLabels/* ./data/labels/train/
   ```

### 模型训练
1. 预训练
```bash
nohup python -m torch.distributed.launch --nproc_per_node 2 train --batch-size 24 --img-size 512 --epochs 200 --data ./data/clear_voc.yaml  --weights '' --hyp ./data/hyp.scratch.yaml --notest --cache --multi-scale > pretrain.log &  
```
2. 微调 
```bash
nohup python -m torch.distributed.launch --nproc_per_node 2 train --batch-size 24 --img-size 512 --epochs 200 --data ./data/clear_voc.yaml  --weights ./runs/train/exp/weights/best.pt --hyp ./data/hyp.finetune.yaml --notest --cache --multi-scale > finetune.log &  
```

### 模型训练
```
python detect.py --weights ./weights/yolov5_clear_noise.pt  --source ..ships/a_test/pic/ --img-size 512 --conf-thres 0.25 --iou-thres 0.65 --device 0 --save-conf --save-result --augment 
```
可得到结果json于`./runs/detect/exp/`