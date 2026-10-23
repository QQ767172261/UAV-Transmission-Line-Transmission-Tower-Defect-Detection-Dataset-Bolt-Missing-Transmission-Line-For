输电线输电塔缺陷检测数据集![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/064823465c7d48aca80e36035f801bf8.png)

，2068张，yolo，voc，coco三种标注方式
图像尺寸:640*640
类别数量:10类![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ac081faaa0e34b24b108416d7c1c974e.png)

训练集图像数量:1875; 验证集图像数量:137； 测试集图像数量:56
类别名称: 每一类图像数 ，每一类标注数![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/909ba4f52b624ec0bbd1207bc952d8fe.png)

Bend Part - 弯曲部件：195,220
Vegetation - 植被：302,588
Missing Plate - 缺失的板材：245,657
Barbed Wire Missing - 缺失的带刺铁丝：289,373
Bolt Missing - 缺失的螺栓：448,742
Foreign Material - 异物：267,320
Split Pin Missing - 缺失的开口销：316,361
Loose Part - 松动的部件：265,459
Position and Condition - 位置与状态：324,388
Missing Part - 缺失的部件（泛指）：16,17
image num: 2068![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d6a1617d8d9e48c0b12b61fd6b4e477d.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/97f2ad487b1543358d023c893b967f7a.png)
# 输电线输电塔缺陷检测数据集
## 数据集总信息表
|项目|详情|
| ---- | ---- |
|数据集名称|输电线输电塔缺陷检测数据集|
|总图像数量|2068张|
|图像分辨率|640×640|
|标注格式|YOLO、VOC、COCO 三种|
|类别总数|10类|
|训练集|1875张|
|验证集|137张|
|测试集|56张|

## 类别分布表
|类别英文|类别中文|包含该类图像数量|标注框总数|
| ---- | ---- | ---- | ---- |
|Bend Part|弯曲部件|195|220|
|Vegetation|植被|302|588|
|Missing Plate|缺失的板材|245|657|
|Barbed Wire Missing|缺失的带刺铁丝|289|373|
|Bolt Missing|缺失的螺栓|448|742|
|Foreign Material|异物|267|320|
|Split Pin Missing|缺失的开口销|316|361|
|Loose Part|松动的部件|265|459|
|Position and Condition|位置与状态|324|388|
|Missing Part|缺失的部件（泛指）|16|17|

## 标签列表
`Bend Part`、`Vegetation`、`Missing Plate`、`Barbed Wire Missing`、`Bolt Missing`、`Foreign Material`、`Split Pin Missing`、`Loose Part`、`Position and Condition`、`Missing Part`

## 数据集yaml配置文件 `power_tower_defect.yaml`
```yaml
path: ./power_tower_defect_dataset
train: images/train
val: images/val
test: images/test

names:
  0: Bend Part
  1: Vegetation
  2: Missing Plate
  3: Barbed Wire Missing
  4: Bolt Missing
  5: Foreign Material
  6: Split Pin Missing
  7: Loose Part
  8: Position and Condition
  9: Missing Part
```

## YOLOv11训练代码 train_power_tower.py
```python
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11n.pt")
    results = model.train(
        data="power_tower_defect.yaml",
        epochs=100,
        imgsz=640,
        batch=12,
        device=0,
        project="power_tower_output",
        name="yolo11_power_tower_defect",
        conf=0.25,
        iou=0.45
    )
```

## 推理测试代码 predict_power_tower.py
```python
from ultralytics import YOLO

model = YOLO("./power_tower_output/yolo11_power_tower_defect/weights/best.pt")

res = model.predict("test.jpg", save=True, conf=0.25)
for r in res:
    print(f"检测目标总数：{len(r.boxes)}")
    for box in r.boxes:
        cls_id = int(box.cls)
        cls_name = model.names[cls_id]
        conf = float(box.conf)
        xyxy = box.xyxy.tolist()[0]
        print(f"类别：{cls_name},置信度:{conf:.3f},坐标:{xyxy}")
```

## 环境安装命令
```bash
pip install ultralytics torch opencv-python
```

关键词：输电塔缺陷检测、无人机电力巡检、输电线异物识别、螺栓缺失检测、铁塔构件故障识别、YOLO目标检测、电力安全运维


