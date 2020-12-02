## 基于注意力机制的OCR超分蒸馏


### Quick Start

1.安装相关包
```
pip install -r requirements.txt
```
2.下载教师网络预训练模型          

3.运行

1)测试教师网络的对单张图片输入的文字检测效果       
```
python Test_craft.py
```

教师网络在文字检测数据集上的测试          
```
python TD_eval.py
```

2)训练好的学生网络RCAN_dat在单张图像上的超分测试           
```
python Test_RCAN_dat.py
```

```
python RCAN_dat_eval.py
```

4.训练










- Download the trained teacher models

 *Model name* | *Used datasets* | *Languages* | *Purpose* | *Model Link* |
 | :--- | :--- | :--- | :--- | :--- |
General | SynthText, IC13, IC17 | Eng + MLT | For general purpose | [Click](https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ)
IC15 | SynthText, IC15 | Eng | For IC15 only | [Click](https://drive.google.com/open?id=1i2R7UIUqmkUtF0jv_3MXTqmQ_9wuAnLf)
LinkRefiner | CTW1500 | - | Used with the General Model | [Click


