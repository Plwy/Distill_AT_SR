## 基于注意力机制的OCR超分蒸馏


### Quick Start

#### 1.安装相关包

```
pip install -r requirements.txt
```
#### 2.下载教师网络预训练模型   
 *Model name* | *Used datasets* | *Languages* | *Purpose* | *Model Link* |
 | :--- | :--- | :--- | :--- | :--- |
General | SynthText, IC13, IC17 | Eng + MLT | For general purpose | [Click](https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ)
IC15 | SynthText, IC15 | Eng | For IC15 only | [Click](https://drive.google.com/open?id=1i2R7UIUqmkUtF0jv_3MXTqmQ_9wuAnLf)
LinkRefiner | CTW1500 | - | Used with the General Model | [Click](https://drive.google.com/open?id=1XSaFwBkOaFOdtk4Ane3DFyJGPRw6v5bO)       

将下好的模型放在目录 ckpts/teacher_ckpts/CRAFT_model/ 下

#### 3.运行

1) 训练

设置bash脚本中的主要参数进行训练
```
bash ./scripts/train_RCAN_dat.sh 
```


2) 测试           
测试教师网络对单张图片输入的文字检测效果       
```
python Test_craft.py
```

教师网络在文字检测数据集上的测试          
```
python eval_craft.py
```




