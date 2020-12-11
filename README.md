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
--dir_data 训练集和验证集所在目录
--pre_train 学生网络预训练模型路径



2) 测试           

测试教师网络对单张图片输入的文字检测效果       
```
python Test_craft.py
```


### todo

1.数据加载      
指定文件夹路径加载。
指定取训练文件数目，测试文件数目。


2.从trainer 分离的单独的test      
只加载训练好的学生模型进行测试 


3.tensorboard内容添加          

4.注意力的热图生成        


