# HBUT-LML生活垃圾分类代码详解

![](https://pandao.github.io/editor.md/images/logos/editormd-logo-180x180.png)

代码编写没有完全符合python规范，亲谅解。

代码由HBUT-李铭龙编写，亲尊重原作者的代码。

项目的最终归属权为湖北工业大学

下面将是项目的完整介绍

## 项目介绍

**目录 (Table of Contents)**
- readme.txt
- HBUT/
  - manzai/
    - CYLJ.png
    - KHSLJ.png
    - QTLJ.png
    - YHLJ.png
  - 0805.pt
  - HBUT.mp4
  - HBUT_lml7.py
  - HBUT1.mp4
  - manzai.png
  - README.md
  - 摄像头帧数查看.py

## 详细介绍

### readme.txt
这个是为了引导读者阅读README.md的
### HBUT
所有的代码集中在此文件夹中
读者不用担心会看不过来，我已经将零散的代码整理到了一起
也就是主要代码就是HBUT_lml7.py

#### manzai
底部存有四个图片文件
* CYLJ.png
* KHSLJ.png
* QTLJ.png
* YHLJ.png

四张图片是四类四类垃圾缩写命名，为了满足满载检测时，对应垃圾桶显示对应的垃圾满载信号
#### 0805.pt
这是国赛的时候最终训练出来的模型，模型大概由12000多张照片训练出的
在不同环境和光照环境可能会有不同程度的影响
#### HBUT.mp4
这是生活垃圾分类宣传视频，但是倍速了8倍左右
#### HBUT_lml7.py
这是垃圾分类主要的python文件
#### HBUT1.mp4
这是生活垃圾分类宣传视频，但是没有倍速
#### manzai.png
这是早期尝试只用一张图片显示满载
#### README.md
这就是说明文档，请使用含有markdown解释器的软件（pycharm,visual code）等来查看
#### 摄像头帧数查看.py
这个是用来查看摄像头帧数的python文件

## 运行要求与步骤
_(1)首先我们需要确定几个地方的路径没有错_
```python
        # 创建一个视频播放器实例，加载指定路径的视频文件，尾的 self 是传递给 VideoPlayer 构造函数的第二个参数
        self.video_player = VideoPlayer(r"/home/mxc/Alml/HBUT/HBUT.mp4", self)
```
这里建议设置绝对路径，虽然比较麻烦，但是稳定
```python
        # 初始化检测工作器，将DetectionWorker传递给self.worker中，也就是下面调用的话就用self.worker就可以了
        self.worker = DetectionWorker(
            cap=cv2.VideoCapture(0),
            model_path=r'/home/mxc/Alml/HBUT/0730.pt',
            names=["CYLJ", "KHSLJ", "QTLJ", "YHLJ"],
            video_player=self.video_player,
            main_window=self
        )
```
这里改成0805.pt的模型路径，也是在同级目录下
```python
        # 加载图片
        if cat == 0:
            image_path = "/home/mxc/Alml/HBUT/manzai.png"  # 替换为你的图片路径
            x_position = 0
            y_position = 0
        elif cat == 1:
            image_path = "/home/mxc/Alml/HBUT/manzai/CYLJ.png"  # 替换为你的图片路径
            x_position = 0
            y_position = 0
        elif cat == 2:
            image_path = "/home/mxc/Alml/HBUT/manzai/KHSLJ.png"
            x_position = 217
            y_position = 0
        elif cat == 3:
            image_path = "/home/mxc/Alml/HBUT/manzai/QTLJ.png"
            x_position = 0
            y_position = 140
        else:
            image_path = "/home/mxc/Alml/HBUT/manzai/YHLJ.png"
            x_position = 217
            y_position = 140
```
这里的所有地方路径均需要修改，绝对路径或者相对路径看你自己<br>

_(2)我们需要插入两个串口，一个作为接收串口_
```python
        self.ser = serial.Serial(
            port='/dev/ttyUSB0',
            baudrate=115200,
            timeout=1
        )
```
串口部分在这里修改，windows使用则为COM，linux上使用则为/dev/ttyUSB<br>
COM号在串口助手中查找，USB号通过 ls /dev/ttyUSB来查找<br>

_(3)摄像头设置_
```python
        self.worker = DetectionWorker(
            cap=cv2.VideoCapture(0),
            model_path=r'/home/mxc/Alml/HBUT/0730.pt',
            names=["CYLJ", "KHSLJ", "QTLJ", "YHLJ"],
            video_player=self.video_player,
            main_window=self
        )
```
1.如果使用intel-D455,或者微软摄像头，或者奥秘中光摄像头请查看详细的说明文档，有对应的调用方式<br>
（1）intel-D455摄像头调用最简洁，只需要调用python文件包即可。<br>
（2）微软摄像头需要调用ROS，比较复杂，静下心慢慢调。<br>
（3）奥秘中光摄像头调用极其复杂，建议耐心调试。<br>
2.如果使用普通Usb摄像头，只需要用opencv库即可，调用即可用cv2.VideoCapture(0)<br>
（1）注意摄像头的代号<br>
（2）笔记本的话自带摄像头，所以USB摄像头编号为0，其他依次递增<br>
（3）视觉开发板没有自带摄像头，所以外接USB摄像头编号为0，其他一次递增<br>

_(4)确认好之后即可开始运行python3 HBUT_lml7.py注意配置的环境问题_
希望高速运行，保证稳定运行效果需要以下环境配置：<br>
numpy                        1.25.2<br>
opencv-python                4.10.0.84<br>
openvino                     2024.6.0<br>
pandas                       2.2.3<br>
pillow                       10.4.0<br>
__torch__                        2.4.1<br>
__torchaudio__                   2.4.1<br>
__torchvision__                  0.19.1<br>
配置过程可能相对复杂，亲耐心配置，不要配置错了，就算你糊里糊涂安装了torch-cpu也是没办法在后续继续使用的，请安装torch-gpu torchvision-gpu

## 杂项
(1)模型训练代码和教程<br>
方法一：YOLOv8_fast_running(https://github.com/FY-4/YOLOv8_fast_running.git)<br>
方法二：官方教程Ultralytics官网<br>
(2)建议使用网上算力跑数据，否则电脑寿命大减


