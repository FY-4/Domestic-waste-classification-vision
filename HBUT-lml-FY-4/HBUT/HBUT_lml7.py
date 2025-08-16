# import sys
import time
import cv2
# import os
import torch
from collections import Counter, defaultdict
from ultralytics import YOLO
import serial
import serial.tools.list_ports
import tkinter as tk
# from tkinter import messagebox
from PIL import Image, ImageTk
import threading
import random

"""
FY-4-LOGGER
这个代码可以实现完整的比赛功能，包括出现垃圾直接跳转，10次垃圾直接结束开始播放视频，
结束开始播放视频的话时间可以设置，代码只给了少部分注释，逻辑自行推理，并非最终代码，
有部分缺陷望海涵
"""


class DetectionSignals:
    """检测信号类，用于存储检测结果"""

    def __init__(self):
        # 存储检测结果，初始化为 None
        self.detection_result = None


class VideoPlayer:
    """视频播放器类，负责视频的播放、暂停等控制"""

    def __init__(self, video_path, main_window):
        """
        初始化视频播放器
        :param video_path: 视频文件路径
        :param main_window: 主窗口实例
        """
        # 存储视频文件的路径
        self.video_path = video_path
        # 创建视频捕获对象，使用 cv2.VideoCapture 打开视频文件
        self.cap = cv2.VideoCapture(self.video_path)
        # 运行状态标志，初始化为 False
        self.running = False
        # 暂停状态标志，初始化为 True
        self.paused = True
        # 存储主窗口实例的引用
        self.main_window = main_window
        # 视频显示标签，初始化为 None
        self.video_label = None

    def start(self):
        """启动视频播放"""
        # 将运行状态标志设置为 True
        self.running = True
        # 将暂停状态标志设置为 False
        self.paused = False
        # 调用 play_video 方法开始播放视频
        self.play_video()

    def play_video(self):
        """视频播放循环"""
        # 当处于运行状态时
        if self.running:
            # 如果没有暂停
            if not self.paused:
                # 读取视频帧
                ret, frame = self.cap.read()
                # 如果读取视频帧失败
                if not ret:
                    # 将视频的当前帧位置设置为 0，即重新开始播放
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.play_video()
                    return

                # 放大帧图像
                frame = cv2.resize(frame, (1024, 600))
                # 将 BGR 颜色空间的帧转换为 RGB 颜色空间
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 将 numpy 数组转换为 PIL 图像
                img = Image.fromarray(rgb_image)
                # 将 PIL 图像转换为 PhotoImage 图像，以便在 Tkinter 中显示
                img = ImageTk.PhotoImage(image=img)
                # 将图像设置到视频显示标签上
                self.video_label.config(image=img)
                # 存储图像引用，防止被垃圾回收
                self.video_label.image = img
                # 更新主窗口
                self.main_window.root.update()
                # 在主窗口的标签上显示当前模式为 "Video Playback Mode"
                self.main_window.video_label_text.set('Video Playback Mode')
                # 约 30fps 的播放速度，使用 root.after 方法在 33 毫秒后调用 play_video 方法继续播放下一帧
                self.main_window.root.after(33, self.play_video)
            else:
                # 如果暂停，在 100 毫秒后重新调用 play_video 方法，保持等待状态
                self.main_window.root.after(100, self.play_video)

    def stop(self):
        """停止视频播放"""
        # 将运行状态标志设置为 False
        self.running = False
        # 释放视频捕获对象的资源
        self.cap.release()

    def pause(self):
        """暂停视频播放"""
        # 将暂停状态标志设置为 True
        self.paused = True

    def resume(self):
        """恢复视频播放"""
        # 将暂停状态标志设置为 False
        self.paused = False


class DetectionWorker:
    """检测工作器类，负责目标检测和串口通信"""

    def __init__(self, cap, model_path, names, video_player, main_window):
        self.cap = cap
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"-----check the device to use-----")
        print(f"-----Using device: {self.device}------")
        self.model = YOLO(model_path)
        self.names = names
        self.video_player = video_player
        self.main_window = main_window
        self.running = True
        self.ser = serial.Serial(
            port='/dev/ttyUSB0',
            baudrate=115200,
            timeout=1
        )
        self.start_detection = False
        self.detection_sequence = 0  # 序号
        self.label_counter = defaultdict(int) 
        self.detection_paused = False
        self.received_e09f = False
        self.detection_start_time = None
        self.random_generator = random.Random()
        self.show_live_frame = False  
        self.first_detection = True

        # 第二层传送带定义是否停止检测
        self.two_covering_stop_signal = True

        # 设置鼠标光标为 spider
        self.main_window.root.config(cursor="spider")
        if hasattr(self.main_window, "detection_window"):
            self.main_window.detection_window.config(cursor="spider")

        # 在初始化时推理一帧，以提前加载模型和摄像头
        self.inference_one_frame()

        self.block_until = 0

        self.manzai_cat = 0

        # 新增后台检测标志
        self.background_detection = True
        self.background_conf = 0.6  # 后台检测置信度阈值
        self.background_iou = 0.55  # 后台检测IOU阈值

    def inference_one_frame(self):
        ret, frame = self.cap.read()
        if ret:
            results = self.model.predict(source=frame, conf=0.9, iou=0.85, device=self.device)
            print("---------------------")
            print("load model and video ")
            print("---------------------")
        else:
            print("Failed to read initial frame.")

    def draw_direction_markers(self, frame):
        h, w = frame.shape[:2]  # 获取图像高、宽

        left_line_x = int(w * 0.3)  # 左侧参考线（图像左1/3处）
        #cv2.line(frame, (left_line_x, 0), (left_line_x, h), (0, 0, 255), 2)
        #cv2.putText(
        #    frame, "cx-", (left_line_x + 10, 30),
        #    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
        #)

        right_line_x = int(w * 0.7)  # 右侧参考线（图像右1/3处）
        #cv2.line(frame, (right_line_x, 0), (right_line_x, h), (255, 0, 0), 2)
        #cv2.putText(
        #    frame, "cx+", (right_line_x - 100, 30),
        #    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2
        #)

        top_line_y = int(h * 0.3)  # 上侧参考线（图像上1/3处）
        #cv2.line(frame, (0, top_line_y), (w, top_line_y), (0, 255, 0), 2)
        #cv2.putText(
        #    frame, "cy-", (w // 2 - 100, top_line_y + 30),
        #    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        #)

        bottom_line_y = int(h * 0.7)  # 下侧参考线（图像下1/3处）
        #cv2.line(frame, (0, bottom_line_y), (w, bottom_line_y), (0, 255, 255), 2)
        #cv2.putText(
        #    frame, "cy+", (w // 2 - 100, bottom_line_y - 10),
        #    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
        #)

        #cv2.line(frame, (w // 2, 0), (w // 2, h), (128, 128, 128), 1)  # 垂直中线
        #cv2.line(frame, (0, h // 2), (w, h // 2), (128, 128, 128), 1)  # 水平中线
        #cv2.putText(
        #    frame, "cx=w/2, cy=h/2", (w // 2 + 10, h // 2 + 30),
        #    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1
        #)

        return frame

    def draw_detection_center(self, frame, box):
        x1, y1, x2, y2 = box.xyxy[0]  
        cx = int((x1 + x2) / 2)  
        cy = int((y1 + y2) / 2)  

        #cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1) 
        #cv2.putText(
        #    frame, f"中心 (cx={cx}, cy={cy})", (cx + 10, cy - 10),
        #    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
        #)
        return frame

    def run(self):
        while self.running:
            current_time = time.time()

            if current_time < self.block_until:
                continue

            if self.ser.in_waiting > 0:
                command = self.ser.readline().decode('utf-8').strip()
                if command == "A1B":
                    print("---------------------")
                    print("receive data: A1B")
                    print("cap detect task begin ")
                    print("---------------------")
                    self.start_detection = True
                    self.show_live_frame = True  
                    self.video_player.pause()
                    self.main_window.show_detection_screen()
                    self.detection_sequence = 0 
                    self.reset_counters()
                    if self.main_window.text_browser:
                        self.main_window.text_browser.delete('1.0', tk.END)
                elif command == "A2B":
                    print("---------------------")
                    print("receive data: A2B")
                    print("A2B return to video playback")
                    print("---------------------")
                    self.start_detection = False
                    self.detection_paused = True
                    self.show_live_frame = False 
                    self.video_player.resume()
                    self.main_window.hide_detection_screen()
                elif command == "A3B":
                    print("---------------------")
                    print("show manzai_warning")
                    print("---------------------")
                    self.main_window.show_alert_popup(0)
                elif command == "A4B":
                    print("---------------------")
                    print("shut up manzai_warning")
                    print("---------------------")
                    self.main_window.close_alert_popup()
                    
                elif command == "A5B":
                    print("---------------------")
                    print("show CYLJ manzai_warning")
                    print("---------------------")
                    self.main_window.show_alert_popup(1)
                elif command == "A6B":
                    print("---------------------")
                    print("show KHSLJ manzai_warning")
                    print("---------------------")
                    self.main_window.show_alert_popup(2)
                elif command == "A7B":
                    print("---------------------")
                    print("show QTLJ manzai_warning")
                    print("---------------------")
                    self.main_window.show_alert_popup(3)
                elif command == "A8B":
                    print("---------------------")
                    print("show YHLJ manzai_warning")
                    print("---------------------")
                    self.main_window.show_alert_popup(4)


                elif command == "A9B":
                    print("---------------------")
                    print("close CYLJ manzai_warning")
                    print("---------------------")
                    self.main_window.close_alert_popup(cat=1)
                elif command == "A10B":
                    print("-    --------------------")
                    print("close KHSLJ manzai_warning")
                    print("---------------------")
                    self.main_window.close_alert_popup(cat=2)
                elif command == "A11B":
                    print("---------------------")
                    print("close QTLJ manzai_warning")
                    print("---------------------")
                    self.main_window.close_alert_popup(cat=3)
                elif command == "A12B":
                    print("---------------------")
                    print("close YHLJ manzai_warning")
                    print("---------------------")
                    self.main_window.close_alert_popup(cat=4)


                elif command == "E09F" and self.start_detection:
                    print("---------------------")
                    print("receive data: E09F")
                    print("start detection")
                    print("---------------------")
                    self.received_e09f = True
                    self.detection_start_time = time.time()
                    self.detection_paused = False
                elif command == "E01F" and self.received_e09f:
                    if current_time < self.block_until:
                        continue
                    self.block_until = current_time + 0.5

                    print("---------------------")
                    print("receive data: E01F")
                    print("stop detection")
                    print("---------------------")
                    self.detection_paused = True  
                    self.process_detection_results()  
                    time.sleep(0.5) 
                    self.reset_counters()
                    self.detection_paused = False 

            if self.background_detection and not self.start_detection:
                ret, frame = self.cap.read()
                if ret:
                    results = self.model.predict(
                        source=frame,
                        conf=self.background_conf,
                        iou=self.background_iou,
                        device=self.device,

                    )

                    if any(len(r.boxes) > 0 for r in results):
                        print("检测到物体，触发正式检测流程")
                        self.background_detection = False
                        self._trigger_formal_detection()

            if self.start_detection and self.show_live_frame: 
                ret, frame = self.cap.read()
                if not ret:
                    print("---------------------")
                    print("can not read frame, check the VideoCapture")
                    print("---------------------")
                    break
                self.show_frame(frame)

            if self.received_e09f and not self.detection_paused:
                ret, frame = self.cap.read()
                if not ret:
                    print("---------------------")
                    print("can not read frame, check the VideoCapture")
                    print("---------------------")
                    break
                self.inference_task(frame)

    def _trigger_formal_detection(self):
        self.start_detection = True
        self.show_live_frame = True
        self.video_player.pause()
        self.main_window.show_detection_screen()
        self.detection_sequence = 0
        self.reset_counters()

        self.received_e09f = True
        self.detection_start_time = time.time()
        self.detection_paused = False

        if self.main_window.text_browser:
            self.main_window.text_browser.delete('1.0', tk.END)

    def show_frame(self, frame):
        frame_with_markers = self.draw_direction_markers(frame)
        frame_with_markers = cv2.resize(frame_with_markers, (800, 600))
        rgb_image = cv2.cvtColor(frame_with_markers, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_image)
        img_tk = ImageTk.PhotoImage(image=img)
        self.main_window.show_img_label.config(image=img_tk)
        self.main_window.show_img_label.image = img_tk

    def reset_counters(self):
        self.label_counter = defaultdict(int)
        self.detection_start_time = time.time()

    # def handle_g09h_command(self):
    #     random_label = self.random_generator.randint(0, 3)
    #     command = f"C{random_label + 1}9D"
    #     print(f"send the data: {command}")
    #     self.ser.write((command + '\n').encode('utf-8'))
    #     i = 0
    #     while True:
    #         i += 1
    #         self.ser.write(("C09D\n").encode('utf-8'))
    #         if i >= 20:
    #             break
    #     if self.main_window.text_browser:
    #         self.detection_sequence += 1
    #         label_text = self.get_label_text(random_label)  
    #         self.main_window.text_browser.insert(tk.END, f"{self.detection_sequence},{label_text},1,OK!\n")
    #         self.main_window.text_browser.see(tk.END)  
    #
    #         # 更新表格中的计数
    #         self.update_table_counter(random_label)
    #
    #     self.received_e09f = False  
    #     self.detection_paused = True 
    #     self.show_live_frame = False  
    #     print("Real-time detection stopped after receiving G09H.")

    def inference_task(self, frame):
        results = self.model.predict(source=frame, conf=0.60, iou=0.55, device=self.device)
        current_labels = []

        for result in results:
            boxes = result.boxes.cpu().numpy()
            if len(boxes) > 0:
                selected_box = max(boxes, key=lambda box: (box.xyxy[0][1] + box.xyxy[0][3]) / 2)

                frame = self.draw_detection_center(frame, selected_box)
                frame = self.draw_bounding_box(frame, selected_box)

                class_idx = int(selected_box.cls[0])
                current_labels.append(class_idx)
                self.label_counter[class_idx] += 1
                self.send_serial_command(class_idx)

        self.show_frame(frame)

    def draw_bounding_box(self, frame, box):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_idx = int(box.cls[0])
        conf = box.conf[0]

        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
        color = colors[class_idx % len(colors)]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{self.names[class_idx]} {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame

    def process_detection_results(self):
        if self.label_counter:
            most_common_label = max(self.label_counter, key=self.label_counter.get)
            command = f"C{most_common_label + 1}9D"
            print(f"send the data: {command}")
            n1 = 0
            while True:
                n1 += 1
                self.ser.write((command + '\n').encode('utf-8'))
                if n1 >= 10:
                    break
            time.sleep(0.05)
            m1 = 0
            while True:
                m1 += 1
                self.ser.write((command + '\n').encode('utf-8'))
                if m1 > 10:
                    break

            if self.main_window.text_browser:
                self.main_window.text_browser.tag_add("black", "1.0", tk.END)
                self.detection_sequence += 1
                self.main_window.text_browser.insert(tk.END,
                                                     f"{self.detection_sequence},{self.get_label_text(most_common_label)},1,OK!\n",
                                                     "green")
                start_index = self.main_window.text_browser.index(tk.END + "-1c")
                end_index = self.main_window.text_browser.index(tk.END)
                self.main_window.text_browser.tag_add("highlight", start_index, end_index)
                self.main_window.text_browser.see(tk.END)  

                self.update_table_counter(most_common_label)

        # 发送结束信号
        print(f"send the data: C09D")
        n = 0
        while True:
            n += 1
            self.ser.write(("C09D\n").encode('utf-8'))
            if n >= 10:
                break

        if self.detection_sequence == 100:
            print("Detection sequence reached 10, returning to video playback in 10 seconds...")
            self.main_window.root.after(5000, self.return_to_video_playback)

    def return_to_video_playback(self):
        self.start_detection = False
        self.detection_paused = True
        self.show_live_frame = False  
        self.video_player.resume()
        self.main_window.hide_detection_screen()

    def send_serial_command(self, label):
        command = f"C{label + 1}9D"
        print(f"send the data: {command}")
        self.ser.write((command + '\n').encode('utf-8'))

    def get_label_text(self, label):
        label_map = {
            0: "厨余垃圾",
            1: "可回收垃圾",
            2: "其他垃圾",
            3: "有害垃圾"
        }
        return label_map.get(label, "未知垃圾")

    def update_table_counter(self, label):
        if hasattr(self.main_window, 'table_frame'):
            label_text = self.get_label_text(label)

            for row in range(4):
                cell1 = self.main_window.table_frame.grid_slaves(row=row, column=0)[0]
                if cell1.cget("text") == label_text:
                    cell2 = self.main_window.table_frame.grid_slaves(row=row, column=1)[0]
                    current_count = int(cell2.cget("text"))
                    cell2.config(text=str(current_count + 1))
                    break


class MainWindow:
    """主窗口类，负责 GUI 界面的创建和管理"""

    def __init__(self):
        """初始化主窗口"""
        # 创建主窗口实例
        # tkinter窗口
        self.root = tk.Tk()
        # 名称
        self.root.title("Video Player and Detection")
        # 设置窗口大小
        width = int(15.5 * 96 / 2)
        height = int(8.5 * 96 / 2)
        # 我通过geometry来设置窗口
        self.root.geometry(f"{width}x{height}")
        # 全屏
        self.root.attributes('-fullscreen', True)
        # 这是一个标志位，确保下面的代码能够正常的调用全屏或者退出全屏
        self.fullscreen = True
        # 修改鼠标光标（tkinter的爱心样式）
        self.root.config(cursor="heart")

        """
        绑定快捷键
        <ESC>退出全屏
        <q>退出程序
        """
        self.root.bind('<Escape>', self.toggle_fullscreen)
        self.root.bind('q', self.quit_app)

        # 创建视频播放器
        # 创建一个视频播放器实例，加载指定路径的视频文件，尾的 self 是传递给 VideoPlayer 构造函数的第二个参数
        self.video_player = VideoPlayer(r"/home/mxc/Alml/HBUT/HBUT.mp4", self)
        # 设置视频标签的文本变量，初始显示"Video Playback Mode" 可以类似为PYQT5中的widge或者textbrowser可以放置视频
        self.video_label_text = tk.StringVar()
        # 创建标签控件并将其添加到主窗口中，用于显示视频内容
        self.video_label_text.set('Video Playback Mode')
        # 创建标签控件并将其添加到主窗口中
        self.video_label = tk.Label(self.root, textvariable=self.video_label_text)
        # 填充标签控件
        self.video_label.pack(expand=True, fill=tk.BOTH)
        # 通过这行代码将视频播放器与界面中的显示控件进行关联，使播放器知道在哪里显示视频内容
        self.video_player.video_label = self.video_label
        # 开始播放视频
        self.video_player.start()

        # 初始化检测工作器，将DetectionWorker传递给self.worker中，也就是下面调用的话就用self.worker就可以了
        self.worker = DetectionWorker(
            cap=cv2.VideoCapture(0),
            model_path=r'/home/mxc/Alml/HBUT/0730.pt',
            names=["CYLJ", "KHSLJ", "QTLJ", "YHLJ"],
            video_player=self.video_player,
            main_window=self
        )

        # 启动检测线程，分配一个检测线程给到检测工作，避免每一次都阻塞主页面
        self.detection_thread = threading.Thread(target=self.worker.run)
        self.detection_thread.start()

        # 创建检测窗口，创建一个顶级窗口实例，用于显示检测结果
        # 窗口实时置顶
        self.detection_window = tk.Toplevel(self.root)
        # 设置窗口名字
        self.detection_window.title("Detection Screen")
        self.detection_window.geometry(f"{width}x{height}")
        # fullscreen的属性只有True和False，所以我们首先定义为True，开始不想让他全屏的话在修改
        self.detection_window.attributes('-fullscreen', True)
        # 绑定快捷键ESC为全屏与否的控件
        self.detection_window.bind('<Escape>', self.toggle_detection_fullscreen)
        self.detection_window.bind('q', self.quit_app)

        # 隐藏实时检测窗口 TK独有方法
        self.detection_window.withdraw()

        # 创建检测界面元素（也就是填充顶级界面上的控件）
        # 创建一个标签控件用于显示图像，放置在检测窗口的左侧
        self.show_img_label = tk.Label(self.detection_window)
        self.show_img_label.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        # 创建一个文本控件用于显示文字内容，放置在检测窗口的右侧，可以修改字体的大小
        self.text_browser = tk.Text(self.detection_window, font=("default", 18))
        self.text_browser.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        # 绑定窗口大小改变事件
        # 目的是为了当窗口大小发生变化的时候，子部件窗口也能够对应的发生该有的变化，保证子部件的大小能够正确匹配
        self.root.bind("<Configure>", self.on_resize)
        self.detection_window.bind("<Configure>", self.on_detection_resize)

        # 绿色底色，高亮，白色前景色，字号16，粗体
        self.text_browser.tag_configure("highlight", background="green", foreground="white",
                                        font=("default", 16, "bold"))

    def show_message_on_video_label(self, message):
        """在 video_label 的底部中间显示消息，并在三秒后清除"""
        # 如果已经存在 message_label，先清除它，以免重复创建
        if hasattr(self, "message_label"):
            self.message_label.destroy()

        # 创建一个标签控件用于显示消息文本
        # 该标签放置在show_img_label容器中，黑色背景和白色前景
        # 文本内容为message变量的值（去除首尾空白字符）
        # 字体设置为默认字体，大小为25号
        self.message_label = tk.Label(
            self.show_img_label,
            text=message.strip(),
            bg="black",
            fg="white",
            font=("default", 25)
        )

        # 将消息 Label 放置在 video_label 的底部中间
        self.message_label.place(relx=0.5, rely=0.95, anchor=tk.CENTER)

        # 设置定时器，在三秒后清除消息
        self.video_label.after(2000, self.clear_message_on_video_label)

    def clear_message_on_video_label(self):
        """清除 video_label 中的消息"""
        # 检查并销毁消息标签控件
        # 如果当前对象存在message_label属性，则销毁该标签控件并从对象中删除该属性
        if hasattr(self, "message_label"):
            self.message_label.destroy()
            del self.message_label


    def show_detection_screen(self):
        """显示检测窗口"""
        # 显示检测窗口
        self.detection_window.deiconify()
        # 设置光标样式为蜘蛛形状
        self.detection_window.config(cursor="spider")
        # 隐藏视频标签组件
        self.video_label.pack_forget()
        # 重新布局视频标签组件，放置在窗口底部并填充剩余空间
        self.video_label.pack(side=tk.BOTTOM, expand=True, fill=tk.BOTH)


    def hide_detection_screen(self):
        """隐藏检测窗口"""
        # 隐藏检测窗口并重新布局视频标签
        # 首先隐藏检测窗口，然后移除视频标签的当前布局，
        # 最后将视频标签重新放置在窗口顶部并填充整个可用空间
        self.detection_window.withdraw()  # 隐藏检测窗口
        self.video_label.pack_forget()
        self.video_label.pack(side=tk.TOP, expand=True, fill=tk.BOTH)


    def resize_video_label(self, width, height):
        video_width = int(width)  # 视频宽度与窗口宽度相同
        video_height = int(height)  # 视频高度与窗口高度相同
        x_offset = 0  # 视频从左上角开始
        y_offset = 0  # 视频从左上角开始
        # 视频窗口放置在左上角开始的整个lable控件中
        self.video_label.place(x=x_offset, y=y_offset, width=video_width, height=video_height)

    def on_resize(self, event):
        # 当主窗口大小改变时，调用 self.resize_video_label() 方法来重新调整视频显示区域的大小，确保视频能够正确填充窗口
        # widget 是当前事件对象关联的控件，这里我们只处理主窗口大小改变的事件，因此 event.widget == self.root
        if event.widget == self.root:
            self.resize_video_label(event.width, event.height)

    """
    这里面我写了有关于表格的设计信息
    """

    def on_detection_resize(self, event):
        if event.widget == self.detection_window:
            img_width = int(event.width * 3 / 4)
            img_height = event.height
            self.show_img_label.place(x=0, y=0, width=img_width, height=img_height)

            text_width = int(event.width / 4)
            text_height = int(event.height * 20 / 26) 
            text_x = img_width  
            text_y = int(event.height * 1 / 30) 
            self.text_browser.place(x=text_x, y=text_y, width=text_width, height=text_height)

            table1_x = text_x
            table1_y = 0
            table1_width = text_width
            table1_height = int(event.height * 0.04)

            if hasattr(self, "table1_frame"):
                self.table1_frame.destroy()

            self.table1_frame = tk.Frame(self.detection_window, width=table1_width, height=table1_height, bg="white")
            self.table1_frame.place(x=table1_x, y=table1_y, width=table1_width, height=table1_height)
            labels = ["序号", "垃圾种类", "数量", "是否分类"]
            for col, label_text in enumerate(labels):
                label = tk.Label(self.table1_frame, text=label_text, borderwidth=1, relief="solid", anchor="center",
                                 font=("default", 10))
                label.grid(row=0, column=col, sticky="nsew")
                self.table1_frame.columnconfigure(col, weight=1)  

            for i in range(4):
                self.table1_frame.rowconfigure(i, weight=1)  
            self.table1_frame.columnconfigure(0, weight=1)  

         
            table_x = text_x  
            table_y = text_y + text_height  
            table_width = text_width  
            table_height = int(event.height * 0.2) 

            if hasattr(self, "table_frame"):
                self.table_frame.destroy()

            self.table_frame = tk.Frame(self.detection_window, width=table_width, height=table_height, bg="white")
            self.table_frame.place(x=table_x, y=table_y, width=table_width, height=table_height)

            labels = ["厨余垃圾", "可回收垃圾", "其他垃圾", "有害垃圾"]
            for row in range(4):
                cell1 = tk.Label(self.table_frame, text=labels[row], borderwidth=1, relief="solid", anchor="center",
                                 font=("default", 18))
                cell1.grid(row=row, column=0, sticky="nsew")
                cell2 = tk.Label(self.table_frame, text="0", borderwidth=1, relief="solid", anchor="center",
                                 font=("default", 18))
                cell2.grid(row=row, column=1, sticky="nsew")

            for i in range(4):
                self.table_frame.rowconfigure(i, weight=1)
            for j in range(2):
                self.table_frame.columnconfigure(j, weight=1)

    def toggle_fullscreen(self, event=None):
        self.fullscreen = not self.fullscreen
        self.root.attributes('-fullscreen', self.fullscreen)
        if self.fullscreen:
            self.resize_video_label(self.root.winfo_screenwidth(), self.root.winfo_screenheight())
        else:
            self.resize_video_label(int(15.5 * 96 / 2), int(8.5 * 96 / 2))

    def toggle_detection_fullscreen(self, event=None):
        # 切换到not self.detection_window.atrributes("-fullscreen")，也就是这个value= not True
        self.detection_window.attributes('-fullscreen', not self.detection_window.attributes('-fullscreen'))

    def quit_app(self, event=None):
        # 退出程序
        self.root.destroy()

    def show_image_in_main_thread(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_image)

        def display_image():
            img_tk = ImageTk.PhotoImage(image=img)
            self.show_img_label.config(image=img_tk)
            self.show_img_label.image = img_tk

        self.detection_window.after(0, display_image)

    """def show_alert_popup(self):
        if not hasattr(self, 'alert_popup') or not self.alert_popup:
            self.alert_popup = tk.Toplevel(self.root)
            self.alert_popup.title("WARNING WARNING WARNING")
            popup_width = 720
            popup_height = 250
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            x_position = int((screen_width - popup_width) / 2)  # 转换为整数
            y_position = int((screen_height - popup_height) / 2)  # 转换为整数
            self.alert_popup.geometry(f"{popup_width}x{popup_height}+{x_position}+{y_position}")
            label = tk.Label(self.alert_popup, text="满载警告!!!", fg="red", font=("黑体",100, "bold"))
            label.pack(expand=True)
            self.alert_popup.attributes('-topmost', True)"""

    def show_alert_popup(self, cat=0):
        if hasattr(self, 'alert_popups'):
            for alert_popup in self.alert_popups:
                if hasattr(alert_popup, 'cat') and alert_popup.cat == cat:
                    print(f"弹窗 cat={cat} 已经存在，不再重复弹出")
                    return

        popup_width = int(750 * 1 / 3)
        popup_height = int(420 * 1 / 3)

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        '''# 计算弹窗的初始位置（左上角）
        x_position = 0
        y_position = 0

        # 如果已经有弹窗，计算下一个弹窗的位置
        if hasattr(self, 'alert_popups') and self.alert_popups:
            last_popup = self.alert_popups[-1]
            last_x = last_popup.winfo_x()
            x_position = last_x + 50  # 下一个弹窗向下偏移 100 像素

            # 如果超出屏幕高度，重置到顶部
            if x_position + popup_height > screen_height:
                x_position =0'''

        if cat == 0:
            image_path = "/home/mxc/Alml/HBUT/manzai.png"  
            x_position = 0
            y_position = 0
        elif cat == 1:
            image_path = "/home/mxc/Alml/HBUT/manzai/CYLJ.png" 
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

        alert_popup = tk.Toplevel(self.root)
        alert_popup.title("WARNING WARNING WARNING")
        alert_popup.geometry(f"{popup_width}x{popup_height}+{x_position}+{y_position}")
        
        try:
            img = Image.open(image_path)
            img = img.resize((popup_width, popup_height), Image.Resampling.LANCZOS)  # 调整图片大小
            img_tk = ImageTk.PhotoImage(img)
            label = tk.Label(alert_popup, image=img_tk)
            label.image = img_tk 
            label.pack(expand=True)
        except Exception as e:
            print(f"Failed to load image: {e}")
            label = tk.Label(alert_popup, text="满载警告!!!", fg="red", font=("宋体", 100, "bold"))
            label.pack(expand=True)

        alert_popup.attributes('-topmost', True)

        alert_popup.cat = cat

        if not hasattr(self, 'alert_popups'):
            self.alert_popups = []
        self.alert_popups.append(alert_popup)

        alert_popup.protocol("WM_DELETE_WINDOW", lambda: self.close_alert_popup(alert_popup))

    def close_alert_popup(self, popup=None, cat=None):
        if popup:
            popup.destroy()
            if hasattr(self, 'alert_popups') and popup in self.alert_popups:
                self.alert_popups.remove(popup)
        elif cat is not None:
            if hasattr(self, 'alert_popups'):
                for alert_popup in self.alert_popups[:]:  
                    if hasattr(alert_popup, 'cat') and alert_popup.cat == cat:
                        alert_popup.destroy()
                        self.alert_popups.remove(alert_popup)
        else:
            if hasattr(self, 'alert_popups'):
                for alert_popup in self.alert_popups:
                    alert_popup.destroy()
                self.alert_popups = []


if __name__ == '__main__':
    app = MainWindow()  
    app.root.mainloop()  
