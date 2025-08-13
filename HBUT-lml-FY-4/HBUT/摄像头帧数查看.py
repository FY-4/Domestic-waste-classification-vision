import cv2
from ultralytics import YOLO
import time


def test_yolo_fps_ultralytics(camera_index=0, test_duration=10, model_name='0319.pt'):

    model = YOLO(model_name)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"摄像头寄了")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"分辨率: {width}x{height}")
    frame_count = 0
    start_time = time.time()
    while (time.time() - start_time) < test_duration:
        ret, frame = cap.read()
        if not ret:
            print("无法读取帧")
            break
        results = model(frame, verbose=False)
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            current_fps = frame_count / elapsed_time
            print(f"\r当前帧率: {current_fps:.2f} FPS", end="")
        annotated_frame = results[0].plot()
        cv2.imshow("YOLO", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0

    print(f"\n测试完成 - 平均帧率: {avg_fps:.2f} FPS")
    print(f"总帧数: {frame_count} 总时间: {total_time:.2f}秒")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_yolo_fps_ultralytics(model_name=r'F:\YOLO\HBUT4\0319whole.pt')