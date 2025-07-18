import pyzed.sl as sl
import numpy as np
import cv2
import time
import os

class ZEDCamera:
    def __init__(self, serial_number=None, device_path=None, resolution=sl.RESOLUTION.HD720, depth_mode=sl.DEPTH_MODE.PERFORMANCE):
        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = resolution
        init_params.depth_mode = depth_mode
        init_params.camera_fps = 30

        # 设置设备输入来源（按优先级选择）
        if serial_number is not None:
            input_type = sl.InputType()
            input_type.set_from_serial_number(serial_number)
            init_params.input = input_type
        elif device_path is not None:
            init_params.input = sl.InputType(device_path)

        # 打开相机
        status = self.zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"无法打开 ZED 相机: {status}")

        # 初始化图像容器
        self.image_left = sl.Mat()
        self.image_right = sl.Mat()
        self.depth = sl.Mat()

    def grab_frame(self):
        right_image = None
        if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(self.image_left, sl.VIEW.LEFT)
            # self.zed.retrieve_image(self.image_right, sl.VIEW.RIGHT)

            left_image = self.image_left.get_data()[:, :, :3].copy()
            # right_image = self.image_right.get_data()[:, :, :3].copy()

            return left_image, right_image
        else:
            raise RuntimeError("获取图像失败")

if __name__ == "__main__":
    zed_camera = ZEDCamera(serial_number=36276705)
    zed_wrist_camera = ZEDCamera(serial_number=13132609)

    left, right = zed_camera.grab_frame()
    left_wrist, right_wrist = zed_wrist_camera.grab_frame()
    # path = "./test_0"
    # path.mkdir()
    cv2.imwrite("left.png", left)
    cv2.imwrite("right.png", right)

    cv2.imwrite("left_wrist.png", left_wrist)
    cv2.imwrite("right_wrist.png", right_wrist)
