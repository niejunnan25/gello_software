import pyzed.sl as sl

def list_zed_cameras():
    # 获取系统中连接的ZED相机列表
    device_list = sl.Camera.get_device_list()
    if not device_list:
        print("未检测到任何 ZED 相机。")
        return

    print(f"检测到 {len(device_list)} 个 ZED 相机:\n")
    for i, device in enumerate(device_list):
        print(f"相机 {i+1}:")
        print(f"  序列号:       {device.serial_number}")
        print()

if __name__ == "__main__":
    list_zed_cameras()
