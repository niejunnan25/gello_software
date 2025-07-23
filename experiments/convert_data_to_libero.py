
import pickle
import cv2
import os
from pathlib import Path
from datetime import datetime
import re
from PIL import Image
import numpy as np
from tqdm import tqdm
import tyro

"""
在运行这个转换脚本之前，我们假设：
1.图像已经写入了: bc_data/gello/XXXXXX/image 文件夹, 并且以时间戳命名
2.pkl 文件已经写入了: bc_data/gello/XXXXXX/ 文件夹
3.main(data_dir): data_dir 是所有数据保存的文件夹，在这个示例中是 bc_data/gello, 再往下是每个episode的文件夹
4.传入参数 data_dir, prompt。
5.data_dir 在3.已经解释过了, prompt: str，表示data_dir这整个文件夹下所有任务的内容，也就是每个 episode 干了啥
"""

from lerobot.src.lerobot.datasets.lerobot_dataset import LeRobotDataset

def count_files(folder_path, end_name):

    if not os.path.exists(folder_path):
        print(f"文件夹 {folder_path} 不存在")
        return 0

    # 统计 .pkl 文件
    count = sum(1 for file in os.listdir(folder_path) if file.endswith(end_name))
    return count

def extract_timestamp(file_path):
    # 假设文件名格式为 left_YYYYMMDD_HHMMSS.pkl 或 right_YYYYMMDD_HHMMSS.pkl
    filename = file_path.stem  # 获取文件名（不含扩展名）
    timestamp_str = filename.split('_')[-1]  # 提取时间戳部分
    try:
        return datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
    except ValueError:
        # 如果时间戳格式不匹配，返回一个很早的时间以便排序
        return datetime.min
    
def match_closest_pkl_fast(image_files, pkl_files):
    """
    双指针匹配
    """
    image_ts = [int(os.path.splitext(f)[0]) for f in image_files]
    pkl_ts = [int(os.path.splitext(f)[0]) for f in pkl_files]

    matched = []
    j = 0

    for image_time in image_ts:
        while j + 1 < len(pkl_ts) and abs(pkl_ts[j + 1] - image_time) < abs(pkl_ts[j] - image_time):
            j += 1
        matched.append(pkl_files[j])

    return matched

def convert_to_float32(obj):
    if isinstance(obj, np.ndarray) and obj.dtype == np.float64:
        return obj.astype(np.float32)
    elif isinstance(obj, np.generic) and obj.dtype == np.float64:
        return np.float32(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_float32(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_float32(i) for i in obj]
    else:
        return obj


def main(data_dir : str, prompt : str):
    """
    参数:
        data_dir : 数据保存的文件夹名称，例如 bc_data/gello
        prompt: 任务的 prompt
    """
    dataset = LeRobotDataset.create(
            repo_id="REPO_0",
            robot_type="panda",
            fps=10,
            features={
                "image": {
                    "dtype": "image",
                    "shape": (256, 256, 3),
                    "names": ["height", "width", "channel"],
                },
                "wrist_image": {
                    "dtype": "image",
                    "shape": (256, 256, 3),
                    "names": ["height", "width", "channel"],
                },
                "state": {
                    "dtype": "float32",
                    "shape": (8,), # 7 - Dof + 1 个夹爪
                    "names": ["state"],
                },
                "actions": {
                    "dtype": "float32",
                    "shape": (7,),
                    "names": ["actions"],
                },
            },
            image_writer_threads=10,
            image_writer_processes=5,
        )

    for episode_name in os.listdir(data_dir):
        
        image_path = os.path.join(data_dir, episode_name, "image")
        wrist_image_path = os.path.join(data_dir, episode_name, "wrist_image")

        pkl_cnt = count_files(os.path.join(data_dir, episode_name), ".pkl")
        image_cnt = count_files(image_path, ".png")
        wrist_image_cnt = count_files(wrist_image_path, ".png")
    
        # 左右一起保存下来的，所以除以 2
        print(f".pkl 文件有 {pkl_cnt} 个，主相机.png 文件有 {image_cnt / 2} 个，腕部相机.png 文件有 {wrist_image_cnt / 2} 个")

        assert image_cnt == wrist_image_cnt, "主视角和腕部视角保存的帧数不匹配"
        
        pkl_files = list(os.path.join(data_dir, episode_name).glob('*.pkl'))

        pattern = re.compile(r"^left_\d+\.png$")
        image_files = [f for f in os.listdir(image_path) if pattern.match(f)]

        wrist_pattern = re.compile(r"^wrist_left_\d+\.png$")
        wrist_image_files = [f for f in os.listdir(wrist_image_path) if wrist_pattern.match(f)]

        pkl_sorted_files = sorted(pkl_files, key=extract_timestamp)
        image_files = sorted(image_files, key=extract_timestamp)
        wrist_image_files = sorted(wrist_image_files, key=extract_timestamp)

        # 双指针匹配
        match_pkl_files = match_closest_pkl_fast(image_files, pkl_sorted_files)

        for idx, image_file in enumerate(tqdm(image_files, desc="正在进行数据转换")):
            image = Image.open(os.path.join(image_path, image_file))
            image_np = np.array(image)
            wrist_image = Image.open(os.path.join(wrist_image_path, image_file))
            wrist_image_np = np.array(wrist_image)

            assert image_np.shape[2] == 3, "主视角图像格式不匹配"
            assert wrist_image_np.shape[2] == 3, "腕部视角图像格式不匹配"
            assert len(match_pkl_files) == len(image_files), "匹配后的 .pkl 文件数量与图像数量不匹配"

            try:
                with open(os.path.join(data_dir, episode_name, pkl_files[idx]), "rb") as f:
                    data_np_float32 = convert_to_float32(pickle.load(f))
            except Exception as e:
                print(e)
            
            """
            {
                'joint_positions': np.ndarray,
                'joint_velocities': np.ndarray,
                'ee_pos_quat': np.ndarray,
                'gripper_position': np.float32,
                'control': np.ndarray
            }
            """
            # image_np, wrist_image_np 应该是 0-255
            dataset.add_frame({
                "image": image_np,
                "wrist_image": wrist_image_np,
                "state": data_np_float32["joint_positions"],
                "actions": data_np_float32["control"],
            })

        dataset.save_episode(task=prompt) 

    dataset.consolidate(run_compute_stats=False) 

if __name__ == "__main__":
    # 指定 data_dir, prompt,
    main(tyro.cli())
