import datetime
import glob
import time
import numpy as np
import tyro
import os
import cv2
import termcolor
import pickle
import copy

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict

from gello.agents.agent import BimanualAgent, DummyAgent
from gello.agents.gello_agent import GelloAgent
from gello.data_utils.format_obs import save_frame
from gello.env import RobotEnv
from gello.robots.robot import PrintRobot
from gello.zmq_core.robot_node import ZMQClientRobot

from ZEDCamera import ZedCamera
from concurrent.futures import ThreadPoolExecutor, as_completed
from gello.data_utils.keyboard_interface import KBReset
from tqdm import tqdm
from PIL import Image

def save_single_pkl(path: Path, data: dict):

    with open(path, "wb") as f:
        pickle.dump(data, f)

import cv2

def save_image(path, image):
    """
    保存图像到指定路径，输入图像为 RGB 格式，直接保存。

    参数：
        path: 保存图像的文件路径（例如 'output.png'）
        image: 输入的 RGB 格式图像（NumPy 数组）
    """
    # 直接保存 RGB 格式图像
    cv2.imwrite(path, image)

@dataclass
class Args:
    agent: str = "none"
    robot_port: int = 6001
    wrist_camera_port: int = 5000
    base_camera_port: int = 5001
    hostname: str = "127.0.0.1"
    robot_type: Optional[str] = None  # only needed for quest agent or spacemouse agent
    hz: int = 100
    start_joints: Optional[Tuple[float, ...]] = None

    gello_port: Optional[str] = None
    mock: bool = False
    use_save_interface: bool = True
    data_dir: str = "~/bc_data"
    bimanual: bool = False
    verbose: bool = False

def print_color(*args, color=None, attrs=(), **kwargs):

    if len(args) > 0:
        args = tuple(termcolor.colored(arg, color=color, attrs=attrs) for arg in args)
    print(*args, **kwargs)

# def save_frame(
#     folder: Path,
#     timestamp: datetime.datetime,
#     obs: Dict[str, np.ndarray],
#     action: np.ndarray,
# ) -> None:
#     obs["control"] = action  # add action to obs

#     # make folder if it doesn't exist
#     folder.mkdir(exist_ok=True, parents=True)
#     recorded_file = folder / (timestamp.isoformat() + ".pkl")

#     with open(recorded_file, "wb") as f:
#         pickle.dump(obs, f)


def main(args):
    if args.mock:
        robot_client = PrintRobot(8, dont_print=True)
        camera_clients = {}
    else:
        camera_clients = {
            # you can optionally add camera nodes here for imitation learning purposes
            # "wrist": ZMQClientCamera(port=args.wrist_camera_port, host=args.hostname),
            # "base": ZMQClientCamera(port=args.base_camera_port, host=args.hostname),
        }
        robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
    env = RobotEnv(robot_client, control_rate_hz=args.hz, camera_dict=camera_clients)

    if args.bimanual:
        if args.agent == "gello":
            # dynamixel control box port map (to distinguish left and right gello)
            right = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT7WBG6A-if00-port0"
            left = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT7WBEIA-if00-port0"
            left_agent = GelloAgent(port=left)
            right_agent = GelloAgent(port=right)
            agent = BimanualAgent(left_agent, right_agent)
        elif args.agent == "quest":
            from gello.agents.quest_agent import SingleArmQuestAgent

            left_agent = SingleArmQuestAgent(robot_type=args.robot_type, which_hand="l")
            right_agent = SingleArmQuestAgent(
                robot_type=args.robot_type, which_hand="r"
            )
            agent = BimanualAgent(left_agent, right_agent)
            # raise NotImplementedError
        elif args.agent == "spacemouse":
            from gello.agents.spacemouse_agent import SpacemouseAgent

            left_path = "/dev/hidraw0"
            right_path = "/dev/hidraw1"
            left_agent = SpacemouseAgent(
                robot_type=args.robot_type, device_path=left_path, verbose=args.verbose
            )
            right_agent = SpacemouseAgent(
                robot_type=args.robot_type,
                device_path=right_path,
                verbose=args.verbose,
                invert_button=True,
            )
            agent = BimanualAgent(left_agent, right_agent)
        else:
            raise ValueError(f"Invalid agent name for bimanual: {args.agent}")

        # System setup specific. This reset configuration works well on our setup. If you are mounting the robot
        # differently, you need a separate reset joint configuration.
        reset_joints_left = np.deg2rad([0, -90, -90, -90, 90, 0, 0])
        reset_joints_right = np.deg2rad([0, -90, 90, -90, -90, 0, 0])
        reset_joints = np.concatenate([reset_joints_left, reset_joints_right])
        curr_joints = env.get_obs()["joint_positions"]
        max_delta = (np.abs(curr_joints - reset_joints)).max()
        steps = min(int(max_delta / 0.01), 100)

        for jnt in np.linspace(curr_joints, reset_joints, steps):
            env.step(jnt)
    else:
        if args.agent == "gello":
            gello_port = args.gello_port
            if gello_port is None:
                usb_ports = glob.glob("/dev/serial/by-id/*")
                print(f"Found {len(usb_ports)} ports")
                if len(usb_ports) > 0:
                    gello_port = usb_ports[0]
                    print(f"using port {gello_port}")
                else:
                    raise ValueError(
                        "No gello port found, please specify one or plug in gello"
                    )
            if args.start_joints is None:
                reset_joints = np.array(
                    [0, 0, 0, -1.57, 0, 1.57, 0, 0]
                )  # Change this to your own reset joints
            else:
                reset_joints = np.array(args.start_joints)
            agent = GelloAgent(port=gello_port, start_joints=reset_joints)
            print("success to connect to gello")
            curr_joints = env.get_obs()["joint_positions"]
            print(f"success to get joint positions")
            if reset_joints.shape == curr_joints.shape:
                max_delta = (np.abs(curr_joints - reset_joints)).max()
                steps = min(int(max_delta / 0.01), 100)

                for jnt in np.linspace(curr_joints, reset_joints, steps):
                    env.step(jnt)
                    time.sleep(0.001)
        elif args.agent == "quest":
            from gello.agents.quest_agent import SingleArmQuestAgent

            agent = SingleArmQuestAgent(robot_type=args.robot_type, which_hand="l")
        elif args.agent == "spacemouse":
            from gello.agents.spacemouse_agent import SpacemouseAgent

            agent = SpacemouseAgent(robot_type=args.robot_type, verbose=args.verbose)
        elif args.agent == "dummy" or args.agent == "none":
            agent = DummyAgent(num_dofs=robot_client.num_dofs())
        elif args.agent == "policy":
            raise NotImplementedError("add your imitation policy here if there is one")
        else:
            raise ValueError("Invalid agent name")

    # going to start position
    print("Going to start position")
    start_pos = agent.act(env.get_obs())
    obs = env.get_obs()
    joints = obs["joint_positions"]

    print(f"Joints: {joints}")

    abs_deltas = np.abs(start_pos - joints)
    id_max_joint_delta = np.argmax(abs_deltas)

    max_joint_delta = 0.8
    if abs_deltas[id_max_joint_delta] > max_joint_delta:
        id_mask = abs_deltas > max_joint_delta
        print()
        ids = np.arange(len(id_mask))[id_mask]
        for i, delta, joint, current_j in zip(
            ids,
            abs_deltas[id_mask],
            start_pos[id_mask],
            joints[id_mask],
        ):
            print(
                f"joint[{i}]: \t delta: {delta:4.3f} , leader: \t{joint:4.3f} , follower: \t{current_j:4.3f}"
            )
        return

    print(f"Start pos: {len(start_pos)}", f"Joints: {len(joints)}")
    assert len(start_pos) == len(
        joints
    ), f"agent output dim = {len(start_pos)}, but env dim = {len(joints)}"

    max_delta = 0.05
    for _ in range(25):
        obs = env.get_obs()
        command_joints = agent.act(obs)
        current_joints = obs["joint_positions"]
        delta = command_joints - current_joints
        max_joint_delta = np.abs(delta).max()
        if max_joint_delta > max_delta:
            delta = delta / max_joint_delta * max_delta
        env.step(current_joints + delta)

    obs = env.get_obs()
    joints = obs["joint_positions"]
    action = agent.act(obs)
    print(joints)
    print(action)
    if (action - joints > 0.5).any():
        print("Action is too big")

        joint_index = np.where(action - joints > 0.8)
        for j in joint_index:
            print(
                f"Joint [{j}], leader: {action[j]}, follower: {joints[j]}, diff: {action[j] - joints[j]}"
            )
        exit()

    kb_interface = KBReset()

    print_color("\nStart 🚀🚀🚀", color="green", attrs=("bold",))

    zed_camera = ZedCamera(serial_number=36276705,fps=30)
    zed_wrist_camera = ZedCamera(serial_number=13132609, fps=30)

    save_path = None
    start_time = time.time()

    left_image_dict = {}
    wrist_left_image_dict = {}
    obs_dict = {}

    while True:
        # num = time.time() - start_time

        # message = f"\rTime passed: {round(num, 2)}"
        # print_color(
        #     message,
        #     color="white",
        #     attrs=("bold",),
        #     end="",
        #     flush=True,
        # )

        action = agent.act(obs)
        dt = datetime.datetime.now()

        if args.use_save_interface:

            state = kb_interface.update()

            # 当打开 pygame , 按下 SPACE 键后执行这个 if 语句
            if state == "start":
                dt_time = datetime.datetime.now()
                save_path = (
                    Path(args.data_dir).expanduser()
                    / args.agent
                    / dt_time.strftime("%m%d_%H%M%S")
                )

                save_path.mkdir(parents=True, exist_ok=True)

            elif state == "save":

                # assert save_path is not None, "save_path is None"

                if save_path is None:
                    print("ERROR：请先按空格键开始录制！")
                    continue  # 跳过本次循环
                
                left_image, right_image = zed_camera.capture_frame()
                wrist_left_image, wrist_right_image = zed_wrist_camera.capture_frame()
                
                # TODO: 把 save_frame 函数改成在下面的 state == "stop" 时，再写入 .pkl 文件
                # 7 月 21 日 12:30 ， 已经修改完了
                # save_frame(save_path, dt, obs, action)

                # timestamp : 20250719_204355_123456
                # timestamp = dt.isoformat().replace(":", "_").replace(".", "_")
                dt = datetime.datetime.now()

                # 深拷贝
                action_copy = copy.deepcopy(action)
                obs_copy = copy.deepcopy(obs)
                obs_copy["control"] = action_copy
                # obs["control"] = action


                timestamp = dt.strftime("%Y%m%d_%H%M%S_%f")

                obs_dict[timestamp] = obs_copy
                left_image_dict[timestamp] = left_image
                wrist_left_image_dict[timestamp] = wrist_left_image
                
            elif state == "normal":
                save_path = None
            elif state == "esc":
                break
            elif state == "stop":
                
                # assert save_path is not None, "save_path is None"

                if save_path is None:
                    print("错误：请先按空格键开始录制！")
                    kb_interface._set_color((128, 128, 128))  # 恢复灰色
                    continue  # 跳过本次循环

                kb_interface._saved = False

                print("停止数据采集...")

                image_path = os.path.join(save_path, "image")
                wrist_image_path = os.path.join(save_path, "wrist_image")

                save_path.mkdir(exist_ok=True, parents=True)

                os.makedirs(image_path, exist_ok=True)
                os.makedirs(wrist_image_path, exist_ok=True)

                #################################################################################################################
                # 保存 .pkl 文件
                print("正在保存 .pkl 文件...")

                pkl_futures = []
                pkl_save_start = time.time()
                with ThreadPoolExecutor(max_workers=8) as executor:
                    for timestamp, single_obs in obs_dict.items():
                        pkl_path = save_path / f"{timestamp}.pkl"
                        pkl_futures.append(executor.submit(save_single_pkl, pkl_path, single_obs))

                    for future in tqdm(as_completed(pkl_futures), total=len(pkl_futures), desc="pkl 多线程写入进度"):
                        try:
                            future.result()
                        except Exception as e:
                            print(f"ERROR: 保存 {future} 出错：{e}")
                
                pkl_save_end = time.time()

                # for timestamp, obs in obs_dict.items():
                #     recorded_file = save_path / f"{timestamp}.pkl"
                #     with open(recorded_file, "wb") as f:
                #         pickle.dump(obs, f)

                #################################################################################################################
                # 保存图像

                print("等待图像保存...")
                print("正在使用多线程进行图像保存...")

                # with ThreadPoolExecutor(max_workers = 8) as executor:

                #     for timestamp, left_image in left_image_dict.items():
                #         left_image_path = os.path.join(image_path, f"left_{timestamp}.png")
                #         futures.append(executor.submit(save_image, left_image_path, left_image))

                #     for timestamp, wrist_left_image in wrist_left_image_dict.items():
                #         wrist_left_image_path = os.path.join(wrist_image_path, f"wrist_left_{timestamp}.png")
                #         futures.append(executor.submit(save_image, wrist_left_image_path, wrist_left_image))

                #     for _ in tqdm(as_completed(futures), total=len(futures), desc="多线程写入进度"):
                #         pass

                img_futures = []
                image_save_start = time.time()

                with ThreadPoolExecutor(max_workers=8) as executor:
                    for ts, img in left_image_dict.items():
                        img_path = os.path.join(image_path, f"left_{ts}.png")
                        img_futures.append(executor.submit(save_image, img_path, img))
                    for ts, wimg in wrist_left_image_dict.items():
                        wimg_path = os.path.join(wrist_image_path, f"wrist_left_{ts}.png")
                        img_futures.append(executor.submit(save_image, wimg_path, wimg))

                    for future in tqdm(as_completed(img_futures), total=len(img_futures), desc="图像多线程写入进度"):
                        try:
                            future.result()
                        except Exception as e:
                            print(f"ERROR: 保存图像时出错：{e}")
                
                image_save_end = time.time()

                print(f".pkl 文件以及 .png 图像均已保存完成：")
                print(f"   - .pkl 文件：{len(obs_dict)}")
                print(f"   - 左视图帧数：{len(left_image_dict)}")
                print(f"   - 手腕图帧数：{len(wrist_left_image_dict)}")
                print(f"   - .pkl 文件保存耗时：{pkl_save_end - pkl_save_start:.2f} 秒")
                print(f"   - 图像保存耗时：{image_save_end - image_save_start:.2f} 秒")

                obs_dict.clear()
                left_image_dict.clear()
                wrist_left_image_dict.clear()
                print(f"缓存已清空")

                kb_interface._set_color((128, 128, 128))
                kb_interface._saved = False
                save_path = None

                print("按下 空格键 继续采集数据")
            else:
                raise ValueError
            
        obs = env.step(action)
    
    kb_interface.close()


if __name__ == "__main__":
    main(tyro.cli(Args))