import os
import sys
import time
import numpy as np
import pandas as pd
import random
from tqdm import tqdm  # 进度条库
from Basilisk.utilities import SimulationBaseClass, macros
from Basilisk.architecture import messaging
from Basilisk.simulation import spacecraft, gravityEffector, eclipse, simpleNav
from Basilisk.simulation import camera
from Basilisk.utilities import orbitalMotion

# --------------------------
# 全局配置参数（用户可按需修改）
# --------------------------
TOTAL_IMAGES = 1200  # 目标生成图像总数（建议1000+）
IMAGES_PER_ORBIT = 10  # 每条轨道生成图像数
ORBIT_COUNT = TOTAL_IMAGES // IMAGES_PER_ORBIT  # 需要的轨道数量
SIM_STEP = macros.sec2nano(0.5)  # 仿真步长（0.5秒）
IMAGE_INTERVAL = macros.sec2nano(60)  # 图像生成间隔（60秒/张）
DATA_ROOT = "dataset"  # 数据存储根目录
IMAGE_DIR = os.path.join(DATA_ROOT, "images")  # 图像目录
CSV_PATH = os.path.join(DATA_ROOT, "train_annotations.csv")  # 真值CSV
LOG_PATH = os.path.join("log", "generate_log.txt")  # 日志文件
MARS_RADIUS_KM = 3396.19  # 火星真实半径（km）
MARS_MU = 4.28284e4  # 火星引力参数（km^3/s^2）

# 轨道参数分散范围（确保数据多样性，论文表6.8/6.9适配）
ORBIT_PARAMS = {
    "a": (15000, 21000),  # 半长轴（km）：15000~21000
    "e": (0.1, 0.5),      # 偏心率：0.1~0.5
    "i": (0, 40),         # 倾角（°）：0~40
    "Omega": (0, 360),    # 升交点赤经（°）：0~360
    "omega": (0, 360),    # 近心点幅角（°）：0~360
    "f": (-90, 90)        # 真近点角（°）：-90~90
}

# 相机参数（论文表5.3适配）
CAMERA_PARAMS = {
    "resolution": [512, 512],  # 分辨率512×512
    "sensorSize": [0.01, 0.01],  # 传感器尺寸10×10mm
    "fov": 40,  # 视场角40°
    "focalLength": 0.01373  # 焦距1.373cm
}

# --------------------------
# 初始化目录与日志
# --------------------------
def init_environment():
    """创建数据目录和日志文件"""
    for dir_path in [DATA_ROOT, IMAGE_DIR, "log"]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    # 初始化CSV文件（表头）
    if not os.path.exists(CSV_PATH):
        df = pd.DataFrame(columns=[
            "image_name", "x_center", "y_center", "radius_px",
            "orbit_a", "orbit_e", "orbit_i", "sc_distance_km"
        ])
        df.to_csv(CSV_PATH, index=False)
    # 初始化日志
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n=== 数据生成开始：{time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

# --------------------------
# 生成随机轨道参数
# --------------------------
def generate_random_orbit_params():
    """生成一条随机分散的火星轨道参数"""
    params = {
        "a": random.uniform(*ORBIT_PARAMS["a"]),
        "e": random.uniform(*ORBIT_PARAMS["e"]),
        "i": np.deg2rad(random.uniform(*ORBIT_PARAMS["i"])),
        "Omega": np.deg2rad(random.uniform(*ORBIT_PARAMS["Omega"])),
        "omega": np.deg2rad(random.uniform(*ORBIT_PARAMS["omega"])),
        "f": np.deg2rad(random.uniform(*ORBIT_PARAMS["f"]))
    }
    return params

# --------------------------
# 初始化单条轨道仿真
# --------------------------
def init_orbit_simulation(orbit_params):
    """创建Basilisk仿真实例，配置单条轨道"""
    # 1. 初始化仿真核心
    sim = SimulationBaseClass.SimBaseClass()
    process = sim.CreateNewProcess("OrbitProcess")
    task = sim.CreateNewTask("OrbitTask", SIM_STEP)
    process.addTask(task)

    # 2. 创建航天器
    sc = spacecraft.Spacecraft()
    sc.ModelTag = f"sc_{int(orbit_params['a'])}"
    sc.hub.mass = 750  # 质量750kg
    sc.hub.inertia = [[900, 0, 0], [0, 800, 0], [0, 0, 600]]  # 转动惯量
    sim.AddModelToTask("OrbitTask", sc)

    # 3. 火星引力模块
    mars_grav = gravityEffector.GravityEffector()
    mars_grav.ModelTag = "MarsGrav"
    mars_grav.mu = MARS_MU
    mars_grav.r_BN_N = [0, 0, 0]  # 火星在惯性系原点
    sim.AddModelToTask("OrbitTask", mars_grav)
    sc.addGravityEffector(mars_grav)

    # 4. 相机模块（生成图像）
    cam = camera.Camera()
    cam.ModelTag = "MarsCamera"
    cam.resolution = CAMERA_PARAMS["resolution"]
    cam.sensorSize = CAMERA_PARAMS["sensorSize"]
    cam.fieldOfView = [np.deg2rad(CAMERA_PARAMS["fov"])]
    cam.focalLength = CAMERA_PARAMS["focalLength"]
    sim.AddModelToTask("OrbitTask", cam)

    # 5. 导航模块（提供姿态/位置真值）
    nav = simpleNav.SimpleNav()
    nav.ModelTag = "SimpleNav"
    sim.AddModelToTask("OrbitTask", nav)
    nav.scStateInMsg.subscribeTo(sc.scStateOutMsg)

    # 6. 设置轨道参数
    oe_msg = messaging.OrbitEpochMsgPayload()
    oe_msg.a = orbit_params["a"]
    oe_msg.e = orbit_params["e"]
    oe_msg.i = orbit_params["i"]
    oe_msg.Omega = orbit_params["Omega"]
    oe_msg.omega = orbit_params["omega"]
    oe_msg.f = orbit_params["f"]
    sc.orbitEpochInMsg.subscribeTo(messaging.OrbitEpochMsg().write(oe_msg))

    return sim, sc, cam

# --------------------------
# 单条轨道数据生成
# --------------------------
def generate_orbit_data(sim, sc, cam, orbit_params, orbit_idx):
    """生成单条轨道的IMAGES_PER_ORBIT张图像和真值"""
    # 初始化仿真
    sim.InitializeSimulation()
    # 计算单条轨道总仿真时间（生成IMAGES_PER_ORBIT张图像）
    total_sim_time = IMAGE_INTERVAL * IMAGES_PER_ORBIT
    sim.ConfigureStopTime(total_sim_time)

    # 记录生成的图像信息
    orbit_data = []
    image_count = 0

    # 运行仿真并提取数据
    while sim.GetCurrentSimTime() < total_sim_time and image_count < IMAGES_PER_ORBIT:
        # 运行一个图像间隔的仿真
        sim.RunOneStep()
        current_time = sim.GetCurrentSimTime()

        # 每隔IMAGE_INTERVAL生成一张图像
        if current_time % IMAGE_INTERVAL == 0:
            # 1. 提取航天器位置和距离
            sc_state = sc.scStateOutMsg.read()
            sc_pos = sc_state.r_BN_N  # 相对火星位置（km）
            sc_dist = np.linalg.norm(sc_pos)  # 距离（km）

            # 2. 生成图像并保存
            img_data = cam.imageOutMsg.read()
            img_array = np.array(img_data.imageData).reshape(
                CAMERA_PARAMS["resolution"][1],
                CAMERA_PARAMS["resolution"][0],
                3
            ).astype(np.uint8)
            image_name = f"mars_orbit_{orbit_idx}_img_{image_count}.png"
            img_path = os.path.join(IMAGE_DIR, image_name)
            cv2.imwrite(img_path, img_array)  # 保存图像

            # 3. 计算真值（中心坐标+半径，像素单位）
            # 焦距（像素）= 物理焦距 / 像素尺寸
            pixel_size = CAMERA_PARAMS["sensorSize"][0] / CAMERA_PARAMS["resolution"][0]
            focal_length_px = CAMERA_PARAMS["focalLength"] / pixel_size
            # 半径（像素）= 火星真实半径 / 距离 × 焦距
            radius_px = (MARS_RADIUS_KM / sc_dist) * focal_length_px
            # 中心坐标（图像中点，因航天器指向火星）
            center_x = CAMERA_PARAMS["resolution"][0] / 2
            center_y = CAMERA_PARAMS["resolution"][1] / 2

            # 4. 记录数据
            orbit_data.append({
                "image_name": image_name,
                "x_center": round(center_x, 2),
                "y_center": round(center_y, 2),
                "radius_px": round(radius_px, 2),
                "orbit_a": round(orbit_params["a"], 2),
                "orbit_e": round(orbit_params["e"], 3),
                "orbit_i": round(np.rad2deg(orbit_params["i"]), 2),
                "sc_distance_km": round(sc_dist, 2)
            })

            image_count += 1
            # 日志输出
            log_msg = f"轨道{orbit_idx}：生成图像{image_name}，距离{sc_dist:.1f}km"
            print(log_msg)
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(log_msg + "\n")

    # 将轨道数据写入CSV
    if orbit_data:
        df = pd.DataFrame(orbit_data)
        df.to_csv(CSV_PATH, mode="a", header=False, index=False)

    return image_count  # 返回实际生成的图像数

# --------------------------
# 主生成流程
# --------------------------
def main():
    # 初始化环境
    init_environment()
    total_generated = 0  # 已生成图像总数
    failed_orbits = 0    # 失败轨道数

    # 进度条
    pbar = tqdm(total=TOTAL_IMAGES, desc="批量生成训练数据")

    # 循环生成每条轨道
    for orbit_idx in range(ORBIT_COUNT):
        if total_generated >= TOTAL_IMAGES:
            break
        try:
            # 1. 生成随机轨道参数
            orbit_params = generate_random_orbit_params()
            # 2. 初始化仿真
            sim, sc, cam = init_orbit_simulation(orbit_params)
            # 3. 生成该轨道数据
            generated = generate_orbit_data(sim, sc, cam, orbit_params, orbit_idx)
            # 4. 更新统计
            total_generated += generated
            pbar.update(generated)
        except Exception as e:
            # 轨道生成失败，记录日志并跳过
            failed_orbits += 1
            err_msg = f"轨道{orbit_idx}生成失败：{str(e)}"
            print(err_msg)
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(err_msg + "\n")
            continue

    # 收尾工作
    pbar.close()
    final_msg = f"""
=== 数据生成完成 ===
目标图像数：{TOTAL_IMAGES}
实际生成数：{total_generated}
失败轨道数：{failed_orbits}
数据目录：{DATA_ROOT}
日志路径：{LOG_PATH}
"""
    print(final_msg)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(final_msg)

if __name__ == "__main__":
    # 安装依赖（若未安装）
    try:
        from tqdm import tqdm
        import cv2
    except ImportError:
        print("正在安装依赖库...")
        os.system(f"{sys.executable} -m pip install tqdm opencv-python")
        from tqdm import tqdm
        import cv2
    # 启动生成
    main()