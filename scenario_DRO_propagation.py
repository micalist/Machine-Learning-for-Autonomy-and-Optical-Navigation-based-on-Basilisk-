# -*- coding: utf-8 -*-
"""
Basilisk Scenario: Earth-Moon DRO Orbit Propagation
---------------------------------------------------
Target:
    Simulate a Distant Retrograde Orbit (DRO) in the Earth-Moon CR3BP system.
    Demonstrate coordinate transformation from Rotating Frame to Inertial Frame.

Author: 陈冠华
Date: 2025-12-06
"""

import os
import urllib.request
import numpy as np
import matplotlib.pyplot as plt

# --- Basilisk Imports ---
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros
from Basilisk.utilities import simIncludeGravBody
from Basilisk.utilities import unitTestSupport
from Basilisk.simulation import spacecraft

# --- 1. 物理常数定义 (Physical Constants) ---
# 必须与 CR3BP 无量纲化基准保持一致
L_EarthMoon = 384400.0 * 1000.0  # Length Unit (LU) [m]
MU_EARTH = 3.986004418e14  # [m^3/s^2]
MU_MOON = 4.9048695e12  # [m^3/s^2]
MU_SYS = MU_EARTH + MU_MOON
# 系统平均角速度 (Mean Motion of Earth-Moon system)
n_sys = np.sqrt(MU_SYS / L_EarthMoon ** 3)

# --- 2. DRO 初始条件 (无量纲, 旋转坐标系) ---
# 这是一个典型的平面 DRO 轨道初值
# 坐标原点：地月系质心 (Barycenter)
# x轴：地月连线，指向月球
# y轴：垂直于地月连线，指向运动方向
x_dimless = 0.8234  # 无量纲位置 x
y_dimless = 0.0  # 无量纲位置 y
z_dimless = 0.0  # 无量纲位置 z
vx_dimless = 0.0  # 无量纲速度 vx
vy_dimless = 0.5180  # 无量纲速度 vy (注意：这是旋转系下的相对速度)
vz_dimless = 0.0  # 无量纲速度 vz


def download_spice_kernel(kernel_name, download_path):
    """
    自动从 NASA NAIF 下载星历文件，并检查文件完整性
    """
    file_path = os.path.join(download_path, kernel_name)

    # 根据文件扩展名确定下载子目录
    subdir = "spk/planets" if kernel_name.endswith(".bsp") else "lsk"

    # 检查文件是否存在
    if os.path.exists(file_path):
        # 简单检查文件大小 (de430 > 100MB, naif0012 ~ 5KB)
        size_kb = os.path.getsize(file_path) / 1024
        if kernel_name.endswith(".bsp") and size_kb < 10000:  # BSP should be large
            print(f"Found incomplete {kernel_name} ({size_kb:.1f} KB). Deleting...")
            try:
                os.remove(file_path)
            except PermissionError:
                print(f"Error: Cannot delete {kernel_name}. Please delete manually.")
                return
        else:
            print(f"Found valid {kernel_name} locally.")
            return

    url = f"https://naif.jpl.nasa.gov/pub/naif/generic_kernels/{subdir}/{kernel_name}"
    print(f"Downloading {kernel_name} from NASA NAIF...")
    try:
        urllib.request.urlretrieve(url, file_path)
        print(f"Successfully downloaded {kernel_name}.")
    except Exception as e:
        print(f"Failed to download {kernel_name}. Error: {e}")
        print(f"Please manually download from: {url}")
        print(f"And place it in: {download_path}")


#main simulation program
def run_dro_simulation(show_plots=True):
    # Create simulation process
    simTaskName = "simTask"
    simProcessName = "simProcess"
    scSim = SimulationBaseClass.SimBaseClass()
    dynProcess = scSim.CreateNewProcess(simProcessName)

    # 步长设为 10秒(10秒内航天器的加速度保持不变)
    # 每隔 10秒钟，重新根据当前的地球月球位置，精确计算一次卫星的受力.保证三体积分精度
    simulationTimeStep = macros.sec2nano(10.0)
    dynProcess.addTask(scSim.CreateNewTask(simTaskName, simulationTimeStep))

    # --- 3. Setup Spacecraft ---
    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "DRO_Sat"
    scSim.AddModelToTask(simTaskName, scObject)

    # --- 4. Setup Gravity Bodies (Earth + Moon) ---
    gravFactory = simIncludeGravBody.gravBodyFactory()
    earth = gravFactory.createEarth()
    earth.isCentralBody = True  # Simulation is centered at Earth
    moon = gravFactory.createMoon()

    gravFactory.addBodiesTo(scObject)

    # --- 5. Setup SPICE (Ephemeris) ---
    # 获取当前脚本所在目录作为数据路径
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 关键修复：确保路径末尾有分隔符 (解决 "CR3BPde430.bsp" 拼接错误)
    # Basilisk 底层 C++ 往往直接拼接 dataPath + fileName
    data_path_with_sep = current_dir + os.sep

    # 1. 下载必须的内核文件
    bsp_name = 'de430.bsp'  # 星历 (位置)
    tls_name = 'naif0012.tls'  # 跳秒 (时间) - 解决 stod 错误的关键

    download_spice_kernel(bsp_name, current_dir)
    download_spice_kernel(tls_name, current_dir)

    # 2. 初始化 SPICE 接口
    timeInitString = "2026 January 1, 00:00:00.0 TDB"
    spiceObject = gravFactory.createSpiceInterface(time=timeInitString, epochInMsg=True)

    # 3. 显式加载内核 (Explicitly load kernels)
    # 必须先加载 LSK (时间)，再加载 SPK (星历)
    # 修复：传递带有分隔符的路径

    #检查与加载时间核
    if os.path.exists(os.path.join(current_dir, tls_name)):
        spiceObject.loadSpiceKernel(tls_name, data_path_with_sep)
    else:
        print(f"Error: {tls_name} missing. Time conversion will fail.")

    #检查与加载星历核
    if os.path.exists(os.path.join(current_dir, bsp_name)):
        spiceObject.loadSpiceKernel(bsp_name, data_path_with_sep)
    else:
        print(f"Error: {bsp_name} missing. Orbit propagation will fail.")
        return

    spiceObject.zeroBase = 'Earth'
    scSim.AddModelToTask(simTaskName, spiceObject)

    # --- 6. 初始状态转换 (Rotating -> Inertial) ---

    # 1. 还原有量纲单位
    pos_rot = np.array([x_dimless, y_dimless, z_dimless]) * L_EarthMoon
    vel_rot = np.array([vx_dimless, vy_dimless, vz_dimless]) * (L_EarthMoon * n_sys)

    # 2. 坐标原点修正：从"质心"移到"地心". CR3BP->质心   Basilisk软件要求->地心
    mu_cr3bp = MU_MOON / MU_SYS
    r_Sat_Earth_rot_x = pos_rot[0] - (-mu_cr3bp * L_EarthMoon)
    r_Sat_Earth_rot = np.array([r_Sat_Earth_rot_x, pos_rot[1], pos_rot[2]])

    # 3. 速度转换 (Velocity Transport Theorem)，V惯性=V相对速度+V牵连速度
    omega_vec = np.array([0, 0, n_sys])  # 假设月球绕Z轴公转
    v_transport = np.cross(omega_vec, r_Sat_Earth_rot)
    v_inertial = vel_rot + v_transport

    print(f"Initial Pos (Inertial): {r_Sat_Earth_rot}")
    print(f"Initial Vel (Inertial): {v_inertial}")


    #赋予卫星在地心惯性系下的初始速度和位置
    scObject.hub.r_CN_NInit = r_Sat_Earth_rot
    scObject.hub.v_CN_NInit = v_inertial

    # --- 7. Setup Data Logging ---
    samplingTime = macros.sec2nano(3600.)  # 1 hour
    rec_sc = scObject.scStateOutMsg.recorder(samplingTime)
    scSim.AddModelToTask(simTaskName, rec_sc)

    # 记录月球状态
    # 修复 AttributeError: 直接使用索引访问月球消息
    # 在 gravFactory 中，我们先创建了 Earth，后创建了 Moon
    # 只要 spiceObject 配置正确，它会包含两个 output msg
    # Index 0 -> Earth (因为 zeroBase='Earth', 这个消息通常是全零，或者不存在，取决于实现)
    # Index 1 -> Moon

    # 安全起见，我们记录所有的 planetStateOutMsgs，然后在绘图时判断哪个是月球
    # 月球的特点：距离原点(地球)约 384400 km

    rec_planet_0 = spiceObject.planetStateOutMsgs[0].recorder(samplingTime)
    scSim.AddModelToTask(simTaskName, rec_planet_0)

    rec_planet_1 = None
    if len(spiceObject.planetStateOutMsgs) > 1:
        rec_planet_1 = spiceObject.planetStateOutMsgs[1].recorder(samplingTime)
        scSim.AddModelToTask(simTaskName, rec_planet_1)

    # --- 8. Run Simulation ---
    scSim.InitializeSimulation()
    # 跑 27 天
    sim_days = 60.0
    simulationTime = macros.sec2nano(sim_days * 86400.0)
    scSim.ConfigureStopTime(simulationTime)

    print("Starting Simulation...")
    scSim.ExecuteSimulation()
    print("Simulation Finished.")

    # --- 9. Post-Processing & Plotting ---
    if show_plots:
        pos_Sat_N = rec_sc.r_BN_N

        # 智能判断哪个是月球数据
        # 检查 rec_planet_0 的平均距离
        dist_0 = np.mean(np.linalg.norm(rec_planet_0.PositionVector, axis=1))

        pos_Moon_N = None
        if dist_0 > 100000.0 * 1000.0:  # 如果大于 10万公里，肯定是月球
            pos_Moon_N = rec_planet_0.PositionVector
            print("Identified Moon at Index 0")
        elif rec_planet_1:
            dist_1 = np.mean(np.linalg.norm(rec_planet_1.PositionVector, axis=1))
            if dist_1 > 100000.0 * 1000.0:
                pos_Moon_N = rec_planet_1.PositionVector
                print("Identified Moon at Index 1")

        if pos_Moon_N is None:
            print("Warning: Could not identify Moon trajectory. Plotting might be incorrect.")
            # Fallback to dummy data to avoid crash
            pos_Moon_N = np.zeros_like(pos_Sat_N)
            pos_Moon_N[:, 0] = 384400.0 * 1000.0  # 假装月球在 X 轴

        # 1. 惯性系轨迹
        plt.figure(1, figsize=(6, 6))
        plt.plot(pos_Sat_N[:, 0] / 1000, pos_Sat_N[:, 1] / 1000, label='Spacecraft')
        plt.plot(pos_Moon_N[:, 0] / 1000, pos_Moon_N[:, 1] / 1000, 'r--', label='Moon Orbit')
        plt.scatter(0, 0, c='b', s=50, label='Earth')
        plt.xlabel('X [km]')
        plt.ylabel('Y [km]')
        plt.title('Inertial Frame Trajectory')
        plt.legend()
        plt.axis('equal')

        # 2. 旋转系轨迹 (Rotating Frame)
        pos_rot_x = []
        pos_rot_y = []

        for i in range(len(pos_Sat_N)):
            r_sat = pos_Sat_N[i]
            r_moon = pos_Moon_N[i]  # 地心指向月球的向量

            norm_r_moon = np.linalg.norm(r_moon)
            if norm_r_moon < 1.0:  # 防止除零（如果月球数据是0）
                i_r = np.array([1.0, 0.0, 0.0])
            else:
                i_r = r_moon / norm_r_moon

            i_h = np.array([0, 0, 1])
            i_theta = np.cross(i_h, i_r)

            x_rot = np.dot(r_sat, i_r)
            y_rot = np.dot(r_sat, i_theta)

            pos_rot_x.append(x_rot)
            pos_rot_y.append(y_rot)

        plt.figure(2, figsize=(6, 6))
        plt.plot(np.array(pos_rot_x) / 1000, np.array(pos_rot_y) / 1000, 'g-', linewidth=2, label='DRO Trajectory')

        moon_dist = np.mean(np.linalg.norm(pos_Moon_N, axis=1)) / 1000
        plt.scatter(moon_dist, 0, c='gray', s=100, label='Moon (Fixed)')
        plt.scatter(0, 0, c='b', s=100, label='Earth (Fixed)')

        plt.scatter(pos_rot_x[0] / 1000, pos_rot_y[0] / 1000, c='k', marker='o', s=100, zorder=5, label='Start')


        plt.scatter(pos_rot_x[-1] / 1000, pos_rot_y[-1] / 1000, c='r', marker='x', s=100, linewidth=3, zorder=5,
                    label='End (60 days)')

        plt.xlabel('Rotating X [km]')
        plt.ylabel('Rotating Y [km]')
        plt.title(f'Rotating Frame Trajectory (DRO Shape)\nInitial: x={x_dimless}, vy={vy_dimless}')
        plt.grid(True)
        plt.legend()
        plt.axis('equal')

        plt.show()

    return


if __name__ == "__main__":
    run_dro_simulation()