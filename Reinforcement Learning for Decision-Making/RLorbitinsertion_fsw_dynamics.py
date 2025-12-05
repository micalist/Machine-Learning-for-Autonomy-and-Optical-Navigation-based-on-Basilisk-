import numpy as np
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector import SyncVectorEnv

# -------------------------- Basilisk 核心模块导入 --------------------------
from Basilisk.utilities import SimulationBaseClass, macros, orbitalMotion, unitTestSupport
from Basilisk.architecture import messaging

# -------------------------- 导入提供的外部 BSK 模块 --------------------------
sys.path.append(os.getcwd())

try:
    from BSK_OpNavDynamics import BSKDynamicModels
    from BSK_OpNavFsw import BSKFswModels
    import BSK_OpNavFsw
except ImportError as e:
    print("错误: 无法导入 BSK_OpNavDynamics 或 BSK_OpNavFsw。请确保这两个文件与本脚本在同一目录下。")
    raise e

# -------------------------- 全局配置 --------------------------
BSK_OpNavFsw.centerRadiusCNNIncluded = False


# -------------------------- 优化后的 BSK 配置类 --------------------------

class OptimizedBSKDynamicModels(BSKDynamicModels):
    """
    轻量化动力学模型
    """

    def SetVizInterface(self, SimBase):
        pass  # 禁用 Vizard

    def SetCamera(self):
        if not hasattr(self, 'dummyImageMsg'):
            self.dummyImageMsg = messaging.CameraImageMsg()
        self.cameraMod.imageInMsg.subscribeTo(self.dummyImageMsg)
        self.cameraMRP_CB = [0., 0., 0.]
        self.cameraMod.cameraIsOn = 0
        self.cameraMod.saveImages = 0

    def SetCamera2(self):
        if not hasattr(self, 'dummyImageMsg2'):
            self.dummyImageMsg2 = messaging.CameraImageMsg()
        self.cameraMod2.imageInMsg.subscribeTo(self.dummyImageMsg2)
        self.cameraMod2.cameraIsOn = 0

    def SetSimpleNavObject(self):
        self.SimpleNavObject.ModelTag = "SimpleNavigation"
        self.SimpleNavObject.PMatrix = np.zeros((18, 18))
        self.SimpleNavObject.walkBounds = np.array([1e20] * 18)
        self.SimpleNavObject.crossTrans = True
        self.SimpleNavObject.crossAtt = False
        self.SimpleNavObject.scStateInMsg.subscribeTo(self.scObject.scStateOutMsg)


class OptimizedBSKFswModels(BSKFswModels):
    """
    极速版飞行软件模型
    """

    def InitAllFSWObjects(self, SimBase):
        self.SetHillPointGuidance(SimBase)
        self.SetVehicleConfiguration()
        self.SetRWConfigMsg()
        self.SetMRPFeedbackRWA(SimBase)
        self.SetRWMotorTorque(SimBase)
        self.SetAttTrackingErrorCam(SimBase)
        self.SetOpNavPointGuidance(SimBase)

        self.SetImageProcessing(SimBase)
        self.SetPixelLineConversion(SimBase)
        self.SetRelativeODFilter()
        self.SetFaultDetection(SimBase)
        self.SetLimbFinding(SimBase)
        self.SetHorizonNav(SimBase)
        self.SetHeadingUKF(SimBase)
        self.SetPixelLineFilter(SimBase)

    def SetImageProcessing(self, SimBase):
        self.imageProcessing.imageInMsg.subscribeTo(SimBase.DynModels.cameraMod.imageOutMsg)
        self.imageProcessing.opnavCirclesOutMsg = self.opnavCirclesMsg

    def SetPixelLineConversion(self, SimBase):
        self.pixelLine.circlesInMsg.subscribeTo(self.opnavCirclesMsg)
        self.pixelLine.cameraConfigInMsg.subscribeTo(SimBase.DynModels.cameraMod.cameraConfigOutMsg)
        self.pixelLine.attInMsg.subscribeTo(SimBase.DynModels.SimpleNavObject.attOutMsg)
        messaging.OpNavMsg_C_addAuthor(self.pixelLine.opNavOutMsg, self.opnavMsg)

    def SetRelativeODFilter(self):
        self.relativeOD.opNavInMsg.subscribeTo(self.opnavMsg)
        self.relativeOD.stateInit = [1e7, 0, 0, 0, 7e3, 0]
        self.relativeOD.covarInit = (np.eye(6) * 1e6).flatten().tolist()

    def SetFaultDetection(self, SimBase):
        self.opNavFault.navMeasPrimaryInMsg.subscribeTo(self.opnavPrimaryMsg)
        self.opNavFault.navMeasSecondaryInMsg.subscribeTo(self.opnavSecondaryMsg)
        self.opNavFault.cameraConfigInMsg.subscribeTo(SimBase.DynModels.cameraMod.cameraConfigOutMsg)
        self.opNavFault.attInMsg.subscribeTo(SimBase.DynModels.SimpleNavObject.attOutMsg)
        messaging.OpNavMsg_C_addAuthor(self.opNavFault.opNavOutMsg, self.opnavMsg)

    def SetLimbFinding(self, SimBase):
        self.limbFinding.imageInMsg.subscribeTo(SimBase.DynModels.cameraMod.imageOutMsg)

    def SetHorizonNav(self, SimBase):
        self.horizonNav.limbInMsg.subscribeTo(self.limbFinding.opnavLimbOutMsg)
        self.horizonNav.cameraConfigInMsg.subscribeTo(SimBase.DynModels.cameraMod.cameraConfigOutMsg)
        self.horizonNav.attInMsg.subscribeTo(SimBase.DynModels.SimpleNavObject.attOutMsg)
        messaging.OpNavMsg_C_addAuthor(self.horizonNav.opNavOutMsg, self.opnavMsg)

    def SetHeadingUKF(self, SimBase):
        self.headingUKF.opnavDataInMsg.subscribeTo(self.opnavMsg)
        self.headingUKF.cameraConfigInMsg.subscribeTo(SimBase.DynModels.cameraMod.cameraConfigOutMsg)

    def SetPixelLineFilter(self, SimBase):
        if not hasattr(self, 'dummyCirclesMsg'):
            self.dummyCirclesMsg = messaging.OpNavCirclesMsg()
        self.pixelLineFilter.circlesInMsg.subscribeTo(self.dummyCirclesMsg)
        self.pixelLineFilter.cameraConfigInMsg.subscribeTo(SimBase.DynModels.cameraMod.cameraConfigOutMsg)
        self.pixelLineFilter.attInMsg.subscribeTo(SimBase.DynModels.SimpleNavObject.attOutMsg)

    def setupGatewayMsgs(self):
        super().setupGatewayMsgs()


# -------------------------- 1. Basilisk 仿真环境封装类 --------------------------
class BasiliskRLWrapper:
    def __init__(self, scenario="MOI"):
        self.scenario = scenario
        self.initialized = False

        self.simBase = SimulationBaseClass.SimBaseClass()
        self.simBase.SetProgressBar(False)
        self.simBase.modeRequest = None

        self.dynRate = 5.0
        self.fswRate = 5.0
        self.step_duration_ns = macros.sec2nano(self.dynRate)

        dynProcessName = "DynamicsProcess"
        fswProcessName = "FSWProcess"
        self.dynProc = self.simBase.CreateNewProcess(dynProcessName)
        self.fswProc = self.simBase.CreateNewProcess(fswProcessName)
        self.simBase.DynamicsProcessName = dynProcessName
        self.simBase.FSWProcessName = fswProcessName
        self.simBase.dynProc = self.dynProc
        self.simBase.fswProc = self.fswProc

        self.DynModels = OptimizedBSKDynamicModels(self.simBase, self.dynRate)
        self.simBase.DynModels = self.DynModels
        self.FSWModels = OptimizedBSKFswModels(self.simBase, self.fswRate)
        self.simBase.FSWModels = self.FSWModels

        self.scObject = self.DynModels.scObject
        self.mars_grav = self.DynModels.gravFactory.gravBodies['mars barycenter']

        self.thrustCmdMsg = messaging.THRArrayOnTimeCmdMsg()
        self.DynModels.thrustersDynamicEffector.cmdsInMsg.subscribeTo(self.thrustCmdMsg)

        tasks_to_disable = [
            "CameraTask", "opNavODTask", "imageProcTask", "opNavODTaskLimb",
            "opNavFaultDet", "opNavODTaskB", "attODFaultDet"
        ]
        for task in tasks_to_disable:
            try:
                self.simBase.disableTask(task)
            except Exception:
                pass

        self.target_a = 2000 * 1000
        self.target_e = 0.01
        self.covariance = np.eye(6) * 100.0

    def setup_initial_conditions(self):
        oe = orbitalMotion.ClassicElements()
        mu = self.mars_grav.mu

        if self.scenario == "MOI":
            oe.a = -2000 * 1000
            oe.e = 2.0
            oe.i = 0.0 * macros.D2R
            oe.Omega = 0.0 * macros.D2R
            oe.omega = 0.0 * macros.D2R
            oe.f = 0.0 * macros.D2R
        else:
            oe.a = 18000 * 1000
            oe.e = 0.6
            oe.i = 10.0 * macros.D2R
            oe.Omega = 25.0 * macros.D2R
            oe.omega = 190.0 * macros.D2R
            oe.f = 80.0 * macros.D2R

        rN, vN = orbitalMotion.elem2rv(mu, oe)

        self.scObject.hub.r_CN_NInit = [[rN[0]], [rN[1]], [rN[2]]]
        self.scObject.hub.v_CN_NInit = [[vN[0]], [vN[1]], [vN[2]]]
        self.scObject.hub.sigma_BNInit = [[0.1], [0.0], [0.1]]
        self.scObject.hub.omega_BN_BInit = [[0.001], [-0.001], [0.001]]

    def get_state(self):
        scStateMsg = self.scObject.scStateOutMsg.read()
        r = np.array(scStateMsg.r_BN_N)
        v = np.array(scStateMsg.v_BN_N)
        mu = self.mars_grav.mu
        oe = orbitalMotion.rv2elem(mu, r, v)

        if self.scenario == "MOI":
            a_error = (oe.a - self.target_a) / 1000
            e_error = oe.e - self.target_e
            currentTime = self.simBase.TotalSim.CurrentNanos * macros.NANO2SEC
            thrust_countdown = max(0, 8 - (currentTime / 120))
            observation = np.array([a_error, e_error, np.trace(self.covariance) / 6, thrust_countdown],
                                   dtype=np.float32)
        else:
            sun_pos = np.array([1.52 * 1e11, 0.0, 0.0])
            sc_dist = np.linalg.norm(r)
            sun_dist = 1.52 * 1e11
            phase_angle = 0.0
            if sc_dist > 1e-6:
                dot_prod = np.dot(r, sun_pos)
                phase_angle = dot_prod / (sc_dist * sun_dist)
            norm_cov = np.trace(self.covariance) / 1000
            observation = np.array([phase_angle, norm_cov], dtype=np.float32)

        return observation

    def execute_action(self, action):
        reward = 0.0
        done = False

        if self.scenario == "MOI" and action == 2:
            thrCmdPayload = messaging.THRArrayOnTimeCmdMsgPayload()
            thrCmdPayload.OnTimeRequest = [self.dynRate] * 8
            self.thrustCmdMsg.write(thrCmdPayload)
        else:
            thrCmdPayload = messaging.THRArrayOnTimeCmdMsgPayload()
            thrCmdPayload.OnTimeRequest = [0.0] * 8
            self.thrustCmdMsg.write(thrCmdPayload)

        if self.scenario == "MOI":
            if action == 0:
                self.simBase.modeRequest = "prepOpNav"
                self.covariance *= 0.95
                reward -= 0.1
            elif action == 1:
                self.simBase.modeRequest = "prepOpNav"
                reward -= 0.1
            elif action == 2:
                self.simBase.modeRequest = "prepOpNav"
                reward += 10.0
        else:
            if action == 0:
                self.simBase.modeRequest = "pointHead"
                reward += 0.1
            elif action == 1:
                self.simBase.modeRequest = "prepOpNav"
                self.covariance *= 0.8
                reward -= 0.05

        stop_time = self.simBase.TotalSim.CurrentNanos + self.step_duration_ns
        self.simBase.ConfigureStopTime(stop_time)
        self.simBase.ExecuteSimulation()

        state = self.get_state()

        if self.simBase.TotalSim.CurrentNanos > macros.sec2nano(3600 * 2):
            done = True

        if self.scenario == "MOI":
            current_rv = self.scObject.scStateOutMsg.read()
            current_r = np.array(current_rv.r_BN_N)
            current_v = np.array(current_rv.v_BN_N)
            current_a = orbitalMotion.rv2elem(self.mars_grav.mu, current_r, current_v).a
            if 2000 * 1000 < current_a < 10000 * 1000:
                reward += 100.0
                done = True

        truncated = False
        info = {}
        return state, reward, done, truncated, info

    def reset(self):
        self.setup_initial_conditions()
        if not self.initialized:
            self.simBase.InitializeSimulation()
            self.initialized = True
        else:
            self.simBase.Reset()

        self.simBase.modeRequest = "standby"
        emptyPayload = messaging.THRArrayOnTimeCmdMsgPayload()
        emptyPayload.OnTimeRequest = [0.0] * 8
        self.thrustCmdMsg.write(emptyPayload)
        self.covariance = np.eye(6) * 100.0
        return self.get_state(), {}

    def close(self):
        pass


# -------------------------- 2. Gymnasium 环境类 --------------------------
class MOIEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, render_mode=None):
        super(MOIEnv, self).__init__()
        self.sim = BasiliskRLWrapper(scenario="MOI")
        self.observation_space = spaces.Box(
            low=np.array([-50000.0, -5.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([50000.0, 5.0, 1000.0, 10.0], dtype=np.float32),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)
        self.render_mode = render_mode

    def step(self, action):
        return self.sim.execute_action(action)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        return self.sim.reset()

    def close(self):
        self.sim.close()


# -------------------------- 3. RL 算法 (DQN) --------------------------
class DQN:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=0.2, device=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = 128
        self.memory = []
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = self._build_network().to(self.device)
        self.target_net = self._build_network().to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def _build_network(self):
        return nn.Sequential(
            nn.Linear(self.state_dim, 64), nn.ReLU(), nn.Linear(64, self.action_dim)
        )

    def store_experience(self, s, a, r, s_, done, truncated):
        self.memory.append((s, a, r, s_, done))
        if len(self.memory) > 50000: self.memory.pop(0)

    def choose_action(self, s):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.action_dim)
        else:
            s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_net(s_tensor)
            return torch.argmax(q_values).item()

    def update_target_net(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def learn(self):
        if len(self.memory) < self.batch_size: return

        # [修复] 正确的采样逻辑：先采样索引，再取数据
        indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[i] for i in indices]

        s_batch = torch.tensor(np.array([x[0] for x in batch]), dtype=torch.float32).to(self.device)
        a_batch = torch.tensor(np.array([x[1] for x in batch]), dtype=torch.long).to(self.device)
        r_batch = torch.tensor(np.array([x[2] for x in batch]), dtype=torch.float32).to(self.device)
        s_next_batch = torch.tensor(np.array([x[3] for x in batch]), dtype=torch.float32).to(self.device)
        done_batch = torch.tensor(np.array([x[4] for x in batch]), dtype=torch.bool).to(self.device)

        with torch.no_grad():
            q_next = self.target_net(s_next_batch)
            max_q_next = q_next.max(dim=1)[0]
            q_target = r_batch + self.gamma * max_q_next * (~done_batch)

        q_pred = self.q_net(s_batch)
        q_pred_selected = q_pred.gather(1, a_batch.unsqueeze(1)).squeeze(1)
        loss = self.loss_fn(q_pred_selected, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# -------------------------- 4. 并行训练逻辑 --------------------------
def make_env():
    return MOIEnv(render_mode=None)


def train_moi_dqn_parallel():
    print("开始训练 MOI-DQN (稳定版 - Sync模式 + Fast Reset + 最终修复)...")

    num_envs = 16
    envs = SyncVectorEnv([make_env for _ in range(num_envs)])

    state_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.n
    dqn = DQN(state_dim, action_dim)

    total_timesteps = 50000
    steps_done = 0
    learn_frequency = 5
    update_target_frequency = 200
    learn_count = 0

    states, _ = envs.reset()

    import time
    start_time = time.time()

    while steps_done < total_timesteps:
        actions = []
        for i in range(num_envs):
            if np.random.uniform(0, 1) < dqn.epsilon:
                actions.append(np.random.choice(action_dim))
            else:
                s_tensor = torch.tensor(states[i], dtype=torch.float32).unsqueeze(0).to(dqn.device)
                with torch.no_grad():
                    q_values = dqn.q_net(s_tensor)
                actions.append(torch.argmax(q_values).item())

        next_states, rewards, terminations, truncations, infos = envs.step(actions)
        dones = terminations | truncations

        for i in range(num_envs):
            dqn.store_experience(states[i], actions[i], rewards[i], next_states[i], dones[i], False)

        states = next_states
        steps_done += num_envs

        if steps_done % learn_frequency == 0:
            dqn.learn()
            learn_count += 1

            dqn.epsilon = max(0.01, dqn.epsilon * 0.9999)

            if learn_count % update_target_frequency == 0:
                dqn.update_target_net()
                current_time = time.time()
                elapsed = current_time - start_time
                fps = steps_done / elapsed if elapsed > 0 else 0
                print(f"Steps: {steps_done}, FPS: {fps:.1f}, Epsilon: {dqn.epsilon:.3f}, Memory: {len(dqn.memory)}")

    print("训练完成，保存模型...")
    torch.save(dqn.q_net.state_dict(), "moi_dqn_parallel.pth")
    envs.close()


if __name__ == "__main__":
    train_moi_dqn_parallel()
