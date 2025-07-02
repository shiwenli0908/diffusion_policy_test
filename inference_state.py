import os
import matplotlib.pyplot as plt
import numpy as np
import json
import imageio
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.encoder import get_resnet, replace_bn_with_gn
from diffusion_policy.encoder import PotentialResNetEncoder
from diffusion_policy.model import ConditionalUnet1D
from dataset.dataset import BlockReachPotentialDataset2
from dataset.utils import normalize_data, unnormalize_data

from env.block_reaching.block_reaching_multimodal import BlockPushMultimodal
from env.block_reaching.utils.adaptation import move_to_position, transform, retransform

pred_horizon = 16
obs_horizon = 2
action_horizon = 8

dataset = BlockReachPotentialDataset2(
    data_dir="FullNPZDataset2",  # 包含 sample_xxxxx 子目录的主目录
    pred_horizon=16,
    obs_horizon=2,
    action_horizon=8
)

# save training data statistics (min, max) for each dim
stats = dataset.stats


# ResNet18 has output dim of 512
#vision_feature_dim = 512
#potential_feature_dim = 512
# agent_pos is 2 dimensional
lowdim_obs_dim = 2
target_dim = 3
# observation feature has 514 dims in total per step
obs_dim = lowdim_obs_dim + target_dim
action_dim = 2

#global_cond_dim = (vision_feature_dim + lowdim_obs_dim + target_dim) * obs_horizon + potential_feature_dim + obs_horizon
global_cond_dim = (lowdim_obs_dim + target_dim) * obs_horizon + 2*obs_horizon

noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=global_cond_dim
)

# the final arch has 2 parts
nets = nn.ModuleDict({
    #'vision_encoder': vision_encoder,
    #'potential_encoder': potential_encoder,
    'noise_pred_net': noise_pred_net
})

# for this demo, we use DDPMScheduler with 100 diffusion iterations
num_diffusion_iters = 100
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    beta_schedule='squaredcos_cap_v2',
    clip_sample=True,
    prediction_type='epsilon'
)

# device transfer
device = torch.device('cuda')
_ = nets.to(device)

load_pretrained = True
if load_pretrained:
  ckpt_path = "/home/wenli/checkpoints/blockreach_vision_cfg_state_200.ckpt"

  state_dict = torch.load(ckpt_path, map_location='cuda')
  ema_nets = nets
  ema_nets.load_state_dict(state_dict)
  print('Pretrained weights loaded.')
else:
  print("Skipped pretrained weight loading.")

# === inference ===

scene_idx = 0  # 选择要测试的场景索引
target_idx = 1  # 选择要测试的目标索引
scoref = 0.6  # 设置一个 scoref 值

target_onehot = np.zeros((3,))
target_onehot[target_idx] = 1.0

root_dir = "FullNPZDataset2"
scene_dir = os.path.join(root_dir, f"sample_{scene_idx:05d}")
config_path = os.path.join(scene_dir, "config_potential.json")

# 读取 JSON 配置文件
with open(config_path, 'r') as f:
    config = json.load(f)

# 提取两个目标的位置信息
target_positions = [t["position"] for t in config["targets"]]
transformed_positions = [transform(pos)[:2].tolist() for pos in target_positions]

# 初始化环境
env = BlockPushMultimodal(custom_block_positions=transformed_positions)

start_pos = np.array([0.4, -0.3])
obs, _ = move_to_position(env, start_pos, render=False)

action = np.array([0.0, 0.0])
obs, reward, done, info = env.step(action)

effector_pos = obs["effector_translation"]

def render_resized(rgb, size=(96, 96)):
    rgb_resized = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)  # (96, 96, 3)
    rgb_transposed = np.transpose(rgb_resized, (2, 0, 1))              # (3, 96, 96)
    return rgb_transposed.astype(np.float32) / 255.0     # [0,1] float32

frame = env.render(mode='rgb_array')                   # (240, 320, 3)
frame_proc = render_resized(frame)
print("RGB shape:", frame.shape)

def extract_obs_vector(obs_dict):
    return np.concatenate([
        obs_dict["effector_translation"],   # (2,)
    ])  # → (2,)

#obs_vec = np.concatenate([start_pos, target_onehot])  # (4,)
obs_vec = extract_obs_vector(obs)
#obs_vec = retransform(obs_vec)  # 将 effector_pos 转换回原始空间
obs_deque = collections.deque(
    [{"obs": obs_vec.copy()}] * obs_horizon,
    maxlen=obs_horizon
)

imgs = [frame]
step_idx = 0
done = False
rewards = []
max_steps = 100

with torch.no_grad():
    while not done and step_idx < max_steps:
        B = 1

        obs_seq = np.stack([x["obs"] for x in obs_deque])  # (obs_h, 2)
        obs_seq = normalize_data(obs_seq, stats['obs'])  # (obs_h, 2)
        target_seq = np.tile(target_onehot, (obs_horizon, 1))  # (obs_h, 2)

        nagent = np.concatenate([obs_seq, target_seq], axis=-1)  # (obs_h, 4)

        nagent = torch.from_numpy(nagent).float().to(device)

        obs_feat = torch.cat([nagent], dim=-1)  # (obs_h, 514)
        obs_cond = obs_feat.unsqueeze(0).flatten(start_dim=1)   # (1, obs_horizon * D)

        '''#potential_feat = nets["potential_encoder"](potential_tensor)  # (1,512)
        score_feat = torch.tensor([[scoref]] * obs_horizon, device=device).float().view(1, -1)

        # CFG dropout
        #drop = (torch.rand(1).item() < 0.1)
        drop = False
        if drop:
            score_feat *= 0.0
            #potential_feat *= 0.0'''

        score_vec = torch.tensor([scoref, 1.0], device=device).float()  # shape: (2,)
        score_seq = score_vec.unsqueeze(0).repeat(obs_horizon, 1)  # (obs_horizon, 2)

        # Classifier-Free Guidance（按需禁用）
        drop = True  # or True if you want to test unconditional
        if drop:
            score_seq[:] = 0.0  # 整段 [score, mask] → [0, 0]

        # reshape to (1, obs_horizon * 2)
        score_feat = score_seq.unsqueeze(0).flatten(start_dim=1)  # shape: (1, obs_horizon * 2)

        #global_cond = torch.cat([obs_cond, potential_feat, score_feat], dim=-1)  # (1, D)
        global_cond = torch.cat([obs_cond, score_feat], dim=-1)  # (1, D)

        naction = torch.randn((1, pred_horizon, 2), device=device)
        noise_scheduler.set_timesteps(num_diffusion_iters)
        for k in noise_scheduler.timesteps:
            noise_pred = nets["noise_pred_net"](naction, k, global_cond)
            naction = noise_scheduler.step(noise_pred, k, naction).prev_sample

        naction = naction[0].cpu().numpy()
        action_pred = unnormalize_data(naction, stats['action'])  # (T,2)
        act_seq = action_pred[obs_horizon-1 : obs_horizon-1 + action_horizon]

        for act in act_seq:
            #act = transform(act)  # 将动作转换回原始空间
            obs, reward, done, info = env.step(act)
            frame = env.render(mode='rgb_array')
            #frame_proc = render_resized(frame)
            #obs_vec = np.concatenate([obs["effector_translation"], target_onehot])
            obs_vec = obs["effector_translation"]  # (2,)
            #obs_vec = retransform(obs_vec)  # 将 effector_pos 转换回原始空间
            obs_deque.append({"obs": obs_vec})
            imgs.append(frame)

            rewards.append(reward)

            step_idx += 1
            if step_idx > max_steps:
                done = True
            if done:
                break

print("Score:", max(rewards))
# Save video
os.makedirs("videos", exist_ok=True)
video_path = f"videos/infer_scene{scene_idx}_target{target_idx}_state.mp4"
imageio.mimsave(video_path, imgs, fps=10)
print("Saved video to:", video_path)



