import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.encoder import get_resnet, replace_bn_with_gn
from diffusion_policy.encoder import PotentialResNetEncoder
from diffusion_policy.model import ConditionalUnet1D
from dataset.dataset import BlockReachPotentialDataset, BlockReachPotentialDataset2

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

# create dataloader
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=64,
    #num_workers=4,
    shuffle=True
)

'''# visualize data in batch
batch = next(iter(dataloader))

print(batch['obs'].shape)         # torch.Size([32, 2, 2])
print(batch['action'].shape)      # torch.Size([32, 8, 2])
print(batch['image'].shape)       # torch.Size([32, 2, 3, 96, 96])
print(batch['target'].shape)      # torch.Size([32, 8, 2])
print(batch['scoref'].shape)      # torch.Size([32, 8, 1])
#print(batch['potential'].shape)   # torch.Size([32, 1, H, W])'''

# construct ResNet18 encoder
vision_encoder = get_resnet('resnet18')
vision_encoder = replace_bn_with_gn(vision_encoder)

'''
# --- instantiate visual encoder for potential map ---
potential_encoder = PotentialResNetEncoder(
    name='resnet18',
    output_dim=512,
    pretrained=False,
    group_norm=True
)
'''

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
global_cond_dim = (lowdim_obs_dim + target_dim) * obs_horizon + 2 * obs_horizon

noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=global_cond_dim
)


'''# create network object
noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim= obs_dim * obs_horizon
)
'''


# the final arch has 2 parts
nets = nn.ModuleDict({
    'noise_pred_net': noise_pred_net
})

# for this demo, we use DDPMScheduler with 100 diffusion iterations
num_diffusion_iters = 100
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    # the choise of beta schedule has big impact on performance
    # we found squared cosine works the best
    beta_schedule='squaredcos_cap_v2',
    # clip output to [-1,1] to improve stability
    clip_sample=True,
    # our network predicts noise (instead of denoised action)
    prediction_type='epsilon'
)

# device transfer
device = torch.device('cuda')
_ = nets.to(device)

num_epochs = 200

# Exponential Moving Average
# accelerates training and improves stability
# holds a copy of the model weights
ema = EMAModel(
    parameters=nets.parameters(),
    power=0.75)

# Standard ADAM optimizer
# Note that EMA parametesr are not optimized
optimizer = torch.optim.AdamW(
    params=nets.parameters(),
    lr=1e-4, weight_decay=1e-6)

# Cosine LR schedule with linear warmup
lr_scheduler = get_scheduler(
    name='cosine',
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(dataloader) * num_epochs
)

all_epoch_losses = []

with tqdm(range(num_epochs), desc='Epoch') as tglobal:
    # epoch loop
    for epoch_idx in tglobal:
        epoch_loss = list()
        # batch loop
        with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
            for nbatch in tepoch:
                # data normalized in dataset
                # device transfer
                #nimage = nbatch['image'][:,:obs_horizon].to(device)
                nagent_pos = nbatch['obs'][:,:obs_horizon].to(device)
                target = nbatch['target'][:,:obs_horizon].to(device)
                naction = nbatch['action'].to(device)
                B = nagent_pos.shape[0]

                # encoder vision features
                #image_features = nets['vision_encoder'](nimage.flatten(end_dim=1))
                #image_features = image_features.reshape(*nimage.shape[:2], -1)  # (B, obs_horizon, 512)

                '''
                # potential encoder
                potential_map = nbatch['potential'].to(device)  # (B, 1, H, W)
                potential_features = nets['potential_encoder'](potential_map)  # (B, 512)
                '''

                scoref = nbatch['scoref'][:, :obs_horizon].to(device)  # (B, obs_horizon, 2)

                '''# (2) Classifier-Free Guidance dropout mask
                mask = (torch.rand(B, device=device) < 0.1).float().view(B, 1, 1)  # (B, 1, 1)

                # (3) Apply masking by multiplication (safe for autograd)
                scoref_masked = scoref * (1.0 - mask)  # (B, obs_horizon, 1)
                #potential_masked = potential_features * (1.0 - mask[:, 0, 0].view(B, 1))  # (B, 512)'''

                # (2) Classifier-Free Guidance dropout mask (per-sample)
                drop_mask = (torch.rand(B, device=device) < 0.1).view(B, 1, 1)  # shape: (B, 1, 1)

                # (3) Expand to match scoref shape
                drop_mask = drop_mask.expand(-1, obs_horizon, 2)  # shape: (B, obs_horizon, 2)

                # (4) Apply dropout: set entire [score, 1] → [0, 0] if masked
                scoref_masked = torch.where(drop_mask.bool(), torch.zeros_like(scoref), scoref)

                # (5) Flatten for input
                score_cond = scoref_masked.flatten(start_dim=1)  # shape: (B, obs_horizon * 2)


                # (4) obs embedding
                obs_features = torch.cat([nagent_pos, target], dim=-1)  # (B, obs_horizon, D)
                obs_cond = obs_features.flatten(start_dim=1)  # (B, obs_horizon * D)

                # (5) concatenate potential + scoref (flattened)
                #score_cond = scoref_masked.flatten(start_dim=1)  # (B, obs_horizon)
                #global_cond = torch.cat([obs_cond, potential_masked, score_cond], dim=-1)  # (B, D)
                global_cond = torch.cat([obs_cond, score_cond], dim=-1)  # (B, D)

                # sample noise to add to actions
                noise = torch.randn(naction.shape, device=device)

                # sample a diffusion iteration for each data point
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (B,), device=device
                ).long()

                # add noise to the clean images according to the noise magnitude at each diffusion iteration
                # (this is the forward diffusion process)
                noisy_actions = noise_scheduler.add_noise(
                    naction, noise, timesteps)

                # predict the noise residual
                noise_pred = noise_pred_net(
                    noisy_actions, timesteps, global_cond=global_cond)

                # L2 loss
                loss = nn.functional.mse_loss(noise_pred, noise)

                # optimize
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # step lr scheduler every batch
                # this is different from standard pytorch behavior
                lr_scheduler.step()

                # update Exponential Moving Average of the model weights
                ema.step(nets.parameters())

                # logging
                loss_cpu = loss.item()
                epoch_loss.append(loss_cpu)
                tepoch.set_postfix(loss=loss_cpu)
        tglobal.set_postfix(loss=np.mean(epoch_loss))

        all_epoch_losses.append(np.mean(epoch_loss))

# Weights of the EMA model
# is used for inference
ema_nets = nets
ema.copy_to(ema_nets.parameters())

torch.save(ema_nets.state_dict(), "blockreach_vision_cfg_state_200.ckpt")
print("EMA model saved to blockreach_ema.ckpt")

# 保存目录
save_dir = "training_plots"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "loss_curve_epoch_cfg_state_200.png")

plt.figure(figsize=(8, 5))
plt.plot(all_epoch_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss (log scale)')
plt.title('Training Loss Curve for Traget-Conditioned Diffusion Policy')
plt.yscale('log')
plt.grid(True, which='both', ls='--', lw=0.5)
plt.legend()
plt.tight_layout()

plt.savefig(save_path, dpi=300)   # 保存图片
plt.close()                       # 关闭以释放内存

print(f"Loss curve saved to: {save_path}")
