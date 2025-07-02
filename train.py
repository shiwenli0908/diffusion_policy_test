
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

'''
# create dataset from file
dataset = BlockReachPotentialDataset(
    action_file="/content/drive/MyDrive/StageM2/Diffusion-Policy-Adapted/action.npz",
    obs_file="/content/drive/MyDrive/StageM2/Diffusion-Policy-Adapted/obs_array_with_goal_score_2.npz",
    frame_file="/content/drive/MyDrive/StageM2/Diffusion_Policy_Block/datasets/train_full/frames_resized.npz",
    pred_horizon=16,
    obs_horizon=2,
    action_horizon=8
)
'''

dataset = BlockReachPotentialDataset2(
    data_dir="FullNPZDataset",  # 包含 sample_xxxxx 子目录的主目录
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

# visualize data in batch
batch = next(iter(dataloader))
'''print("batch['image'].shape:", batch['image'].shape)
print("batch['obs'].shape:", batch['obs'].shape)
print("batch['action'].shape", batch['action'].shape)'''

print(batch['obs'].shape)         # torch.Size([32, 2, 2])
print(batch['action'].shape)      # torch.Size([32, 8, 2])
print(batch['image'].shape)       # torch.Size([32, 2, 3, 96, 96])
print(batch['target'].shape)      # torch.Size([32, 8, 2])
print(batch['scoref'].shape)      # torch.Size([32, 8, 1])
print(batch['potential'].shape)   # torch.Size([32, 1, H, W])


# construct ResNet18 encoder
vision_encoder = get_resnet('resnet18')
vision_encoder = replace_bn_with_gn(vision_encoder)

# --- instantiate visual encoder for potential map ---
potential_encoder = PotentialResNetEncoder(
    name='resnet18',
    output_dim=512,
    pretrained=False,
    group_norm=True
)

# ResNet18 has output dim of 512
vision_feature_dim = 512
# agent_pos is 2 dimensional
lowdim_obs_dim = 7
# observation feature has 514 dims in total per step
obs_dim = vision_feature_dim + lowdim_obs_dim
action_dim = 2

# create network object
noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim*obs_horizon
)

# the final arch has 2 parts
nets = nn.ModuleDict({
    'vision_encoder': vision_encoder,
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
                nimage = nbatch['image'][:,:obs_horizon].to(device)
                nagent_pos = nbatch['obs'][:,:obs_horizon].to(device)
                naction = nbatch['action'].to(device)
                B = nagent_pos.shape[0]

                # encoder vision features
                image_features = nets['vision_encoder'](
                    nimage.flatten(end_dim=1))
                image_features = image_features.reshape(
                    *nimage.shape[:2],-1)
                # (B,obs_horizon,D)

                # concatenate vision feature and low-dim obs
                obs_features = torch.cat([image_features, nagent_pos], dim=-1)
                obs_cond = obs_features.flatten(start_dim=1)
                # (B, obs_horizon * obs_dim)

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
                    noisy_actions, timesteps, global_cond=obs_cond)

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
