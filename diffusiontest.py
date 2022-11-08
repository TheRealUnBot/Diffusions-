%matplotlib inline
import torch
from denoising_diffusion_pytorch import(Unet,
                                        GaussianDiffusion,
                                        Trainer)
model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
)
diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 2000,   # number of steps
    loss_type = 'l1'    # L1 or L2
)
trainer = Trainer(
    diffusion,
    '.',
    train_batch_size = 1,
    train_lr = 2e-5,
    # total training steps
    train_num_steps = 4000, 
    # gradient accumulation steps      
    gradient_accumulate_every = 2,
    # exponential moving average decay
    ema_decay = 0.995,
    # turn on mixed precision training with apex              
    fp16 = False
)
trainer.train()
sampled_images = diffusion.sample(batch_size = 1)
sampled_images.shape # (4, 3, 128, 128)