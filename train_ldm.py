import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
from lightning.pytorch import LightningModule
import torchvision
from torch.amp import autocast
from monai.networks.nets import DiffusionModelUNet, SPADEDiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler, DDIMScheduler
from monai.inferers import DiffusionInferer, LatentDiffusionInferer
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import math
from vqgan import *
from monai.utils import optional_import
from functools import partial
from typing import Optional, Tuple, List
import torch
from tqdm import tqdm
import warnings
from ulf_dataset import *
from utils.trainer import create_ddpmtrainer

# 忽略所有用户警告
warnings.filterwarnings("ignore", category=UserWarning)


class DiffusionModel(LightningModule):
    def __init__(self, num_training_steps=None, vae_path=None):
        super().__init__()
        self.embedding_dim = 8
        # 条件编码器
        self.con_encoder = WatEncoder(in_channels=3, basedim=32, downdeepth=2, model_type='3d',
                                      embedding_dim=self.embedding_dim)
        # spade diffusion model
        self.model = SPADEDiffusionModelUNet(
            spatial_dims=3,
            in_channels=8,
            out_channels=8,
            num_res_blocks=2,
            channels=(64, 128, 256, 512),
            attention_levels=(False, False, True, True),
            num_head_channels=8,
            norm_num_groups=32,
            label_nc=8,
        )
        # VAE 模型
        self.vae = VQVAE(
            spatial_dims=3,
            in_channels=3,
            out_channels=3,
            channels=(32, 64),
            num_res_channels=(64, 128),
            downsample_parameters=((2, 4, 1, 1), (2, 4, 1, 1)),
            upsample_parameters=((2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
            num_res_layers=2,
            num_embeddings=8192,
            embedding_dim=8,
        ).eval()
        checkpoint = torch.load("new_checkpoints/vqgan/last.ckpt")
        state_dict = checkpoint.get("state_dict", checkpoint)
        model_weights = {k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")}
        missing_model = self.vae.load_state_dict(model_weights, strict=False)
        print("Loaded vae weights:", missing_model)

        # 采样/训练参数
        self.num_training_steps = num_training_steps
        self.scheduler = DDPMScheduler(num_train_timesteps=1000,
                                       schedule="scaled_linear_beta")  # linear_beta scaled_linear_beta cosine
        self.inferer = LatentDiffusionInferer(self.scheduler, scale_factor=2.2384, )
        self.sample_steps = 2000
        self.inference_steps = 1000

    def configure_optimizers(self):
        # 1. 定义优化器
        optimizer = torch.optim.AdamW(params=list(self.model.parameters()) + list(self.con_encoder.parameters()),
                                      lr=1e-4, betas=(0.95, 0.999),
                                      weight_decay=1e-6, eps=1e-08)

        # 2. 定义学习率调度器
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_training_steps,
                                                                  eta_min=1e-7)
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def training_step(self, batch, batch_idx):
        # 原始
        Tlow_T1_raw = batch['Tlow_T1'][tio.DATA] / 255.0
        Tlow_T2_raw = batch['Tlow_T2'][tio.DATA] / 255.0
        Tlow_FLAIR_raw = batch['Tlow_FLAIR'][tio.DATA] / 255.0
        Tup_T1_raw = batch['Tup_T1'][tio.DATA] / 255.0
        Tup_T2_raw = batch['Tup_T2'][tio.DATA] / 255.0
        Tup_FLAIR_raw = batch['Tup_FLAIR'][tio.DATA] / 255.0
        # 沿通道维度拼接
        input_tensor = torch.cat([Tlow_T1_raw, Tlow_T2_raw, Tlow_FLAIR_raw], dim=1)
        target_tensor = torch.cat([Tup_T1_raw, Tup_T2_raw, Tup_FLAIR_raw], dim=1)
        print("输入图像:", input_tensor.shape, input_tensor.min().item(), input_tensor.max().item())
        # 潜在空间表示
        # with torch.no_grad():
        #     latent_tensor = self.vae.encode_stage_2_inputs(target_tensor)
        # print("潜在空间:", latent_tensor.shape)
        # 条件输入
        condition_tensor = self.con_encoder(input_tensor)
        print("条件输入:", condition_tensor.shape)
        # 噪声
        noise = torch.randn_like(condition_tensor)
        # 时间步长
        timesteps = torch.randint(0, self.inferer.scheduler.num_train_timesteps, (condition_tensor.shape[0],),
                                  device=self.device).long()
        # Get model prediction
        noise_pred = self.inferer(
            inputs=target_tensor,
            autoencoder_model=self.vae,
            diffusion_model=self.model,
            noise=noise,
            timesteps=timesteps,
            seg=condition_tensor,
        )
        # 损失
        loss = F.mse_loss(noise_pred, noise)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        print(f"Step {self.global_step}, Loss: {loss.item()}")
        if self.global_step % self.sample_steps == 0 and self.global_step > 0:
            with torch.no_grad():
                samples = self.sample(noise, condition=condition_tensor, num_inference_steps=self.inference_steps,
                                      mode="crossattn")
                samples = torch.clamp(samples, 0.0, 1.0)
                gen_mse = F.mse_loss(target_tensor, samples)
            self.log('gen_mse', gen_mse, on_step=True, on_epoch=True, prog_bar=True)
            self.save_3d_image_slices(samples[0], "generated_images", self.global_step)
            self.save_3d_image_slices(input_tensor[0], "input_images", self.global_step)
            self.save_3d_image_slices(target_tensor[0], "target_images", self.global_step)
        return loss

    @torch.no_grad()
    def sample(self, noise, condition, num_inference_steps=1000, mode="crossattn"):  # , mode="crossattn"
        self.scheduler.set_timesteps(num_inference_steps=num_inference_steps)
        with torch.no_grad():
            image = self.inferer.sample(input_noise=noise, autoencoder_model=self.vae, diffusion_model=self.model,
                                        scheduler=self.scheduler, seg=condition, mode=mode)
        return image

    def save_3d_image_slices(self, image_3d, tag, global_step):
        """
        :param image_3d: 3D 图像张量，形状为 (C, D, H, W)
        :param tag: 图像的标签（用于日志记录）
        :param global_step: 当前训练步数
        """
        # 获取中间切片
        depth = image_3d.shape[1]
        height = image_3d.shape[2]
        width = image_3d.shape[3]

        axial_slice = image_3d[:, depth // 2, :, :]  # 轴向切片，形状为 (C, H, W)
        sagittal_slice = image_3d[:, :, height // 2, :]  # 矢状面切片，形状为 (C, D, W)
        coronal_slice = image_3d[:, :, :, width // 2]  # 冠状面切片，形状为 (C, D, H)

        # 拼接：形状为 (3, max_dim, max_dim)
        concat_slice = torch.cat([axial_slice, sagittal_slice, coronal_slice], dim=0)  # (3, max_dim, max_dim)
        concat_slice = concat_slice.unsqueeze(1)  # (3, 1, max_dim, max_dim)

        # 堆叠并生成网格
        grid = torchvision.utils.make_grid(
            concat_slice,
            nrow=3,
            normalize=True,
            value_range=(0, 1),
        )

        # 记录图像
        self.logger.experiment.add_image(tag, grid, global_step)
        return


def run():
    print(torch.cuda.is_available())

    path = r'/data/zyxiang/datasets/Training data'
    dataset = load_Multi_modal_dataset(path, is_train=True, out_min_max=(0, 1), load_getitem=True)
    dataloader = patch_train_maskdataloader(dataset, patch_size=(128, 128, 128), batch_size=4, samples_per_volume=8)
    # 加载模型
    num_training_steps = len(dataloader) * 6000
    model = DiffusionModel(num_training_steps=num_training_steps, vae_path="new_checkpoints/vqgan/last.ckpt")
    # model.load_checkpoint()
    print('加载成功')

    trainer = create_ddpmtrainer(name='vqgan_spadeldm', save_dir="./new_logs",
                                 checkpoint_dir="./new_checkpoints/vqgan_spadeldm",
                                 precision='16', max_epoch=6000, monitor='train_loss',
                                 strategy="ddp_find_unused_parameters_true", )
    # 训练
    trainer.fit(model, dataloader)  # , ckpt_path='last'


if __name__ == '__main__':
    # export HIP_VISIBLE_DEVICES=1
    # tensorboard --logdir medicaldiffusion/lightning_logs
    run()