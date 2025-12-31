import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
import lightning as L  # 主要变化：替换为lightning
from lightning.pytorch import LightningModule  # 改为从lightning.pytorch导入
import torchvision
from torch.amp import autocast
from monai.networks.nets import DiffusionModelUNet, SPADEDiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler, DDIMScheduler
from monai.inferers import DiffusionInferer
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import math
from vqgan import *
from monai.utils import optional_import
from functools import partial
from typing import Optional, Tuple, List
from tqdm import tqdm
import warnings
from ulf_dataset import *
from utils.trainer import create_ddpmtrainer

# 忽略所有用户警告
warnings.filterwarnings("ignore", category=UserWarning)


class DiffusionModel(LightningModule):  # 继承自LightningModule
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
            num_head_channels=32,
            norm_num_groups=32,
            label_nc=self.embedding_dim,
        )
        # VAE 模型
        self.vae = ResidualVQGAN.load_from_checkpoint(vae_path).eval()
        for param in self.vae.parameters():
            param.requires_grad = False
            self.vae.freeze()
        # 采样/训练参数
        self.num_training_steps = num_training_steps
        self.scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta")
        self.inferer = DiffusionInferer(self.scheduler)
        self.sample_steps = 5000
        self.inference_steps = 1000

        self.scale_factor = 3.2725  # 与 VAE 的缩放因子相匹配

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
        # 潜在空间表示
        with torch.no_grad():
            latent_tensor = self.vae.encode(target_tensor, quantize=True).detach() * self.scale_factor
        # print("潜在空间:", latent_tensor.shape)
        # 条件输入
        condition_tensor = self.con_encoder(input_tensor)
        # print("条件输入:", condition_tensor.shape)
        # 噪声
        noise = torch.randn_like(latent_tensor)
        # 时间步长
        timesteps = torch.randint(0, self.inferer.scheduler.num_train_timesteps, (latent_tensor.shape[0],),
                                  device=self.device).long()
        # Get model prediction
        noise_pred = self.inferer(
            inputs=latent_tensor,
            diffusion_model=self.model,
            noise=noise,
            timesteps=timesteps,
            seg=condition_tensor,
        )
        # 损失
        loss = F.mse_loss(noise_pred, noise)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('global_step', self.global_step,
                 on_step=True,
                 on_epoch=False,
                 prog_bar=True,
                 logger=False)

        return loss

    def validation_step(self, batch, batch_idx):
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
        # 潜在空间表示
        with torch.no_grad():
            # 潜在空间表示
            latent_tensor = self.vae.encode(target_tensor, quantize=True).detach() * self.scale_factor
            # 条件输入
            condition_tensor = self.con_encoder(input_tensor)
            # 噪声
            noise = torch.randn_like(latent_tensor)
            # 采样
            samples = self.sample(noise, condition=condition_tensor, num_inference_steps=self.inference_steps,
                                  mode="crossattn")
            print(samples.shape, samples.min(), samples.max())
            mse = F.mse_loss(samples, latent_tensor)
            self.log('latent_mse', mse, on_step=True, on_epoch=True, prog_bar=True)
            generated_tensor = self.vae.decode(samples / self.scale_factor, quantize=True)
            print(generated_tensor.min(), generated_tensor.max())
            if torch.isnan(generated_tensor).any():
                print(f"⚠️ generated_tensor has NaN! Shape={generated_tensor.shape}")
                return
            # 生成
            generated_tensor = torch.clamp(generated_tensor, 0.0, 1.0)
            gen_l1 = F.l1_loss(generated_tensor, target_tensor)
            self.log('gen_l1', gen_l1, on_step=True, on_epoch=True, prog_bar=True)
            self.save_3d_image_slices(generated_tensor[0], "generated_images", self.global_step)
            self.save_3d_image_slices(input_tensor[0], "input_images", self.global_step)
            self.save_3d_image_slices(target_tensor[0], "target_images", self.global_step)
            with torch.no_grad():
                vq_tensor = self.vae.decode(latent_tensor, quantize=True)
            self.save_3d_image_slices(vq_tensor[0], "vq_images", self.global_step)
        return mse

    @torch.no_grad()
    def sample(self, noise, condition, num_inference_steps=1000, mode="crossattn"):  # , mode="crossattn"
        self.scheduler.set_timesteps(num_inference_steps=num_inference_steps)
        with torch.no_grad():
            image = self.inferer.sample(input_noise=noise, diffusion_model=self.model, scheduler=self.scheduler,
                                        seg=condition, mode=mode)  # , mode=mode
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
    dataloader = patch_train_maskdataloader(dataset, patch_size=(128, 128, 128), batch_size=10, samples_per_volume=16)
    val_dataset = load_val_dataset(path, is_train=False, out_min_max=(0, 1))
    val_dataloader = patch_train_maskdataloader(val_dataset, patch_size=(128, 128, 128), batch_size=8,
                                                samples_per_volume=8)
    # 加载模型
    # num_training_steps = len(dataloader) * 10000
    model = DiffusionModel(num_training_steps=4000000, vae_path='checkpoints/residual_vqgan_old/last.ckpt')
    # model = DiffusionModel.load_from_checkpoint(checkpoint_path="new_checkpoints/spade_ldm2/last.ckpt", num_training_steps=1000000, vae_path='checkpoints/residual_vqgan_old/last.ckpt')
    # compiled_model = torch.compile(model, mode="max-autotune")
    print('加载成功')

    trainer = create_ddpmtrainer(name='spade_ldm', save_dir="./new_logs", checkpoint_dir="./new_checkpoints/spade_ldm",
                                 precision='16', max_epoch=100000, monitor='train_loss',
                                 strategy="ddp_find_unused_parameters_true", )
    # 训练
    trainer.fit(model, dataloader, val_dataloader)


if __name__ == '__main__':
    # export HIP_VISIBLE_DEVICES=1
    # tensorboard --logdir medicaldiffusion/lightning_logs
    run()