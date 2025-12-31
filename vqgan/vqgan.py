import math
import random
import argparse
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torchvision
import torchio as tio
from typing import Union
from vector_quantize_pytorch import FSQ, ResidualFSQ, LFQ, ResidualVQ
from monai.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator, ControlNet, VQVAE
from monai.losses import PerceptualLoss
from monai.networks.layers import Act
from typing import *
from einops import rearrange
from monai.metrics import SSIMMetric, PSNRMetric

def shift_dim(x, src_dim=-1, dest_dim=-1, make_contiguous=True):
    n_dims = len(x.shape)
    if src_dim < 0:
        src_dim = n_dims + src_dim
    if dest_dim < 0:
        dest_dim = n_dims + dest_dim

    assert 0 <= src_dim < n_dims and 0 <= dest_dim < n_dims

    dims = list(range(n_dims))
    del dims[src_dim]

    permutation = []
    ctr = 0
    for i in range(n_dims):
        if i == dest_dim:
            permutation.append(src_dim)
        else:
            permutation.append(dims[ctr])
            ctr += 1
    x = x.permute(permutation)
    if make_contiguous:
        x = x.contiguous()
    return x


def r_reg(d_out, x_in):
    """
    # zero-centered gradient penalty for real images

    参数:
        d_out: Discriminator out
        x_in: real images
    """
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg


class GANLossComps(nn.Module):
    """Define GAN loss.

    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge',
            'wgan-logistic-ns'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(self,
                 gan_type: str,
                 real_label_val: float = 1.0,
                 fake_label_val: float = 0.0,
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            self.loss = self._wgan_loss
        elif self.gan_type == 'wgan-logistic-ns':
            self.loss = self._wgan_logistic_ns_loss
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError(
                f'GAN type {self.gan_type} is not implemented.')

    def _wgan_loss(self, input: torch.Tensor, target: bool) -> torch.Tensor:
        """wgan loss.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return -input.mean() if target else input.mean()

    def _wgan_logistic_ns_loss(self, input: torch.Tensor,
                               target: bool) -> torch.Tensor:
        """WGAN loss in logistically non-saturating mode.

        This loss is widely used in StyleGANv2.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """

        return F.softplus(-input).mean() if target else F.softplus(
            input).mean()

    def get_target_label(self, input: torch.Tensor,
                         target_is_real: bool) -> Union[bool, torch.Tensor]:
        """Get target label.

        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.

        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise, \
                return Tensor.
        """

        if self.gan_type in ['wgan', 'wgan-logistic-ns']:
            return target_is_real
        target_val = (
            self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val

    def forward(self,
                input: torch.Tensor,
                target_is_real: bool,
                is_disc: bool = False) -> torch.Tensor:
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.

        Returns:
            Tensor: GAN loss value.
        """
        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == 'hinge':
            if is_disc:  # for discriminators in hinge-gan
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                loss = -input.mean()
        else:  # other gan types
            loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight


class Autoencoder(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.cfg = cfg
        self.startd_step = cfg.model.startd_step
        self.sample_step = cfg.model.sample_step

        self.model = AutoencoderKL(
            spatial_dims=3,
            in_channels=3,
            out_channels=3,
            channels=(16, 32, 64),
            norm_num_groups=16,
            latent_channels=4,
            num_res_blocks=1,
            attention_levels=(False, False, False),
        )
        self.discriminator = PatchDiscriminator(spatial_dims=3, in_channels=cfg.model.in_channels,
                                                channels=cfg.model.disc_channels,
                                                out_channels=cfg.model.out_channels,
                                                num_layers_d=cfg.model.disc_downdeepth)
        # loss weight
        self.kl_weight = 1e-6
        self.gan_feat_weight = cfg.model.gan_feat_weight
        self.adv_weight = cfg.model.adv_weight
        self.recon_weight = cfg.model.recon_weight
        self.ganloss = GANLossComps(gan_type=cfg.model.gan_type, loss_weight=self.adv_weight)

    def configure_optimizers(self):
        lr = self.cfg.model.lr

        opt_g = torch.optim.Adam(self.model.parameters(), lr=lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr)
        return [opt_g, opt_d], []

    def training_step(self, batch):
        optimizer_g, optimizer_d = self.optimizers()

        Tup_T1_raw = batch['Tup_T1'][tio.DATA] / 255.0
        Tup_T2_raw = batch['Tup_T2'][tio.DATA] / 255.0
        Tup_FLAIR_raw = batch['Tup_FLAIR'][tio.DATA] / 255.0

        x = torch.concat([Tup_T1_raw, Tup_T2_raw, Tup_FLAIR_raw], dim=1)
        print(x.min(), x.max())
        # =================================================================================== #
        #                               2. generator loss                                         #
        # =================================================================================== #
        self.toggle_optimizer(optimizer_g)
        x_recon, z_mu, z_sigma = self.model(x)
        # recon loss
        recon_loss = F.l1_loss(x_recon, x)
        # kl loss
        kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        # features_loss and adv_loss
        if self.global_step > self.startd_step:
            out_recon = self.discriminator(x_recon)
            logits_recon = out_recon[-1]
            features_recon = out_recon[:-1]
            out_orig = self.discriminator(x)
            features_real = out_orig[:-1]
            adv_loss = self.ganloss(logits_recon, True, is_disc=False)

            features_loss = torch.tensor(0.0, device=x.device)
            if self.gan_feat_weight > 0:
                for i in range(0, len(features_recon) - 1):
                    features_loss += F.l1_loss(features_recon[i], features_real[i].detach())
            self.log("train/adv_loss", adv_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("train/features_loss", features_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        else:
            adv_loss = torch.tensor(0.0, device=x.device)
            features_loss = torch.tensor(0.0, device=x.device)

        # generator loss
        g_loss = recon_loss + (kl_loss * self.kl_weight) + adv_loss * self.adv_weight + features_loss * self.gan_feat_weight
        self.manual_backward(g_loss)  # 手动反向传播
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)  # 关闭优化器
        self.log("train_recon_loss", recon_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/kl_losss", kl_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/g_loss", g_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        # =================================================================================== #
        #                             3. discriminator                                            #
        # =================================================================================== #
        if self.global_step > self.startd_step:
            self.toggle_optimizer(optimizer_d)
            out_real = self.discriminator(x)
            logits_real = out_real[-1]
            x_fake = x_recon.detach()
            out_fake = self.discriminator(x_fake)
            logits_fake = out_fake[-1]
            d_loss = (self.ganloss(logits_real, True, is_disc=True) + self.ganloss(logits_fake, False,
                                                                                   is_disc=True)) / 2
            self.clip_gradients(optimizer_d, gradient_clip_val=0.001, gradient_clip_algorithm="norm")
            optimizer_d.step()
            optimizer_d.zero_grad()
            self.untoggle_optimizer(optimizer_d)
            self.log("train/d_loss", d_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        if self.global_step % self.sample_step == 0:
            self.save_3d_image_slices(x_recon[0], "generated_reconst_images", self.global_step)
        return g_loss

    def save_3d_image_slices(self, image_3d, tag, global_step):
        """
        :param image_3d: 3D images，shape (C, D, H, W)
        :param tag: tag label
        :param global_step:
        """
        # 获取中间切片
        depth = image_3d.shape[1]
        height = image_3d.shape[2]
        width = image_3d.shape[3]

        axial_slice = image_3d[:, depth // 2, :, :]  #  (C, H, W)
        sagittal_slice = image_3d[:, :, height // 2, :]  # (C, D, W)
        coronal_slice = image_3d[:, :, :, width // 2]  # (C, D, H)

        concat_slice = torch.cat([axial_slice, sagittal_slice, coronal_slice], dim=0)  # (3, max_dim, max_dim)
        concat_slice = concat_slice.unsqueeze(1)  # (3, 1, max_dim, max_dim)
        grid = torchvision.utils.make_grid(
            concat_slice,
            nrow=3,
            normalize=True,
            value_range=(0, 1),
        )
        self.logger.experiment.add_image(tag, grid, global_step)
        return


class VQGAN(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.cfg = cfg
        self.startd_step = cfg.model.startd_step
        self.sample_step = cfg.model.sample_step

        self.model = VQVAE(
            spatial_dims=3,
            in_channels=3,
            out_channels=3,
            channels=(32, 64),
            num_res_channels=(32, 64),
            downsample_parameters=((2, 4, 1, 1), (2, 4, 1, 1)),
            upsample_parameters=((2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
            num_res_layers=3,
            num_embeddings=8192,
            embedding_dim=8,
            commitment_cost=1,
            decay=0.99,
        )

        # 判别器
        self.discriminator = PatchDiscriminator(spatial_dims=3, in_channels=cfg.model.in_channels,
                                                channels=cfg.model.disc_channels,
                                                out_channels=cfg.model.out_channels,
                                                num_layers_d=cfg.model.disc_downdeepth)
        # loss权重
        self.gan_feat_weight = cfg.model.gan_feat_weight
        self.adv_weight = cfg.model.adv_weight
        self.recon_weight = cfg.model.recon_weight
        self.ganloss = GANLossComps(gan_type=cfg.model.gan_type, loss_weight=self.adv_weight)
        print("loss weights:", self.recon_weight, self.adv_weight, self.gan_feat_weight)

    def configure_optimizers(self):
        lr = self.cfg.model.lr

        opt_g = torch.optim.Adam(self.model.parameters(), lr=lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr)
        return [opt_g, opt_d], []

    def training_step(self, batch):
        optimizer_g, optimizer_d = self.optimizers()

        Tup_T1_raw = batch['Tup_T1'][tio.DATA] / 255.0
        Tup_T2_raw = batch['Tup_T2'][tio.DATA] / 255.0
        Tup_FLAIR_raw = batch['Tup_FLAIR'][tio.DATA] / 255.0

        x = torch.concat([Tup_T1_raw, Tup_T2_raw, Tup_FLAIR_raw], dim=1)
        print(self.global_step, x.min(), x.max())
        # =================================================================================== #
        #                               2. 训练生成器                                          #
        # =================================================================================== #
        self.toggle_optimizer(optimizer_g)
        x_recon, quantization_losses = self.model(images=x)

        # 计算重建损失
        recon_loss = F.l1_loss(x_recon, x)

        # features_loss and adv_loss
        if self.global_step > self.startd_step:
            # 计算感知loss和对抗loss
            out_recon = self.discriminator(x_recon)
            logits_recon = out_recon[-1]
            features_recon = out_recon[:-1]
            out_orig = self.discriminator(x)
            features_real = out_orig[:-1]
            adv_loss = self.ganloss(logits_recon, True, is_disc=False)

            features_loss = torch.tensor(0.0, device=x.device)
            if self.gan_feat_weight > 0:
                for i in range(0, len(features_recon) - 1):
                    features_loss += F.l1_loss(features_recon[i], features_real[i].detach())
            self.log("train/adv_loss", adv_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("train/features_loss", features_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        else:
            adv_loss = torch.tensor(0.0, device=x.device)
            features_loss = torch.tensor(0.0, device=x.device)

        g_loss = quantization_losses + recon_loss * self.recon_weight + adv_loss * self.adv_weight + features_loss * self.gan_feat_weight
        self.manual_backward(g_loss)  # 手动反向传播
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)  # 关闭优化器
        self.log("train_recon_loss", recon_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/quantization_losses", quantization_losses, prog_bar=True, logger=True, on_step=True,
                 on_epoch=True)
        self.log("train/g_loss", g_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        # with torch.no_grad():  # 这是监控，不需要梯度
        #     # 获取当前batch所有空间位置对应的码本索引
        #     encoding_indices = self.model.index_quantize(x)
        #     # 展平所有索引以便统计
        #     flat_indices = encoding_indices.flatten().long()  # 形状: [N]
        #     # 1. 计算码本使用率：有多少个不同的码本向量被使用了？
        #     unique_indices = torch.unique(flat_indices)
        #     usage_ratio = len(unique_indices) / self.model.num_embeddings
        #     self.log("train/codebook_usage_ratio", usage_ratio, prog_bar=False, on_step=True, on_epoch=True)

        #     # 2. 计算困惑度 (Perplexity)：更科学的指标，越接近8192越好
        #     # 统计每个码本向量被使用的次数
        #     counts = torch.bincount(flat_indices, minlength=self.model.num_embeddings)
        #     probs = counts.float() / counts.sum()
        #     # 避免log(0)，加一个极小值
        #     perplexity = torch.exp(-torch.sum(probs * torch.log(probs + 1e-10)))
        #     self.log("train/codebook_perplexity", perplexity, prog_bar=True, on_step=True, on_epoch=True)

        #     # 3.统计std
        #     z = self.model.encode_stage_2_inputs(x)
        #     batch_std = z.std().item()
        #     print('Latent Std:', batch_std)
        #     self.log("train/latent_std", batch_std, prog_bar=False, on_step=True, on_epoch=True)

        # =================================================================================== #
        #                             3. 训练判别器                                            #
        # =================================================================================== #
        if self.global_step > self.startd_step:
            self.toggle_optimizer(optimizer_d)
            out_real = self.discriminator(x)
            logits_real = out_real[-1]
            x_fake = x_recon.detach()
            out_fake = self.discriminator(x_fake)
            logits_fake = out_fake[-1]
            d_loss = (self.ganloss(logits_real, True, is_disc=True) + self.ganloss(logits_fake, False,
                                                                                   is_disc=True)) / 2
            self.clip_gradients(optimizer_d, gradient_clip_val=0.015, gradient_clip_algorithm="norm")
            self.manual_backward(d_loss)
            optimizer_d.step()
            optimizer_d.zero_grad()
            self.untoggle_optimizer(optimizer_d)
            self.log("train/d_loss", d_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        # if self.global_step % self.sample_step == 0:
        #     self.save_3d_image_slices(x_recon[0], "generated_reconst_images", self.global_step)
        #     ssim = SSIM(x_recon[0], x[0])
        #     self.log("train/ssim", ssim, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        return g_loss

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


class ResidualVQGAN(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.cfg = cfg
        self.startd_step = cfg.model.startd_step
        self.sample_step = 1000
        self.encoder = Encoder(in_channels=cfg.model.in_channels, basedim=cfg.model.basedim,
                               downdeepth=cfg.model.downdeepth, model_type=cfg.model.model_type,
                               embedding_dim=cfg.model.embedding_dim, num_groups=cfg.model.num_groups)

        self.decoder = Decoder(out_channels=cfg.model.out_channels, basedim=cfg.model.basedim,
                               downdeepth=cfg.model.downdeepth, model_type=cfg.model.model_type,
                               embedding_dim=cfg.model.embedding_dim, num_groups=cfg.model.num_groups)
        # codebook
        self.codebook = ResidualVQ(dim=cfg.model.embedding_dim, codebook_size=cfg.model.n_codes,
                                   num_quantizers=cfg.model.num_quantizers,
                                   decay=0.99, kmeans_init=True, commitment_weight=0.25, rotation_trick=True,
                                   threshold_ema_dead_code=2)

        # 判别器
        self.discriminator = PatchDiscriminator(spatial_dims=3, in_channels=cfg.model.in_channels,
                                                channels=cfg.model.disc_channels,
                                                out_channels=cfg.model.out_channels,
                                                num_layers_d=cfg.model.disc_downdeepth)

        # loss权重
        self.gan_feat_weight = cfg.model.gan_feat_weight
        self.adv_weight = cfg.model.adv_weight
        self.recon_weight = cfg.model.recon_weight
        self.perceptual_weight = cfg.model.perceptual_weight
        self.ganloss = GANLossComps(gan_type=cfg.model.gan_type, loss_weight=self.adv_weight)
        self.perceptualloss = PerceptualLoss(
            spatial_dims=3,  # 3D数据
            network_type="vgg",
            is_fake_3d=True,
            fake_3d_ratio=0.2,
            cache_dir="./pretrained_weights",  # 权重缓存目录
        ).eval()

        self.ssim_metric = SSIMMetric(spatial_dims=3, data_range=1.0)

    def configure_optimizers(self):
        lr = self.cfg.model.lr

        opt_g = torch.optim.Adam(list(self.encoder.parameters()) +
                                 list(self.codebook.parameters()) +
                                 list(self.decoder.parameters()),
                                 lr=lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr)
        return opt_g, opt_d

    def encode(self, x, quantize=True):
        z = self.encoder(x)
        if quantize:
            z, indices, _ = self.vector_quantize(z)
        return z

    def decode(self, z, quantize=True):
        if quantize:
            z, indices, _ = self.vector_quantize(z)
        x_recon = self.decoder(z)
        return x_recon

    def vector_quantize(self, z):
        # Step 1: 转换为 (B, H*W*D, C)
        B, C, H, W, D = z.shape
        z_flat = rearrange(z, 'b c h w d -> b (h w d) c')

        # Step 2: 通过 codebook
        quantized_flat, indices, commit_loss = self.codebook(z_flat)

        # Step 3: 还原为 (B, C, H, W, D)
        quantized = rearrange(quantized_flat, 'b (h w d) c -> b c h w d', h=H, w=W, d=D)

        return quantized, indices, commit_loss.mean()

    def forward(self, x):
        z = self.encoder(x)
        z, indices, _ = self.vector_quantize(z)
        x_recon = self.decoder(z)
        return x_recon

    def training_step(self, batch):
        optimizer_g, optimizer_d = self.optimizers()
        Tup_T1_raw = batch['Tup_T1'][tio.DATA] / 255.0
        Tup_T2_raw = batch['Tup_T2'][tio.DATA] / 255.0
        Tup_FLAIR_raw = batch['Tup_FLAIR'][tio.DATA] / 255.0

        x = torch.concat([Tup_T1_raw, Tup_T2_raw, Tup_FLAIR_raw], dim=1)
        print(x.min(), x.max())
        # =================================================================================== #
        #                               2. 训练生成器                                          #
        # =================================================================================== #
        self.toggle_optimizer(optimizer_g)
        z = self.encoder(x)
        z, indices, quantization_losses = self.vector_quantize(z)
        print('z shape:', z.shape, 'global step:', self.global_step)
        x_recon = self.decoder(z)

        # 计算重建损失
        recon_loss = F.l1_loss(x_recon, x)

        # perceptual loss
        perceptual_loss = self.perceptualloss(x_recon, x)

        # features_loss and adv_loss
        if self.global_step > self.startd_step:
            # 计算感知loss和对抗loss
            out_recon = self.discriminator(x_recon)
            logits_recon = out_recon[-1]
            features_recon = out_recon[:-1]
            out_orig = self.discriminator(x)
            features_real = out_orig[:-1]
            adv_loss = self.ganloss(logits_recon, True, is_disc=False)

            features_loss = torch.tensor(0.0, device=x.device)
            if self.gan_feat_weight > 0:
                for i in range(0, len(features_recon) - 1):
                    features_loss += F.l1_loss(features_recon[i], features_real[i].detach())
            self.log("train/adv_loss", adv_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("train/features_loss", features_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        else:
            adv_loss = torch.tensor(0.0, device=x.device)
            features_loss = torch.tensor(0.0, device=x.device)

        # 计算总损失
        g_loss = recon_loss * self.recon_weight + quantization_losses + perceptual_loss * self.perceptual_weight + adv_loss + features_loss * self.gan_feat_weight
        self.manual_backward(g_loss)  # 手动反向传播
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)  # 关闭优化器
        self.log("train_recon_loss", recon_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/perceptual_loss", perceptual_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("g_loss", g_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/quantization_losses", quantization_losses, prog_bar=True, logger=True, on_step=True,
                 on_epoch=True)

        # =================================================================================== #
        #                             3. 训练判别器                                            #
        # =================================================================================== #
        if self.global_step > self.startd_step:
            self.toggle_optimizer(optimizer_d)
            x_real = x.detach().requires_grad_(True)
            out_real = self.discriminator(x_real)
            logits_real = out_real[-1]
            x_fake = x_recon.detach()
            out_fake = self.discriminator(x_fake)
            logits_fake = out_fake[-1]

            d_loss = (self.ganloss(logits_real, True, is_disc=True) + self.ganloss(logits_fake, False,
                                                                                   is_disc=True)) / 2
            self.log("train/d_loss", d_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

            r1_penalty = r_reg(logits_real, x_real)
            self.log("train/r_penalty", r1_penalty, prog_bar=True, logger=True, on_step=True, on_epoch=True)

            self.manual_backward(d_loss)
            # self.clip_gradients(optimizer_d, gradient_clip_val=0.001, gradient_clip_algorithm="norm")
            optimizer_d.step()
            optimizer_d.zero_grad()
            self.untoggle_optimizer(optimizer_d)
        return g_loss

    def validation_step(self, batch, batch_idx):
        Tup_T1_raw = batch['Tup_T1'][tio.DATA] / 255.0
        Tup_T2_raw = batch['Tup_T2'][tio.DATA] / 255.0
        Tup_FLAIR_raw = batch['Tup_FLAIR'][tio.DATA] / 255.0

        x = torch.concat([Tup_T1_raw, Tup_T2_raw, Tup_FLAIR_raw], dim=1)
        print(x.min(), x.max())
        # =================================================================================== #
        #                               2. 训练生成器                                          #
        # =================================================================================== #
        z = self.encoder(x)
        z, indices, quantization_losses = self.vector_quantize(z)
        print('z shape:', z.shape, 'global step:', self.global_step)
        x_recon = self.decoder(z)

        print('indices shape:', indices.shape)
        B, num_tokens, num_quant = indices.shape

        layer_perplexities = []
        layer_usage_ratios = []

        # for q in range(num_quant):
        #     # 获取第q层所有样本、所有位置的量化索引
        #     layer_indices = indices[..., q].flatten().long()  # 形状: [B * num_tokens]

        #     # 1. 计算该层码本使用率
        #     unique_indices = torch.unique(layer_indices)
        #     layer_codebook_size = self.cfg.model.n_codes # 该层的码本大小
        #     usage_ratio = len(unique_indices) / layer_codebook_size
        #     layer_usage_ratios.append(usage_ratio)

        #     # 2. 计算该层困惑度
        #     counts = torch.bincount(layer_indices, minlength=layer_codebook_size)
        #     probs = counts.float() / counts.sum()
        #     perplexity = torch.exp(-torch.sum(probs * torch.log(probs + 1e-10)))
        #     print(f"Layer {q} - Perplexity: {perplexity:.2f}, Usage Ratio: {usage_ratio:.4f}")
        #     layer_perplexities.append(perplexity)

        #     # 记录每一层的指标（可选，但非常有助于诊断）
        #     self.log(f"train/codebook_q{q}_perplexity", perplexity, on_epoch=True)
        #     self.log(f"train/codebook_q{q}_usage_ratio", usage_ratio, on_epoch=True)

        # # 3. 计算整体平均指标（简单平均）
        # avg_perplexity = torch.stack(layer_perplexities).mean()
        # avg_usage_ratio = sum(layer_usage_ratios) / len(layer_usage_ratios)

        # self.log("train/codebook_avg_perplexity", avg_perplexity, prog_bar=True)
        # self.log("train/codebook_avg_usage_ratio", avg_usage_ratio, prog_bar=False)

        # 3.统计std
        batch_std = z.std().item()
        print('Latent Std:', batch_std)
        self.log("train/latent_std", batch_std, prog_bar=False, on_step=True, on_epoch=True)

        x_recon = torch.clamp(x_recon, 0.0, 1.0)
        self.save_3d_image_slices(x_recon[0], "generated_reconst_images", self.global_step)
        self.save_3d_image_slices(x[0], "input_images", self.global_step)
        ssim = self.ssim_metric(x_recon, x).mean()
        self.log("train/ssim", ssim, prog_bar=True, logger=True, on_epoch=True)
        return ssim

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

    def load_dict(self, checkpoint_path, strict=False):
        """
        加载模型权重（支持非严格模式）

        参数:
            checkpoint_path: 权重文件路径
            strict: 是否严格匹配权重（False时忽略不匹配的键）
        """
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint.get('state_dict', checkpoint)  # 兼容不同格式的checkpoint

        # 提取各模块权重
        encoder_weights = {k.replace('encoder.', ''): v
                           for k, v in state_dict.items() if k.startswith('encoder.')}
        decoder_weights = {k.replace('decoder.', ''): v
                           for k, v in state_dict.items() if k.startswith('decoder.')}
        codebook_weights = {k.replace('codebook.', ''): v
                            for k, v in state_dict.items() if k.startswith('codebook.')}
        discriminator_weights = {k.replace('discriminator.', ''): v
                                 for k, v in state_dict.items() if k.startswith('discriminator.')}

        # 非严格加载各模块
        print("\n===== 权重加载情况 =====")
        missing = self.encoder.load_state_dict(encoder_weights, strict=strict)
        print(missing)
        missing = self.decoder.load_state_dict(decoder_weights, strict=strict)
        print(missing)
        missing = self.codebook.load_state_dict(codebook_weights, strict=strict)
        print(missing)
        missing = self.discriminator.load_state_dict(discriminator_weights, strict=strict)
        print(missing)


def Normalize(in_channels, norm_type='group', num_groups=16):
    assert norm_type in ['group', 'instance', 'batch']
    if norm_type == 'group':
        return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)
    elif norm_type == 'batch':
        return torch.nn.SyncBatchNorm(in_channels)
    elif norm_type == 'instance':
        return torch.nn.InstanceNorm3d(in_channels, affine=True, track_running_stats=False)


def extend_by_dim(krnlsz, model_type='3d', half_dim=1):
    if model_type == '2d':
        outsz = [krnlsz] * 2
    elif model_type == '3d':
        outsz = [krnlsz] * 3
    elif model_type == '2.5d':
        outsz = [(np.array(krnlsz) * 0 + 1) * half_dim] + [krnlsz] * 2
    else:
        outsz = [krnlsz]
    return tuple(outsz)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding='same', mid_channels=None,
                 model_type='3d', residualskip=True, num_groups=16, norm_type='group'):
        super(ResBlock, self).__init__()

        self.change_dimension = in_channels != out_channels
        self.model_type = model_type
        self.residualskip = residualskip
        padding = {'same': kernel_size // 2, 'valid': 0}[padding] if padding in ['same', 'valid'] else padding
        mid_channels = out_channels if mid_channels is None else mid_channels

        if self.model_type == '3d':
            self.Conv = nn.Conv3d
        elif self.model_type == '2.5d':
            self.Conv = nn.Conv3d
        else:
            self.Conv = nn.Conv2d

        def extdim(krnlsz, halfdim=1):
            return extend_by_dim(krnlsz, model_type=model_type.lower(), half_dim=halfdim)

        self.short_cut_conv = self.Conv(in_channels, out_channels, 1, extdim(stride))
        self.norm0 = Normalize(out_channels, norm_type, num_groups=num_groups)
        self.conv1 = self.Conv(in_channels, mid_channels, extdim(kernel_size, 3), extdim(stride),
                               padding=extdim(padding, 1), padding_mode='reflect')
        self.norm1 = Normalize(mid_channels, norm_type, num_groups=num_groups)
        self.silu1 = nn.SiLU()
        self.conv2 = self.Conv(mid_channels, out_channels, extdim(kernel_size, 3), extdim(1),
                               padding=extdim(padding, 1), padding_mode='reflect')
        self.norm2 = Normalize(out_channels, norm_type, num_groups=num_groups)
        self.silu2 = nn.SiLU()

    def forward(self, x):
        if self.residualskip and self.change_dimension:
            short_cut_conv = self.norm0(self.short_cut_conv(x))
        else:
            short_cut_conv = x
        o_c1 = self.silu1(self.norm1(self.conv1(x)))
        o_c2 = self.norm2(self.conv2(o_c1))
        if self.residualskip:
            out_res = self.silu2(o_c2 + short_cut_conv)
        else:
            out_res = self.silu2(o_c2)
        return out_res


class StackConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding='same', mid_channels=None,
                 model_type='3d', residualskip=False, device=None, dtype=None):
        super(StackConvBlock, self).__init__()

        self.change_dimension = in_channels != out_channels
        self.model_type = model_type
        self.residualskip = residualskip
        padding = {'same': kernel_size // 2, 'valid': 0}[padding] if padding in ['same', 'valid'] else padding
        mid_channels = out_channels if mid_channels is None else mid_channels

        if self.model_type == '3d':
            self.ConvBlock, self.InstanceNorm = nn.Conv3d, nn.InstanceNorm3d
        elif self.model_type == '2.5d':
            self.ConvBlock, self.InstanceNorm = nn.Conv3d, nn.InstanceNorm3d
        else:
            self.ConvBlock, self.InstanceNorm = nn.Conv2d, nn.InstanceNorm2d

        def extdim(krnlsz, halfdim=1):
            return extend_by_dim(krnlsz, model_type=model_type.lower(), half_dim=halfdim)

        self.short_cut_conv = self.ConvBlock(in_channels, out_channels, 1, extdim(stride))
        self.norm0 = self.InstanceNorm(out_channels, affine=True)
        self.conv1 = self.ConvBlock(in_channels, mid_channels, extdim(kernel_size, 3), extdim(stride),
                                    padding=extdim(padding, 1), padding_mode='reflect')
        self.norm1 = self.InstanceNorm(mid_channels, affine=True)
        self.relu1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.conv2 = self.ConvBlock(mid_channels, out_channels, extdim(kernel_size, 3), extdim(1),
                                    padding=extdim(padding, 1), padding_mode='reflect')
        self.norm2 = self.InstanceNorm(out_channels, affine=True, track_running_stats=False)
        self.relu2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x):
        if self.residualskip and self.change_dimension:
            short_cut_conv = self.norm0(self.short_cut_conv(x))
        else:
            short_cut_conv = x
        o_c1 = self.relu1(self.norm1(self.conv1(x)))
        o_c2 = self.norm2(self.conv2(o_c1))
        if self.residualskip:
            out_res = self.relu2(o_c2 + short_cut_conv)
        else:
            out_res = self.relu2(o_c2)
        return out_res


class Encoder(nn.Module):
    def __init__(self, in_channels=1, basedim=32, downdeepth=2, num_res_layers=3, model_type='3d', embedding_dim=8,
                 num_groups=32):
        super().__init__()
        if model_type == '3d':
            self.conv = nn.Conv3d
        else:
            self.conv = nn.Conv2d

        self.begin_conv = ResBlock(in_channels=in_channels, out_channels=basedim, model_type=model_type,
                                   num_groups=num_groups)
        self.encoding_block = nn.ModuleList()
        for convidx in range(0, downdeepth):
            for layeridx in range(0, num_res_layers - 1):
                self.encoding_block.append(
                    ResBlock(in_channels=basedim * 2 ** convidx, out_channels=basedim * 2 ** convidx,
                             model_type=model_type, ))
            self.encoding_block.append(ResBlock(basedim * 2 ** convidx, basedim * 2 ** (convidx + 1), 3, 2,
                                                model_type=model_type, num_groups=num_groups))
        self.enc_out = basedim * 2 ** downdeepth
        self.pre_vq_conv = self.conv(self.enc_out, embedding_dim, 1, )

    def forward(self, x):
        x = self.begin_conv(x)
        for block in self.encoding_block:
            x = block(x)
        x = self.pre_vq_conv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, out_channels=1, basedim=32, downdeepth=2, num_res_layers=3,
                 model_type='3d', embedding_dim=8, num_groups=32):
        super().__init__()
        self.model_type = model_type
        if self.model_type == '3d':
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = nn.Conv3d
        else:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = nn.Conv2d

        self.enc_out = basedim * 2 ** downdeepth
        self.post_vq_conv = nn.Sequential(
            self.conv(embedding_dim, self.enc_out, 3, 1, 1, padding_mode='reflect'),
            Normalize(self.enc_out, norm_type='group', num_groups=num_groups),
            nn.SiLU(),
        )

        # 修改为支持num_res_layers
        self.decoding_blocks = nn.ModuleList()
        for convidx in reversed(range(1, downdeepth + 1)):
            # 步骤1：通道压缩（in_channels -> out_channels）
            in_ch = basedim * 2 ** convidx
            out_ch = basedim * 2 ** (convidx - 1)

            block = nn.Module()

            # 创建num_res_layers个残差块
            res_blocks = nn.ModuleList()
            for layer_idx in range(num_res_layers):
                # 第一个残差块处理通道变化
                if layer_idx == 0 and in_ch != out_ch:
                    res_blocks.append(
                        ResBlock(in_ch, out_ch, 3, 1, model_type=self.model_type, num_groups=num_groups)
                    )
                else:
                    # 后续残差块保持通道数不变
                    res_blocks.append(
                        ResBlock(out_ch, out_ch, 3, 1, model_type=self.model_type, num_groups=num_groups)
                    )

            block.res_blocks = res_blocks

            # 上采样层
            block.upsample = self.up

            self.decoding_blocks.append(block)

        self.final_conv = self.conv(basedim, out_channels, 3, 1, 1)

    def forward(self, x):
        x = self.post_vq_conv(x)

        for i, block in enumerate(self.decoding_blocks):
            # 执行num_res_layers个残差块
            for res_block in block.res_blocks:
                x = res_block(x)
            # 上采样
            x = block.upsample(x)
        x = self.final_conv(x)
        return x

from .DWT_IDWT_layer import DWT_3D, IDWT_3D
from .squeeze_and_excitation_3D import ProjectExciteLayer

class WATConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dwt_channels=1, kernel_size=3, stride=2, model_type='3d',
                 residualskip=True):
        super(WATConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dwt = DWT_3D("haar")
        self.resblock = StackConvBlock(in_channels, out_channels, kernel_size, stride, model_type=model_type,
                                       residualskip=residualskip)
        self.peblock = ProjectExciteLayer(out_channels + 7 * dwt_channels)
        self.conv = nn.Conv3d(out_channels + 7 * dwt_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x1, x2):
        # 进行小波变换
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = self.dwt(x2)
        # 残差结构处理
        o1 = self.resblock(x1)
        wat_tensor = torch.cat([o1, LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)
        o2 = self.peblock(wat_tensor)
        o3 = self.conv(o2) + o1
        return o3, LLL


class WatEncoder(nn.Module):
    def __init__(self, in_channels=1, basedim=16, downdeepth=2, model_type='3d', embedding_dim=8):
        super().__init__()
        if model_type == '3d':
            self.conv = nn.Conv3d
        else:
            self.conv = nn.Conv2d

        self.begin_conv = StackConvBlock(in_channels=in_channels, out_channels=basedim, model_type=model_type)
        self.encoding_block = nn.ModuleList([
            WATConvBlock(basedim * 2 ** convidx,
                         basedim * 2 ** (convidx + 1), dwt_channels=in_channels,
                         kernel_size=3, stride=2, model_type=model_type)
            for convidx in range(0, downdeepth)
        ])
        self.enc_out = basedim * 2 ** downdeepth
        self.pre_vq_conv = self.conv(self.enc_out, embedding_dim, 1, )

    def forward(self, x):
        o1 = self.begin_conv(x)
        LLL = x
        for block in self.encoding_block:
            o1, LLL = block(o1, LLL)
        o2 = self.pre_vq_conv(o1)
        return o2


def build_end_activation(input, activation='linear', alpha=None):
    if activation == 'softmax':
        output = F.softmax(input, dim=1)
    elif activation == 'sigmoid':
        output = torch.sigmoid(input)
    elif activation == 'elu':
        if alpha is None: alpha = 0.01
        output = F.elu(input, alpha=alpha)
    elif activation == 'lrelu':
        if alpha is None: alpha = 0.01
        output = F.leaky_relu(input, negative_slope=alpha)
    elif activation == 'relu':
        output = F.relu(input)
    elif activation == 'tanh':
        output = F.tanh(input)  # * (3-2.0*torch.relu(1-torch.relu(input*100)))
    else:
        output = input
    return output


class RVQVAE(VQVAE):
    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            channels: Sequence[int] = (96, 96, 192),  # 需要提供默认值或确保调用时传入
            num_res_layers: int = 3,  # 需要提供默认值或确保调用时传入
            num_res_channels: Sequence[int] | int = (96, 96, 192),  # 需要提供默认值或确保调用时传入
            downsample_parameters: Sequence[Tuple[int, int, int, int]] | Tuple[int, int, int, int] = (
                    (2, 4, 1, 1),
                    (2, 4, 1, 1),
                    (2, 4, 1, 1),
            ),  # 需要提供默认值或确保调用时传入
            upsample_parameters: Sequence[Tuple[int, int, int, int, int]] | Tuple[int, int, int, int, int] = (
                    (2, 4, 1, 1, 0),
                    (2, 4, 1, 1, 0),
                    (2, 4, 1, 1, 0),
            ),  # 需要提供默认值或确保调用时传入
            num_quantizers: int = 8,
            codebook_size: int = 1024,
            embedding_dim: int = 64,
            commitment_cost: float = 0.25,
            decay: float = 0.99,
            act: tuple | str | None = 'leakyrelu',
            **kwargs  # 接收其他父类参数
    ):
        # 先调用父类的初始化方法
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            num_res_layers=num_res_layers,
            num_res_channels=num_res_channels,
            downsample_parameters=downsample_parameters,
            upsample_parameters=upsample_parameters,
            num_embeddings=codebook_size,
            embedding_dim=embedding_dim,
            act=act,
            commitment_cost=commitment_cost,
            decay=decay,
            **kwargs
        )

        # 替换量化器为ResidualVQ
        self.quantizer = ResidualVQ(
            dim=embedding_dim,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            commitment_weight=commitment_cost,
            decay=decay,
            kmeans_init=True,
            rotation_trick=True,
            threshold_ema_dead_code=2,
        )

    def quantize(self, z: torch.Tensor):
        """
        z:  (B, C, H, W)      if spatial_dims==2
            (B, C, H, W, D)  if spatial_dims==3
        return  与 z 同形状的量化结果
        """
        # 1. 摊平空间维
        #    2-D -> 'b c h w   -> b (h w) c'
        #    3-D -> 'b c h w d -> b (h w d) c'
        spatial_pattern = 'h w' if self.spatial_dims == 2 else 'h w d'
        z_flat = rearrange(z, f'b c {spatial_pattern} -> b ({spatial_pattern}) c')

        # 2. 量化
        quantized_flat, indices, commit_loss = self.quantizer(z_flat)

        # 3. 还原空间形状
        quantized = rearrange(
            quantized_flat,
            f'b ({spatial_pattern}) c -> b c {spatial_pattern}',
            **{k: z.shape[i + 2] for i, k in enumerate(spatial_pattern.split())}
        )
        return quantized, indices, commit_loss.mean()

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(images)
        quantized, indices, commit_loss = self.quantize(z)
        reconstruction = self.decode(quantized)

        return reconstruction, commit_loss

    def encode_stage_2_inputs(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        e, _, _ = self.quantize(z)
        return e

    def decode_stage_2_outputs(self, z: torch.Tensor) -> torch.Tensor:
        e, _, _ = self.quantize(z)
        image = self.decode(e)
        return image


class RVQGAN(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters(ignore=['vqvae'])
        self.cfg = cfg
        self.startd_step = cfg.model.startd_step  # step to start training discriminator

        # generator
        self.vqvae = RVQVAE(
            spatial_dims=3,
            in_channels=3,
            out_channels=3,
            channels=(32, 64),
            num_res_layers=3,
            num_res_channels=(32, 64),
            downsample_parameters=((2, 4, 1, 1), (2, 4, 1, 1)),
            upsample_parameters=((2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
            num_quantizers=8,
            codebook_size=2048,
            embedding_dim=8,
            commitment_cost=0.25,
            decay=0.99,
        )

        # discriminator
        self.discriminator = PatchDiscriminator(spatial_dims=3, in_channels=cfg.model.in_channels,
                                                channels=cfg.model.disc_channels,
                                                out_channels=cfg.model.out_channels,
                                                num_layers_d=cfg.model.disc_downdeepth)

        # loss weights
        self.gan_feat_weight = cfg.model.gan_feat_weight
        self.adv_weight = cfg.model.adv_weight
        self.recon_weight = cfg.model.recon_weight
        self.perceptual_weight = cfg.model.perceptual_weight
        self.ganloss = GANLossComps(gan_type=cfg.model.gan_type, loss_weight=self.adv_weight)
        self.perceptual_loss = PerceptualLoss(spatial_dims=3, network_type="vgg", is_fake_3d=True, fake_3d_ratio=0.2,
                                              cache_dir="./pretrained_weights", ).eval()

        # metrics
        self.ssim_metric = SSIMMetric(spatial_dims=3, data_range=1.0)
        # self.psnr_metric = PSNRMetric(spatial_dims=cfg.dims, data_range=1.0)

    def configure_optimizers(self):
        lr = self.cfg.model.lr
        opt_g = torch.optim.Adam(self.vqvae.parameters(), lr=lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        return opt_g, opt_d

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.vqvae.encode_stage_2_inputs(x)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        image = self.vqvae.decode_stage_2_outputs(z)
        return image

    def training_step(self, batch):
        optimizer_g, optimizer_d = self.optimizers()

        # =================================================================================== #
        #                               1. DATA                                               #
        # =================================================================================== #
        Tup_T1_raw = batch['Tup_T1'][tio.DATA] / 255.0
        Tup_T2_raw = batch['Tup_T2'][tio.DATA] / 255.0
        Tup_FLAIR_raw = batch['Tup_FLAIR'][tio.DATA] / 255.0
        x = torch.concat([Tup_T1_raw, Tup_T2_raw, Tup_FLAIR_raw], dim=1)
        print(x.min(), x.max())

        # =================================================================================== #
        #                               2. train generator                                    #
        # =================================================================================== #
        self.toggle_optimizer(optimizer_g)
        x_recon, quantization_losses = self.vqvae(x)

        # recon_loss
        recon_loss = F.l1_loss(x_recon, x)

        # perceptual loss
        perceptual_loss = self.perceptual_loss(x_recon, x)

        print(self.global_step)
        # features_loss and adv_loss
        if self.global_step > self.startd_step:
            out_recon = self.discriminator(x_recon)
            logits_recon = out_recon[-1]
            features_recon = out_recon[:-1]
            out_orig = self.discriminator(x)
            features_real = out_orig[:-1]
            adv_loss = self.ganloss(logits_recon, True, is_disc=False)

            features_loss = torch.tensor(0.0, device=x.device)
            if self.gan_feat_weight > 0:
                for i in range(0, len(features_recon) - 1):
                    features_loss += F.l1_loss(features_recon[i], features_real[i].detach())
            self.log("train/adv_loss", adv_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("train/features_loss", features_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        else:
            adv_loss = torch.tensor(0.0, device=x.device)
            features_loss = torch.tensor(0.0, device=x.device)

        # 计算总损失
        g_loss = recon_loss * self.recon_weight + quantization_losses + adv_loss + features_loss * self.gan_feat_weight + perceptual_loss * self.perceptual_weight
        self.manual_backward(g_loss)  # 手动反向传播
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)  # 关闭优化器
        self.log("g_loss", g_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train_recon_loss", recon_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/quantization_losses", quantization_losses, prog_bar=True, logger=True, on_step=True,
                 on_epoch=True)
        self.log("train/perceptual_loss", perceptual_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        # =================================================================================== #
        #                             3. 训练判别器                                            #
        # =================================================================================== #
        if self.global_step > self.startd_step:
            self.toggle_optimizer(optimizer_d)
            x_real = x.detach().requires_grad_(True)
            out_real = self.discriminator(x_real)
            logits_real = out_real[-1]
            x_fake = x_recon.detach()
            out_fake = self.discriminator(x_fake)
            logits_fake = out_fake[-1]

            d_loss = (self.ganloss(logits_real, True, is_disc=True) + self.ganloss(logits_fake, False,
                                                                                   is_disc=True)) / 2
            self.log("train/d_loss", d_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

            r1_penalty = r_reg(logits_real, x_real)
            self.log("train/r_penalty", r1_penalty, prog_bar=True, logger=True, on_step=True, on_epoch=True)

            self.manual_backward(d_loss)
            # self.clip_gradients(optimizer_d, gradient_clip_val=0.001, gradient_clip_algorithm="norm")
            optimizer_d.step()
            optimizer_d.zero_grad()
            self.untoggle_optimizer(optimizer_d)
        return g_loss

    def validation_step(self, batch):
        # =================================================================================== #
        #                               1. DATA                                               #
        # =================================================================================== #
        Tup_T1_raw = batch['Tup_T1'][tio.DATA] / 255.0
        Tup_T2_raw = batch['Tup_T2'][tio.DATA] / 255.0
        Tup_FLAIR_raw = batch['Tup_FLAIR'][tio.DATA] / 255.0
        x = torch.concat([Tup_T1_raw, Tup_T2_raw, Tup_FLAIR_raw], dim=1)

        # =================================================================================== #
        #                               2. gen                                                #
        # =================================================================================== #
        z = self.vqvae.encoder(x)
        z, indices, quantization_losses = self.vqvae.quantize(z)
        x_recon = self.vqvae.decoder(z)

        # =================================================================================== #
        #                               2. metric                                             #
        # =================================================================================== #
        # std
        batch_std = z.std().item()
        self.log("val/latent_std", batch_std, prog_bar=False, on_step=True, on_epoch=True)

        x_recon = torch.clamp(x_recon, 0.0, 1.0)
        ssim = self.ssim_metric(x_recon, x).mean()
        self.log("val/ssim", ssim, prog_bar=True, logger=True, on_epoch=True)

        # psnr = self.psnr_metric(x_recon, x).mean()
        # self.log("val/psnr", psnr, prog_bar=True, logger=True, on_epoch=True)

        recon_loss = F.l1_loss(x_recon, x)
        self.log("val/recon_loss", recon_loss, prog_bar=True, logger=True, on_epoch=True)

        self.save_3d_image_slices(x_recon[0], "val/generated_reconst_images", self.global_step)
        self.save_3d_image_slices(x[0], "val/input_images", self.global_step)

        # =================================================================================== #
        #                               3. perplexity and  usage_ratio                        #
        #               TODO: This needs to be adjusted according to the actual situation.    #
        # =================================================================================== #
        # B, num_tokens, num_quant = indices.shape
        # layer_perplexities = []
        # layer_usage_ratios = []
        # for q in range(num_quant):
        #     layer_indices = indices[..., q].flatten().long()  # shape: [B * num_tokens]
        #
        #     # 1. computer usage_ratio
        #     unique_indices = torch.unique(layer_indices)
        #     layer_codebook_size = self.cfg.model.n_codes
        #     usage_ratio = len(unique_indices) / layer_codebook_size
        #     layer_usage_ratios.append(usage_ratio)
        #
        #     # 2. computer perplexity
        #     counts = torch.bincount(layer_indices, minlength=layer_codebook_size)
        #     probs = counts.float() / counts.sum()
        #     perplexity = torch.exp(-torch.sum(probs * torch.log(probs + 1e-10)))
        #     layer_perplexities.append(perplexity)
        #
        #     self.log(f"val/codebook_q{q}_perplexity", perplexity, on_epoch=True)
        #     self.log(f"val/codebook_q{q}_usage_ratio", usage_ratio, on_epoch=True)
        #
        # avg_perplexity = torch.stack(layer_perplexities).mean()
        # avg_usage_ratio = sum(layer_usage_ratios) / len(layer_usage_ratios)
        # self.log("train/codebook_avg_perplexity", avg_perplexity, prog_bar=True)
        # self.log("train/codebook_avg_usage_ratio", avg_usage_ratio, prog_bar=False)

        return recon_loss

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


if __name__ == '__main__':
    tensor1 = torch.randn(1, 1, 128, 128, 128)  # 输入张量
    # # 创建 Encoder 实例
    # encoder = Encoder(in_channels=1, basedim=32, downdeepth=2, model_type='3d', embedding_dim=256)
    # # 前向传播
    # output_tensor = encoder(tensor1)
    # # 输出张量的形状
    # print(output_tensor.shape)
    #
    # # 创建 Decoder 实例
    # decoder = Decoder(out_channels=1, basedim=32, downdeepth=2, model_type='3d', embedding_dim=256)
    # # 前向传播
    # output_tensor = decoder(output_tensor)
    # # 输出张量的形状
    # print(output_tensor.shape)