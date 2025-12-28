import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
import lightning as L
from einops import rearrange
from typing import *
from collections.abc import Sequence
from monai.networks.nets.vqvae import Encoder, Decoder, VQVAE
from monai.networks.nets import AutoencoderKL, PatchDiscriminator
from monai.networks.layers import Act
from monai.losses import PerceptualLoss, SSIMLoss, DiceLoss
from monai.metrics import SSIMMetric, PSNRMetric, MAEMetric, MSEMetric
from vector_quantize_pytorch import FSQ, LFQ, ResidualVQ

import torchio as tio

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


class SimpleEncoder(nn.Module):
    def __init__(self, in_channels=1, basedim=32, downdeepth=2, model_type='3d', embedding_dim=256, num_groups=8):
        super().__init__()
        if model_type == '3d':
            self.conv = nn.Conv3d
        else:
            self.conv = nn.Conv2d

        self.begin_conv = ResBlock(in_channels=in_channels, out_channels=basedim, model_type=model_type,
                                   num_groups=num_groups)
        self.encoding_block = nn.ModuleList([nn.Sequential(
            ResBlock(basedim * 2 ** convidx, basedim * 2 ** (convidx + 1), 3, 2,
                     model_type=model_type, num_groups=num_groups)) for
            convidx in range(0, downdeepth)])
        self.enc_out = basedim * 2 ** downdeepth
        self.pre_vq_conv = self.conv(self.enc_out, embedding_dim, 1, )

    def forward(self, x):
        x = self.begin_conv(x)
        for block in self.encoding_block:
            x = block(x)
        x = self.pre_vq_conv(x)
        return x


class SimpleDecoder(nn.Module):
    def __init__(self, out_channels=1, basedim=32, downdeepth=2, model_type='3d', embedding_dim=256, num_groups=8):
        super().__init__()
        self.model_type = model_type
        if self.model_type == '3d':
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = nn.Conv3d
        else:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = nn.Conv2d

        self.enc_out = basedim * 2 ** downdeepth
        # self.post_vq_conv = ResBlock(embedding_dim, self.enc_out, 1, model_type=model_type, num_groups=num_groups)
        self.post_vq_conv = self.conv(embedding_dim, self.enc_out, 1, )

        self.decoding_block = nn.ModuleList([nn.Sequential(
            self.up,
            ResBlock(basedim * 2 ** convidx, basedim * 2 ** (convidx - 1), 3, 1,
                     model_type=self.model_type, num_groups=num_groups), ) for convidx in
            reversed(range(1, downdeepth + 1))])
        self.final_conv = self.conv(basedim, out_channels, 1, )

    def forward(self, x):
        x = self.post_vq_conv(x)
        for block in self.decoding_block:
            x = block(x)
        x = self.final_conv(x)
        return x

def r_reg(d_out, x_in):
    """
    计算R1正则化损失。

    参数:
        d_out: 判别器输出。
        x_in: 输入图像。

    返回:
        reg: 正则化损失。
    """
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
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
            act: tuple | str | None = Act.RELU,
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
            **{k: z.shape[i+2] for i, k in enumerate(spatial_pattern.split())}
        )
        return quantized, indices, commit_loss.mean()

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(images)
        print(z.shape)
        quantized, indices, commit_loss = self.quantize(z)
        print(quantized.shape)
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

class VQVAE_Wrapper(L.LightningModule):
    def __init__(self, cfg, vqvae: nn.Module):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters(ignore=['vqvae'])
        self.cfg = cfg
        self.startd_step = cfg.model.startd_step    # step to start training discriminator

        # generator
        self.vqvae = vqvae

        # discriminator
        self.discriminator = PatchDiscriminator(spatial_dims=cfg.dims, in_channels=cfg.model.in_channels,
                                                channels=cfg.model.disc_channels, out_channels=cfg.model.out_channels,
                                                num_layers_d=cfg.model.disc_downdeepth)

        # loss weights
        self.gan_feat_weight = cfg.model.gan_feat_weight
        self.adv_weight = cfg.model.adv_weight
        self.recon_weight = cfg.model.recon_weight
        self.perceptual_weight = cfg.model.perceptual_weight
        self.ganloss = GANLossComps(gan_type=cfg.model.gan_type, loss_weight=self.adv_weight)
        self.perceptualloss = PerceptualLoss(
            spatial_dims=cfg.dims,
            network_type="vgg",
            is_fake_3d=True,
            fake_3d_ratio=0.2,
            cache_dir="./pretrained_weights",
        ).eval()

        # metrics
        self.ssim_metric = SSIMMetric(spatial_dims=cfg.dims, data_range=1.0)
        self.psnr_metric = PSNRMetric(spatial_dims=cfg.dims, data_range=1.0)

    def configure_optimizers(self):
        lr = self.cfg.model.lr
        opt_g = torch.optim.Adam(self.vqvae.parameters(), lr=lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        return opt_g, opt_d

    def training_step(self, batch):
        optimizer_g, optimizer_d = self.optimizers()

        # =================================================================================== #
        #                               1. DATA                                               #
        # =================================================================================== #
        Tup_T1_raw = batch['Tup_T1'][tio.DATA] / 255.0
        Tup_T2_raw = batch['Tup_T2'][tio.DATA] / 255.0
        Tup_FLAIR_raw = batch['Tup_FLAIR'][tio.DATA] / 255.0
        x = torch.concat([Tup_T1_raw, Tup_T2_raw, Tup_FLAIR_raw], dim=1)

        # =================================================================================== #
        #                               2. train generator                                    #
        # =================================================================================== #
        self.toggle_optimizer(optimizer_g)
        x_recon, quantization_losses = self.vqvae(x)

        # recon_loss
        recon_loss = F.l1_loss(x_recon, x)

        # perceptual loss
        perceptual_loss = self.perceptualloss(x_recon, x)

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
        g_loss = recon_loss * self.recon_weight + quantization_losses + perceptual_loss * self.perceptual_weight  + adv_loss + features_loss * self.gan_feat_weight
        self.manual_backward(g_loss)  # 手动反向传播
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)  # 关闭优化器
        self.log("g_loss", g_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("recon_loss", recon_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/quantization_losses", quantization_losses, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/perceptual_loss", perceptual_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        # =================================================================================== #
        #                             3. 训练判别器                                            #
        # =================================================================================== #
        if self.global_step > self.startd_step:
            self.toggle_optimizer(optimizer_d)
            out_real = self.discriminator(x.detach())
            logits_real = out_real[-1]
            x_fake = x_recon.detach()
            out_fake = self.discriminator(x_fake.detach())
            logits_fake = out_fake[-1]
            d_loss = (self.ganloss(logits_real, True, is_disc=True) + self.ganloss(logits_fake, False, is_disc=True)) / 2
            self.manual_backward(d_loss)
            self.clip_gradients(optimizer_d, gradient_clip_val=1, gradient_clip_algorithm="norm")
            optimizer_d.step()
            optimizer_d.zero_grad()
            self.untoggle_optimizer(optimizer_d)
            self.log("train/d_loss", d_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
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
        z, indices, quantization_losses = self.vqvae.vector_quantize(z)
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

        psnr = self.psnr_metric(x_recon, x).mean()
        self.log("val/psnr", psnr, prog_bar=True, logger=True, on_epoch=True)

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


if __name__ == "__main__":
    vqvae = RVQVAE(
        spatial_dims=2,
        in_channels=3,
        out_channels=3,
        channels=(64, 128),  # 添加channels参数
        num_res_layers=2,
        num_res_channels=(64, 128),  # 添加num_res_channels参数
        downsample_parameters=((2, 4, 1, 1), (2, 4, 1, 1)),
        upsample_parameters=((2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
        num_quantizers=8,
        codebook_size=2048,
        embedding_dim=8,
        act='leakyrelu',
        commitment_cost=0.25,
        decay=0.99,
    ).cuda()

    x = torch.randn(1, 3, 128, 128).cuda()
    z, loss = vqvae(x)
    print(z.shape)