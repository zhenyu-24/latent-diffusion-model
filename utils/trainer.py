import lightning as L  # 主要变化：导入统一为lightning
from lightning.pytorch.loggers import TensorBoardLogger  # 子模块路径变化
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, TQDMProgressBar
from lightning.pytorch.callbacks import WeightAveraging
from torch.optim.swa_utils import get_ema_avg_fn


class EMAWeightAveraging(WeightAveraging):
    def __init__(self):
        super().__init__(avg_fn=get_ema_avg_fn(decay=0.99))

    def should_update(self, step_idx=None, epoch_idx=None):
        # Start after 10000 steps.
        return (step_idx is not None) and (step_idx >= 1000)


def create_ddpmtrainer(
        name='model_1',
        save_dir='logs/',
        checkpoint_dir='checkpoints/',
        precision=16,
        max_epoch=1000,
        strategy='auto',
        is_earlystopping=False,
        monitor='val_loss',
        check_val_every_n_epoch=1000,
        gradient_clip_val=0.5
):
    print(f"name: {name}, checkpoint_dir: {checkpoint_dir}, "
          f"is_earlystopping: {is_earlystopping}, monitor: {monitor}")

    # 创建 TensorBoardLogger 日志记录器
    logger = TensorBoardLogger(save_dir=save_dir, name=name)

    # 创建 ModelCheckpoint 回调
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,  # 保存模型的路径
        save_last=True,  # 是否保存最后一个模型
        save_top_k=10,  # 保存最好的几个模型
        monitor=monitor,  # 监控的指标
        filename=f'{name}-epoch={{epoch}}-{{{monitor}:.4f}}',
        mode='min',  # 保存模型的模式
        every_n_train_steps=1000,  # 每多少训练步保存一次模型
        save_on_train_epoch_end=True,
    )

    # 学习率监控
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # 进度条
    progress_bar = TQDMProgressBar(refresh_rate=1)

    callbacks = [checkpoint_callback, progress_bar, lr_monitor, EMAWeightAveraging()]

    if is_earlystopping:
        # 创建 EarlyStopping 回调
        earlystopping = EarlyStopping(monitor, mode="min")
        callbacks.append(earlystopping)

    # 创建 Trainer 实例 - 使用 L.Trainer
    trainer = L.Trainer(
        max_epochs=max_epoch,
        max_steps=1000000,
        precision=precision,  # 精度
        logger=logger,  # 日志记录器
        callbacks=callbacks,  # 回调函数
        strategy=strategy,  # 分布式策略
        check_val_every_n_epoch=check_val_every_n_epoch,  # 每隔多少个 epoch 验证一次
        # gradient_clip_val=gradient_clip_val,  # 梯度裁剪
        accumulate_grad_batches=1,  # 梯度累积
    )

    return trainer