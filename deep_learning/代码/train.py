import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v2
from PIL import Image
import numpy as np
import random
import argparse


# ============================================================
# 模型定义
# ============================================================

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        # 两层卷积，用于特征融合和 refinement
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x, skip):
        # 上采样到与跳跃连接相同的尺寸
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        # 拼接深层特征和浅层特征
        return self.conv(torch.cat([x, skip], dim=1))


class MobileUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载预训练的MobileNetV2作为编码器
        backbone = mobilenet_v2(pretrained=True)
        # 编码器各层（下采样路径）- 变量名与预训练权重保持一致
        self.s1 = backbone.features[0:2]        # 输出16通道
        self.s2 = backbone.features[2:4]       # 输出24通道
        self.s3 = backbone.features[4:7]       # 输出32通道
        self.s4 = backbone.features[7:14]      # 输出96通道
        self.bot = backbone.features[14:19]    # 输出1280通道
        # 瓶颈层：压缩通道数
        self.bot_conv = nn.Sequential(
            nn.Conv2d(1280, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        # 解码器各层（上采样路径）
        self.dec4 = DecoderBlock(256, 96, 128)
        self.dec3 = DecoderBlock(128, 32, 64)
        self.dec2 = DecoderBlock(64, 24, 32)
        self.dec1 = DecoderBlock(32, 16, 16)
        # 最终输出层：1通道（二分类：滑坡/背景）
        self.final = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        # 编码路径：逐层提取特征
        s1 = self.s1(x)
        s2 = self.s2(s1)
        s3 = self.s3(s2)
        s4 = self.s4(s3)
        b = self.bot_conv(self.bot(s4))
        # 解码路径：逐层恢复分辨率，融合多尺度特征
        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        # 输出与输入相同尺寸的预测图
        return self.final(F.interpolate(d1, x.shape[2:], mode='bilinear'))


# ============================================================
# 损失函数
# ============================================================

class HighRecallCombinedLoss(nn.Module):
    def __init__(self, bce_w=0.15, dice_w=0.55, focal_w=0.30,
                 focal_gamma=3.0, pos_weight=15.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        self.bce_w = bce_w
        self.dice_w = dice_w
        self.focal_w = focal_w
        self.gamma = focal_gamma
    def forward(self, logits, target):
        # BCE损失
        bce = self.bce(logits, target)
        # Dice损失
        pred = torch.sigmoid(logits)
        p = pred.contiguous().view(-1)
        t = target.contiguous().view(-1)
        intersection = (p * t).sum()
        dice = 1 - (2. * intersection + 1.0) / (p.sum() + t.sum() + 1.0)
        # Focal损失
        p_t = t * p + (1 - t) * (1 - p)
        focal = (-(1 - p_t) ** self.gamma * torch.log(p_t + 1e-8)).mean()
        return self.bce_w * bce + self.dice_w * dice + self.focal_w * focal


# ============================================================
# 数据集
# ============================================================

class LandslideDataset(Dataset):
    def __init__(self, data_roots, splits=['train'], augment=True, img_size=256):
        self.augment = augment
        self.img_size = img_size
        self.items = []
        # 扫描所有数据目录
        for root in data_roots:
            for split in splits:
                img_dir = os.path.join(root, split, 'images')
                lbl_dir = os.path.join(root, split, 'labels')
                if not os.path.exists(img_dir):
                    continue
                for f in sorted(os.listdir(img_dir)):
                    if not f.lower().endswith(('.jpg', '.png')):
                        continue
                    img_path = os.path.join(img_dir, f)
                    lbl_path = os.path.join(lbl_dir, os.path.splitext(f)[0] + '.png')
                    if os.path.exists(lbl_path):
                        self.items.append((img_path, lbl_path))
        print(f"Loaded {len(self.items)} samples")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, lbl_path = self.items[idx]
        # 读取并缩放图像
        img = Image.open(img_path).convert('RGB').resize(
            (self.img_size, self.img_size), Image.BILINEAR)
        lbl = Image.open(lbl_path).convert('L').resize(
            (self.img_size, self.img_size), Image.NEAREST)
        # 数据增强
        if self.augment:
            # 随机水平翻转
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                lbl = lbl.transpose(Image.FLIP_LEFT_RIGHT)
            # 随机垂直翻转
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                lbl = lbl.transpose(Image.FLIP_TOP_BOTTOM)
            # 随机旋转（0°/90°/180°/270°）
            rot = random.choice([0, 90, 180, 270])
            if rot > 0:
                img = img.rotate(rot, Image.BILINEAR)
                lbl = lbl.rotate(rot, Image.NEAREST)
        # 转换为Tensor，归一化到[0,1]
        img_np = np.array(img, dtype=np.float32) / 255.0
        lbl_np = (np.array(lbl) > 128).astype(np.float32)
        # 增加通道维度 [H,W] -> [1,H,W]
        lbl_np = lbl_np[np.newaxis, :]
        return torch.from_numpy(img_np.transpose(2, 0, 1)), torch.from_numpy(lbl_np)


# ============================================================
# 评估函数
# ============================================================

def evaluate(model, loader, device):
    model.eval()
    tp, fp, fn = 0, 0, 0
    with torch.no_grad():
        for img, lbl in loader:
            img, lbl = img.to(device), lbl.to(device)
            logits = model(img)
            pred = (torch.sigmoid(logits) > 0.5).float()
            # 统计TP、FP、FN
            tp += ((pred == 1) & (lbl == 1)).sum().item()
            fp += ((pred == 1) & (lbl == 0)).sum().item()
            fn += ((pred == 0) & (lbl == 1)).sum().item()
    # 计算各指标
    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    return p, r, f1, iou


# ============================================================
# 主函数
# ============================================================

def main():
    # 命令行参数（默认值为已验证的最佳配置，直接运行 python train.py 即可）
    parser = argparse.ArgumentParser(description='滑坡分割 - 训练 MobileUNet')
    parser.add_argument('--epochs', type=int, default=150, help='训练轮数 (默认: 150)')
    parser.add_argument('--lr', type=float, default=0.0001, help='学习率 (默认: 0.0001)')
    parser.add_argument('--bs', type=int, default=8, help='批次大小 (默认: 8)')
    args = parser.parse_args()

    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("滑坡分割 - 训练 MobileUNet")
    print("=" * 60)
    print(f"使用设备: {device}")
    print(f"训练轮数: {args.epochs}")
    print(f"学习率: {args.lr}")
    print(f"批次大小: {args.bs}")
    print("=" * 60)

    # 数据集路径
    # 训练集：横断2008(train) + 横断2018(train+val) + 汶川(train+val)
    # 验证集/测试集：横断2008(val) — 横断2008的val不参与训练，保证测试公平

    print("\n构建数据集...")
    train_dataset = LandslideDataset(
        data_roots=[
            './landslide_data/横断_2008',  # 只用train部分
            './landslide_data/横断_2018',
            './landslide_data/汶川',
        ],
        splits=['train'],
        augment=True,
    )
    # 横断2018和汶川的val也加入训练
    train_dataset_extra = LandslideDataset(
        data_roots=[
            './landslide_data/横断_2018',
            './landslide_data/汶川',
        ],
        splits=['val'],
        augment=True,
    )
    train_dataset.items.extend(train_dataset_extra.items)
    print(f"训练集总计: {len(train_dataset.items)} samples")

    val_dataset = LandslideDataset(
        data_roots=['./landslide_data/横断_2008'],
        splits=['val'],
        augment=False,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.bs)
    # 创建模型
    print(f"\n构建 MobileUNet...")
    model = MobileUNet().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"训练参数: {n_params:,}")
    # 训练配置
    criterion = HighRecallCombinedLoss(pos_weight=15.0)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    # 创建保存目录
    os.makedirs('./checkpoints', exist_ok=True)
    save_path = './checkpoints/best_final.pth'

    # 训练循环
    print(f"\n训练 {args.epochs} epochs...")
    best_f1 = 0.0
    best_iou = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        # 前向传播 + 反向传播
        for img, lbl in train_loader:
            img, lbl = img.to(device), lbl.to(device)
            optimizer.zero_grad()
            loss = criterion(model(img), lbl)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * img.size(0)
        train_loss /= len(train_loader.dataset)
        scheduler.step()
        # 每5个epoch评估一次
        if epoch % 5 == 0 or epoch == 1:
            p, r, f1, iou = evaluate(model, val_loader, device)
            print(f"Epoch {epoch:4d}: loss={train_loss:.4f} "
                  f"P={p:.3f} R={r:.3f} F1={f1:.3f} IoU={iou:.3f}")
            if f1 > best_f1:
                best_f1, best_iou = f1, iou
                torch.save(model.state_dict(), save_path)
                print(f"  -> saved best (F1={f1:.4f})")

    # 训练完成
    print(f"\n{'='*50}")
    print(f"训练完成！最佳F1={best_f1:.3f}, IoU={best_iou:.3f}")
    print(f"模型保存在: {save_path}")
    print(f"{'='*50}")

    # 最终验证
    print(f"\n{'='*50}")
    print(f"最终验证结果")
    print(f"{'='*50}")

    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()

    p, r, f1, iou = evaluate(model, val_loader, device)
    print(f"P={p:.3f} R={r:.3f} F1={f1:.3f} IoU={iou:.3f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
