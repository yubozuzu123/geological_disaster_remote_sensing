import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torchvision.models import mobilenet_v2
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # 无GUI后端，避免显示问题
import matplotlib.pyplot as plt
# 中文字体配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 模型定义
# ============================================================

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class MobileUNet(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = mobilenet_v2(pretrained=False)  # 测试时不需要下载预训练权重
        self.s1 = backbone.features[0:2]
        self.s2 = backbone.features[2:4]
        self.s3 = backbone.features[4:7]
        self.s4 = backbone.features[7:14]
        self.bot = backbone.features[14:19]
        self.bot_conv = nn.Sequential(
            nn.Conv2d(1280, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.dec4 = DecoderBlock(256, 96, 128)
        self.dec3 = DecoderBlock(128, 32, 64)
        self.dec2 = DecoderBlock(64, 24, 32)
        self.dec1 = DecoderBlock(32, 16, 16)
        self.final = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        s1 = self.s1(x)
        s2 = self.s2(s1)
        s3 = self.s3(s2)
        s4 = self.s4(s3)
        b = self.bot_conv(self.bot(s4))
        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        d0 = F.interpolate(d1, size=x.shape[2:], mode='bilinear', align_corners=True)
        return self.final(d0)


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("滑坡分割 - 测试")
    print("=" * 60)

    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # 加载模型
    print("\n加载模型...")
    model = MobileUNet().to(device)
    state_dict = torch.load('./checkpoints/best_final.pth',
                           map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    # 测试集路径：横断2008的val
    test_img_dir = './landslide_data/横断_2008/val/images'
    test_lbl_dir = './landslide_data/横断_2008/val/labels'
    # 扫描测试样本
    test_samples = []
    for f in sorted(os.listdir(test_img_dir)):
        if not f.lower().endswith(('.jpg', '.png')):
            continue
        base = os.path.splitext(f)[0]
        lbl_path = os.path.join(test_lbl_dir, base + '.png')
        if os.path.exists(lbl_path):
            test_samples.append((os.path.join(test_img_dir, f), lbl_path))
    print(f"测试样本数: {len(test_samples)}")

    # 评估（阈值0.5）
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)

    tp, fp, fn = 0, 0, 0
    with torch.no_grad():
        for img_path, lbl_path in test_samples:
            # 读取图像和标签
            img = Image.open(img_path).convert('RGB').resize(
                (256, 256), Image.BILINEAR)
            lbl = Image.open(lbl_path).convert('L').resize(
                (256, 256), Image.NEAREST)
            # 转换为Tensor
            img_t = transforms.ToTensor()(img).unsqueeze(0).to(device)
            gt = (np.array(lbl) > 128).astype(np.uint8)
            prob = torch.sigmoid(model(img_t))
            pred = (prob > 0.5).long().squeeze().cpu().numpy().astype(np.uint8)

            # 统计TP、FP、FN
            tp += int(np.sum((pred == 1) & (gt == 1)))
            fp += int(np.sum((pred == 1) & (gt == 0)))
            fn += int(np.sum((pred == 0) & (gt == 1)))

    # 计算指标
    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    print(f"P={p:.3f}  R={r:.3f}  F1={f1:.3f}  IoU={iou:.3f}")
    print("=" * 60)

    # 生成三联图可视化
    print("\n生成三联图...")
    os.makedirs('./detections_simple', exist_ok=True)

    # 按滑坡占比排序，选择代表性样本
    sample_data = []
    for img_path, lbl_path in test_samples:
        lbl = Image.open(lbl_path).convert('L').resize((256, 256), Image.NEAREST)
        ratio = np.sum(np.array(lbl) > 128) / (256 * 256)
        sample_data.append((img_path, lbl_path, ratio))
    sample_data.sort(key=lambda x: x[2])
    # 选取10个样本
    indices = np.linspace(0, len(sample_data) - 1, 10).astype(int)
    selected = [sample_data[i] for i in indices]
    for i, (img_path, lbl_path, ratio) in enumerate(selected):
        # 读取图像
        img = Image.open(img_path).convert('RGB').resize((256, 256), Image.BILINEAR)
        lbl = Image.open(lbl_path).convert('L').resize((256, 256), Image.NEAREST)
        # 提取
        img_t = transforms.ToTensor()(img).unsqueeze(0).to(device)
        gt_bin = (np.array(lbl) > 128).astype(np.uint8)
        with torch.no_grad():
            prob = torch.sigmoid(model(img_t))
            pred_bin = (prob > 0.5).long().squeeze().cpu().numpy().astype(np.uint8)
        img_np = np.array(img)

        # 标签：黑白图
        gt = gt_bin * 255

        # 提取结果：白色=正确提取，红色=错提，绿色=漏提，黑色=背景
        fp = (pred_bin == 1) & (gt_bin == 0)
        fn = (pred_bin == 0) & (gt_bin == 1)
        tp = (pred_bin == 1) & (gt_bin == 1)
        pred_color = np.zeros((256, 256, 3), dtype=np.uint8)
        pred_color[tp] = [255, 255, 255]
        pred_color[fp] = [255, 0, 0]
        pred_color[fn] = [0, 255, 0]

        # 绘制三联图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_np)
        axes[0].set_title('图像', fontsize=14)
        axes[0].axis('off')
        axes[1].imshow(gt, cmap='gray')
        axes[1].set_title('标签', fontsize=14)
        axes[1].axis('off')
        axes[2].imshow(pred_color)
        axes[2].set_title('提取结果', fontsize=14)
        axes[2].axis('off')
        plt.tight_layout()
        plt.savefig(f'./detections_simple/triplet_{i:03d}.png',
                   dpi=150, bbox_inches='tight')
        plt.close()

    print(f"保存了 {len(selected)} 个三联图到 ./detections_simple/")

    # 总结
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    print(f"测试集: 横断_2008 val (共 {len(test_samples)} 样本)")
    print(f"阈值: 0.5 (固定)")
    print(f"IoU: {iou:.3f}")
    print(f"F1: {f1:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
