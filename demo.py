import cv2
import torch
import os
import utils.config as config
from model import build_segmenter
from utils.dataset import tokenize
import numpy as np
import matplotlib.pyplot as plt

# 加载配置文件和模型
cfg = config.load_cfg_from_cfg_file("./config/ref-zom/cris_r50.yaml")
PATH = "exp/ref-zom/last_model.pth"
model, _ = build_segmenter(cfg)
checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'], strict=True)
model.eval()
print("=> Loaded checkpoint '{}'".format(PATH))

# 输入尺寸 (416, 416)
input_size = (416, 416)

def resize_and_pad(img, input_size):
    """等比例缩放并填充图像"""
    ori_h, ori_w = img.shape[:2]
    inp_h, inp_w = input_size

    # 计算缩放比例
    scale = min(inp_w / ori_w, inp_h / ori_h)
    new_w, new_h = int(round(ori_w * scale)), int(round(ori_h * scale))

    # 缩放图像
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # 计算填充值
    pad_w = inp_w - new_w
    pad_h = inp_h - new_h
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    # 填充图像
    img_padded = cv2.copyMakeBorder(
        resized_img, top, bottom, left, right,
        cv2.BORDER_CONSTANT,
        value=[int(0.48145466 * 255), int(0.4578275 * 255), int(0.40821073 * 255)]
    )

    # 打印调试信息
    print(f"Original image size: {ori_w}x{ori_h}")
    print(f"Resized image size: {new_w}x{new_h}")
    print(f"Padding: left={left}, right={right}, top={top}, bottom={bottom}")
    print(f"ROI in padded image: x1={left}, x2={left + new_w}, y1={top}, y2={top + new_h}")

    # 返回填充图像以及原始图像在填充图像中的ROI
    return img_padded, (left, top, left + new_w, top + new_h)

def convert(img):
    """预处理图像：缩放、填充、归一化"""
    img_padded, roi = resize_and_pad(img, input_size)
    img_tensor = torch.from_numpy(img_padded.transpose((2, 0, 1))).float()

    # 归一化
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).reshape(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).reshape(3, 1, 1)
    img_tensor = img_tensor.div(255.).sub(mean).div(std)

    return img_tensor, roi

if __name__ == '__main__':
    while True:
        try:
            # 提示用户输入文件名
            print("Please insert image file name (default folder is './img'): ", end="")
            file_name = input().strip()  # 获取输入并去除前后空格

            # 拼接完整路径
            img_path = os.path.join("./img", file_name)

            # 检查文件是否存在
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"File '{img_path}' does not exist. Please try again.")

            # 如果文件存在，跳出循环
            print(f"File '{img_path}' loaded successfully.")
            break

        except FileNotFoundError as e:
            # 提示用户重新输入
            print(e)
        except Exception as e:
            # 捕获其他异常并退出
            print(f"Unexpected error: {e}")
            break

    # 加载文件
    print(f"Proceeding with file: {img_path}")
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ori_h, ori_w = img.shape[:2]

    # 预处理图像
    img_tensor, roi = convert(img)
    img_tensor = img_tensor.unsqueeze(0)  # 添加 batch 维度
    print('Input tensor shape:', img_tensor.shape)

    # 模型预测
    print("Please insert text: ", end='')
    sent = input()
    text = tokenize(sent, 17, True)
    pred = model(img_tensor, text)
    pred = torch.sigmoid(pred)

    # 将模型输出的遮罩插值到输入图像的尺寸
    pred = torch.nn.functional.interpolate(
        pred, size=input_size, mode='bilinear', align_corners=False
    )
    pred = (pred > 0.35)  # 阈值处理
    pred = pred.cpu().numpy()[0, 0].astype(np.uint8)  # 转换为 NumPy 格式
    print('Predicted mask shape (after upscaling):', pred.shape)

    # 裁剪预测的遮罩
    x1, y1, x2, y2 = roi
    pred_cropped = pred[y1:y2, x1:x2]
    print('Predicted mask shape (after cropping):', pred_cropped.shape)
    cv2.imwrite('pred_cropped.png', pred_cropped * 255)  # 保存裁剪后的遮罩

    # 缩放到原始图像尺寸
    pred_resized = cv2.resize(pred_cropped, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)
    print('Predicted mask shape (after resizing):', pred_resized.shape)
    cv2.imwrite('pred_resized.png', pred_resized * 255)  # 保存最终的遮罩

    # 将遮罩叠加在原始图像上
    mask_rgb = np.stack([pred_resized] * 3, axis=-1) * 255  # 将单通道遮罩转为RGB格式
    overlay = cv2.addWeighted(img, 0.7, mask_rgb.astype(np.uint8), 0.3, 0)  # 叠加图像

    # 保存叠加图像（不包含文本）
    cv2.imwrite('overlay_debug.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))  # 保存叠加图像

    # 在显示图像时添加文本
    plt.figure(figsize=(10, 10))
    plt.imshow(overlay)
    plt.title(f"Input Prompt: {sent}", fontsize=14, color='red')  # 添加标题作为文本
    plt.axis('off')
    plt.show()
