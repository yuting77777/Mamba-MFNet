import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import numpy as np
import torchvision.transforms.functional as TF
from models.loss_ssim import ssim

# 添加RGB到灰度转换函数
def rgb_to_grayscale(img):
    """
    将RGB图像转换为灰度图像
    img: [B, 3, H, W] 或 [B, 1, H, W]
    returns: [B, 1, H, W]
    """
    if img.size(1) == 1:
        return img  # 已经是灰度图
    
    # 使用标准的RGB到灰度转换公式: Y = 0.299*R + 0.587*G + 0.114*B
    grayscale = 0.299 * img[:, 0:1, :, :] + 0.587 * img[:, 1:2, :, :] + 0.114 * img[:, 2:3, :, :]
    return grayscale

class L_color(nn.Module):
    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape
        
        # 对于多通道图像，使用RGB到灰度转换后再计算
        if c > 1:
            x_gray = rgb_to_grayscale(x)
            mean_gray = torch.mean(x_gray, [2, 3], keepdim=True)
            k = torch.zeros_like(mean_gray)
            return k  # 对于灰度图，颜色差异为0
        
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr - mg, 2)
        Drb = torch.pow(mr - mb, 2)
        Dgb = torch.pow(mb - mg, 2)
        k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)
        return k

class L_SSIM(nn.Module):
    def __init__(self):
        super(L_SSIM, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        # 检查通道数，如果大于1则转换为灰度图
        if image_A.size(1) > 1 or image_B.size(1) > 1 or image_fused.size(1) > 1:
            image_A_gray = rgb_to_grayscale(image_A)
            image_B_gray = rgb_to_grayscale(image_B)
            image_fused_gray = rgb_to_grayscale(image_fused)
            
            # 计算灰度图的SSIM
            Loss_SSIM = 0.5 * ssim(image_A_gray, image_fused_gray) + 0.5 * ssim(image_B_gray, image_fused_gray)
        else:
            # 单通道图像直接计算
            Loss_SSIM = 0.5 * ssim(image_A, image_fused) + 0.5 * ssim(image_B, image_fused)
        
        return Loss_SSIM

class L_Grad(nn.Module):
    def __init__(self):
        super(L_Grad, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        # 统一转换为灰度图计算梯度
        if image_A.size(1) > 1:
            image_A_gray = rgb_to_grayscale(image_A)
        else:
            image_A_gray = image_A
            
        if image_B.size(1) > 1:
            image_B_gray = rgb_to_grayscale(image_B)
        else:
            image_B_gray = image_B
            
        if image_fused.size(1) > 1:
            image_fused_gray = rgb_to_grayscale(image_fused)
        else:
            image_fused_gray = image_fused
        
        gradient_A = self.sobelconv(image_A_gray)
        gradient_B = self.sobelconv(image_B_gray)
        gradient_fused = self.sobelconv(image_fused_gray)
        
        gradient_joint = torch.max(gradient_A, gradient_B)
        Loss_gradient = F.l1_loss(gradient_fused, gradient_joint)
        return Loss_gradient

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0, 0, 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False)
        
    def forward(self, x):
        # 确保权重在与输入相同的设备上
        if x.device != self.weightx.device:
            self.weightx = self.weightx.to(x.device)
            self.weighty = self.weighty.to(x.device)
            
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)

class L_Intensity(nn.Module):
    def __init__(self):
        super(L_Intensity, self).__init__()

    def forward(self, image_A, image_B, image_fused):
        # 统一转换为灰度图计算强度损失
        if image_A.size(1) > 1:
            image_A_gray = rgb_to_grayscale(image_A)
        else:
            image_A_gray = image_A
            
        if image_B.size(1) > 1:
            image_B_gray = rgb_to_grayscale(image_B)
        else:
            image_B_gray = image_B
            
        if image_fused.size(1) > 1:
            image_fused_gray = rgb_to_grayscale(image_fused)
        else:
            image_fused_gray = image_fused
        
        intensity_joint = torch.max(image_A_gray, image_B_gray)
        Loss_intensity = F.l1_loss(image_fused_gray, intensity_joint)
        return Loss_intensity

class fusion_loss_med(nn.Module):
    def __init__(self):
        super(fusion_loss_med, self).__init__()
        self.L_Grad = L_Grad()
        self.L_Inten = L_Intensity()
        self.L_SSIM = L_SSIM()

    def forward(self, image_A, image_B, image_fused):
        # image_A represents MRI image
        loss_l1 = 20 * self.L_Inten(image_A, image_B, image_fused)
        loss_gradient = 100 * self.L_Grad(image_A, image_B, image_fused)
        loss_SSIM = 50 * (1 - self.L_SSIM(image_A, image_B, image_fused))
        fusion_loss = loss_l1 + loss_gradient + loss_SSIM
        return fusion_loss, loss_gradient, loss_l1, loss_SSIM