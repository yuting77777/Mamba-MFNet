import argparse
import numpy as np
from collections import OrderedDict
import os
import torch
import time
import sys
from torchinfo import summary
import math
from scipy.ndimage import sobel
from scipy import ndimage
from scipy.signal import convolve2d
from skimage.metrics import structural_similarity as ssim
import json
from datetime import datetime
import cv2
from scipy.ndimage import gaussian_filter
from PIL import Image
from skimage.measure import shannon_entropy

from models.network import MambaMFNet as net
from utils import utils_image as util
from data.dataloder import Dataset as D
from torch.utils.data import DataLoader
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# 评价指标计算函数
def calculate_en(image):
    """计算熵 (EN)"""
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    hist, _ = np.histogram(image, bins=256, range=(0, 255))
    hist = hist.astype(float)
    hist /= hist.sum()
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))

def calculate_sd(image):
    """计算标准差 (SD)"""
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = image * 255.0
        image = image.astype(np.float32)
    return np.std(image)

def calculate_sf(image):
    """计算空间频率 (SF)"""
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = image * 255.0
        image = image.astype(np.float32)
    
    rf = np.sqrt(np.mean(np.diff(image, axis=0)**2))
    cf = np.sqrt(np.mean(np.diff(image, axis=1)**2))
    return np.sqrt(rf**2 + cf**2)

def ComEntropy(img1, img2):
    """计算联合熵"""
    width = img1.shape[0]
    height = img1.shape[1]
    tmp = np.zeros((256, 256))
    res = 0
    
    for i in range(width):
        for j in range(height):
            val1 = img1[i][j]
            val2 = img2[i][j]
            tmp[val1][val2] = float(tmp[val1][val2] + 1)
    
    tmp = tmp / (width * height)
    
    for i in range(width):
        for j in range(height):
            if tmp[i][j] == 0:
                res = res
            else:
                res = res - tmp[i][j] * (math.log(tmp[i][j]) / math.log(2.0))
    return res

def calculate_mi(img1, img2, fused):
    def calc_mi(x, y):
        # 使用直方图2D计算联合分布
        hist_2d, _, _ = np.histogram2d(x.flatten(), y.flatten(), bins=256)
        p_xy = hist_2d / np.sum(hist_2d)
        p_x = np.sum(p_xy, axis=1)
        p_y = np.sum(p_xy, axis=0)
        
        # 计算互信息
        mi = 0
        for i in range(p_xy.shape[0]):
            for j in range(p_xy.shape[1]):
                if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                    mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))
        return mi
    
    # 归一化到0-255
    img1 = (img1 - img1.min()) / (img1.max() - img1.min() + 1e-8) * 255
    img2 = (img2 - img2.min()) / (img2.max() - img2.min() + 1e-8) * 255
    fused = (fused - fused.min()) / (fused.max() - fused.min() + 1e-8) * 255
    
    mi_af = calc_mi(img1.astype(np.uint8).flatten(), fused.astype(np.uint8).flatten())
    mi_bf = calc_mi(img2.astype(np.uint8).flatten(), fused.astype(np.uint8).flatten())
    
    return mi_af + mi_bf

def calculate_scd(img1, img2, fused):
    """计算差异相关和 (SCD) - 修正负值处理版本"""
    
    def normalized_correlation(x, y):
        """计算归一化相关系数"""
        x = x.astype(np.float64)
        y = y.astype(np.float64)
        
        # 零均值化
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        x_zero_mean = x - x_mean
        y_zero_mean = y - y_mean
        
        numerator = np.sum(x_zero_mean * y_zero_mean)
        denominator = np.sqrt(np.sum(x_zero_mean**2) * np.sum(y_zero_mean**2))
        
        # 添加小值避免除零，并限制在[-1, 1]范围
        corr = numerator / (denominator + 1e-10)
        return np.clip(corr, -1.0, 1.0)
    
    # 统一数据范围和类型
    def normalize_image(img):
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = img * 255.0
        return img.astype(np.float32)
    
    img1_norm = normalize_image(img1)
    img2_norm = normalize_image(img2)
    fused_norm = normalize_image(fused)
    
    # 正确的SCD计算公式
    scd1 = normalized_correlation(img1_norm, fused_norm - img2_norm)
    scd2 = normalized_correlation(img2_norm, fused_norm - img1_norm)
    
    scd_total = scd1 + scd2
    
    # 修改这里：允许返回负值，只保留警告信息
    if scd_total < 0:
        print(f"Warning: Negative SCD detected: {scd_total:.4f} (scd1={scd1:.4f}, scd2={scd2:.4f})")
        # 不再截断为0，而是直接返回负值
        # 这表示融合质量可能有问题
    
    return scd_total  # 直接返回计算值，可以是负数

def calculate_vif(img1, img2, fused):
    """
    使用 sewar 库的标准多尺度 VIF 实现。
    若未安装 sewar，则回退到自定义实现并给出警告。
    """
    try:
        from sewar.full_ref import vifp
        use_sewar = True
    except ImportError:
        print("Warning: sewar not installed. Falling back to custom VIF implementation (may not be standard).")
        use_sewar = False

    if use_sewar:
        # 预处理图像：转换为 float64 并缩放到 [0,1] 范围
        def prepare(img):
            img = img.astype(np.float64)
            if img.max() > 1.0:
                img = img / 255.0
            return np.clip(img, 0, 1)

        img1_prep = prepare(img1)
        img2_prep = prepare(img2)
        fused_prep = prepare(fused)

        # 计算 VIF（两个源图像分别与融合图像比较）
        vif1 = vifp(img1_prep, fused_prep)
        vif2 = vifp(img2_prep, fused_prep)

        # 平均并限制在 [0,1] 范围内
        vif_avg = (vif1 + vif2) / 2.0
        return np.clip(vif_avg, 0.0, 1.0)
    else:
        # 回退到自定义实现（原代码保留为后备）
        return _calculate_vif_fallback(img1, img2, fused)

# 将原 calculate_vif 函数完整移到此处作为后备实现
def _calculate_vif_fallback(img1, img2, fused):
    """
    改进的多尺度VIF实现，特别处理SPECT-MRI等特殊情况
    参考：Sheikh, H. R., & Bovik, A. C. (2006). Image information and visual quality.
    修改：针对SPECT-MRI等模态的特殊处理
    """
    def vifp_mscale(ref, dist, modality='generic'):
        """多尺度VIF计算，添加模态特异性处理"""
        # 模态特定的噪声方差调整
        if modality.lower() == 'spect':
            sigma_nsq = 0.5  # SPECT图像噪声较低，降低噪声方差估计
        else:
            sigma_nsq = 2.0  # 默认值，适用于PET和CT
        
        eps = 1e-10
        
        # 确保图像为float64
        ref = ref.astype(np.float64)
        dist = dist.astype(np.float64)
        
        # 预处理：去除异常值
        def clip_outliers(image, lower_percentile=0.1, upper_percentile=99.9):
            lower = np.percentile(image, lower_percentile)
            upper = np.percentile(image, upper_percentile)
            return np.clip(image, lower, upper)
        
        ref = clip_outliers(ref)
        dist = clip_outliers(dist)
        
        # 4个尺度
        scales = 4
        num = np.zeros(scales)
        den = np.zeros(scales)
        
        for scale in range(scales):
            # 当前尺度的图像
            if scale == 0:
                ref_scaled = ref.copy()
                dist_scaled = dist.copy()
            else:
                # 高斯模糊后下采样
                ref_scaled = cv2.GaussianBlur(ref_scaled, (5, 5), 1.0)
                dist_scaled = cv2.GaussianBlur(dist_scaled, (5, 5), 1.0)
                ref_scaled = ref_scaled[::2, ::2]
                dist_scaled = dist_scaled[::2, ::2]
            
            # 计算局部统计量
            window = cv2.getGaussianKernel(7, 7/6)
            window = window @ window.T
            
            mu1 = cv2.filter2D(ref_scaled, -1, window, borderType=cv2.BORDER_REFLECT)
            mu2 = cv2.filter2D(dist_scaled, -1, window, borderType=cv2.BORDER_REFLECT)
            
            sigma1_sq = cv2.filter2D(ref_scaled*ref_scaled, -1, window, 
                                    borderType=cv2.BORDER_REFLECT) - mu1*mu1
            sigma2_sq = cv2.filter2D(dist_scaled*dist_scaled, -1, window, 
                                    borderType=cv2.BORDER_REFLECT) - mu2*mu2
            sigma12 = cv2.filter2D(ref_scaled*dist_scaled, -1, window, 
                                  borderType=cv2.BORDER_REFLECT) - mu1*mu2
            
            # 防止负值
            sigma1_sq = np.maximum(sigma1_sq, 0)
            sigma2_sq = np.maximum(sigma2_sq, 0)
            
            # 计算增益和噪声方差
            g = sigma12 / (sigma1_sq + eps)
            sigma_v_sq = sigma2_sq - g * sigma12
            sigma_v_sq = np.maximum(sigma_v_sq, 0)
            
            # SPECT特异性处理：限制异常大的增益值
            if modality.lower() == 'spect':
                g = np.clip(g, 0, 5)  # 限制增益最大值
            
            # 计算当前尺度的VIF贡献
            num[scale] = np.sum(np.log10(1 + g**2 * sigma1_sq / (sigma_v_sq + sigma_nsq + eps)))
            den[scale] = np.sum(np.log10(1 + sigma1_sq / (sigma_nsq + eps)))
        
        # 加权平均
        vif = np.sum(num) / (np.sum(den) + eps)
        
        # VIF理论上应在0-1之间，但由于计算误差可能略高于1
        # 对于SPECT，我们更严格地限制范围
        if modality.lower() == 'spect':
            return min(vif, 1.2)  # 更严格的上限
        else:
            return min(vif, 1.5)
    
    # 改进的归一化函数
    def normalize_img(img, modality='generic'):
        img = img.astype(np.float64)
        
        # 检查图像是否已经归一化
        if img.max() > 1.0:
            img = img / 255.0
        
        # 模态特定的归一化
        if modality.lower() == 'spect':
            # SPECT图像通常有较低的背景值和较高的对比度
            # 使用自适应归一化
            non_zero = img[img > 0]
            if len(non_zero) > 0:
                p99 = np.percentile(non_zero, 99)
                if p99 > 0:
                    img = np.clip(img / p99, 0, 1)
        else:
            # 对于PET和CT，使用标准归一化
            img_min = img.min()
            img_max = img.max()
            if img_max > img_min:
                img = (img - img_min) / (img_max - img_min)
        
        return img
    
    # 检测图像类型
    def detect_modality(img1, img2):
        """简单的图像类型检测"""
        # 基于强度分布判断
        img1_flat = img1.flatten()
        img2_flat = img2.flatten()
        
        # 计算统计特征
        img1_nonzero = img1_flat[img1_flat > 0]
        img2_nonzero = img2_flat[img2_flat > 0]
        
        if len(img1_nonzero) > 0 and len(img2_nonzero) > 0:
            # SPECT通常有更高的峰度（更尖锐的分布）
            from scipy.stats import kurtosis
            try:
                k1 = kurtosis(img1_nonzero)
                k2 = kurtosis(img2_nonzero)
                
                # 如果任一图像具有高峰度，可能是SPECT
                if k1 > 10 or k2 > 10:
                    return 'spect'
            except:
                pass
        
        return 'generic'
    
    # 检测模态
    modality = detect_modality(img1, img2)
    
    # 预处理
    img1_norm = normalize_img(img1, modality)
    img2_norm = normalize_img(img2, modality)
    fused_norm = normalize_img(fused, modality)
    
    # 计算VIF，传入模态信息
    vif1 = vifp_mscale(img1_norm, fused_norm, modality)
    vif2 = vifp_mscale(img2_norm, fused_norm, modality)
    
    # 平均VIF值
    vif_avg = (vif1 + vif2) / 2.0
    
    # 最终范围限制
    vif_avg = min(max(vif_avg, 0), 1.0)
    
    return vif_avg


def calculate_qabf(img1, img2, fused):
    """计算融合因子 (Q_abf) - 按照您提供的MATLAB代码转换为Python"""
    # 模型参数
    L = 1
    Tg = 0.9994
    kg = -15
    Dg = 0.5
    Ta = 0.9879
    ka = -22
    Da = 0.8
    
    # Sobel算子
    h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    h2 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])
    h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    
    # 转换为double类型
    def to_double(img):
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = img * 255.0
        return img.astype(np.float64)
    
    pA = to_double(img1)
    pB = to_double(img2)
    pF = to_double(fused)
    
    # 如果是彩色图像，转换为灰度
    if len(pA.shape) > 2 and pA.shape[2] > 1:
        pA = cv2.cvtColor(pA.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)
    if len(pB.shape) > 2 and pB.shape[2] > 1:
        pB = cv2.cvtColor(pB.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)
    if len(pF.shape) > 2 and pF.shape[2] > 1:
        pF = cv2.cvtColor(pF.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)
    
    # 计算梯度幅度和方向
    SAx = convolve2d(pA, h3, mode='same')
    SAy = convolve2d(pA, h1, mode='same')
    gA = np.sqrt(SAx**2 + SAy**2)
    
    M, N = SAx.shape
    aA = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            if SAx[i, j] == 0:
                aA[i, j] = np.pi / 2
            else:
                aA[i, j] = np.arctan(SAy[i, j] / SAx[i, j])
    
    SBx = convolve2d(pB, h3, mode='same')
    SBy = convolve2d(pB, h1, mode='same')
    gB = np.sqrt(SBx**2 + SBy**2)
    
    aB = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            if SBx[i, j] == 0:
                aB[i, j] = np.pi / 2
            else:
                aB[i, j] = np.arctan(SBy[i, j] / SBx[i, j])
    
    SFx = convolve2d(pF, h3, mode='same')
    SFy = convolve2d(pF, h1, mode='same')
    gF = np.sqrt(SFx**2 + SFy**2)
    
    aF = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            if SFx[i, j] == 0:
                aF[i, j] = np.pi / 2
            else:
                aF[i, j] = np.arctan(SFy[i, j] / SFx[i, j])
    
    # 计算QAF
    GAF = np.zeros((M, N))
    AAF = np.zeros((M, N))
    QgAF = np.zeros((M, N))
    QaAF = np.zeros((M, N))
    QAF = np.zeros((M, N))
    
    for i in range(M):
        for j in range(N):
            if gA[i, j] > gF[i, j]:
                GAF[i, j] = gF[i, j] / gA[i, j]
            else:
                if gA[i, j] == gF[i, j]:
                    GAF[i, j] = gF[i, j]
                else:
                    GAF[i, j] = gA[i, j] / gF[i, j]
            
            AAF[i, j] = 1 - abs(aA[i, j] - aF[i, j]) / (np.pi / 2)
            
            QgAF[i, j] = Tg / (1 + np.exp(kg * (GAF[i, j] - Dg)))
            QaAF[i, j] = Ta / (1 + np.exp(ka * (AAF[i, j] - Da)))
            
            QAF[i, j] = QgAF[i, j] * QaAF[i, j]
    
    # 计算QBF
    GBF = np.zeros((M, N))
    ABF = np.zeros((M, N))
    QgBF = np.zeros((M, N))
    QaBF = np.zeros((M, N))
    QBF = np.zeros((M, N))
    
    for i in range(M):
        for j in range(N):
            if gB[i, j] > gF[i, j]:
                GBF[i, j] = gF[i, j] / gB[i, j]
            else:
                if gB[i, j] == gF[i, j]:
                    GBF[i, j] = gF[i, j]
                else:
                    GBF[i, j] = gB[i, j] / gF[i, j]
            
            ABF[i, j] = 1 - abs(aB[i, j] - aF[i, j]) / (np.pi / 2)
            
            QgBF[i, j] = Tg / (1 + np.exp(kg * (GBF[i, j] - Dg)))
            QaBF[i, j] = Ta / (1 + np.exp(ka * (ABF[i, j] - Da)))
            
            QBF[i, j] = QgBF[i, j] * QaBF[i, j]
    
    # 计算最终的Qabf
    deno = np.sum(gA) + np.sum(gB)
    nume = np.sum(QAF * gA) + np.sum(QBF * gB)
    Qabf1 = nume / deno if deno != 0 else 0
    
    return Qabf1

def calculate_ssim_metric(img1, img2, fused):
    """计算结构相似性 (SSIM) - 改进版本"""
    
    def prepare_for_ssim(img):
        """准备图像用于SSIM计算"""
        if len(img.shape) > 2:
            # 如果是彩色图像，转换为灰度
            if img.shape[2] == 3:
                img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
        
        # 确保是二维数组
        img = np.squeeze(img)
        
        # 检查并转换数据类型
        if img.dtype == np.uint8:
            return img.astype(np.float32) / 255.0
        elif img.max() > 1.0:
            return img.astype(np.float32) / 255.0
        else:
            return img.astype(np.float32)
    
    def get_window_size(img_size):
        """根据图像大小确定合适的窗口大小"""
        min_dim = min(img_size)
        if min_dim < 7:
            return min_dim if min_dim % 2 == 1 else min_dim - 1
        else:
            return 7  # skimage默认值
    
    # 准备图像
    img1_prep = prepare_for_ssim(img1)
    img2_prep = prepare_for_ssim(img2)
    fused_prep = prepare_for_ssim(fused)
    
    # 检查图像尺寸
    h1, w1 = img1_prep.shape
    h2, w2 = img2_prep.shape
    hf, wf = fused_prep.shape
    
    if h1 != h2 or w1 != w2 or h1 != hf or w1 != wf:
        # 如果不一致，调整到最小尺寸
        h_min = min(h1, h2, hf)
        w_min = min(w1, w2, wf)
        img1_prep = cv2.resize(img1_prep, (w_min, h_min))
        img2_prep = cv2.resize(img2_prep, (w_min, h_min))
        fused_prep = cv2.resize(fused_prep, (w_min, h_min))
        print(f"Warning: Resized images to {h_min}x{w_min} for SSIM calculation")
    
    # 确定窗口大小
    win_size = get_window_size(img1_prep.shape)
    
    # 计算SSIM - 使用skimage的ssim，但指定合适的数据范围
    try:
        # SSIM between img1 and fused
        ssim1 = ssim(
            img1_prep, 
            fused_prep, 
            data_range=1.0,
            win_size=win_size,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False
        )
        
        # SSIM between img2 and fused
        ssim2 = ssim(
            img2_prep, 
            fused_prep, 
            data_range=1.0,
            win_size=win_size,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False
        )
    except Exception as e:
        print(f"Error computing SSIM: {e}")
        # 如果标准方法失败，使用更简单的设置
        try:
            ssim1 = ssim(img1_prep, fused_prep, data_range=1.0)
            ssim2 = ssim(img2_prep, fused_prep, data_range=1.0)
        except:
            # 如果仍然失败，尝试手动计算
            print("Using manual SSIM fallback")
            ssim1 = compute_simple_ssim(img1_prep, fused_prep)
            ssim2 = compute_simple_ssim(img2_prep, fused_prep)
    
    # 处理NaN值
    if np.isnan(ssim1):
        ssim1 = 0.0
    if np.isnan(ssim2):
        ssim2 = 0.0
    
    # 返回平均值
    return (ssim1 + ssim2) / 2.0

def compute_simple_ssim(img1, img2, K1=0.01, K2=0.03):
    """简单的SSIM计算（备用方法）"""
    C1 = (K1 * 1.0) ** 2
    C2 = (K2 * 1.0) ** 2
    
    # 计算均值
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # 计算方差和协方差
    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
    
    # SSIM公式
    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    
    ssim_map = num / den
    return np.mean(ssim_map)

def calculate_ag(image):
    """计算平均梯度 (AG)"""
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = image * 255.0
        image = image.astype(np.float32)
    
    gy, gx = np.gradient(image.astype(float))
    gradient_magnitude = np.sqrt(gx**2 + gy**2)
    return np.mean(gradient_magnitude)

def calculate_all_metrics(img1, img2, fused):
    """计算所有九个评价指标"""
    metrics = {}
    
    try:
        metrics['EN'] = calculate_en(fused)
        metrics['SD'] = calculate_sd(fused)
        metrics['SF'] = calculate_sf(fused)
        metrics['MI'] = calculate_mi(img1, img2, fused)
        metrics['SCD'] = calculate_scd(img1, img2, fused)
        metrics['VIF'] = calculate_vif(img1, img2, fused)
        metrics['Q_abf'] = calculate_qabf(img1, img2, fused)
        metrics['SSIM'] = calculate_ssim_metric(img1, img2, fused)
        metrics['AG'] = calculate_ag(fused)
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        for key in ['EN', 'SD', 'SF', 'MI', 'SCD', 'VIF', 'Q_abf', 'SSIM', 'AG']:
            metrics[key] = 0.0
    
    return metrics

def save_metrics_to_file(metrics_data, save_dir, filename="evaluation_metrics.txt"):
    """将指标结果保存到文件"""
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    
    with open(filepath, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("Medical Image Fusion Evaluation Metrics\n")
        f.write("=" * 100 + "\n")
        f.write(f"Evaluation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {metrics_data['dataset']}\n")
        f.write(f"Model: {metrics_data['model']}\n")
        f.write(f"Total Images: {len(metrics_data['individual_metrics'])}\n")
        f.write("=" * 100 + "\n\n")
        
        f.write("INDIVIDUAL IMAGE METRICS:\n")
        f.write("-" * 100 + "\n")
        f.write("{:<20} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8}\n".format(
            "Image", "EN", "SD", "SF", "MI", "SCD", "VIF", "Q_abf", "SSIM", "AG"))
        f.write("-" * 100 + "\n")
        
        for img_name, metrics in metrics_data['individual_metrics'].items():
            f.write("{:<20} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f}\n".format(
                img_name, metrics['EN'], metrics['SD'], metrics['SF'], metrics['MI'], 
                metrics['SCD'], metrics['VIF'], metrics['Q_abf'], metrics['SSIM'], metrics['AG']))
        
        f.write("\n" + "=" * 100 + "\n")
        f.write("AVERAGE METRICS:\n")
        f.write("=" * 100 + "\n")
        avg_metrics = metrics_data['average_metrics']
        f.write("EN (Entropy): {:.4f}\n".format(avg_metrics['EN']))
        f.write("SD (Standard Deviation): {:.4f}\n".format(avg_metrics['SD']))
        f.write("SF (Spatial Frequency): {:.4f}\n".format(avg_metrics['SF']))
        f.write("MI (Mutual Information): {:.4f}\n".format(avg_metrics['MI']))
        f.write("SCD (Sum of Correlation Differences): {:.4f}\n".format(avg_metrics['SCD']))
        f.write("VIF (Visual Information Fidelity): {:.4f}\n".format(avg_metrics['VIF']))
        f.write("Q_abf (Fusion Factor): {:.4f}\n".format(avg_metrics['Q_abf']))
        f.write("SSIM (Structural Similarity): {:.4f}\n".format(avg_metrics['SSIM']))
        f.write("AG (Average Gradient): {:.4f}\n".format(avg_metrics['AG']))
        f.write("=" * 100 + "\n")
    
    # json_filepath = os.path.join(save_dir, "evaluation_metrics.json")
    # with open(json_filepath, 'w') as f:
    #     json.dump(metrics_data, f, indent=4)
    
    return filepath

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='fusion', help='classical_sr, lightweight_sr, real_sr, '
                                                                     'gray_dn, color_dn, jpeg_car')
    parser.add_argument('--scale', type=int, default=1, help='scale factor: 1, 2, 3, 4, 8')
    parser.add_argument('--large_model', action='store_true', help='use large model, only provided for real image sr')
    parser.add_argument('--model_path', type=str,
                        default='./Model/Medical_Fusion-PET-CT/Medical_Fusion/models/')
    parser.add_argument('--iter_number', type=str,
                        default='10000')
    parser.add_argument('--data_root', type=str, default='./datasets/',
                        help='root directory of datasets')
    parser.add_argument('--dataset', type=str, default='PET-CT',
                        help='dataset name')
    parser.add_argument('--A_dir', type=str, default='CT',
                        help='directory for modality A')
    parser.add_argument('--B_dir', type=str, default='PET',
                        help='directory for modality B')
    parser.add_argument('--tile', type=int, default=None, help='Tile size, None for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
    parser.add_argument('--in_channel', type=int, default=1, help='3 means color image and 1 means gray image')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 设置模型
    model_path = os.path.join(args.model_path, args.iter_number + '_E.pth')
    if os.path.exists(model_path):
        print(f'loading model from {model_path}')
    else:
        print('Target model path: {} not existing!!!'.format(model_path))
        sys.exit()
        
    model = define_model(args)
    model.eval()
    model = model.to(device)
    
    # 设置window_size
    window_size = 8
    
    # 存储所有图像的指标
    all_metrics = []
    individual_metrics = {}
    
    # 只处理测试集，不处理训练集
    phase = 'testsets'  # 只处理测试集
    
    # 构建完整路径 - 直接读取CT和MRI目录，没有类别子目录
    base_path = os.path.join(args.data_root, phase, args.dataset)
    a_dir = os.path.join(base_path, args.A_dir)
    b_dir = os.path.join(base_path, args.B_dir)
    
    # 检查路径是否存在
    if not os.path.exists(a_dir) or not os.path.exists(b_dir):
        print(f"Error: Test dataset paths don't exist!")
        print(f"A_dir: {a_dir} - exists: {os.path.exists(a_dir)}")
        print(f"B_dir: {b_dir} - exists: {os.path.exists(b_dir)}")
        sys.exit(1)
        
    # 创建保存目录
    save_dir = os.path.join('results', f'{args.dataset}_{phase}')
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建数据集
    test_set = D(a_dir, b_dir, args.in_channel)
    test_loader = DataLoader(test_set, batch_size=1,
                             shuffle=False, num_workers=1,
                             drop_last=False, pin_memory=True)
    
    print(f"Processing test set: {len(test_set)} images found")
    
    for i, test_data in enumerate(test_loader):
        imgname = os.path.basename(test_data['A_path'][0])
        img_a = test_data['A'].to(device)
        img_b = test_data['B'].to(device)
        start = time.time()
        
        # 推理
        with torch.no_grad():
            # 填充输入图像
            _, _, h_old, w_old = img_a.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            
            img_a = torch.cat([img_a, torch.flip(img_a, [2])], 2)[:, :, :h_old + h_pad, :]
            img_a = torch.cat([img_a, torch.flip(img_a, [3])], 3)[:, :, :, :w_old + w_pad]
            img_b = torch.cat([img_b, torch.flip(img_b, [2])], 2)[:, :, :h_old + h_pad, :]
            img_b = torch.cat([img_b, torch.flip(img_b, [3])], 3)[:, :, :, :w_old + w_pad]
            
            output = test(img_a, img_b, model, args, window_size)
            output = output[..., :h_old * args.scale, :w_old * args.scale]
            output = output.detach()[0].float().cpu()
        
        end = time.time()
        output_img = util.tensor2uint(output)
        
        # 保存结果
        save_name = os.path.join(save_dir, imgname)
        util.imsave(output_img, save_name)
        
        # 计算评价指标
        img_a_np = util.tensor2uint(img_a[..., :h_old, :w_old].detach()[0].float().cpu())
        img_b_np = util.tensor2uint(img_b[..., :h_old, :w_old].detach()[0].float().cpu())
        
        # 转换为float32类型用于计算
        img_a_np = img_a_np.astype(np.float32)
        img_b_np = img_b_np.astype(np.float32)
        output_np = output_img.astype(np.float32)
        
        # 计算所有指标
        metrics = calculate_all_metrics(img_a_np, img_b_np, output_np)
        all_metrics.append(metrics)
        individual_metrics[imgname] = metrics
        
        print(f"[Test] {i+1}/{len(test_loader)} Saved {save_name}, Time: {end - start:.4f}s")
        print(f"Metrics for {imgname}: EN={metrics['EN']:.4f}, SD={metrics['SD']:.4f}, SF={metrics['SF']:.4f}, "
              f"MI={metrics['MI']:.4f}, SCD={metrics['SCD']:.4f}, VIF={metrics['VIF']:.4f}, "
              f"Q_abf={metrics['Q_abf']:.4f}, SSIM={metrics['SSIM']:.4f}, AG={metrics['AG']:.4f}")
    
    # 计算平均指标
    if all_metrics:
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        print("\n" + "="*80)
        print("AVERAGE METRICS FOR TEST DATASET: {}".format(args.dataset))
        print("="*80)
        print("EN (Entropy): {:.4f}".format(avg_metrics['EN']))
        print("SD (Standard Deviation): {:.4f}".format(avg_metrics['SD']))
        print("SF (Spatial Frequency): {:.4f}".format(avg_metrics['SF']))
        print("MI (Mutual Information): {:.4f}".format(avg_metrics['MI']))
        print("SCD (Sum of Correlation Differences): {:.4f}".format(avg_metrics['SCD']))
        print("VIF (Visual Information Fidelity): {:.4f}".format(avg_metrics['VIF']))
        print("Q_abf (Fusion Factor): {:.4f}".format(avg_metrics['Q_abf']))
        print("SSIM (Structural Similarity): {:.4f}".format(avg_metrics['SSIM']))
        print("AG (Average Gradient): {:.4f}".format(avg_metrics['AG']))
        print("="*80)
        
        # 保存指标到文件
        metrics_data = {
            'dataset': args.dataset,
            'model': f'MambaDFuse_iter{args.iter_number}',
            'evaluation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'individual_metrics': individual_metrics,
            'average_metrics': avg_metrics
        }
        
        # 保存到results目录
        results_dir = os.path.join('results', args.dataset)
        os.makedirs(results_dir, exist_ok=True)
        
        txt_filepath = save_metrics_to_file(metrics_data, results_dir)
        print(f"\nMetrics saved to:")
        print(f"Text file: {txt_filepath}")
        # print(f"JSON file: {json_filepath}")

def define_model(args):
    model = net(upscale=args.scale, in_chans=args.in_channel, img_size=128, window_size=8,
                img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                mlp_ratio=2, upsampler=None, resi_connection='1conv')
    param_key_g = 'params'
    
    model_path = os.path.join(args.model_path, args.iter_number + '_E.pth')
    pretrained_model = torch.load(model_path)
    
    model.load_state_dict(pretrained_model[param_key_g] 
                         if param_key_g in pretrained_model.keys() 
                         else pretrained_model, strict=True)
    
    return model

def test(img_a, img_b, model, args, window_size):
    if args.tile is None:
        # 整图处理
        output = model(img_a, img_b)
    else:
        # 分块处理
        b, c, h, w = img_a.size()
        tile = min(args.tile, h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"
        tile_overlap = args.tile_overlap
        sf = args.scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*sf, w*sf).type_as(img_a)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_a[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.div_(W)

    return output

if __name__ == '__main__':
    main()
