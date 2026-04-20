import os.path
import random
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util


class Dataset(data.Dataset): 
    """
    # -----------------------------------------
    # Get L/H for denosing on AWGN with fixed sigma.
    # Only dataroot_H is needed.
    # -----------------------------------------
    # e.g., DnCNN
    # -----------------------------------------
    """
    def __init__(self, root_A, root_B, in_channels):
        super(Dataset, self).__init__()
        # ------------------------------------
        # get path of H
        # return None if input is None
        # ------------------------------------
        self.paths_A = self._get_filtered_image_paths(root_A)
        self.paths_B = self._get_filtered_image_paths(root_B)
        self.inchannels = in_channels

    def _get_filtered_image_paths(self, root):
        """获取过滤后的图像路径，排除checkpoint等不需要的文件"""
        all_paths = util.get_image_paths(root)
        # 过滤掉包含checkpoint的文件和其他不需要的文件
        filtered_paths = []
        for path in all_paths:
            filename = os.path.basename(path)
            # 排除包含checkpoint的文件
            if 'checkpoint' in filename.lower():
                continue
            # 排除其他可能的问题文件
            if filename.startswith('.') or filename.startswith('~'):
                continue
            # 确保文件存在且是图像文件
            if os.path.exists(path) and any(path.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']):
                filtered_paths.append(path)
        
        print(f"Filtered {len(all_paths) - len(filtered_paths)} invalid files from {root}")
        return filtered_paths

    def __getitem__(self, index):

        # ------------------------------------
        # get under-exposure image, over-exposure image
        # and norm-exposure image
        # ------------------------------------
        # print('input channels:', self.n_channels)
        A_path = self.paths_A[index]
        B_path = self.paths_B[index]
        img_A = util.imread_uint(A_path, self.inchannels)
        img_B = util.imread_uint(B_path, self.inchannels)
       
        """
        # --------------------------------
        # get testing image pairs
        # --------------------------------
        """
        img_A = util.uint2single(img_A)
        img_B = util.uint2single(img_B)
        # --------------------------------
        # HWC to CHW, numpy to tensor
        # --------------------------------
        img_A = util.single2tensor3(img_A)
        img_B = util.single2tensor3(img_B)

        return {'A': img_A, 'B': img_B, 'A_path': A_path, 'B_path': B_path}

    def __len__(self):
        return len(self.paths_A)