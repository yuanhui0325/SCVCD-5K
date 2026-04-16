##训练
#import os
#import torch
#import logging
#import cv2
#from PIL import Image
#import imageio
#import numpy as np
#import torch.utils.data as data
#from os.path import join, exists
#import math
#import random
#import sys
#import json
#import random
#from subnet.basics import *
#from subnet.ms_ssim_torch import ms_ssim
#from augmentation import random_flip, random_crop_and_pad_image_and_labels
#
#class UVGDataSet(data.Dataset):
#    def __init__(self, root="data/UVG/images/", filelist="data/UVG/originalv.txt", refdir='L12000', testfull=False):
#        with open(filelist) as f:
#            folders = f.readlines()
#        self.ref = []
#        self.refbpp = []
#        self.input = []
#        self.hevcclass = []
#        AllIbpp = self.getbpp(refdir)
#        ii = 0
#        for folder in folders:
#            seq = folder.rstrip()
#            seqIbpp = AllIbpp[ii]
#            imlist = os.listdir(os.path.join(root, seq))
#            cnt = 0
#            for im in imlist:
#                if im[-4:] == '.png':
#                    cnt += 1
#            if testfull:
#                framerange = cnt // 12
#            else:
#                framerange = 1
#            for i in range(framerange):
#                refpath = os.path.join(root, seq, refdir, 'im'+str(i * 12 + 1).zfill(4)+'.png')
#                inputpath = []
#                for j in range(12):
#                    inputpath.append(os.path.join(root, seq, 'im' + str(i * 12 + j + 1).zfill(3)+'.png'))
#                self.ref.append(refpath)
#                self.refbpp.append(seqIbpp)
#                self.input.append(inputpath)
#            ii += 1
#
#
#    def getbpp(self, ref_i_folder):
#        Ibpp = None
#        if ref_i_folder == 'H265L20':
#            print('use H265L20')
#            Ibpp = []# you need to fill bpps after generating crf=20
#        elif ref_i_folder == 'H265L23':
#            print('use H265L23')
#            Ibpp = []# you need to fill bpps after generating crf=23
#        elif ref_i_folder == 'H265L26':
#            print('use H265L26')
#            Ibpp = []# you need to fill bpps after generating crf=26
#        elif ref_i_folder == 'H265L29':
#            print('use H265L29')
#            Ibpp = []# you need to fill bpps after generating crf=29
#        else:
#            print('cannot find ref : ', ref_i_folder)
#            exit()
#        if len(Ibpp) == 0:
#            print('You need to generate I frames and fill the bpps above!')
#            exit()
#        return Ibpp
#
#    
#    def __len__(self):
#        return len(self.ref)
#
#    def __getitem__(self, index):
#        ref_image = imageio.imread(self.ref[index]).transpose(2, 0, 1).astype(np.float32) / 255.0
#        h = (ref_image.shape[1] // 64) * 64
#        w = (ref_image.shape[2] // 64) * 64
#        ref_image = np.array(ref_image[:, :h, :w])
#        input_images = []
#        refpsnr = None
#        refmsssim = None
#        for filename in self.input[index]:
#            input_image = (imageio.imread(filename).transpose(2, 0, 1)[:, :h, :w]).astype(np.float32) / 255.0
#            if refpsnr is None:
#                refpsnr = CalcuPSNR(input_image, ref_image)
#                refmsssim = ms_ssim(torch.from_numpy(input_image[np.newaxis, :]), torch.from_numpy(ref_image[np.newaxis, :]), data_range=1.0).numpy()
#            else:
#                input_images.append(input_image[:, :h, :w])
#
#        input_images = np.array(input_images)
#        return input_images, ref_image, self.refbpp[index], refpsnr, refmsssim
#
##class DataSet(data.Dataset):
##    def __init__(self, path="data/vimeo_septuplet/test.txt", im_height=256, im_width=256):
##        self.image_input_list, self.image_ref_list = self.get_vimeo(filefolderlist=path)
##        self.im_height = im_height
##        self.im_width = im_width
##        
##        self.featurenoise = torch.zeros([out_channel_M, self.im_height // 16, self.im_width // 16])
##        self.znoise = torch.zeros([out_channel_N, self.im_height // 64, self.im_width // 64])
##        self.mvnois = torch.zeros([out_channel_mv, self.im_height // 16, self.im_width // 16])
##        print("dataset find image: ", len(self.image_input_list))
##
##    def get_vimeo(self, rootdir="data/vimeo_septuplet/sequences/", filefolderlist="data/vimeo_septuplet/test.txt"):
##        with open(filefolderlist) as f:
##            data = f.readlines()
##            
##        fns_train_input = []
##        fns_train_ref = []
##
##        for n, line in enumerate(data, 1):
##            y = os.path.join(rootdir, line.rstrip())
##            fns_train_input += [y]
##            refnumber = int(y[-5:-4]) - 2
##            refname = y[0:-5] + str(refnumber) + '.png'
##            fns_train_ref += [refname]
##
##        return fns_train_input, fns_train_ref
##
##    def __len__(self):
##        return len(self.image_input_list)
##
##    def __getitem__(self, index):
##        input_image = imageio.imread(self.image_input_list[index])
##        ref_image = imageio.imread(self.image_ref_list[index])
##
##        input_image = input_image.astype(np.float32) / 255.0
##        ref_image = ref_image.astype(np.float32) / 255.0
##
##        input_image = input_image.transpose(2, 0, 1)
##        ref_image = ref_image.transpose(2, 0, 1)
##        
##        input_image = torch.from_numpy(input_image).float()
##        ref_image = torch.from_numpy(ref_image).float()
##
##        input_image, ref_image = random_crop_and_pad_image_and_labels(input_image, ref_image, [self.im_height, self.im_width])
##        input_image, ref_image = random_flip(input_image, ref_image)
##
##        quant_noise_feature, quant_noise_z, quant_noise_mv = torch.nn.init.uniform_(torch.zeros_like(self.featurenoise), -0.5, 0.5), torch.nn.init.uniform_(torch.zeros_like(self.znoise), -0.5, 0.5), torch.nn.init.uniform_(torch.zeros_like(self.mvnois), -0.5, 0.5)
##        return input_image, ref_image, quant_noise_feature, quant_noise_z, quant_noise_mv
##        
#class DataSet(data.Dataset):
#    def __init__(self, path, im_height=256, im_width=256, augment=True):
#        self.im_height = im_height
#        self.im_width = im_width
#        self.augment = augment
#
#        TRAIN_ROOT = "/data/lichaofei/data/SCVCD-NEW/train/sequences"
#        VAL_ROOT   = "/data/lichaofei/data/SCVCD-NEW/val/sequences"
#        TEST_ROOT  = "/data/lichaofei/data/SCVCD-NEW/test/sequences"
#
#        if "train" in path:
#            self.root = TRAIN_ROOT
#        elif "val" in path:
#            self.root = VAL_ROOT
#        elif "test" in path:
#            self.root = TEST_ROOT
#        else:
#            self.root = TRAIN_ROOT
#
#        self.image_input_list, self.image_ref_list = self.get_vimeo(self.root, path)
#
#        self.featurenoise = torch.zeros([out_channel_M, self.im_height // 16, self.im_width // 16])
#        self.znoise = torch.zeros([out_channel_N, self.im_height // 64, self.im_width // 64])
#        self.mvnois = torch.zeros([out_channel_mv, self.im_height // 16, self.im_width // 16])
#
#        print("Dataset Loaded.")
#        print("  Txt:", path)
#        print("  Root:", self.root)
#        print("  Images:", len(self.image_input_list))
#        print("  Augment:", self.augment)
#
#    def get_vimeo(self, rootdir, filefolderlist):
#        with open(filefolderlist) as f:
#            data = f.readlines()
#
#        fns_input = []
#        fns_ref = []
#
#        for line in data:
#            line = line.strip()
#            if not line:
#                continue
#
#            # 兼容绝对/相对路径
#            if os.path.isabs(line):
#                input_path = line
#            else:
#                input_path = os.path.join(rootdir, line)
#
#            folder_path, file_name = os.path.split(input_path)
#
#            try:
#                frame_num = int(file_name.split('.')[0].replace('im', ''))
#                ref_num = frame_num - 2
#                if ref_num < 1:
#                    continue
#                ref_path = os.path.join(folder_path, f"im{ref_num}.png")
#            except Exception:
#                continue
#
#            fns_input.append(input_path)
#            fns_ref.append(ref_path)
#
#        return fns_input, fns_ref
#
#    def __len__(self):
#        return len(self.image_input_list)
#
#    def __getitem__(self, index):
#        input_image = imageio.imread(self.image_input_list[index]).astype(np.float32) / 255.0
#        ref_image = imageio.imread(self.image_ref_list[index]).astype(np.float32) / 255.0
#
#        input_image = torch.from_numpy(input_image.transpose(2, 0, 1)).float()
#        ref_image = torch.from_numpy(ref_image.transpose(2, 0, 1)).float()
#
#        if self.augment:
#            input_image, ref_image = random_crop_and_pad_image_and_labels(
#                input_image, ref_image, [self.im_height, self.im_width]
#            )
#            input_image, ref_image = random_flip(input_image, ref_image)
#
#            quant_noise_feature = torch.nn.init.uniform_(torch.zeros_like(self.featurenoise), -0.5, 0.5)
#            quant_noise_z = torch.nn.init.uniform_(torch.zeros_like(self.znoise), -0.5, 0.5)
#            quant_noise_mv = torch.nn.init.uniform_(torch.zeros_like(self.mvnois), -0.5, 0.5)
#        else:
#            # 验证：固定裁剪（左上角），避免随机性
#            input_image = input_image[:, :self.im_height, :self.im_width]
#            ref_image = ref_image[:, :self.im_height, :self.im_width]
#
#            # 验证：噪声固定为0
#            quant_noise_feature = torch.zeros_like(self.featurenoise)
#            quant_noise_z = torch.zeros_like(self.znoise)
#            quant_noise_mv = torch.zeros_like(self.mvnois)





##测试
#import os
#import torch
#import logging
#import cv2
#from PIL import Image
#import imageio
#import numpy as np
#import torch.utils.data as data
#from os.path import join, exists
#import math
#import random
#import sys
#import json
#import random
#from subnet.basics import *
#from subnet.ms_ssim_torch import ms_ssim
#from augmentation import random_flip, random_crop_and_pad_image_and_labels
#
#class UVGDataSet(data.Dataset):
#    def __init__(self, root="data/UVG/images/", filelist="data/UVG/originalv.txt", refdir='L12000', testfull=False):
#        with open(filelist) as f:
#            folders = f.readlines()
#        self.ref = []
#        self.refbpp = []
#        self.input = []
#        self.hevcclass = []
#        AllIbpp = self.getbpp(refdir)
#        ii = 0
#        for folder in folders:
#            seq = folder.rstrip()
#            seqIbpp = AllIbpp[ii]
#            imlist = os.listdir(os.path.join(root, seq))
#            cnt = 0
#            for im in imlist:
#                if im[-4:] == '.png':
#                    cnt += 1
#            if testfull:
#                framerange = cnt // 12
#            else:
#                framerange = 1
#            for i in range(framerange):
#                refpath = os.path.join(root, seq, refdir, 'im'+str(i * 12 + 1).zfill(4)+'.png')
#                inputpath = []
#                for j in range(12):
#                    inputpath.append(os.path.join(root, seq, 'im' + str(i * 12 + j + 1).zfill(3)+'.png'))
#                self.ref.append(refpath)
#                self.refbpp.append(seqIbpp)
#                self.input.append(inputpath)
#            ii += 1
#
#
#    def getbpp(self, ref_i_folder):
#        Ibpp = None
#        if ref_i_folder == 'H265L20':
#            print('use H265L20')
#            Ibpp = []# you need to fill bpps after generating crf=20
#        elif ref_i_folder == 'H265L23':
#            print('use H265L23')
#            Ibpp = []# you need to fill bpps after generating crf=23
#        elif ref_i_folder == 'H265L26':
#            print('use H265L26')
#            Ibpp = []# you need to fill bpps after generating crf=26
#        elif ref_i_folder == 'H265L29':
#            print('use H265L29')
#            Ibpp = []# you need to fill bpps after generating crf=29
#        else:
#            print('cannot find ref : ', ref_i_folder)
#            exit()
#        if len(Ibpp) == 0:
#            print('You need to generate I frames and fill the bpps above!')
#            exit()
#        return Ibpp
#
#    
#    def __len__(self):
#        return len(self.ref)
#
#    def __getitem__(self, index):
#        ref_image = imageio.imread(self.ref[index]).transpose(2, 0, 1).astype(np.float32) / 255.0
#        h = (ref_image.shape[1] // 64) * 64
#        w = (ref_image.shape[2] // 64) * 64
#        ref_image = np.array(ref_image[:, :h, :w])
#        input_images = []
#        refpsnr = None
#        refmsssim = None
#        for filename in self.input[index]:
#            input_image = (imageio.imread(filename).transpose(2, 0, 1)[:, :h, :w]).astype(np.float32) / 255.0
#            if refpsnr is None:
#                refpsnr = CalcuPSNR(input_image, ref_image)
#                refmsssim = ms_ssim(torch.from_numpy(input_image[np.newaxis, :]), torch.from_numpy(ref_image[np.newaxis, :]), data_range=1.0).numpy()
#            else:
#                input_images.append(input_image[:, :h, :w])
#
#        input_images = np.array(input_images)
#        return input_images, ref_image, self.refbpp[index], refpsnr, refmsssim
#
#class SCVCD_DVC_Test7(data.Dataset):
#    def __init__(self,
#                 seq_root="/data/lichaofei/data/SCVCD-NEW/test/sequences",
#                 rec_root="/data/lichaofei/data/SCVCD-NEW/test/DVC_rec_I",
#                 filelist="/data/lichaofei/data/SCVCD-NEW/test/DVC_test",
#                 refdir="H265L20"):
#        self.seq_root = seq_root
#        self.rec_root = rec_root
#        self.refdir = refdir
#
#        with open(filelist, "r") as f:
#            self.folders = [x.strip() for x in f.readlines() if x.strip()]
#
#        bpp_json = os.path.join(rec_root, f"bpp_{refdir}.json")  # e.g. bpp_H265L20.json
#        if not os.path.isfile(bpp_json):
#            raise FileNotFoundError(f"Missing {bpp_json}. Run CreateI script first.")
#        with open(bpp_json, "r") as f:
#            self.bpp_map = json.load(f)
#
#    def __len__(self):
#        return len(self.folders)
#
#    def __getitem__(self, index):
#        rel = self.folders[index]
#
#        # reconstructed I
#        ref_path = os.path.join(self.rec_root, rel, self.refdir, "im0001.png")
#        if not os.path.isfile(ref_path):
#            raise FileNotFoundError(ref_path)
#
#        if rel not in self.bpp_map:
#            raise KeyError(f"{rel} not found in bpp json {self.refdir}")
#        ref_bpp = float(self.bpp_map[rel]["bpp"])
#
#        ref_image = imageio.imread(ref_path).transpose(2, 0, 1).astype(np.float32) / 255.0
#        h = (ref_image.shape[1] // 64) * 64
#        w = (ref_image.shape[2] // 64) * 64
#        ref_image = np.array(ref_image[:, :h, :w])
#
#        input_images = []
#        refpsnr = None
#        refmsssim = None
#
#        # original frames im1..im7
#        for i in range(1, 8):
#            img_path = os.path.join(self.seq_root, rel, f"im{i}.png")
#            if not os.path.isfile(img_path):
#                raise FileNotFoundError(img_path)
#
#            input_image = (imageio.imread(img_path).transpose(2, 0, 1)[:, :h, :w]).astype(np.float32) / 255.0
#            if refpsnr is None:
#                refpsnr = CalcuPSNR(input_image, ref_image)
#                refmsssim = ms_ssim(torch.from_numpy(input_image[np.newaxis, :]),
#                                    torch.from_numpy(ref_image[np.newaxis, :]),
#                                    data_range=1.0).numpy()
#            else:
#                input_images.append(input_image)
#
#        input_images = np.array(input_images)  # (6,3,h,w)
#        return input_images, ref_image, ref_bpp, refpsnr, refmsssim
#
##class DataSet(data.Dataset):
##    def __init__(self, path="data/vimeo_septuplet/test.txt", im_height=256, im_width=256):
##        self.image_input_list, self.image_ref_list = self.get_vimeo(filefolderlist=path)
##        self.im_height = im_height
##        self.im_width = im_width
##        
##        self.featurenoise = torch.zeros([out_channel_M, self.im_height // 16, self.im_width // 16])
##        self.znoise = torch.zeros([out_channel_N, self.im_height // 64, self.im_width // 64])
##        self.mvnois = torch.zeros([out_channel_mv, self.im_height // 16, self.im_width // 16])
##        print("dataset find image: ", len(self.image_input_list))
##
##    def get_vimeo(self, rootdir="data/vimeo_septuplet/sequences/", filefolderlist="data/vimeo_septuplet/test.txt"):
##        with open(filefolderlist) as f:
##            data = f.readlines()
##            
##        fns_train_input = []
##        fns_train_ref = []
##
##        for n, line in enumerate(data, 1):
##            y = os.path.join(rootdir, line.rstrip())
##            fns_train_input += [y]
##            refnumber = int(y[-5:-4]) - 2
##            refname = y[0:-5] + str(refnumber) + '.png'
##            fns_train_ref += [refname]
##
##        return fns_train_input, fns_train_ref
##
##    def __len__(self):
##        return len(self.image_input_list)
##
##    def __getitem__(self, index):
##        input_image = imageio.imread(self.image_input_list[index])
##        ref_image = imageio.imread(self.image_ref_list[index])
##
##        input_image = input_image.astype(np.float32) / 255.0
##        ref_image = ref_image.astype(np.float32) / 255.0
##
##        input_image = input_image.transpose(2, 0, 1)
##        ref_image = ref_image.transpose(2, 0, 1)
##        
##        input_image = torch.from_numpy(input_image).float()
##        ref_image = torch.from_numpy(ref_image).float()
##
##        input_image, ref_image = random_crop_and_pad_image_and_labels(input_image, ref_image, [self.im_height, self.im_width])
##        input_image, ref_image = random_flip(input_image, ref_image)
##
##        quant_noise_feature, quant_noise_z, quant_noise_mv = torch.nn.init.uniform_(torch.zeros_like(self.featurenoise), -0.5, 0.5), torch.nn.init.uniform_(torch.zeros_like(self.znoise), -0.5, 0.5), torch.nn.init.uniform_(torch.zeros_like(self.mvnois), -0.5, 0.5)
##        return input_image, ref_image, quant_noise_feature, quant_noise_z, quant_noise_mv
##        
#class DataSet(data.Dataset):
#    def __init__(self, path="data/vimeo_septuplet/test.txt", im_height=256, im_width=256):
#        self.im_height = im_height
#        self.im_width = im_width
#        
#        # ====================================================
#        # [修改 1] 定义你的数据绝对路径
#        # ====================================================
#        TRAIN_ROOT = "/data/lichaofei/data/SCVCD-NEW/train/sequences"
#        VAL_ROOT   = "/data/lichaofei/data/SCVCD-NEW/val/sequences"
#        TEST_ROOT  = "/data/lichaofei/data/SCVCD-NEW/test/sequences"
#        
#        # ====================================================
#        # [修改 2] 根据传入的 txt 路径，自动判断使用哪个 Root
#        # ====================================================
#        if "train" in path:
#            self.root = TRAIN_ROOT
#        elif "val" in path:
#            self.root = VAL_ROOT
#        elif "test" in path:
#            self.root = TEST_ROOT
#        else:
#            # 默认 fallback，防止报错
#            print(f"[Warning] Unknown dataset type in {path}, defaulting to TRAIN_ROOT")
#            self.root = TRAIN_ROOT
#
#        # 调用 get_vimeo 时传入确定的 root
#        self.image_input_list, self.image_ref_list = self.get_vimeo(self.root, path)
#        
#        # 初始化噪声 (保持原样)
#        self.featurenoise = torch.zeros([out_channel_M, self.im_height // 16, self.im_width // 16])
#        self.znoise = torch.zeros([out_channel_N, self.im_height // 64, self.im_width // 64])
#        self.mvnois = torch.zeros([out_channel_mv, self.im_height // 16, self.im_width // 16])
#        print(f"Dataset Loaded.\n  Txt: {path}\n  Root: {self.root}\n  Images: {len(self.image_input_list)}")
#
#    def get_vimeo(self, rootdir, filefolderlist):
#        with open(filefolderlist) as f:
#            data = f.readlines()
#            
#        fns_train_input = []
#        fns_train_ref = []
#
#        for line in data:
#            line = line.strip()
#            if not line: continue
#            
#            # ====================================================
#            # [修改 3] 拼接绝对路径
#            # ====================================================
#            # line 是 00001/0001/im3.png -> 拼接成完整路径
#            input_path = os.path.join(rootdir, line)
#            fns_train_input.append(input_path)
#
#            # ====================================================
#            # [修改 4] 更稳健的参考帧查找逻辑
#            # 原代码: int(y[-5:-4]) - 2 容易在长路径下出错
#            # 新逻辑: 解析文件名 im3.png -> 3 -> 1 -> im1.png
#            # ====================================================
#            folder_path, file_name = os.path.split(input_path) # file_name: im3.png
#            
#            try:
#                # 去掉 'im' 和 '.png'，拿到数字 3
#                frame_num = int(file_name.split('.')[0].replace('im', ''))
#                
#                # 计算参考帧序号: 3 - 2 = 1
#                ref_num = frame_num - 2
#                
#                # 拼回 im1.png
#                ref_name = f"im{ref_num}.png"
#                ref_path = os.path.join(folder_path, ref_name)
#                
#                fns_train_ref.append(ref_path)
#            except Exception as e:
#                print(f"[Error] 解析文件名失败: {input_path}, 错误: {e}")
#                # 如果解析失败，为了不让程序崩，可以移除刚加入的 input，或者报错退出
#                # 这里简单处理：移除最后一个 input 保持对齐
#                fns_train_input.pop()
#
#        return fns_train_input, fns_train_ref
#
#    def __len__(self):
#        return len(self.image_input_list)
#
#    def __getitem__(self, index):
#        # 读取图片
#        input_image = imageio.imread(self.image_input_list[index])
#        ref_image = imageio.imread(self.image_ref_list[index])
#
#        input_image = input_image.astype(np.float32) / 255.0
#        ref_image = ref_image.astype(np.float32) / 255.0
#
#        input_image = input_image.transpose(2, 0, 1)
#        ref_image = ref_image.transpose(2, 0, 1)
#        
#        input_image = torch.from_numpy(input_image).float()
#        ref_image = torch.from_numpy(ref_image).float()
#
#        input_image, ref_image = random_crop_and_pad_image_and_labels(input_image, ref_image, [self.im_height, self.im_width])
#        input_image, ref_image = random_flip(input_image, ref_image)
#
#        quant_noise_feature, quant_noise_z, quant_noise_mv = torch.nn.init.uniform_(torch.zeros_like(self.featurenoise), -0.5, 0.5), torch.nn.init.uniform_(torch.zeros_like(self.znoise), -0.5, 0.5), torch.nn.init.uniform_(torch.zeros_like(self.mvnois), -0.5, 0.5)
#        return input_image, ref_image, quant_noise_feature, quant_noise_z, quant_noise_mv





#测试
import os
import torch
import logging
import cv2
from PIL import Image
import imageio
import numpy as np
import torch.utils.data as data
from os.path import join, exists
import math
import random
import sys
import json
import random
from subnet.basics import *
from subnet.ms_ssim_torch import ms_ssim
from augmentation import random_flip, random_crop_and_pad_image_and_labels

class UVGDataSet(data.Dataset):
    def __init__(self, root="data/UVG/images/", filelist="data/UVG/originalv.txt", refdir='L12000', testfull=False):
        with open(filelist) as f:
            folders = f.readlines()
        self.ref = []
        self.refbpp = []
        self.input = []
        self.hevcclass = []
        AllIbpp = self.getbpp(refdir)
        ii = 0
        for folder in folders:
            seq = folder.rstrip()
            seqIbpp = AllIbpp[ii]
            imlist = os.listdir(os.path.join(root, seq))
            cnt = 0
            for im in imlist:
                if im[-4:] == '.png':
                    cnt += 1
            if testfull:
                framerange = cnt // 12
            else:
                framerange = 1
            for i in range(framerange):
                refpath = os.path.join(root, seq, refdir, 'im'+str(i * 12 + 1).zfill(4)+'.png')
                inputpath = []
                for j in range(12):
                    inputpath.append(os.path.join(root, seq, 'im' + str(i * 12 + j + 1).zfill(3)+'.png'))
                self.ref.append(refpath)
                self.refbpp.append(seqIbpp)
                self.input.append(inputpath)
            ii += 1


    def getbpp(self, ref_i_folder):
        Ibpp = None
        if ref_i_folder == 'H265L20':
            print('use H265L20')
            Ibpp = []# you need to fill bpps after generating crf=20
        elif ref_i_folder == 'H265L23':
            print('use H265L23')
            Ibpp = []# you need to fill bpps after generating crf=23
        elif ref_i_folder == 'H265L26':
            print('use H265L26')
            Ibpp = []# you need to fill bpps after generating crf=26
        elif ref_i_folder == 'H265L29':
            print('use H265L29')
            Ibpp = []# you need to fill bpps after generating crf=29
        else:
            print('cannot find ref : ', ref_i_folder)
            exit()
        if len(Ibpp) == 0:
            print('You need to generate I frames and fill the bpps above!')
            exit()
        return Ibpp

    
    def __len__(self):
        return len(self.ref)

    def __getitem__(self, index):
        rel = self.folders[index]

        if self.oracle_i:
            # Oracle I: 直接用原始 im1.png 当参考帧
            ref_path = os.path.join(self.seq_root, rel, "im1.png")
            ref_bpp = np.float32(0.0)
        else:
            # 正常：用 H.265 重建 I
            ref_path = os.path.join(self.rec_root, rel, self.refdir, "im0001.png")
            if rel not in self.bpp_map:
                raise KeyError(f"{rel} not found in bpp json {self.refdir}")
            ref_bpp = np.float32(self.bpp_map[rel]["bpp"])

        if not os.path.isfile(ref_path):
            raise FileNotFoundError(ref_path)

        ref_image = imageio.imread(ref_path).transpose(2, 0, 1).astype(np.float32) / 255.0
        h = (ref_image.shape[1] // 64) * 64
        w = (ref_image.shape[2] // 64) * 64
        ref_image = np.array(ref_image[:, :h, :w])
        input_images = []
        refpsnr = None
        refmsssim = None
        for filename in self.input[index]:
            input_image = (imageio.imread(filename).transpose(2, 0, 1)[:, :h, :w]).astype(np.float32) / 255.0
            if refpsnr is None:
                refpsnr = CalcuPSNR(input_image, ref_image)
                refmsssim = ms_ssim(torch.from_numpy(input_image[np.newaxis, :]), torch.from_numpy(ref_image[np.newaxis, :]), data_range=1.0).numpy()
            else:
                input_images.append(input_image[:, :h, :w])

        input_images = np.array(input_images)
        return input_images, ref_image, self.refbpp[index], refpsnr, refmsssim
class SCVCD_DVC_Test7(data.Dataset):
    def __init__(self,
                 seq_root="/data/lichaofei/data/SCVCD-NEW/test/sequences",
                 rec_root="/data/lichaofei/data/SCVCD-NEW/test/DVC_rec_I",
                 filelist="/data/lichaofei/data/SCVCD-NEW/test/DVC_test",
                 refdir="H265L20",
                 oracle_i=False):
        self.seq_root = seq_root
        self.rec_root = rec_root
        self.filelist = filelist
        self.refdir = refdir
        self.oracle_i = oracle_i

        with open(filelist, "r") as f:
            self.folders = [x.strip() for x in f.readlines() if x.strip()]

        if not self.oracle_i:
            bpp_json = os.path.join(rec_root, f"bpp_{refdir}.json")  # e.g. bpp_H265L20.json
            if not os.path.isfile(bpp_json):
                raise FileNotFoundError(f"Missing {bpp_json}. Run CreateI first.")
            with open(bpp_json, "r") as f:
                self.bpp_map = json.load(f)
        else:
            self.bpp_map = None

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, index):
        rel = self.folders[index]

        # 1) reference frame
        if self.oracle_i:
            ref_path = os.path.join(self.seq_root, rel, "im1.png")
            ref_bpp = np.float32(0.0)
        else:
            ref_path = os.path.join(self.rec_root, rel, self.refdir, "im0001.png")
            if rel not in self.bpp_map:
                raise KeyError(f"{rel} not found in bpp json {self.refdir}")
            ref_bpp = np.float32(self.bpp_map[rel]["bpp"])

        if not os.path.isfile(ref_path):
            raise FileNotFoundError(ref_path)

        ref_image = imageio.imread(ref_path).transpose(2, 0, 1).astype(np.float32) / 255.0

        # crop to multiple of 64
        h = (ref_image.shape[1] // 64) * 64
        w = (ref_image.shape[2] // 64) * 64
        ref_image = np.array(ref_image[:, :h, :w])

        # 2) input frames (im1..im7); return frames2..7 as P-frames
        input_images = []
        refpsnr = None
        refmsssim = None

        for i in range(1, 8):
            img_path = os.path.join(self.seq_root, rel, f"im{i}.png")
            if not os.path.isfile(img_path):
                raise FileNotFoundError(img_path)

            input_image = (imageio.imread(img_path).transpose(2, 0, 1)[:, :h, :w]).astype(np.float32) / 255.0

            if refpsnr is None:
                if self.oracle_i:
                    # Oracle I: ref == im1，所以 PSNR 理论上无穷大，这里给个大值即可
                    refpsnr = np.float32(100.0)
                    refmsssim = np.float32(1.0)
                else:
#                    refpsnr = CalcuPSNR(input_image, ref_image)
#                    refmsssim = ms_ssim(torch.from_numpy(input_image[np.newaxis, :]),
#                                        torch.from_numpy(ref_image[np.newaxis, :]),
#                                        data_range=1.0).numpy()
                    # input_image/ref_image: numpy (3,H,W), float32 in [0,1]
                    y_org = (0.299 * input_image[0] + 0.587 * input_image[1] + 0.114 * input_image[2]).astype(np.float32)
                    y_ref = (0.299 * ref_image[0]   + 0.587 * ref_image[1]   + 0.114 * ref_image[2]).astype(np.float32)

                    mse = float(np.mean((y_org - y_ref) ** 2))
                    refpsnr = np.float32(10.0 * np.log10(1.0 / max(mse, 1e-10)))

                    y_org_t = torch.from_numpy(y_org)[None, None, :, :]  # 1x1xHxW
                    y_ref_t = torch.from_numpy(y_ref)[None, None, :, :]
                    refmsssim = ms_ssim(y_org_t, y_ref_t, data_range=1.0).numpy()
            else:
                input_images.append(input_image)

        input_images = np.array(input_images)  # (6,3,h,w)
        return input_images, ref_image, ref_bpp, refpsnr, refmsssim

#class DataSet(data.Dataset):
#    def __init__(self, path="data/vimeo_septuplet/test.txt", im_height=256, im_width=256):
#        self.image_input_list, self.image_ref_list = self.get_vimeo(filefolderlist=path)
#        self.im_height = im_height
#        self.im_width = im_width
#        
#        self.featurenoise = torch.zeros([out_channel_M, self.im_height // 16, self.im_width // 16])
#        self.znoise = torch.zeros([out_channel_N, self.im_height // 64, self.im_width // 64])
#        self.mvnois = torch.zeros([out_channel_mv, self.im_height // 16, self.im_width // 16])
#        print("dataset find image: ", len(self.image_input_list))
#
#    def get_vimeo(self, rootdir="data/vimeo_septuplet/sequences/", filefolderlist="data/vimeo_septuplet/test.txt"):
#        with open(filefolderlist) as f:
#            data = f.readlines()
#            
#        fns_train_input = []
#        fns_train_ref = []
#
#        for n, line in enumerate(data, 1):
#            y = os.path.join(rootdir, line.rstrip())
#            fns_train_input += [y]
#            refnumber = int(y[-5:-4]) - 2
#            refname = y[0:-5] + str(refnumber) + '.png'
#            fns_train_ref += [refname]
#
#        return fns_train_input, fns_train_ref
#
#    def __len__(self):
#        return len(self.image_input_list)
#
#    def __getitem__(self, index):
#        input_image = imageio.imread(self.image_input_list[index])
#        ref_image = imageio.imread(self.image_ref_list[index])
#
#        input_image = input_image.astype(np.float32) / 255.0
#        ref_image = ref_image.astype(np.float32) / 255.0
#
#        input_image = input_image.transpose(2, 0, 1)
#        ref_image = ref_image.transpose(2, 0, 1)
#        
#        input_image = torch.from_numpy(input_image).float()
#        ref_image = torch.from_numpy(ref_image).float()
#
#        input_image, ref_image = random_crop_and_pad_image_and_labels(input_image, ref_image, [self.im_height, self.im_width])
#        input_image, ref_image = random_flip(input_image, ref_image)
#
#        quant_noise_feature, quant_noise_z, quant_noise_mv = torch.nn.init.uniform_(torch.zeros_like(self.featurenoise), -0.5, 0.5), torch.nn.init.uniform_(torch.zeros_like(self.znoise), -0.5, 0.5), torch.nn.init.uniform_(torch.zeros_like(self.mvnois), -0.5, 0.5)
#        return input_image, ref_image, quant_noise_feature, quant_noise_z, quant_noise_mv
#        
class DataSet(data.Dataset):
    def __init__(self, path="data/vimeo_septuplet/test.txt", im_height=256, im_width=256):
        self.im_height = im_height
        self.im_width = im_width
        
        # ====================================================
        # [修改 1] 定义你的数据绝对路径
        # ====================================================
        TRAIN_ROOT = "/data/lichaofei/data/SCVCD-NEW/train/sequences"
        VAL_ROOT   = "/data/lichaofei/data/SCVCD-NEW/val/sequences"
        TEST_ROOT  = "/data/lichaofei/data/SCVCD-NEW/test/sequences"
        
        # ====================================================
        # [修改 2] 根据传入的 txt 路径，自动判断使用哪个 Root
        # ====================================================
        if "train" in path:
            self.root = TRAIN_ROOT
        elif "val" in path:
            self.root = VAL_ROOT
        elif "test" in path:
            self.root = TEST_ROOT
        else:
            # 默认 fallback，防止报错
            print(f"[Warning] Unknown dataset type in {path}, defaulting to TRAIN_ROOT")
            self.root = TRAIN_ROOT

        # 调用 get_vimeo 时传入确定的 root
        self.image_input_list, self.image_ref_list = self.get_vimeo(self.root, path)
        
        # 初始化噪声 (保持原样)
        self.featurenoise = torch.zeros([out_channel_M, self.im_height // 16, self.im_width // 16])
        self.znoise = torch.zeros([out_channel_N, self.im_height // 64, self.im_width // 64])
        self.mvnois = torch.zeros([out_channel_mv, self.im_height // 16, self.im_width // 16])
        print(f"Dataset Loaded.\n  Txt: {path}\n  Root: {self.root}\n  Images: {len(self.image_input_list)}")

    def get_vimeo(self, rootdir, filefolderlist):
        with open(filefolderlist) as f:
            data = f.readlines()
            
        fns_train_input = []
        fns_train_ref = []

        for line in data:
            line = line.strip()
            if not line: continue
            
            # ====================================================
            # [修改 3] 拼接绝对路径
            # ====================================================
            # line 是 00001/0001/im3.png -> 拼接成完整路径
            input_path = os.path.join(rootdir, line)
            fns_train_input.append(input_path)

            # ====================================================
            # [修改 4] 更稳健的参考帧查找逻辑
            # 原代码: int(y[-5:-4]) - 2 容易在长路径下出错
            # 新逻辑: 解析文件名 im3.png -> 3 -> 1 -> im1.png
            # ====================================================
            folder_path, file_name = os.path.split(input_path) # file_name: im3.png
            
            try:
                # 去掉 'im' 和 '.png'，拿到数字 3
                frame_num = int(file_name.split('.')[0].replace('im', ''))
                
                # 计算参考帧序号: 3 - 2 = 1
                ref_num = frame_num - 2
                
                # 拼回 im1.png
                ref_name = f"im{ref_num}.png"
                ref_path = os.path.join(folder_path, ref_name)
                
                fns_train_ref.append(ref_path)
            except Exception as e:
                print(f"[Error] 解析文件名失败: {input_path}, 错误: {e}")
                # 如果解析失败，为了不让程序崩，可以移除刚加入的 input，或者报错退出
                # 这里简单处理：移除最后一个 input 保持对齐
                fns_train_input.pop()

        return fns_train_input, fns_train_ref

    def __len__(self):
        return len(self.image_input_list)

    def __getitem__(self, index):
        # 读取图片
        input_image = imageio.imread(self.image_input_list[index])
        ref_image = imageio.imread(self.image_ref_list[index])

        input_image = input_image.astype(np.float32) / 255.0
        ref_image = ref_image.astype(np.float32) / 255.0

        input_image = input_image.transpose(2, 0, 1)
        ref_image = ref_image.transpose(2, 0, 1)
        
        input_image = torch.from_numpy(input_image).float()
        ref_image = torch.from_numpy(ref_image).float()

        input_image, ref_image = random_crop_and_pad_image_and_labels(input_image, ref_image, [self.im_height, self.im_width])
        input_image, ref_image = random_flip(input_image, ref_image)

        quant_noise_feature, quant_noise_z, quant_noise_mv = torch.nn.init.uniform_(torch.zeros_like(self.featurenoise), -0.5, 0.5), torch.nn.init.uniform_(torch.zeros_like(self.znoise), -0.5, 0.5), torch.nn.init.uniform_(torch.zeros_like(self.mvnois), -0.5, 0.5)
        return input_image, ref_image, quant_noise_feature, quant_noise_z, quant_noise_mv