import os
import cv2
import math
import copy
import random
import argparse
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.spatial import distance
from skimage.feature import canny
from sklearn.metrics import accuracy_score
from skimage.measure import compare_ssim, compare_psnr
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.loss import GANLoss, AdversarialLoss, PerceptualLoss, StyleLoss
from utils.RTV import RTV

gpu_id = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
device = 'cuda:'+gpu_id if torch.cuda.is_available() else 'cpu'
sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.03)
structure = 'LBP'


class SIFTDataset(Dataset):
    def __init__(self, num, file, choice='SLI'):
        self.num = num
        self.file = file
        self.filelist = sorted(os.listdir(self.file))
        self.structure = structure
        self.choice = choice

    def __getitem__(self, idx):
        return self.load_item(idx)

    def __len__(self):
        return self.num

    def load_item(self, idx):
        Ig = cv2.imread(self.file + self.filelist[idx])
        Ig = cv2.resize(Ig, (256, 256))

        Si = np.zeros([256, 256, 128], dtype=float)
        if self.choice == 'SLI':
            Si = self.load_sift(Ig)
        elif self.choice == 'SLI-R':
            Si = self.load_sift_ref_1(Ig)
        elif self.choice == 'SLI-L':
            Si = self.load_sift_landmark(Ig, self.filelist[idx])
        elif self.choice == 'SLI-B':
            Si = self.load_sift_only_location(Ig)

        rtn = np.zeros((256, 256, 1))
        if self.structure == 'LBP':
            Lg = self.load_lbp(Ig)
            Lg = Lg.astype('float') / 127.5 - 1.
            Lg = np.reshape(Lg, (256, 256, 1))
            rtn = Lg
        elif self.structure == 'RTV':
            Rg = RTV(Ig)
            Rg = Rg.astype('float') * 2. - 1.
            rtn = Rg
        elif self.structure == 'Edge':
            Eg = self.load_edge(Ig).reshape((256, 256, 1))
            Eg = Eg * 2. - 1.
            rtn = Eg

        Ig = Ig.astype('float') / 127.5 - 1.
        return self.tensor(Ig), self.tensor(Si), self.tensor(rtn), self.filelist[idx]

    def load_sift(self, img):
        size = 256
        fealen = 128
        feature = np.zeros([size, size, fealen], dtype=float)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        meta, des = sift.detectAndCompute(np.uint8(img), None)
        if len(meta) == 0:
            return feature
        des = des.astype('float') / 127.5 - 1.
        used = []
        for i in range(len(meta)):
            a = int(math.ceil(meta[i].pt[1]) - 1)
            b = int(math.ceil(meta[i].pt[0])) - 1
            fea = list(des[i])
            if self.isEmpty(feature[a][b][:128]):
                feature[a][b][:128] = fea
                used.append(i)
        if True:  # Reduce collisions, can be skip
            meta = np.delete(meta, used)
            for i in range(len(meta)):
                a = int(math.ceil(meta[i].pt[1]) - 1)
                b = int(math.ceil(meta[i].pt[0])) - 1
                ra, rb = self.search_ab(feature, a, b, size)
                if ra == -1:
                    continue
                feature[ra][rb][:128] = list(des[i])
        return feature

    def load_sift_ref_1(self, img):
        ref = cv2.imread('data/celebahq_ref/ref.png')
        ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        ref_meta, ref_des = sift.detectAndCompute(np.uint8(ref), None)
        img = cv2.cvtColor(copy.deepcopy(img), cv2.COLOR_BGR2GRAY)
        meta, des = sift.detectAndCompute(np.uint8(img), None)
        feature = np.zeros([256, 256, 128], dtype=float)
        for i in range(0, len(meta)):
            distances = distance.cdist([des[i]], ref_des, "cosine")[0]
            cos, idx = torch.topk(torch.from_numpy(distances), 2, dim=0, largest=False)
            a = int(math.ceil(ref_meta[idx[0]].pt[1]) - 1)
            b = int(math.ceil(ref_meta[idx[0]].pt[0])) - 1
            fea = list(des[i])
            feature[a][b][:128] = fea
        return feature

    def load_sift_landmark(self, img, fname):
        img = cv2.cvtColor(copy.deepcopy(img), cv2.COLOR_BGR2GRAY)
        meta, des = sift.detectAndCompute(np.uint8(img), None)
        feature = np.zeros([256, 256, 128], dtype=float)
        name = 'train' if 'train' in self.file else 'test'
        pred_classified = np.load('data/celebahq_' + name + '_classified_predict/' + fname[:-4] + '.npy')
        JAW_POINTS = list(range(0, 17))
        RIGHT_BROW_POINTS = list(range(17, 22))
        LEFT_BROW_POINTS = list(range(22, 27))
        NOSE_POINTS = list(range(27, 35))
        RIGHT_EYE_POINTS = list(range(36, 42))
        LEFT_EYE_POINTS = list(range(42, 48))
        MOUTH_POINTS = list(range(48, 68))
        LAND_POINTS = [JAW_POINTS, RIGHT_BROW_POINTS, LEFT_BROW_POINTS, NOSE_POINTS, RIGHT_EYE_POINTS, LEFT_EYE_POINTS,
                       MOUTH_POINTS]
        LANDMARKS = [[68, 117], [70, 135], [73, 153], [77, 171], [81, 189], [90, 205], [101, 218], [113, 230],
                     [128, 234], [144, 231], [160, 221], [174, 209], [186, 193], [193, 176], [197, 156], [201, 136],
                     [204, 116], [74, 104], [82, 100], [92, 101], [101, 105], [111, 110], [142, 108], [153, 103],
                     [164, 101], [176, 101], [186, 106], [125, 124], [124, 136], [124, 149], [123, 162], [114, 168],
                     [118, 171], [124, 173], [131, 170], [137, 168], [86, 123], [93, 118], [103, 119], [111, 126],
                     [101, 128], [92, 127], [147, 126], [154, 119], [164, 119], [172, 124], [164, 128], [155, 128],
                     [103, 188], [111, 184], [119, 183], [126, 184], [133, 182], [143, 184], [154, 186], [144, 199],
                     [134, 205], [127, 206], [119, 205], [111, 201], [108, 190], [119, 187], [126, 187], [133, 187],
                     [150, 188], [134, 196], [126, 197], [119, 196]]
        for i in range(len(meta)):
            pred = pred_classified[i]
            if pred < 7:
                [x, y] = LANDMARKS[random.sample(LAND_POINTS[pred], 1)[0]]
                x += random.randint(-3, 3)
                y += random.randint(-3, 3)
                feature[y, x, :] = des[i]
        return feature

    def load_sift_only_location(self, img):
        feature = np.zeros([256, 256, 1], dtype=float)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        meta, des = sift.detectAndCompute(np.uint8(img), None)
        if len(meta) == 0:
            return feature
        for i in range(len(meta)):
            a = int(math.ceil(meta[i].pt[1]) - 1)
            b = int(math.ceil(meta[i].pt[0])) - 1
            feature[a][b][0] = 1
        return feature

    def load_lbp(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lbp = np.zeros((256, 256))
        for i in range(256):
            for j in range(256):
                lbp[i][j] = self.lbp_calculated(img, i, j)
        return lbp

    def lbp_calculated(self, img, x, y):
        def get_pixel(img, center, x, y):
            new_value = 0
            try:
                if img[x][y] >= center:
                    new_value = 1
            except:
                pass
            return new_value
        center = img[x][y]
        val_ar = []
        val_ar.append(get_pixel(img, center, x - 1, y + 1))  # top_right
        val_ar.append(get_pixel(img, center, x, y + 1))  # right
        val_ar.append(get_pixel(img, center, x + 1, y + 1))  # bottom_right
        val_ar.append(get_pixel(img, center, x + 1, y))  # bottom
        val_ar.append(get_pixel(img, center, x + 1, y - 1))  # bottom_left
        val_ar.append(get_pixel(img, center, x, y - 1))  # left
        val_ar.append(get_pixel(img, center, x - 1, y - 1))  # top_left
        val_ar.append(get_pixel(img, center, x - 1, y))  # top

        power_val = [1, 2, 4, 8, 16, 32, 64, 128]
        val = 0
        for i in range(len(val_ar)):
            val += val_ar[i] * power_val[i]
        return val

    def load_edge(self, img, masking=False):
        mask = (1 - self.mask).astype(np.bool) if masking else None
        return canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0, sigma=2, mask=mask).astype(float)

    def isEmpty(self, feature):
        for i in range(min(len(feature), 128)):
            if feature[i] != 0:
                return False
        return True

    def search_ab(self, feature, a, b, size=256):
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                ra = a + i
                rb = b + j
                if 0 <= ra <= size - 1 and 0 <= rb <= size - 1 and self.isEmpty(feature[ra][rb]):
                    return ra, rb
        return -1, -1

    def tensor(self, img):
        return torch.from_numpy(img).float().permute(2, 0, 1)


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='xavier', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)
        self.apply(init_func)


class LBPGenerator(BaseNetwork):
    def __init__(self, in_channels=128, out_channels=1):
        super(LBPGenerator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1)
        )
        self.layer2 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
        )
        self.layer3 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
        )
        self.layer4 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
        )
        self.layer5 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
        )
        self.layer6 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
        )
        self.layer7 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512)
        )
        # It's feasible to use UpsamplingNearest2d + Conv2d or ConvTranspose2d directly.
        self.layer8 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(512)
        )
        self.layer9 = nn.Sequential(
            nn.ReLU(True),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(512)
        )
        self.layer10 = nn.Sequential(
            nn.ReLU(True),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(512)
        )
        self.layer11 = nn.Sequential(
            nn.ReLU(True),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(512)
        )
        self.layer12 = nn.Sequential(
            nn.ReLU(True),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(256),
        )
        self.layer13 = nn.Sequential(
            nn.ReLU(True),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(128),
        )
        self.layer14 = nn.Sequential(
            nn.ReLU(True),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(64)
        )
        self.layer15 = nn.Sequential(
            nn.ReLU(True),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=128, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.init_weights()

    def forward(self, Si):
        layer1 = self.layer1(Si)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        layer5 = self.layer5(layer4)
        layer6 = self.layer6(layer5)
        layer7 = self.layer7(layer6)
        layer8 = self.layer8(layer7)
        layer9 = self.layer9(torch.cat([layer8, layer7], dim=1))
        layer10 = self.layer10(torch.cat([layer9, layer6], dim=1))
        layer11 = self.layer11(torch.cat([layer10, layer5], dim=1))
        layer12 = self.layer12(torch.cat([layer11, layer4], dim=1))
        layer13 = self.layer13(torch.cat([layer12, layer3], dim=1))
        layer14 = self.layer14(torch.cat([layer13, layer2], dim=1))
        layer15 = self.layer15(torch.cat([layer14, layer1], dim=1))
        Lo = torch.tanh(layer15)
        return Lo


class ImgGenerator(BaseNetwork):
    def __init__(self, in_channels=129):
        super(ImgGenerator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1)
        )
        self.layer2 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
        )
        self.layer3 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
        )
        self.layer4 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
        )
        self.layer5 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
        )
        self.layer6 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
        )
        self.layer7 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512)
        )
        self.layer8 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512)
        )
        self.layer9 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512)
        )
        self.layer10 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512)
        )
        self.layer11 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512)
        )
        self.layer12 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=1024, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
        )
        self.layer13 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
        )
        self.layer14 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64)
        )
        self.layer15 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2, padding=1)
        )
        self.init_weights()

    def forward(self, Si, Lo, choice):
        if choice == 'SLI-B':
            layer1 = self.layer1(Si)  # Omit the input of Lo
        else:
            layer1 = self.layer1(torch.cat([Lo, Si], dim=1))
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        layer5 = self.layer5(layer4)
        layer6 = self.layer6(layer5)
        layer7 = self.layer7(layer6)
        layer8 = self.layer8(layer7)
        layer9 = self.layer9(torch.cat([layer8, layer7], dim=1))
        layer10 = self.layer10(torch.cat([layer9, layer6], dim=1))
        layer11 = self.layer11(torch.cat([layer10, layer5], dim=1))
        layer12 = self.layer12(torch.cat([layer11, layer4], dim=1))
        layer13 = self.layer13(torch.cat([layer12, layer3], dim=1))
        layer14 = self.layer14(torch.cat([layer13, layer2], dim=1))
        layer15 = self.layer15(torch.cat([layer14, layer1], dim=1))
        Io = torch.tanh(layer15)
        return Io


class Discriminator(BaseNetwork):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        self.use_sigmoid = True
        self.conv1 = self.features = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=True),
        )

        self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.save_dir = 'weights/'


class LBPModel(BaseModel):
    def __init__(self):
        super(LBPModel, self).__init__()
        self.lr = 1e-4
        self.gan_type = 're_avg_gan'
        self.channels = 3 if structure == 'RTV' else 1
        self.gen = nn.DataParallel(LBPGenerator(in_channels=128, out_channels=self.channels)).cuda()
        self.dis = nn.DataParallel(Discriminator(in_channels=self.channels)).cuda()
        self.l1_loss = nn.L1Loss()
        self.adversarial_loss = AdversarialLoss()
        self.perceptual_loss = PerceptualLoss()
        self.style_loss = StyleLoss()
        self.criterionGAN = GANLoss(gan_type=self.gan_type)

        self.gen_optimizer = optim.Adam(self.gen.parameters(), lr=self.lr, betas=(0.9, 0.999))
        self.dis_optimizer = optim.Adam(self.dis.parameters(), lr=self.lr, betas=(0.9, 0.999))

        self.ADV_LOSS_WEIGHT = 0.2
        self.L1_LOSS_WEIGHT = 100
        self.PERC_LOSS_WEIGHT = 1

    def process(self, Si, Lg):
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()

        Lo = self(Si)

        if self.gan_type == 're_avg_gan':
            dis_fake, _ = self.dis(Lo.detach())
            gen_fake, _ = self.dis(Lo)
            dis_real, _ = self.dis(Lg)
            gen_real, _ = self.dis(Lg)
            dis_loss = self.criterionGAN(dis_real - dis_fake, True)
            gen_gan_loss = (self.criterionGAN(gen_real - torch.mean(gen_fake), False) +
                            self.criterionGAN(gen_fake - torch.mean(gen_real), True)) / 2. * self.ADV_LOSS_WEIGHT
        else:
            dis_input_real = Lg
            dis_input_fake = Lo.detach()
            dis_real, _ = self.dis(dis_input_real)
            dis_fake, _ = self.dis(dis_input_fake)
            dis_real_loss = self.adversarial_loss(dis_real, True, True)
            dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
            dis_loss = (dis_real_loss + dis_fake_loss) / 2

            gen_input_fake = Lo
            gen_fake, _ = self.dis(gen_input_fake)
            gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.ADV_LOSS_WEIGHT
        if self.channels == 1:
            gen_perceptual_loss = self.perceptual_loss(torch.cat([Lo, Lo, Lo], dim=1), torch.cat([Lg, Lg, Lg], dim=1)) * self.PERC_LOSS_WEIGHT
        else:
            gen_perceptual_loss = self.perceptual_loss(Lo, Lg) * self.PERC_LOSS_WEIGHT
        gen_l1_loss = self.l1_loss(Lo, Lg) * self.L1_LOSS_WEIGHT
        gen_loss = gen_gan_loss + gen_perceptual_loss + gen_l1_loss
        return Lo, gen_loss, dis_loss

    def forward(self, Si):
        return self.gen(Si)

    def backward(self, gen_loss=None, dis_loss=None, retain_graph=False):
        if dis_loss is not None:
            dis_loss.backward(retain_graph=retain_graph)
            self.dis_optimizer.step()

        if gen_loss is not None:
            gen_loss.backward(retain_graph=retain_graph)
            self.gen_optimizer.step()

    def save(self, path):
        if not os.path.exists(self.save_dir + path):
            os.makedirs(self.save_dir + path)
        torch.save(self.gen.state_dict(), self.save_dir + path + 'lbp_gen.pth')
        torch.save(self.dis.state_dict(), self.save_dir + path + 'lbp_dis.pth')

    def load(self, path):
        self.gen.load_state_dict(torch.load(self.save_dir + path + 'lbp_gen.pth'))


class ImgModel(BaseModel):
    def __init__(self, choice='SLI'):
        super(ImgModel, self).__init__()
        self.choice = choice
        self.lr = 1e-4
        self.gan_type = 're_avg_gan'
        self.channels = 3 if structure == 'RTV' else 1
        self.in_channels = 128 + self.channels if self.choice != 'SLI-B' else 1
        self.gen = nn.DataParallel(ImgGenerator(in_channels=self.in_channels)).cuda()
        self.dis = nn.DataParallel(Discriminator(3)).cuda()
        self.l1_loss = nn.L1Loss()
        self.adversarial_loss = AdversarialLoss()
        self.perceptual_loss = PerceptualLoss()
        self.style_loss = StyleLoss()
        self.criterionGAN = GANLoss(gan_type=self.gan_type)
        self.gen_optimizer = optim.Adam(self.gen.parameters(), lr=self.lr, betas=(0.9, 0.999))
        self.dis_optimizer = optim.Adam(self.dis.parameters(), lr=self.lr, betas=(0.9, 0.999))

        self.ADV_LOSS_WEIGHT = 0.2
        self.L1_LOSS_WEIGHT = 100
        self.PERC_LOSS_WEIGHT = 1
        self.STYLE_LOSS_WEIGHT = 10

    def process(self, Si, Lo, Ig):
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()

        Io = self(Si, Lo, self.choice)

        if self.gan_type == 're_avg_gan':
            dis_fake, _ = self.dis(Io.detach())
            gen_fake, _ = self.dis(Io)
            dis_real, _ = self.dis(Ig)
            gen_real, _ = self.dis(Ig)
            dis_loss = self.criterionGAN(dis_real - dis_fake, True)
            gen_gan_loss = (self.criterionGAN(gen_real - torch.mean(gen_fake), False) +
                            self.criterionGAN(gen_fake - torch.mean(gen_real), True)) / 2. * self.ADV_LOSS_WEIGHT
        else:
            dis_input_real = Ig
            dis_input_fake = Io.detach()
            dis_real, _ = self.dis(dis_input_real)
            dis_fake, _ = self.dis(dis_input_fake)
            dis_real_loss = self.adversarial_loss(dis_real, True, True)
            dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
            dis_loss = (dis_real_loss + dis_fake_loss) / 2

            gen_input_fake = Io
            gen_fake, _ = self.dis(gen_input_fake)
            gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.ADV_LOSS_WEIGHT

        gen_style_loss = self.style_loss(Io, Ig) * self.STYLE_LOSS_WEIGHT
        gen_perceptual_loss = self.perceptual_loss(Io, Ig) * self.PERC_LOSS_WEIGHT
        gen_l1_loss = self.l1_loss(Io, Ig) * self.L1_LOSS_WEIGHT
        gen_loss = gen_gan_loss + gen_l1_loss + gen_perceptual_loss + gen_style_loss
        return Io, gen_loss, dis_loss

    def forward(self, Si, Lo, choice):
        return self.gen(Si, Lo, choice)

    def backward(self, gen_loss=None, dis_loss=None, retain_graph=False):
        if dis_loss is not None:
            dis_loss.backward(retain_graph=retain_graph)
            self.dis_optimizer.step()

        if gen_loss is not None:
            gen_loss.backward(retain_graph=retain_graph)
            self.gen_optimizer.step()

    def save(self, path):
        if not os.path.exists(self.save_dir + path):
            os.makedirs(self.save_dir + path)
        torch.save(self.gen.state_dict(), self.save_dir + path + 'img_gen.pth')
        torch.save(self.dis.state_dict(), self.save_dir + path + 'img_dis.pth')

    def load(self, path):
        self.gen.load_state_dict(torch.load(self.save_dir + path + 'img_gen.pth'))


class SIFTReconstruction():
    def __init__(self, choice='SLI'):
        self.choice = choice
        self.dataset = 'celebahq'
        self.train_num = 28000
        self.test_num = 10
        self.batch_size = 1
        train_file = 'data/celebahq_train/'
        test_file = 'data/celebahq_test/'
        train_dataset = SIFTDataset(self.train_num, train_file, self.choice)
        test_dataset = SIFTDataset(self.test_num, test_file, self.choice)

        self.lbp_model = LBPModel().cuda()
        self.img_model = ImgModel(self.choice).cuda()
        self.n_epochs = 50
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8)

        # The performance would be better if the LBP network is trained first,
        # however, it's feasible to directly train both networks.
        self.TRAIN_BOTH = True

    def train(self):
        print('\nTrain/Val ' + self.choice + ' model:')
        for epoch in range(self.n_epochs):
            gen_losses, ssim, psnr = [], [], []
            for cnt, items in enumerate(self.train_loader):
                self.lbp_model.train()
                self.img_model.train()
                Ig, Si, Lg = (item.cuda() for item in items[:-1])
                if self.choice in ['SLI', 'SLI-L', 'SLI-R']:
                    Lo, gen_lbp_loss, dis_lbp_loss = self.lbp_model.process(Si, Lg)
                    Io, gen_img_loss, dis_img_loss = self.img_model.process(Si, Lo, Ig)
                    self.lbp_model.backward(gen_lbp_loss, dis_lbp_loss, self.TRAIN_BOTH)
                    if self.TRAIN_BOTH:
                        self.img_model.backward(gen_img_loss, dis_img_loss)
                elif self.choice == 'SLI-B':
                    Io, gen_img_loss, dis_img_loss = self.img_model.process(Si, Lg, Ig)
                    self.img_model.backward(gen_img_loss, dis_img_loss)

                s, p = self.metrics(Ig, Io) if self.TRAIN_BOTH else self.metrics(Lg, Lo)
                ssim.append(s)
                psnr.append(p)
                if not self.TRAIN_BOTH:
                    gen_losses.append(gen_lbp_loss.item())
                else:
                    gen_losses.append(gen_img_loss.item())
                print('Tra (%d/%d) Loss:%5.4f, SSIM:%4.4f, PSNR:%4.2f' %
                      (cnt, self.train_num, np.mean(gen_losses), np.mean(ssim), np.mean(psnr)), end='\r')
                if cnt % (self.test_num // 2) == 0:
                    val_ssim, val_psnr = self.test()
                    self.lbp_model.save('latest_' + self.choice + '/')
                    if self.TRAIN_BOTH:
                        self.img_model.save('latest_' + self.choice + '/')
                    print('Val (%d/%d) SSIM:%4.4f, PSNR:%4.2f' % (cnt, self.train_num, val_ssim, val_psnr))

    def test(self, pretrained=False):
        if pretrained:
            print('\nTest ' + self.choice + ' model:')
            if self.choice != 'SLI-B':
                self.lbp_model.load('celebahq_' + self.choice + '/')
            self.img_model.load('celebahq_' + self.choice + '/')
        self.lbp_model.eval()
        self.img_model.eval()
        ssim, psnr = [], []
        for cnt, items in enumerate(self.test_loader):
            Ig, Si, Lg = (item.cuda() for item in items[:-1])
            if self.choice in ['SLI', 'SLI-L', 'SLI-R']:
                Lo = self.lbp_model(Si)
                Io = self.img_model(Si, Lo, self.choice)
            elif self.choice == 'SLI-B':
                Io = self.img_model(Si, Lg, self.choice)

            s, p = self.metrics(Ig, Io) if self.TRAIN_BOTH else self.metrics(Lg, Lo)
            ssim.append(s)
            psnr.append(p)
            if cnt < 100:
                # Lo = self.postprocess(Lo)
                # cv2.imwrite('res/' + self.choice + '_results/Lo_%06d.jpg' % (cnt+1), Lo[0])
                Io = self.postprocess(Io)
                cv2.imwrite('res/' + self.choice + '_results/Io_%06d.jpg' % (cnt+1), Io[0])
        if pretrained:
            print(self.choice + ' Evaluation: SSIM:%4.4f, PSNR:%4.2f' % (np.mean(ssim), np.mean(psnr)))
        return np.mean(ssim), np.mean(psnr)

    def postprocess(self, img):
        img = img * 127.5 + 127.5
        img = img.permute(0, 2, 3, 1)
        return img.int().cpu().detach().numpy()

    def metrics(self, Ig, Io):
        a = self.postprocess(Ig)
        b = self.postprocess(Io)
        ssim, psnr = [], []
        for i in range(len(a)):
            ssim.append(compare_ssim(a[i], b[i], win_size=11, data_range=255.0, multichannel=True))
            psnr.append(compare_psnr(a[i], b[i], data_range=255))
        return np.mean(ssim), np.mean(psnr)


class ClassifierDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, idx):
        return self.tensor(np.array(list(self.x[idx]))).view([128]), self.tensor(np.array(self.y[idx]))

    def __len__(self):
        return len(self.x)

    def tensor(self, x):
        return torch.from_numpy(x).float()


class ClassifierNetwork(nn.Module):
    def __init__(self):
        super(ClassifierNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features=128, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=8),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


class ClassifierModel(nn.Module):
    def __init__(self):
        super(ClassifierModel, self).__init__()
        self.lr = 1e-4
        self.gen = ClassifierNetwork()
        self.optimizer = optim.Adam(self.gen.parameters(), lr=self.lr, betas=(0.9, 0.999))
        self.log_loss = nn.CrossEntropyLoss()
        self.save_dir = 'weights/'

    def process(self, x, y):
        self.gen.zero_grad()
        pred = self(x)
        loss = self.log_loss(pred, y.long())
        return pred, loss

    def forward(self, x):
        return self.gen(x)

    def backward(self, loss=None, retain_graph=None):
        if loss is not None:
            loss.backward(retain_graph=retain_graph)
            self.optimizer.step()

    def save(self, path):
        if not os.path.exists(self.save_dir + path):
            os.makedirs(self.save_dir + path)
        torch.save(self.gen.state_dict(), self.save_dir + path + 'sift_classifier_gen.pth')

    def load(self, path):
        self.gen.load_state_dict(torch.load(self.save_dir + path + 'sift_classifier_gen.pth'))


class Classifier():
    def __init__(self):
        des_label = np.load('data/des_label.npy', allow_pickle=True)
        x_train, x_val, y_train, y_val = train_test_split(des_label[:, :1], des_label[:, 1], test_size=0.1, random_state=123)
        self.train_size = len(x_train)
        train_dataset = ClassifierDataset(x_train, y_train)
        test_dataset = ClassifierDataset(x_val, y_val)
        self.model = ClassifierModel().to(device)
        self.batch_size = 256
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False)

    def train(self):
        best_val_acc = 0
        scheduler = ReduceLROnPlateau(self.model.optimizer, factor=0.5, patience=5)
        for epoch in range(1, 20):
            losses, acces = [], []
            cnt = 0
            for items in self.train_loader:
                cnt += self.batch_size
                self.model.train()
                x, y = (item.to(device) for item in items)
                output, loss = self.model.process(x, y)
                self.model.backward(loss)
                pred = np.zeros(len(output))
                for i in range(len(output)):
                    pred[i] = np.argmax(output[i].detach().cpu().numpy())
                acc = accuracy_score(pred, y.detach().cpu())

                losses.append(loss.item())
                acces.append(acc.item())
                print('Tra %d/%d G:%5.3f A:%.4f' % (cnt, self.train_size, np.mean(losses), np.mean(acces)), end='\r')
            val_losses, val_acces = self.test()
            scheduler.step(val_losses)
            if val_acces > best_val_acc:
                self.model.save('celebahq_SIFT_classifier/')
            print('Tes %d/%d G:%5.3f A:%.4f' % (cnt, self.train_size, val_losses, val_acces))

    def test(self):
        self.model.eval()
        losses, acces = [], []
        for items in self.test_loader:
            x, y = (item.to(device) for item in items)
            output, loss = self.model.process(x, y)
            pred = np.zeros(len(output))
            for i in range(len(output)):
                pred[i] = np.argmax(output[i].detach().cpu().numpy())
            acc = accuracy_score(pred, y.detach().cpu())
            losses.append(loss.item())
            acces.append(acc.item())
        return np.mean(losses), np.mean(acces)

    def predict(self):
        self.model.load('celebahq_SIFT_classifier/')
        self.model.eval()
        path = 'data/celebahq_train/'
        flist = sorted(os.listdir(path))
        for idx in range(len(flist)):
            print(idx, end='\r')
            file = flist[idx]
            res = []
            img = cv2.imread(path + file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            meta, des = sift.detectAndCompute(np.uint8(img), None)
            for i in range(len(meta)):
                input = torch.from_numpy(np.array(list(des[i]))).float().view([1, 128]).to(device)
                output = self.model.gen(input)
                pred = np.argmax(output.detach().cpu().numpy())
                res.append(pred)
            np.save('data/celebahq_train_classified_predict/' + file[:-4] + '.npy', res)


def generate_sift_classifier_training_data(path='data/celebahq_train/'):
    PREDICTOR_PATH = 'utils/shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    flist = sorted(os.listdir(path))
    np.random.shuffle(flist)
    flist = flist[:1000]
    JAW_POINTS = list(range(0, 17))
    RIGHT_BROW_POINTS = list(range(17, 22))
    LEFT_BROW_POINTS = list(range(22, 27))
    NOSE_POINTS = list(range(27, 35))
    RIGHT_EYE_POINTS = list(range(36, 42))
    LEFT_EYE_POINTS = list(range(42, 48))
    MOUTH_POINTS = list(range(48, 68))
    LAND_POINTS = [JAW_POINTS, RIGHT_BROW_POINTS, LEFT_BROW_POINTS,
                   NOSE_POINTS, RIGHT_EYE_POINTS, LEFT_EYE_POINTS, MOUTH_POINTS]
    des_label = []
    for file in flist:
        img = cv2.imread(path + file)
        img = cv2.resize(img, (256, 256))
        rects = detector(img, 1)
        if len(rects) != 1:
            continue
        landmarks = np.array([[p.x, p.y] for p in predictor(img, rects[0]).parts()])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        meta, des = SIFT.detectAndCompute(np.uint8(img), None)
        cnt = 0
        sample = random.sample(range(len(meta)), len(meta))
        for i in sample:
            x = int(meta[i].pt[1])
            y = int(meta[i].pt[0])
            distances = distance.cdist([[y, x]], landmarks, "euclidean")[0]
            dis, idx = torch.topk(torch.from_numpy(distances), 1, dim=0, largest=False)
            if dis[0] < 10:
                for j in range(len(LAND_POINTS)):
                    if idx[0] in LAND_POINTS[j]:
                        break
                label = j
                des_label.append((des[i], label))
            else:
                cnt += 1
                label = 7
                if cnt <= 13: # To avoid uneven data distribution
                    des_label.append((des[i], label))
    np.save('data/des_label.npy', np.array(des_label))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('type', type=str, help='train or test the model', choices=['train', 'test'])
    parser.add_argument('--model', type=str, default='SLI', help='SLI, SLI-L, SLI-R or SLI-B',
                        choices=['SLI', 'SLI-L', 'SLI-R', 'SLI-B'])
    args = parser.parse_args()

    model = SIFTReconstruction(choice=args.model)
    if args.type == 'train':
        model.train()
    elif args.type == 'test':
        model.test(True)
    print('End.')
