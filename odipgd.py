#coding=utf-8
import os
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from torchvision import transforms as T
import torch.nn.functional as F
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable as V
from torchvision import models
import time
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from PIL import Image
import argparse

    
class ImageNet_txt(Dataset):
    
    def __init__(self, img_dir, img_txt_dir, transforms=None):
        self.img_dir = img_dir
        self.img_txt_dir = img_txt_dir
        self.transforms = transforms
        img_label_list = self.read_txt(img_txt_dir)
        self.img_label_list = img_label_list

    def read_txt(self, img_txt_dir):
        with open(self.img_txt_dir, 'r') as f:
            lines = f.readlines()
            img_label_list = []
            for i in lines:
                img_label_list.append([os.path.join(self.img_dir, \
                           i.split()[0]), int(i.split()[1])])
        return img_label_list

    def __getitem__(self, index):
        img_path, gt_label = self.img_label_list[index]
        pil_image = Image.open(img_path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(pil_image)
        return img_path, img, gt_label

    def __len__(self):
        return len(self.img_label_list)


def clip_by_tensor(x, x_min, x_max):
    result = x.clamp(x_min, x_max)
    return result


def clip_by_image(img, img_min, img_max):
    img[img < img_min] = img_min[img < img_min]
    img[img > img_max] = img_max[img > img_max]
    return img


def margin_loss(output, orig_label):
    olab = torch.eye(1000)[orig_label].to(device)
    pert_label = output.argmax(1)
    real = torch.max(output * olab)
    other = torch.max((1 - olab) * output)
    loss = other - real
    return pert_label, loss


def one_step(image, grad, step_size, target):
    if target:
        return image.data - step_size * torch.sign(grad)
    else:
        return image.data + step_size * torch.sign(grad)


def odipgd_attack(img, orig_label, model, restart_number, target, eps, step_size, max_epoch, ODI_num_steps=4, ODI_step_size=0.02):
    img_min = clip_by_tensor(img - 2.0 * eps, -1.0, 1.0)
    img_max = clip_by_tensor(img + 2.0 * eps, -1.0, 1.0)
    gt_output = model(img)
    target_label = gt_output.argmin(1).to(device)
    '''
    _, b = gt_output.sort(descending=True)
    i = torch.randint(1, 10, (1,)).to(device)
    target_label = b[0,i]
    '''
    if target:
        print('The target label is: {}'.format(target_label.item()))
    flag = True
    for i in range(restart_number):
        adv_img = V(img.data, requires_grad=True)
        randVector = torch.FloatTensor(model(adv_img).shape).uniform_(-1.0,1.0).to(device)
        random_noise = torch.FloatTensor(adv_img.shape).uniform_(-2*eps, 2*eps).to(device)
        adv_img = V(clip_by_image(adv_img.data + random_noise, img_min, img_max), requires_grad=True)
        for j in range(ODI_num_steps + max_epoch):
            zero_gradients(adv_img)
            output = model(adv_img)
            if target:
                pert_label = output.argmax(1)
                if pert_label == target_label:
                    flag = False
                    break
                with torch.enable_grad():
                    if j < ODI_num_steps:
                        loss = (output * randVector).sum()
                    else:
                        loss = F.cross_entropy(output, target_label)
            else:
                with torch.enable_grad():
                    if j < ODI_num_steps:
                        loss = (output * randVector).sum()
                    else:
                        pert_label, loss = margin_loss(output, orig_label)
                        if pert_label != orig_label:
                            flag = False
                            break
            loss.backward()
            grad = adv_img.grad.data
            if j < ODI_num_steps: 
                adv_img = one_step(adv_img, grad, ODI_step_size, target)
            else:
                adv_img = one_step(adv_img, grad, step_size, target)
            adv_img = V(clip_by_image(adv_img.data, img_min, img_max), requires_grad=True)
        if not flag:
            break
    return adv_img.detach(), pert_label, target_label, i, j+1
    

def save_image(img_path, img, perturbed_img, output_dir):
    img_name = img_path[0].split('/')[-1]
    im_ori = img.clone().detach().to(torch.device('cpu')).squeeze(0).\
    permute(1, 2, 0).div_(2).add_(0.5).mul_(255).type(torch.uint8).numpy()
    im_adv = perturbed_img.clone().detach().to(torch.device('cpu')).\
    squeeze(0).permute(1, 2, 0).div_(2).add_(0.5).mul_(255).type(torch.uint8).numpy()
    pert_img = np.maximum((im_adv - im_ori), (im_ori - im_adv))
    im_ori, im_adv, pert_img = Image.fromarray(im_ori), Image.fromarray(im_adv), Image.fromarray(pert_img)
    im = Image.new('RGB', (3*opt.imgsize, opt.imgsize))
    im.paste(im_ori, (0, 0, opt.imgsize, opt.imgsize))
    im.paste(im_adv, (opt.imgsize, 0, 2*opt.imgsize, opt.imgsize))
    im.paste(pert_img, (2*opt.imgsize, 0, 3*opt.imgsize, opt.imgsize))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    im.save(os.path.join(output_dir, img_name), quality=95)


def main():
    start_time = time.time()
    transforms = T.Compose([T.Resize(size=(opt.imgsize, opt.imgsize)), T.ToTensor(), T.Normalize(opt.mean, opt.std)])
    X = ImageNet_txt(opt.img_dir, opt.img_txt_dir, transforms)
    data_loader = DataLoader(X, batch_size=opt.batch_size, shuffle=False)
    model_type = 'models.' + opt.classifier
    model = eval(model_type)(pretrained=True).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    total = 0
    incorrect = 0
    success = 0
    for img_path, img, label in tqdm(data_loader):
        img = img.to(device)
        label = label.to(device)
        total += label.size(0)
        if (model(img).argmax(1) != label):
            incorrect += 1
            continue
        perturbed_img, final_label, target_label, re_num, iter_num = odipgd_attack(img, label, model, \
                opt.restart_number, opt.target, opt.eps, opt.step_size, opt.max_epoch)
        if opt.target:
            if final_label == target_label:
                success += 1
                save_image(img_path, img, perturbed_img, opt.output_dir)
        else:
            if final_label != label:
                success += 1
                save_image(img_path, img, perturbed_img, opt.output_dir)
        print('Imagename: {}\tlabel_orig: {}\tfinal_label: {}\trestart_number: {}\titer_num: {}'.format(\
            img_path[0].split('/')[-1], label.item(), final_label.item(), re_num, iter_num))
    total_time = time.time() - start_time
    print('class error rate of test images is: {}%.'.format(100 * incorrect / total))
    if total-incorrect != 0:
        print('attack accuracy of test images is: {}%.'.format(100 * success / (total - incorrect)))
    else:
        print('Error!')
    print('Time cost is: {}s.'.format(total_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, \
       default='/home/guest/liu_hanwen/project_dataset/dataset', help='原始图像路径')
    parser.add_argument('--output_dir', type=str, \
       default='/home/guest/liu_hanwen/test_images/odipgd_vgg16_t_sign', help='保存图片路径')
    parser.add_argument('--img_txt_dir', type=str, \
       default='/home/guest/liu_hanwen/project_dataset/vggnet2000_select.txt', help='图像名称及标签的txt文本路径')
    parser.add_argument('--imgsize', type=int, default=224, help='输入分类器的图像大小')
    parser.add_argument('--mean', type=float, default=np.array([0.5, 0.5, 0.5]), help='mean.')
    parser.add_argument('--std', type=float, default=np.array([0.5, 0.5, 0.5]), help='std.')
    parser.add_argument('--classifier', type=str, default='vgg16', help='分类器名称')
    parser.add_argument('--eps', type=float, default=0.03125, help='8/256')
    parser.add_argument('--step_size', type=float, default=0.005, help='单步迭代更新步长')
    parser.add_argument('--max_epoch', type=int, default=100, help='最大迭代次数')
    parser.add_argument('--restart_number', type=int, default=100, help='')
    parser.add_argument('--batch_size', type=int, default=1, help='同时处理图像个数，仅支持1')
    parser.add_argument('--target', type=bool, default=True, help='是否执行有目标攻击')
    parser.add_argument('--GPU', type=str, default='2', help='GPU 编号')
    parser.add_argument('--seed', type=int, default=7923, help='随机数种子')
    opt = parser.parse_args()
    device = torch.device('cuda:{}'.format(opt.GPU) if torch.cuda.is_available() else 'cpu')
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    main()
