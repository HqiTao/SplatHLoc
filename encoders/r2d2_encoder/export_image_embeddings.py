# from segment_anything import sam_model_registry, SamPredictor

import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn.functional as F
import torch.nn as nn
from pathlib import Path
from torchvision import transforms
import torchvision.transforms as tvf
from PIL import Image
import subprocess
import argparse
import os

class R2D2Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.ops = nn.ModuleList([])
        self.dilation = 1
        self.add_conv(3, 32)
        self.add_conv(32, 32)
        self.add_conv(32, 64, stride=2)
        self.add_conv(64, 64)
        self.add_conv(64, 128, stride=2)
        self.add_conv(128, 128)
        self.add_conv(128, 128, k=2, stride=2, relu=False)
        self.add_conv(128, 128, k=2, stride=2, relu=False)
        self.add_conv(128, 128, k=2, stride=2, bn=False, relu=False)
        self.clf = nn.Conv2d(128, 2, kernel_size=1)
        self.sal = nn.Conv2d(128, 1, kernel_size=1)
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
    
    
    def add_conv(self, in_channels, out_channels, k=3, stride=1, bn=True, relu=True):
        conv_params = dict(padding=((k-1) * self.dilation) // 2, dilation=self.dilation, stride=1)
        self.dilation *= stride
        self.ops.append( nn.Conv2d(in_channels, out_channels, kernel_size=k, **conv_params) )
        if bn: 
            self.ops.append( nn.BatchNorm2d(out_channels, affine=False) )
        if relu: 
            self.ops.append( nn.ReLU(inplace=True) )
    
    
    def softmax(self, ux):
        if ux.shape[1] == 1:
            x = F.softplus(ux)
            return x / (1 + x) 
        elif ux.shape[1] == 2:
            return F.softmax(ux, dim=1)[:,1:2]
    
    
    def forward(self, x):
        x = x - x.new_tensor(self.mean).view(1, -1, 1, 1)
        x = x / x.new_tensor(self.std).view(1, -1, 1, 1)
        
        for op in self.ops:
            x = op(x)
        ureliability = self.clf(x**2)
        urepeatability = self.sal(x**2)
        
        descriptors = F.normalize(x, p=2, dim=1)
        repeatability = self.softmax( urepeatability )
        reliability = self.softmax( ureliability )
        
        return descriptors, repeatability, reliability
    

parser = argparse.ArgumentParser(
    description=(
        "Get image embeddings of an input image or directory of images."
    )
)

parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to either a single input image or folder of images.",
)

parser.add_argument(
    "--output",
    type=str,
    required=True,
    help=(
        "Path to the directory where embeddings will be saved. Output will be either a folder "
        "of .pt per image or a single .pt representing image embeddings."
    ),
)


parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")

def get_pretrained_model():
    model = R2D2Net().cuda()
    model_name = 'r2d2_WASF_N16.pt'
    model_dir = Path(os.path.dirname(os.path.abspath(__file__)))/'weights'
    model_path = model_dir / model_name
    if not model_path.exists():
        weight_url = 'https://github.com/naver/r2d2/raw/refs/heads/master/models/'+model_name
        download_model(weight_url, str(model_dir), model_name)
    checkpoint = torch.load(model_path)
    weights = checkpoint['state_dict']
    model.load_state_dict({k.replace('module.',''):v for k,v in weights.items()})
    
    return model.eval().cuda()

def download_model(url, model_dir, model_name):
    model_path = os.path.join(model_dir, model_name)
    if os.path.exists(model_path):
        return
    print('Downloading the {} model from {}.'.format(model_name, url))
    command = ['wget', '--no-check-certificate', url, '-O', model_name]
    subprocess.run(command, check=True)
    os.makedirs(model_dir, exist_ok=True)
    command = ['mv', model_name, model_dir]
    subprocess.run(command, check=True)


def main(args: argparse.Namespace) -> None:
    if not os.path.isdir(args.input):
        targets = [args.input]
    else:
        print(os.listdir(args.input))
        seqs = [f for f in os.listdir(args.input) if "seq" in f and "zip" not in f]
    
    print(seqs)
    os.makedirs(args.output, exist_ok=True)
    
    print("Loading model...")
    model = R2D2Net().cuda().eval()
    model_name = 'r2d2_WASF_N16.pt'
    model_dir = Path(os.path.dirname(os.path.abspath(__file__)))/'weights'
    model_path = model_dir / model_name
    if not model_path.exists():
        weight_url = 'https://github.com/naver/r2d2/raw/refs/heads/master/models/'+model_name
        download_model(weight_url, str(model_dir), model_name)
    checkpoint = torch.load(model_path)
    weights = checkpoint['state_dict']
    model.load_state_dict({k.replace('module.',''):v for k,v in weights.items()})
    
    
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    norm_image = tvf.Compose([tvf.ToTensor(), tvf.Normalize(mean=mean, std=std)])


    for seq in seqs:
        targets = [
            f"{seq}/{f}" for f in os.listdir(os.path.join(args.input, seq)) if "color" in f
        ]
        targets = [os.path.join(args.input, f) for f in targets]

        output_dir = os.path.join(args.output, seq)
        os.makedirs(output_dir, exist_ok=True)

        for t in targets:
            print(f"Processing '{t}'...")
            img_name = t.split(os.sep)[-1]
            image = Image.open(t).convert('RGB')
            if image is None:
                print(f"Could not load '{t}' as an image, skipping...")
                continue
            
            image = norm_image(image)[None].cuda()
            with torch.no_grad():
                descriptors, repeatability, reliability = model(image)
            # tensor_image = torch.from_numpy(np.array(image))
            # input_image = tensor_image.to(
            #     device="cuda", dtype=torch.float32, non_blocking=True
            # )
            # input_image = input_image / 255.0
            # img_features, scores = model(input_image.permute(2, 0, 1)[None])
            # print(scores.shape)
            
            img_features = descriptors.squeeze() # (128, H, W)
            img_scores = reliability.squeeze() # (H, W)
            print(img_features.shape, img_scores.shape)

            torch.save(img_features, os.path.join(output_dir, f"{img_name}_fmap_CxHxW.pt"))
            torch.save(img_scores, os.path.join(output_dir, f"{img_name}_smap_CxHxW.pt"))
        

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

