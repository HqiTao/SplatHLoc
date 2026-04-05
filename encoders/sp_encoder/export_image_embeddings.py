# from segment_anything import sam_model_registry, SamPredictor

import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn.functional as F
import torch.nn as nn
from pathlib import Path
from torchvision import transforms

import argparse
import os

class SuperPoint(nn.Module):
    def __init__(self):
        super().__init__()
        out_channels = 256

        self.transform = transforms.Grayscale(num_output_channels=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(
            c5, out_channels,
            kernel_size=1, stride=1, padding=0)

        path = Path(__file__).parent / 'weights/superpoint_v1.pth'
        self.load_state_dict(torch.load(str(path)), strict=False)

        print('Loaded SuperPoint model')

    def forward(self, x):
        """ Compute keypoints, scores, descriptors for image """
        # Shared Encoder
        # print(x.shape)
        x = self.transform(x)
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Compute the dense keypoint scores
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8)

        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)
        return descriptors, scores
    

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


def main(args: argparse.Namespace) -> None:
    print("Loading model...")

    model = SuperPoint().cuda()

    if not os.path.isdir(args.input):
        targets = [args.input]
    else:
        print(os.listdir(args.input))
        seqs = [f for f in os.listdir(args.input) if "seq" in f and "zip" not in f]
    
    print(seqs)
    os.makedirs(args.output, exist_ok=True)


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
            image = cv2.imread(t)
            if image is None:
                print(f"Could not load '{t}' as an image, skipping...")
                continue
            
            tensor_image = torch.from_numpy(np.array(image))
            input_image = tensor_image.to(
                device="cuda", dtype=torch.float32, non_blocking=True
            )
            input_image = input_image / 255.0
            img_features, scores = model(input_image.permute(2, 0, 1)[None])
            # print(scores.shape)

            img_features = img_features.squeeze(0) # (256, 60, 80)
            img_scores = scores[0] # (60, 80)

            torch.save(img_features, os.path.join(output_dir, f"{img_name}_fmap_CxHxW.pt"))
            torch.save(img_scores, os.path.join(output_dir, f"{img_name}_smap_CxHxW.pt"))
        

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

