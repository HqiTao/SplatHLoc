import torch.nn as nn
import torch
from encoders.r2d2_encoder.export_image_embeddings import get_pretrained_model
import torchvision.transforms as tvf
from encoders.sp_encoder.export_image_embeddings import SuperPoint

class FeatureExtractor(nn.Module):
    def __init__(self, feature_type):
        super(FeatureExtractor, self).__init__()
        self.feature_type = feature_type
        if feature_type == "sp":
            print("Loading SuperPoint model...")
            self.model = SuperPoint().cuda().eval()
            self.feature_dim = 256
        elif feature_type == "r2d2":
            print("Loading R2D2 model...")
            self.model = get_pretrained_model()
            self.feature_dim = 128
            mean = [0.485, 0.456, 0.406]
            std  = [0.229, 0.224, 0.225]
            self.norm_image = tvf.Compose([tvf.Normalize(mean=mean, std=std)])
        else:
            raise ValueError("Foundation model not supported")
        

    @torch.no_grad()
    def forward(self, image):
        if self.feature_type == "sp":
            features, scores = self.model(image)
            return {
                "feature_map": features,
                "scores": scores
            }
        elif self.feature_type == "r2d2":
            image = self.norm_image(image)
            features, repeatability, reliability = self.model(image)
            return {
                "feature_map": features,
                "repeatability": repeatability,
                "reliability": reliability
            }
