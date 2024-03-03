import os
from cog import BasePredictor, Input, Path
from typing import List
import sys
sys.path.append('/content/DSINE')
os.chdir('/content/DSINE')

import os
import sys
import glob
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import utils.utils as utils

def test_samples(img_path, model, intrins=None, device='cpu'):
    # normalize
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    with torch.no_grad():
        print(img_path)
        ext = os.path.splitext(img_path)[1]
        img = Image.open(img_path).convert('RGB')
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
        _, _, orig_H, orig_W = img.shape

        # zero-pad the input image so that both the width and height are multiples of 32
        l, r, t, b = utils.pad_input(orig_H, orig_W)
        img = F.pad(img, (l, r, t, b), mode="constant", value=0.0)
        img = normalize(img)

        intrins_path = img_path.replace(ext, '.txt')
        if os.path.exists(intrins_path):
            # NOTE: camera intrinsics should be given as a txt file
            # it should contain the values of fx, fy, cx, cy
            intrins = utils.get_intrins_from_txt(intrins_path, device=device).unsqueeze(0)
        else:
            # NOTE: if intrins is not given, we just assume that the principal point is at the center
            # and that the field-of-view is 60 degrees (feel free to modify this assumption)
            intrins = utils.get_intrins_from_fov(new_fov=60.0, H=orig_H, W=orig_W, device=device).unsqueeze(0)

        intrins[:, 0, 2] += l
        intrins[:, 1, 2] += t

        pred_norm = model(img, intrins=intrins)[-1]
        pred_norm = pred_norm[:, :, t:t+orig_H, l:l+orig_W]

        # save to output folder
        # NOTE: by saving the prediction as uint8 png format, you lose a lot of precision
        # if you want to use the predicted normals for downstream tasks, we recommend saving them as float32 NPY files
        pred_norm_np = pred_norm.cpu().detach().numpy()[0,:,:,:].transpose(1, 2, 0) # (H, W, 3)
        pred_norm_np = ((pred_norm_np + 1.0) / 2.0 * 255.0).astype(np.uint8)
        im = Image.fromarray(pred_norm_np)
        im.save('/content/test.png')

class Predictor(BasePredictor):
    def setup(self) -> None:
        from models.dsine import DSINE
        self.device = torch.device('cuda')
        self.model = DSINE().to(self.device)
        self.model.pixel_coords = model.pixel_coords.to(self.device)
        self.model = utils.load_checkpoint('/content/DSINE/checkpoints/dsine.pt', model)
        self.model.eval()
    def predict(
        self,
        image: Path = Input(description="Input Image"),
    ) -> Path:
        test_samples(str(image), self.model, intrins=None, device=self.device)
        return Path('/content/test.png')