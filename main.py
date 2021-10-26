
import os

os.chdir('/content/drive/MyDrive/Github/facial-time-lapse-video')


from argparse import Namespace
import time
import sys
import pprint
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

sys.path.append(".")
sys.path.append("..")

from pixel2style2pixel.datasets import augmentations
from pixel2style2pixel.utils.common import tensor2im, log_input_image
from pixel2style2pixel.models.psp import pSp

import dlib
from pixel2style2pixel.scripts.align_all_parallel import align_face



experiment_type = 'ffhq_encode'

EXPERIMENT_DATA_ARGS = {
    "ffhq_encode": {
        "model_path": "pixel2style2pixel/pretrained_models/psp_ffhq_encode.pt",
        #"image_path": "notebooks/images/input_img.jpg",
        "image_path": "/content/drive/MyDrive/me_more_res.jpg",
        
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    },
    "ffhq_frontalize": {
        "model_path": "pretrained_models/psp_ffhq_frontalization.pt",
        #"image_path": "notebooks/images/input_img.jpg",
        "image_path": "/content/drive/MyDrive/me_frontal.jpg",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    },
    "celebs_sketch_to_face": {
        "model_path": "pretrained_models/psp_celebs_sketch_to_face.pt",
        "image_path": "notebooks/images/input_sketch.jpg",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()])
    },
    "celebs_seg_to_face": {
        "model_path": "pretrained_models/psp_celebs_seg_to_face.pt",
        "image_path": "notebooks/images/input_mask.png",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            augmentations.ToOneHot(n_classes=19),
            transforms.ToTensor()])
    },
    "celebs_super_resolution": {
        "model_path": "pretrained_models/psp_celebs_super_resolution.pt",
        "image_path": "notebooks/images/input_img.jpg",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            augmentations.BilinearResize(factors=[16]),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    },
    "toonify": {
        "model_path": "pretrained_models/psp_ffhq_toonify.pt",
        "image_path": "/content/drive/MyDrive/me_more_res.jpg",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    },
}

EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS[experiment_type]

if os.path.getsize(EXPERIMENT_ARGS['model_path']) < 1000000:
  raise ValueError("Pretrained model was unable to be downlaoded correctly!")


model_path = EXPERIMENT_ARGS['model_path']
ckpt = torch.load(model_path, map_location='cpu')

opts = ckpt['opts']
pprint.pprint(opts)

# update the training options
opts['checkpoint_path'] = model_path
if 'learn_in_w' not in opts:
    opts['learn_in_w'] = False
if 'output_size' not in opts:
    opts['output_size'] = 1024

opts = Namespace(**opts)
net = pSp(opts)
net.eval()
net.cuda()
print('Model successfully loaded!')

image_path = EXPERIMENT_DATA_ARGS[experiment_type]["image_path"]
original_image = Image.open(image_path)
if opts.label_nc == 0:
    original_image = original_image.convert("RGB")
else:
    original_image = original_image.convert("L")

original_image.resize((256, 256))

original_image.save("test.jpg")

def run_alignment(image_path):  
  predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
  aligned_image = align_face(filepath=image_path, predictor=predictor)
  print("Aligned image has shape: {}".format(aligned_image.size))
  return aligned_image

if experiment_type not in ["celebs_sketch_to_face", "celebs_seg_to_face"]:
  input_image = run_alignment(image_path)
else:
  input_image = original_image

img_transforms = EXPERIMENT_ARGS['transform']
transformed_image = img_transforms(input_image)

def run_on_batch(inputs, net):
    result_batch = net(inputs.to("cuda").float())
    return result_batch

with torch.no_grad():
    tic = time.time()
    result_image = run_on_batch(transformed_image.unsqueeze(0), net)[0]
    toc = time.time()
    print(result_image.shape)
    print('Inference took {:.4f} seconds.'.format(toc - tic))

input_vis_image = log_input_image(transformed_image, opts)
output_image = tensor2im(result_image)

res = np.concatenate([np.array(input_vis_image.resize((256, 256))),
                     np.array(output_image.resize((256, 256)))], axis=1)


res_image = Image.fromarray(res)
res_image.save("result.jpg")
