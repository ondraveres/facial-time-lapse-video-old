
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
        "image1_path": "/content/drive/MyDrive/me_more_res.jpg",
        "image2_path": "/content/drive/MyDrive/me_frontal.jpg",
        
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

image_path1 = EXPERIMENT_DATA_ARGS[experiment_type]["image1_path"]
original_image1 = Image.open(image_path1)
if opts.label_nc == 0:
    original_image1 = original_image1.convert("RGB")
else:
    original_image1 = original_image1.convert("L")

original_image1.resize((256, 256))

image_path2 = EXPERIMENT_DATA_ARGS[experiment_type]["image2_path"]
original_image2 = Image.open(image_path2)
if opts.label_nc == 0:
    original_image2 = original_image2.convert("RGB")
else:
    original_image2 = original_image2.convert("L")

original_image2.resize((256, 256))

def run_alignment(image_path):  
  predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
  aligned_image = align_face(filepath=image_path, predictor=predictor)
  print("Aligned image has shape: {}".format(aligned_image.size))
  return aligned_image


input_image1 = run_alignment(image_path1)
input_image2 = run_alignment(image_path2)

img_transforms = EXPERIMENT_ARGS['transform']
transformed_image1 = img_transforms(input_image1)
transformed_image2 = img_transforms(input_image2)



with torch.no_grad():
    tic = time.time()
    result_image1, result_latent1 = net(transformed_image1.unsqueeze(0).to("cuda").float(), return_latents = True)
    result_image2, result_latent2 = net(transformed_image2.unsqueeze(0).to("cuda").float(), return_latents = True)
    images = []
    for i in range(10):
        t1 = i/10
        t2 = 1- t1
        avg = result_latent1* t1 + result_latent2*t2
        mixed_image, result_latent = net(avg, return_latents = True, input_code = True)
        images.append(mixed_image)
    toc = time.time()
    print('Inference took {:.4f} seconds.'.format(toc - tic))

input_vis_image = log_input_image(transformed_image1, opts)
output_images = []
for img in images:
    output_images.append(np.array(tensor2im(img[0]).resize((256, 256))))


res = np.concatenate(output_images, axis=1)


res_image = Image.fromarray(res)
for index, output_image in enumerate(output_images):
   res_image = Image.fromarray(output_image)
   res_image.save(str(index) +"result.jpg")
