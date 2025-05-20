#!/usr/bin/env python3
import cv2
import torch
import numpy as np
from PIL import Image

from diffusers.utils import load_image
from diffusers.models import ControlNetModel

from insightface.app import FaceAnalysis
from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps

from pathlib import Path

import argparse

def resize_img(input_image, max_side=1280, min_side=1024, size=None, 
               pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):

    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio*w), round(ratio*h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image

def load_face(fn, debug=False):
	img = Image.open(fn)
	print(f'Loading face: {fn} - image size: {img.size}')

	face_image = load_image(img)
	face_image = resize_img(face_image)

	face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
	n_faces = len(face_info)
	print(f'Found {n_faces} faces')
	if debug:
		print(f'face_info: {face_info}')
		print(f'face_info[0][bbox]: {face_info[0]["bbox"]}')
	if n_faces == 0:
		return None, None
	if n_faces > 1:
		return None, None
	face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] # only use the maximum face
	face_emb = face_info['embedding']
	face_kps = draw_kps(face_image, face_info['kps'])

	return face_emb, face_kps

def infer_face(pipe, face_emb, face_kps, prompt, n_prompt, out_fn,
        controlnet_conditioning_scale=0.8,
        ip_adapter_scale=0.8,
        num_inference_steps=30,
        guidance_scale=4,
):
	print(f"Infering face: {prompt}")
	image = pipe(
		prompt=prompt,
		negative_prompt=n_prompt,
		image_embeds=face_emb,
		image=face_kps,
		controlnet_conditioning_scale=controlnet_conditioning_scale,
		ip_adapter_scale=ip_adapter_scale,
		num_inference_steps=num_inference_steps,
		guidance_scale=guidance_scale,
	).images[0]

	print(f"Saving image: {Path(out_fn).name}")
	image.save(out_fn)


if __name__ == "__main__":

	argparser = argparse.ArgumentParser()
	argparser.add_argument('--input_dir',	type=str, default='.')
	argparser.add_argument('--output_dir',	type=str, default='/tmp')
	argparser.add_argument('--ext',		type=str, default='jpg')
	args = argparser.parse_args()

	search_path = Path(args.input_dir)
	flist = list(search_path.glob('*.' + args.ext))
	print(f'\nFound {len(flist)} images in {search_path}...\n')


	# Load face encoder
	app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
	app.prepare(ctx_id=0, det_size=(640, 640))

	# Path to InstantID models
	face_adapter = f'./checkpoints/ip-adapter.bin'
	controlnet_path = f'./checkpoints/ControlNetModel'

	# Load pipeline
	controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)

	base_model_path = 'stabilityai/stable-diffusion-xl-base-1.0'

	pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
		base_model_path,
		controlnet=controlnet,
		torch_dtype=torch.float16,
	)
	pipe.cuda()
	pipe.load_ip_adapter_instantid(face_adapter)

	# Infer setting
	#prompt = "analog film photo of a man. faded film, desaturated, 35mm photo, grainy, vignette, vintage, Kodachrome, Lomography, stained, highly detailed, found footage, masterpiece, best quality"
	#prompt = "marble statue of man"
	prompt = "marble statue"
	#n_prompt = "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured (lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch,deformed, mutated, cross-eyed, ugly, disfigured"
	n_prompt = "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured, helmet, eyeglasses, beard, hat, brown hair, blonde hair, black hair, pink face, brown face, black face, white skin, brown skin, black skin"

	controlnet_conditioning_scale=0.8
	ip_adapter_scale=0.8
	num_inference_steps=60
	guidance_scale=6

	out_path = Path(args.output_dir)
	out_path.mkdir(parents=True, exist_ok=True)

	for fn in search_path.glob('*.' + args.ext):
		print(f'Reading image {str(fn.name)}...')

		out_fn = out_path / (fn.stem + '-result.jpg')

		#fn = "./examples/yann-lecun_resize.jpg"
		face_emb, face_kps = load_face(fn)
		if face_emb is None or face_kps is None:
			print(f'Failed to load face: {fn} - no faces or several faces found...')
			continue

		infer_face(pipe, face_emb, face_kps, prompt, n_prompt, out_fn,
				controlnet_conditioning_scale=controlnet_conditioning_scale,
				ip_adapter_scale=ip_adapter_scale,
				num_inference_steps=num_inference_steps,
				guidance_scale=guidance_scale
			)

	'''
	for i in range(5):
		for j in range(5):
			for k in range(5):
				for w in range(5):
					controlnet_conditioning_scale=i*0.1+0.5
					ip_adapter_scale=j*0.1+0.5
					num_inference_steps=20+k*10
					guidance_scale=w+3
					print(f'controlnet_conditioning_scale={controlnet_conditioning_scale}, ip_adapter_scale={ip_adapter_scale}, num_inference_steps={num_inference_steps}, guidance_scale={guidance_scale}')
					infer_face(pipe, face_emb, face_kps, prompt, n_prompt,
							f'result-{i}-{j}-{k}-{w}.jpg',
							controlnet_conditioning_scale=controlnet_conditioning_scale,
							ip_adapter_scale=ip_adapter_scale,
							num_inference_steps=num_inference_steps,
							guidance_scale=guidance_scale
					)
	'''

	'''
	face_image = load_image("./examples/yann-lecun_resize.jpg")
	face_image = resize_img(face_image)

	face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
	face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] # only use the maximum face
	face_emb = face_info['embedding']
	face_kps = draw_kps(face_image, face_info['kps'])

	image = pipe(
		prompt=prompt,
		negative_prompt=n_prompt,
		image_embeds=face_emb,
		image=face_kps,
		controlnet_conditioning_scale=0.8,
		ip_adapter_scale=0.8,
		num_inference_steps=30,
		guidance_scale=4,
	).images[0]

	image.save('result.jpg')
	'''
