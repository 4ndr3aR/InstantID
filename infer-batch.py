#!/usr/bin/env python3
import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw

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

# 1. Calculate areas
def calculate_area(bbox):
    x_min, y_min, x_max, y_max = bbox
    return (x_max - x_min) * (y_max - y_min)

# 2. Draw on PIL Image
def draw_bboxes(img, bboxes, colors=['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta', 'black', 'white']):
	# Open image
	#img = Image.open(image_path)
	draw = ImageDraw.Draw(img)
	
	# Draw each bounding box
	for i, bbox in enumerate(bboxes):
		# Convert coordinates to integers and tuple format
		x_min, y_min, x_max, y_max = map(int, bbox)
		color = colors[i % len(colors)]
		draw.rectangle(
			[(x_min, y_min), (x_max, y_max)],
			outline=color,
			width=3
		)
		print(f'Drawing rectangle: {bbox} with color: {color}')
	
	return img


def load_face(fn, debug=False):
	img = Image.open(fn)
	print(f'Loading face: {Path(fn).name} - image size: {img.size}')

	face_image = load_image(img)
	face_image = resize_img(face_image)

	print(f'Face image size: {face_image.size}')

	face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
	n_faces = len(face_info)
	print(f'Found {n_faces} faces in {Path(fn).name}')
	# 'bbox': array([187.35774, 159.91435, 537.2371 , 638.2024 ]
	# 'bbox': array([1035.5541, 59.45189, 1327.8179 , 470.8462 ]
	# 'det_score': np.float32(0.8763549)
	# 'det_score': np.float32(0.67827106)
	# 'gender': np.int64(0)
	# 'gender': np.int64(1)
	# 'age': 32
	# 'age': 31
	# 'kps': array([[341.6794 , 349.55255],
	#       [490.8568 , 369.94705],
	#       [446.57422, 432.40735],
	#       [355.07123, 521.9619 ],
	#       [466.65866, 537.26965]]


	#Loading face: 0000578.jpg - image size: (500, 375)
	#Face image size: (1280, 960)
	#Found 1 faces
	#face_info[0][age]      : 9
	#face_info[0][gender]   : female
	#face_info[0][det_score]: 0.7817953824996948
	#face_info[0][bbox]     : [ 595.3876   336.22235 1115.4376   997.30316]
	#face_info[0][area]     : 343795.09375

	#Loading face: 0000455.jpg - image size: (500, 375)
	#Face image size: (1280, 960)
	#Found 2 faces
	#face_info[0][age]      : 25
	#face_info[0][gender]   : female
	#face_info[0][det_score]: 0.808878481388092
	#face_info[0][bbox]     : [ 934.7908  396.7399 1280.1788  838.1504]
	#face_info[0][area]     : 152457.921875
	#face_info[1][age]      : 29
	#face_info[1][gender]   : female
	#face_info[1][det_score]: 0.7108510732650757
	#face_info[1][bbox]     : [376.503   313.85577 681.81934 645.2223 ]
	#face_info[1][area]     : 101171.6171875

	#Loading face: 0000449.jpg - image size: (375, 500)
	#Face image size: (960, 1280)
	#Found 3 faces
	#face_info[0][age]      : 26
	#face_info[0][gender]   : female
	#face_info[0][det_score]: 0.9073766469955444
	#face_info[0][bbox]     : [258.0275  538.6953  364.71014 675.03784]
	#face_info[0][area]     : 14545.3818359375
	#face_info[1][age]      : 26
	#face_info[1][gender]   : female
	#face_info[1][det_score]: 0.8729305863380432
	#face_info[1][bbox]     : [481.24216 574.84705 610.04846 737.2518 ]
	#face_info[1][area]     : 20918.751953125
	#face_info[2][age]      : 43
	#face_info[2][gender]   : female
	#face_info[2][det_score]: 0.8362807631492615
	#face_info[2][bbox]     : [744.5608 503.105  880.8741 690.3573]
	#face_info[2][area]     : 25524.9765625

	#Loading face: 0000369.jpg - image size: (452, 339)
	#Face image size: (1280, 960)
	#Found 2 faces
	#face_info[0][age]      : 4
	#face_info[0][gender]   : male
	#face_info[0][det_score]: 0.7998645305633545
	#face_info[0][bbox]     : [290.9958  329.63104 533.794   613.28345]
	#face_info[0][area]     : 68870.296875
	#face_info[1][age]      : 2
	#face_info[1][gender]   : male
	#face_info[1][det_score]: 0.6979250311851501
	#face_info[1][bbox]     : [743.5659  337.15533 953.0863  592.8567 ]
	#face_info[1][area]     : 53574.6484375

	#Loading face: 0000341.jpg - image size: (500, 375)
	#Face image size: (1280, 960)
	#Found 2 faces
	#face_info[0][age]      : 33
	#face_info[0][gender]   : female
	#face_info[0][det_score]: 0.6943148970603943
	#face_info[0][bbox]     : [194.8554  326.64514 561.1378  687.39307]
	#face_info[0][area]     : 132135.625
	#face_info[1][age]      : 29
	#face_info[1][gender]   : female
	#face_info[1][det_score]: 0.6711225509643555
	#face_info[1][bbox]     : [ 721.6593  270.7194 1056.9781  647.5462]
	#face_info[1][area]     : 126357.1328125

	#Loading face: 0000312.jpg - image size: (500, 332)
	#Face image size: (1280, 832)
	#Found 3 faces
	#face_info[0][age]      : 56
	#face_info[0][gender]   : female
	#face_info[0][det_score]: 0.895619809627533
	#face_info[0][bbox]     : [608.5094 223.2362 726.9273 382.7912]
	#face_info[0][area]     : 18894.16796875
	#face_info[1][age]      : 70
	#face_info[1][gender]   : female
	#face_info[1][det_score]: 0.8937674164772034
	#face_info[1][bbox]     : [264.73187 256.75763 400.95172 445.40475]
	#face_info[1][area]     : 25697.482421875
	#face_info[2][age]      : 71
	#face_info[2][gender]   : female
	#face_info[2][det_score]: 0.8793427348136902
	#face_info[2][bbox]     : [841.94653 265.58212 948.15344 403.6313 ]
	#face_info[2][area]     : 14661.775390625

	#Loading face: 0000256.jpg - image size: (333, 500)
	#Face image size: (832, 1280)
	#Found 6 faces
	#face_info[0][age]      : 42
	#face_info[0][gender]   : female
	#face_info[0][det_score]: 0.9029721021652222
	#face_info[0][bbox]     : [317.55173 434.65982 409.30124 563.9895 ]
	#face_info[0][area]     : 11865.935546875
	#face_info[1][age]      : 46
	#face_info[1][gender]   : male
	#face_info[1][det_score]: 0.9011645913124084
	#face_info[1][bbox]     : [441.07666 458.23315 533.0517  575.4988 ]
	#face_info[1][area]     : 10785.509765625
	#face_info[2][age]      : 32
	#face_info[2][gender]   : male
	#face_info[2][det_score]: 0.870965838432312
	#face_info[2][bbox]     : [667.6632  399.64606 710.393   461.50406]
	#face_info[2][area]     : 2643.179931640625
	#face_info[3][age]      : 26
	#face_info[3][gender]   : female
	#face_info[3][det_score]: 0.6505197286605835
	#face_info[3][bbox]     : [782.8309  450.5908  803.59644 478.04962]
	#face_info[3][area]     : 570.1981201171875
	#face_info[4][age]      : 29
	#face_info[4][gender]   : male
	#face_info[4][det_score]: 0.6010982990264893
	#face_info[4][bbox]     : [ 38.237938 585.4833    50.822636 601.8659  ]
	#face_info[4][area]     : 206.17044067382812
	#face_info[5][age]      : 29
	#face_info[5][gender]   : female
	#face_info[5][det_score]: 0.5899161696434021
	#face_info[5][bbox]     : [600.2865  466.35168 615.6171  488.49084]
	#face_info[5][area]     : 339.4072265625

	#Loading face: 0000146.jpg - image size: (500, 333)
	#Face image size: (1280, 832)
	#Found 2 faces
	#face_info[0][age]      : 36
	#face_info[0][gender]   : male
	#face_info[0][det_score]: 0.8491812348365784
	#face_info[0][bbox]     : [303.37067 316.75925 532.38794 631.45337]
	#face_info[0][area]     : 72070.390625
	#face_info[1][age]      : 61
	#face_info[1][gender]   : male
	#face_info[1][det_score]: 0.8030573129653931
	#face_info[1][bbox]     : [529.7052  228.32375 878.59686 705.4302 ]
	#face_info[1][area]     : 166458.46875

	#Loading face: 0000143.jpg - image size: (276, 218)
	#Face image size: (1280, 960)
	#Found 2 faces
	#face_info[0][age]      : 72
	#face_info[0][gender]   : male
	#face_info[0][det_score]: 0.8558583855628967
	#face_info[0][bbox]     : [ 905.088    174.95691 1086.5654   403.53546]
	#face_info[0][area]     : 41481.84375
	#face_info[1][age]      : 68
	#face_info[1][gender]   : male
	#face_info[1][det_score]: 0.8269023299217224
	#face_info[1][bbox]     : [301.3625    91.138016 486.01376  329.60632 ]
	#face_info[1][area]     : 44033.4765625

	#Loading face: 0000122.jpg - image size: (500, 375)
	#Face image size: (1280, 960)
	#Found 3 faces
	#face_info[0][age]      : 47
	#face_info[0][gender]   : female
	#face_info[0][det_score]: 0.9011006355285645
	#face_info[0][bbox]     : [648.11646 268.23007 733.44574 384.6452 ]
	#face_info[0][area]     : 9933.6201171875
	#face_info[1][age]      : 72
	#face_info[1][gender]   : male
	#face_info[1][det_score]: 0.8780383467674255
	#face_info[1][bbox]     : [834.2933  271.5739  917.4496  383.37616]
	#face_info[1][area]     : 9297.0625
	#face_info[2][age]      : 60
	#face_info[2][gender]   : male
	#face_info[2][det_score]: 0.8733167052268982
	#face_info[2][bbox]     : [553.0698  328.75064 635.87427 437.54486]
	#face_info[2][area]     : 9008.64453125

	#Loading face: 0000084.jpg - image size: (500, 333)
	#Face image size: (1280, 832)
	#Found 2 faces
	#face_info[0][age]      : 32
	#face_info[0][gender]   : male
	#face_info[0][det_score]: 0.8763548731803894
	#face_info[0][bbox]     : [187.35774 159.91435 537.2371  638.2024 ]
	#face_info[0][area]     : 167343.125
	#face_info[1][age]      : 31
	#face_info[1][gender]   : female
	#face_info[1][det_score]: 0.6782710552215576
	#face_info[1][bbox]     : [1035.5541    59.45189 1327.8179   470.8462 ]
	#face_info[1][area]     : 120235.65625

	if n_faces == 0:
		# we literally found nothing in the image
		return None, None

	faces  = []
	bboxes = []
	for i, face in enumerate(face_info):
		bboxes.append(face_info[i]['bbox'])
		face_info[i]["area"] = calculate_area(face["bbox"])
		print(f'face_info[{i}][age]      : {face["age"]}')
		print(f'face_info[{i}][gender]   : {"female" if face["gender"] == 0 else "male"}')
		print(f'face_info[{i}][det_score]: {face["det_score"]}')
		print(f'face_info[{i}][bbox]     : {face["bbox"]}')
		print(f'face_info[{i}][area]     : {face["area"]}')
		if calculate_area(face["bbox"]) > 100000:
			faces.append(face_info[i])

	if debug or True:
		# Example usage
		image_with_bboxes = draw_bboxes(face_image, bboxes)
		#image_with_bboxes.show()  # or save with .save('output.jpg')
		image_with_bboxes.save('/tmp/output.jpg')

	if debug:
		print(f'face_info: {face_info}')
		print(f'face_info[0][bbox]: {face_info[0]["bbox"]}')

	n_faces = len(faces)
	if n_faces == 0:
		# our face detector found something, but it was too small to be used
		return None, None

	#face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] # only use the maximum face
	face_info = sorted(face_info, key=lambda x:x['area'])[-1]		# only use the larger face
	if n_faces > 1 and False:
		return None, None
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
