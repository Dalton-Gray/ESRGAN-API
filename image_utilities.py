import torch
import numpy as np
import cv2
import os
from PIL import Image
import PIL


def prepare_image(img_numpy, device):
	try:
		img_torch = torch.from_numpy(np.transpose(img_numpy[:, :, [2, 1, 0]], (2, 0, 1))).float()
		img_torch = img_torch.unsqueeze(0)
		img_torch = img_torch.to(device)
	except TypeError:		
		return "TypeError when preparing image for upscaling"
	except Exception:
		return "Exception when preparing image for upscaling" 

	return img_torch

def image_to_numpy(image_file):	
	try:
		image_location = './static/input/' 
		image_location += os.path.join(image_file.filename)
		img_numpy = cv2.imread(image_location, cv2.IMREAD_COLOR)
		img_numpy = img_numpy * 1.0 / 255
	except AttributeError:
		return "AttributeError when converting image to numpy"
	except Exception:
		return "Exception when converting image to numpy" 


	return img_numpy

def output_to_image(img_numpy):
	try:
		image_file = np.transpose(img_numpy[[2, 1, 0], :, :], (1, 2, 0))
		image_file = (image_file * 255.0).round()
	except TypeError as error:
		return "TypeError when converting model output to image"
	except Exception:
		return "Exception when converting model output to image" 

	return image_file

def save_image_to_input_folder(image_file):
	image_location = './static/input/' 
	image_location += os.path.join(image_file.filename)
	image_file.save(image_location)

	return image_file

def save_image_to_output_folder(image_file, upscaled_image, output_folder):
	try:
		filename = output_folder + image_file.filename
		print("output_folder", output_folder)
		print("output filename", filename)
		cv2.imwrite(filename, upscaled_image)
	except TypeError:
		return "TypeError when saving image to folder" 
	except AttributeError:
		return "AttributeError when saving image to folder"
	except Exception:
		return "Exception when saving image to folder"
	
	return "Image saved to folder"