import os.path as osp
import glob
import numpy as np
import torch
from nn import RRDBNet_arch as arch


def RRDB_PSNR(device):
	model_path = 'nn/RRDB_PSNR_x4.pth' 

	model = arch.RRDBNet(3, 3, 64, 23, gc=32)
	model.load_state_dict(torch.load(model_path), strict=True)
	model.eval()
	model = model.to(device)
	return model


def interp08(device):
	model_path = 'nn/interp_08.pth' 

	model = arch.RRDBNet(3, 3, 64, 23, gc=32)
	model.load_state_dict(torch.load(model_path), strict=True)
	model.eval()
	model = model.to(device)
	return model

def upscale_image(img_torch, device, model):
	try:
		with torch.no_grad():
			output = model(img_torch).data.squeeze().float().cpu().clamp_(0, 1).numpy()
	
	except Exception:
		return "Exception when upscaling image"
		
	return output
