from flask import Flask
from flask import render_template
from flask import request
import os
import cv2
from PIL import Image
from prepare_images import mse, compare_images
import os.path as osp
import glob
import numpy as np
import torch
import RRDBNet_arch as arch


app=Flask(__name__)


UPLOAD_FOLDER = './static/input'

OUTPUT_FOLDER = './static/output'

model_path = 'models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
# device = torch.device('cpu')

model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)


#pred=0.0
@app.route("/",methods=['GET','POST'])
def upload_predict_ESRGAN():
	if request.method == 'POST':
		image_file = request.files["image"]
		if image_file:
			

			image_location = os.path.join(UPLOAD_FOLDER,image_file.filename)
			image_file.save(image_location)

			img = cv2.imread(image_location, cv2.IMREAD_COLOR)
			img = img * 1.0 / 255

			# converts image from array to tensor
			img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
			img_LR = img.unsqueeze(0)
			img_LR = img_LR.to(device)
			
			with torch.no_grad():
				# upscales and image converts it to array
				output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
			output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
			output = (output * 255.0).round()
			cv2.imwrite('./static/output/{:s}_ESRGAN.png'.format(image_file.filename.split(".")[0]), output)

			# scores = compare_images(output, img)

			return render_template('index.html',image_name=image_file.filename)#, mse=scores[0] )#psnr=scores[1][0],mse=scores[1][1],ssim=scores[1][2])
			
	return render_template('index.html',image_name=None,psnr=0,mse=0,ssim=0)



if __name__ == "__main__": 
	app.run(host ='0.0.0.0', port = 5001, debug = True)
