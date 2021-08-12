from flask import Flask
from flask import render_template
from flask import request
import os
import cv2
import numpy as np
import torch
from image_utilities import prepare_image, image_to_numpy, output_to_image, save_image_to_input_folder, save_image_to_output_folder
from models import upscale_image, interp08, RRDB_PSNR
from flask import send_file
import time


app=Flask(__name__)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
interp08 = interp08(device)
RRDB_PSNR = RRDB_PSNR(device)


@app.errorhandler(404)
def not_found(e):
	return render_template('404.html')


@app.route("/",methods=['GET','POST'])
@app.route("/quality",methods=['GET','POST'])
def upload_predict_quality():
	if request.method == 'POST':
		image_file = request.files["image"]
		if image_file:

			# 1. Save image to input folder
			image_file = save_image_to_input_folder(image_file)
			# 2. Convert image to numpy
			img_numpy = image_to_numpy(image_file)
			# 3. Prepare image for upscaling model
			img_torch = prepare_image(img_numpy, device)
			# 4. Predict/Upscale image
			output = upscale_image(img_torch, device, interp08)
			# 5. Convert model output to image 
			upscaled_image = output_to_image(output)
			# 6. Save image to output folder
			output_folder = './static/output/quality/'
			save_image_to_output_folder(image_file, upscaled_image, output_folder)
			
			return render_template('quality.html',image_name=image_file.filename)
			
	return render_template('quality.html',image_name=None)


@app.route("/similarity",methods=['GET','POST'])
def upload_predict_similarity():
	if request.method == 'POST':
		start_time = time.time()
		image_file = request.files["image"]
		if image_file:

			# 1. Save image to input folder
			image_file = save_image_to_input_folder(image_file)

			# 2. Convert image to numpy 
			img_numpy = image_to_numpy(image_file)

			# 3. Prepare image for upscaling model
			img_torch = prepare_image(img_numpy, device)

			# 4. Predict/Upscale image
			output = upscale_image(img_torch, device, RRDB_PSNR)

			# 5. Convert model output to image 
			upscaled_image = output_to_image(output)
		
			# 6. Save image to output folder
			output_folder = './static/output/similarity/'
			save_image_to_output_folder(image_file, upscaled_image, output_folder)
			
			print("--- %s seconds ---" % (time.time() - start_time))
			return render_template('similarity.html',image_name=image_file.filename)
			
	return render_template('similarity.html',image_name=None)

if __name__ == "__main__": 
	app.run(host ='0.0.0.0', port = 5001, debug = True)
