# api imports
from flask import Flask
from flask import jsonify
# image prep imports
import io 
import torchvision.transforms as transforms
from PIL import Image 
# model imports
from torchvision import models
import json

app = Flask(__name__)

model = models.densenet121(pretrained=True)
model.eval()





def transform_image(image_bytes):
	my_transforms = transforms.Compose([transforms.Resize(255),
										transforms.CenterCrop(244),
										transforms.ToTensor(),
										transforms.Normalize(
											[0.485, 0.456, 0.406],
											[0.229, 0.224, 0.225])])
	image = Image.open(io.BytesIO(image_bytes))
	return my_transforms(image).unsqueeze(0)

# with open("img059.jpg", 'rb') as f:
# 	image_bytes = f.read()
# 	tensor = transform_image(image_bytes=image_bytes)
# 	print(tensor)


imagenet_class_index = json.load(open('./static/imagenet_class_index.json'))

def get_prediction(image_bytes):
	tensor = transform_image(image_bytes=image_bytes)
	outputs = model.forward(tensor)
	_, y_hat = outputs.max(1)
	predicted_idx = str(y_hat.item())
	return imagenet_class_index[predicted_idx]

with open("img059.jpg", 'rb') as f:
	image_bytes = f.read()
	print(get_prediction(image_bytes = image_bytes))


@app.route('/')
def hello():
	return 'Greetings, Traveller!'

@app.route('/predict', methods=['GET','POST'])
def predict():
	return jsonify({'class_id': 'IMAGE_NET_XXX', 'class_name': 'Cat'})





if __name__ == '__main__':
    app.run()