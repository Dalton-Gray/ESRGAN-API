import requests

resp = requests.post("http://localhost:5000/predict",
                     files={"file": open('img059.jpg','rb')})

result = resp.json()
print("Class Id:{}, Class Name: {}".format(result["class_id"],result["class_name"]))
