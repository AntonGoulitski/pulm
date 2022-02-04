import pprint
import os
import requests

print(os.getcwd())
DETECTION_URL = "http://localhost:5000"

for file in os.listdir():
    if file.endswith('.png'):
        TEST_IMAGE = file
        image_data = open(TEST_IMAGE, "rb").read()
        response = requests.post(DETECTION_URL, files={"image": image_data}).json()
        pprint.pprint(response)


