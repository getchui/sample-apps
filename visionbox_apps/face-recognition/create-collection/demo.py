import base64
import requests
from imutils import paths


base_url = "http://localhost:8085"

images = paths.list_images("./collection")

for image in images:
    data = {
        "collection_id":"office1",
        "namespace":"jakarta",
        "label":image.split("/")[-2],
        "source":base64.b64encode(open(image, 'rb').read())
    }

    r = requests.post(base_url+'/enroll', json=data)
    print(r.json())