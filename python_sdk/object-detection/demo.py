"""Simple Object Detection Demo"""
from trueface.object_detection import ObjectRecognizer
import json
import os

#init object recognizer
object_recognition = ObjectRecognizer(ctx='cpu',
                      model_path="./tf-object_detection-mobilenet/model.trueface",
                      params_path="./tf-object_detection-mobilenet/model.params",
                      license=os.environ['TF_TOKEN'],
                      classes="./tf-object_detection-mobilenet/classes.names")

#predict single image
result = object_recognition.predict("./test.jpg")
print(result)

#predict multiple images
result = object_recognition.batch_predict(["./test.jpg", "./test.jpg"])
print(result)
