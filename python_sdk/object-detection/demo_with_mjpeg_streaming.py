"""Demo with MJPEG streaming"""
from trueface.recognition import FaceRecognizer
from trueface.video import VideoStream, QVideoStream
from trueface.server import create_server
from trueface.object_detection import ObjectRecognizer
from multiprocessing import Process, Queue, Value
import os

#init object recognizer
object_recognition = ObjectRecognizer(ctx='gpu',
                      model_path="./tf-object_detection-mobilenet/model.trueface",
                      params_path="./tf-object_detection-mobilenet/model.params",
                      license=os.environ['TF_TOKEN'],
                      classes="./tf-object_detection-mobilenet/classes.names")

#simple streaming server
def start_server(port, q):
    """starts a simple MJPEG streaming server"""
    app = create_server()
    app.config['q'] = q
    p = Process(target=app.run, kwargs={"host":'0.0.0.0',"port":port, "threaded":True})
    p.daemon = True
    p.start()

#initialize video capture from your webcam
cap = VideoStream(src=0).start()

#create a queue
q = Queue(maxsize=10)

#start streaming server
start_server(8086, q)
print('navigate to http://localhost:8086/ to view your stream.')

while(True):
	frame = cap.read()
	result = object_recognition.predict(frame)
	print(result)
	for i, box in enumerate(result['boxes']):
		object_recognition.draw_label(frame, (int(box[0]), int(box[1])), result['classes'][i])
		object_recognition.draw_box(frame, box)
	if q.full():
		q.get()
	q.put(frame)