"""Main Face Recognition Processor"""
from trueface.recognition import FaceRecognizer
from trueface.video import VideoStream, QVideoStream
from trueface.server import create_server
import cv2
import time
import configparser
from multiprocessing import Process, Queue, Value
import traceback
import requests

#configuration
config = configparser.ConfigParser()
config.read('config.ini')
url = config['config']['url']
url = int(url) if len(url) == 1 else url
threshold = config['config']['threshold']
webhook_url = config['config']['webhook_url']
webhook_holdout = config['config']['webhook_holdout']
license = config['config']['license']


#webhook function
def webhook(identity):
    """performs webhook"""
    data = []
    r = requests.post(webhook_url)
    print r.text

#simple streaming server
def start_server(port, q):
    """starts a simple MJPEG streaming server"""
    app = create_server()
    app.config['q'] = q
    p = Process(target=app.run, kwargs={"host":'0.0.0.0',"port":port, "threaded":True})
    p.daemon = True
    p.start()

#last webhook timestamp for holdout
last_webhook_timestamp = 0

def stream_capture(q, fr_q, fr_q_results):
  #start camera capture
  vcap = VideoStream(src=url).start()

  #main loop
  fr = FaceRecognizer(ctx='gpu',
                     fd_model_path='./fd_model',
                     gpu=0,
                     license=license)
  fr_results = None
  while(True):
      try:
          frame = vcap.read()
          #initialize FR class
          
          if not fr_q_results.empty():
            fr_results = fr_q_results.get_nowait()
          if fr_results:
              for i,box in enumerate(fr_results['boxes']):
                  fr.draw_box(frame, box)
                  if fr_results['labels'][i]['predicted_label'] is not None:
                      fr.draw_label(frame, 
                           (int(box[0]), 
                            int(box[1])), 
                           fr_results['labels'][i]['predicted_label'], 2, 2)

          #pass frame to streaming service and fr queue
          if q.full():
              q.get()
          if fr_q.full():
              fr_q.get()
          q.put(frame)
          fr_q.put(frame)
          # time.sleep(0.01)
      except Exception as e:
        print traceback.format_exc()

def fr_process(fr_q, fr_q_results):
  
  #initialize FR class
  fr = FaceRecognizer(ctx='gpu',
                     fd_model_path='./fd_model',
                     fr_model_path='./model-tfv2/model.trueface', 
                     params_path='./model-tfv2/model.params',
                     gpu=0,
                     license=license)

  #main loop
  while(True):
      try:
          frame = fr_q.get()
          timestamp = time.time()
          bounding_boxes, points, chips = fr.find_faces(frame, return_chips=True, return_binary=True)
          fr_results = {"labels":[]}
          if bounding_boxes is not None:
              for i,chip in enumerate(chips):
                  identity = fr.identify(chip, threshold=float(threshold), collection='./trueface-collection.npz')
                  # print(identity)
                  fr_results['labels'].append(identity)
                      # if int(config['config']['webhook']) == 1 and \
                      #   time.time() - last_webhook_timestamp > int(webhook_holdout):
                      #     p = Process(target=webhook, args=(identity,))
                      #     p.daemon = True
                      #     p.start()
                      #     last_webhook_timestamp = time.time()
              fr_results['boxes'] = bounding_boxes
              fr_q_results.put(fr_results)
          print 'face detect time', time.time() - timestamp
            
      except Exception as e:
        print traceback.format_exc()

processes = []

fr_q = Queue(maxsize=10)
fr_q_results = Queue(maxsize=10)
q = Queue(maxsize=10)


p = Process(target=fr_process, args=(fr_q, fr_q_results))
processes.append(p)
p.daemon = True
p.start()

p = Process(target=stream_capture, args=(q, fr_q, fr_q_results))
processes.append(p)
p.daemon = True
p.start()


start_server(8086, q)

while len(processes) > 0:
    for i,p in enumerate(processes):
        time.sleep(0.5)
        print p.exitcode
        if p.exitcode is None and not p.is_alive(): # Not finished and not running
            # Do your error handling and restarting here assigning the new process to processes[n]
            print(a, 'is gone as if never born!')
            del processes[i] # Removed finished items from the dictionary 
            del servers[i]
        else:
            print (p, 'finished')
            p.join() # Allow tidyup
            del processes[i] # Removed finished items from the dictionary 
print ('FINISHED')

