from trueface.recognition import FaceRecognizer
from trueface.video import VideoStream, QVideoStream
import cv2
import time


fr = FaceRecognizer(ctx='gpu',
                   gpu=0,
                   fd_model_path='./fd_model',
                   fr_model_path='./model-tfv2/model.trueface', 
                   params_path='./model-tfv2/model.params',
                   license=os.environ['TF_TOKEN'])

vcap = VideoStream(src=0).start()

#make output full screen
cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while(True):
    frame = vcap.read()

    #find faces
    bounding_boxes, points, chips = fr.find_faces(frame, return_chips=True, return_binary=True)
    
    if bounding_boxes is not None:
        #loop over extracted faces
        for i,chip in enumerate(chips):
            #identify each face chip
            identity = fr.identify(chip, threshold=0.3, collection='./trump_collection.npz')
            print(identity)
            #if identity, draw name
            if identity['predicted_label']:
                fr.draw_label(frame, 
                             (int(bounding_boxes[i][0]), 
                              int(bounding_boxes[i][1])), 
                             identity['predicted_label'], 2, 2)
            #else, blur face
            else:
                fr.blur_region(bounding_boxes[i], frame)
            fr.draw_box(frame, bounding_boxes[i])

    cv2.imshow('image', frame)
    if cv2.waitKey(33) == ord('q'):
        break