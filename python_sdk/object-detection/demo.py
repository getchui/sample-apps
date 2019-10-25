"""Simple Object Detection Demo"""
from trueface.object_detection import ObjectRecognizer
import argparse
import os
import sys
import cv2
from PIL import Image

def predict_image(frame, recognizer):

    #predict single image
    result = recognizer.predict(frame)
    #print(result)
    return result

    #predict multiple images
    #result = recognizer.batch_predict(["./test.jpg", "./test.jpg"])

    #print(result)

def draw_on_image(frame, result):
    scores = result['scores']
    classes = result['classes']
    boxes = result['boxes']

    for i, score in enumerate(scores):
        if score < 0.5:
            continue

        label = "{}".format(classes[i])
        # cast into integers
        box = [ int(element) for element in boxes[i]]
        cv2.rectangle(frame, (box[0], box[1]),
                      (box[2], box[3]),
                      (0, 255, 0), 2)
        cv2.putText(frame, label, (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image",
                    help="path to input image")
    ap.add_argument("-v", "--video",
                    help="path to input video")

    args = vars(ap.parse_args())

    if not any(args.values()):
        print("Usage: {} [-v <path to video> | -i <path to image>]".format(sys.argv[0]))
        sys.exit(1)

    obj_rec = ObjectRecognizer(ctx='gpu',
                                          model_path="./tf-object_detection-mobilenet/model.trueface",
                                          params_path="./tf-object_detection-mobilenet/model.params",
                                          license=os.environ['TF_TOKEN'],
                                          classes="./tf-object_detection-mobilenet/classes.names")

    if args['image']:
        # make sure it's an actual image
        try:
            frame = cv2.imread(args['image'])
        except:
            print("ERROR: Please enter a video or image file")
            sys.exit(1)
        result = predict_image(args['image'], obj_rec)
        frame = draw_on_image(frame, result)

        file_without_ending = args['image'].split('.')[:-1][0]
        file_ending = args['image'].split('.')[-1]
        output_file = "{}.detected.{}".format(file_without_ending, file_ending)

        cv2.imwrite(output_file, frame)

    if args['video']:
        try:
            cap = cv2.VideoCapture(args['video'])
        except:
            print("ERROR: Please enter a video or image file")
            sys.exit(1)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        file_without_ending = args['video'].split('.')[:-1][0]
        file_ending = args['video'].split('.')[-1]
        output_file = "{}.detected.{}".format(file_without_ending, file_ending)
        out = cv2.VideoWriter(output_file ,fourcc, fps, (width,height))

        i = 0
        while(cap.isOpened()):
            i += 1
            ret, frame = cap.read()
            if not ret:
                break
            if cv2.waitKey(18) & 0xFF == ord('q'):
                break
            result = predict_image(frame, obj_rec)
            frame = draw_on_image(frame, result)
            out.write(frame)
            cv2.imshow('Trueface.ai', frame)


        cap.release()
        out.release()
        cv2.destroyAllWindows()




if __name__ == "__main__":
    main()
