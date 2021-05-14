#!/usr/bin/env python3

import gi
import cv2
import tfsdk
import os
import argparse
from threading import Thread
from threading import Thread, Lock
from time import sleep
import datetime

gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GLib

# Utility function for drawing a label on the image.
def draw_label(image, point, label, color_code = (194,134,58),
               font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=1.0, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x_label, y_label = point
    cv2.rectangle(
        image,
        (x_label, y_label - size[1] - 10),
        (x_label + size[0], y_label),
        color_code,
        cv2.FILLED)

    cv2.putText(
        image, label.capitalize(), (x_label, y_label - 5), font, font_scale,
        (0, 0, 0), thickness, cv2.LINE_AA)

# Utility function for drawing a rectangle on the image.
def draw_rectangle(frame, bounding_box, color_code = (194,134,58)):
    cv2.rectangle(frame,
                  (int(bounding_box.top_left.x), int(bounding_box.top_left.y)),
                  (int(bounding_box.bottom_right.x), int(bounding_box.bottom_right.y)), color_code, 3)

# Sensor Factory class which inherits the GstRtspServer base class and add
# properties to it.
class SensorFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, camera_thread, **properties):
        super(SensorFactory, self).__init__(**properties)
        
        self.camera_thread = camera_thread
        width, height = self.camera_thread.getFrameDims()

        self.timestamp = 0
        self.number_frames = 0

        self.launch_string = 'appsrc name=source is-live=true block=true format=GST_FORMAT_TIME ' \
                             'caps=video/x-raw,format=BGR,width={},height={} ' \
                             '! videoconvert ! video/x-raw,format=I420 ' \
                             '! x264enc speed-preset=ultrafast tune=zerolatency ' \
                             '! rtph264pay config-interval=1 name=pay0 pt=96' \
                             .format(width, height)

        print("RTSP server initialized")

        # Initialize the SDK
        options = tfsdk.ConfigurationOptions()
        options.smallest_face_height = 40 # TODO Can change the smallest_face_height: https://reference.trueface.ai/cpp/dev/latest/py/general.html#tfsdk.ConfigurationOptions.smallest_face_height
        options.fr_model = tfsdk.FACIALRECOGNITIONMODEL.TFV5 # Use the most accurate face recognition model
        options.dbms = tfsdk.DATABASEMANAGEMENTSYSTEM.SQLITE # Use the SQLite backend option
        options.enable_GPU = True # Use the GPU

        self.sdk = tfsdk.SDK(options)

        # Set and validate the license token
        is_valid = self.sdk.set_license(os.environ['TRUEFACE_TOKEN'])
        if (is_valid == False):
            print("Invalid License Provided")
            print("Be sure to export your license token as TRUEFACE_TOKEN")
            quit()

        # Load the database and collection which we previously populated from disk.
        res = self.sdk.create_database_connection("my_database.db")
        if (res != tfsdk.ERRORCODE.NO_ERROR):
          print("Unable to create database connection")
          quit()

        res = self.sdk.create_load_collection("my_collection")
        if (res != tfsdk.ERRORCODE.NO_ERROR):
            print("Unable to create collection")
            quit()

        # Run a single identification query to load the model into memory before the main loop
        # because the module uses lazy initialization.
        res = self.sdk.set_image("../../images/armstrong/armstrong1.jpg")
        if (res != tfsdk.ERRORCODE.NO_ERROR):
            print("Unable to set image 1")
            quit()

        # Extract the feature vector.
        res, v1, found = self.sdk.get_largest_face_feature_vector()
        if (res != tfsdk.ERRORCODE.NO_ERROR or found == False):
            print("Unable to generate feature vector 1, no face detected")
            quit()

        print("SDK has been initialized")
        print("Ready to accept connections...")


    # Method for grabbing frames from the video capture, running face recognition, then pushing annotated images to streaming buffer.
    def on_need_data(self, src, lenght):
        # Allow for variable frame rate by manually computing the timestamps
        start_time = datetime.datetime.now()

        # Grab a frame from the input rtsp stream.
        ret, frame = self.camera_thread.getFrame()
        if ret:
            # Set the image with the SDK.
            res = self.sdk.set_image(frame, frame.shape[1], frame.shape[0], tfsdk.COLORCODE.bgr)
            if (res == tfsdk.ERRORCODE.NO_ERROR):

                # The following code detects all the faces in the image and runs face recognition on all the detected faces.
                # Runing FR on all the detected faces can cause a variable output frame rate depending on the number of detected faces 
                # as FR template generation can be slow on some devices (ex. nvidia jetson).

                ###########################################
                ########## Run FR on all faces in the image
                ###########################################

                faceboxes = self.sdk.detect_faces()
                for facebox in faceboxes:
                    did_find_match = False

                    # Extract a feature vector for all detected faces in the image
                    res, faceprint = self.sdk.get_face_feature_vector(facebox)
                    if (res == tfsdk.ERRORCODE.NO_ERROR):
                        
                        # Run a 1 to N identification query against our collection
                        res, match_found, candidate = self.sdk.identify_top_candidate(faceprint, threshold=0.35)

                        if (res == tfsdk.ERRORCODE.NO_ERROR and match_found):
                            # We found a match
                            # Draw the label and a green box around the face
                            did_find_match = True
                            draw_rectangle(frame, facebox, color_code=(0, 255, 0))   
                            draw_label(frame,
                                       (int(facebox.top_left.x), int(facebox.top_left.y)),
                                       "{} {}%".format(
                                           candidate.identity,
                                           int(candidate.match_probability*100)),
                                       color_code=(0, 255, 0))
                                    
                    if not did_find_match:
                        # Draw a blue box around the detected face
                        draw_rectangle(frame, facebox)

                ###########################################
                ########## Run FR on only the largest face
                ###########################################   

                # found, facebox = self.sdk.detect_largest_face()
                # if found:
                #     found_face_id = False
                #     # Extract the feature vector for the largest face
                #     res, faceprint = self.sdk.get_face_feature_vector(facebox)
                #     if (res == tfsdk.ERRORCODE.NO_ERROR):
                #         # Run the identification query
                #         res, match_found, candidate = self.sdk.identify_top_candidate(faceprint, threshold=0.35)
                #         if (res == tfsdk.ERRORCODE.NO_ERROR and match_found):
                #             found_face_id = True
                #             # Draw a green rectangle around the face and the corresponding label
                #             draw_rectangle(frame, facebox, color_code=(0, 255, 0))   
                #             draw_label(frame,
                #                    (int(facebox.top_left.x), int(facebox.top_left.y)),
                #                    "{} {}%".format(
                #                        candidate.identity,
                #                        int(candidate.match_probability*100)),
                #                    color_code=(0, 255, 0))


                #     if not found_face_id:
                #         # Draw a blue rectangle around the face
                #         draw_rectangle(frame, facebox)

            data = frame.tobytes()
            buf = Gst.Buffer.new_allocate(None, len(data), None)
            buf.fill(0, data)

            # Compute the end timestamp to get the frame timestamp and duration
            end_time = datetime.datetime.now()
            time_diff = (end_time - start_time)
            duration = time_diff.total_seconds() * 1000

            buf.duration = duration
            timestamp = self.timestamp
            self.timestamp += duration * 1000000
            buf.pts = buf.dts = int(timestamp)
            buf.offset = timestamp
            self.number_frames += 1
            retval = src.emit('push-buffer', buf)

            print('pushed buffer, frame {}, durations {} s'.format(self.number_frames, duration / 1000))
            if retval != Gst.FlowReturn.OK:
                print(retval)

    # attach the launch string to the override method
    def do_create_element(self, url):
        return Gst.parse_launch(self.launch_string)
    
    # attaching the source element to the rtsp media
    def do_configure(self, rtsp_media):
        self.number_frames = 0
        appsrc = rtsp_media.get_element().get_child_by_name('source')
        appsrc.connect('need-data', self.on_need_data)

# Rtsp server implementation where we attach the factory sensor with the stream uri
class GstServer(GstRtspServer.RTSPServer):
    def __init__(self, camera_thread, stream_uri, **properties):
        super(GstServer, self).__init__(**properties)
        self.factory = SensorFactory(camera_thread)
        self.factory.set_shared(True)
        self.get_mount_points().add_factory(stream_uri, self.factory)
        self.attach(None)


# Class for grabbing frames from the camera and ensuring we always have the "latest" frame
# Basically ensures there are no frames accumulating in the video buffer
class ThreadedCamera():
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        
        self.frame = None
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
        self.mutex = Lock()

        self.thread = Thread(target = self.update, args = ())
        self.thread.daemon = True
        self.thread.start()


    def update(self):
        # Grabs frames from the buffer
        # Only need to grab frames at the frame rate
        while True:
            self.mutex.acquire()
            _ = self.cap.grab()
            self.mutex.release()
            sleep(1 / self.fps / 10)

    def getFrameDims(self):
        # Returns the frame width and height
        return self.width, self.height

    def getFrame(self):
        # Retrieve and return the latest frame
        self.mutex.acquire()
        ret, frame = self.cap.retrieve()
        self.mutex.release()
        return ret, frame            


# Get the reqired parameters
parser = argparse.ArgumentParser()
parser.add_argument("--input_rtsp_stream", required=True, help="The url for the input RTSP stream")
parser.add_argument("--stream_uri", default = "/video_stream", help="output rtsp video stream uri")

opt = parser.parse_args()

camera_thread = ThreadedCamera(opt.input_rtsp_stream)

# initializing the threads and running the stream on loop.
Gst.init(None)
server = GstServer(camera_thread, opt.stream_uri)
loop = GLib.MainLoop()
loop.run()

while True:
    sleep(1)


# Ouput RTSP url:
# rtsp://localhost:8554/trueface_stream