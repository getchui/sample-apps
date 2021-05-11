#!/usr/bin/env python3

import gi
import cv2
import tfsdk
import os
import argparse

# import required library like Gstreamer and GstreamerRtspServer
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GLib

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


def draw_rectangle(frame, bounding_box, color_code = (194,134,58)):
    # Draw the rectangle on the frame
    cv2.rectangle(frame,
                  (int(bounding_box.top_left.x), int(bounding_box.top_left.y)),
                  (int(bounding_box.bottom_right.x), int(bounding_box.bottom_right.y)), color_code, 3)

# Sensor Factory class which inherits the GstRtspServer base class and add
# properties to it.
class SensorFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, **properties):
        super(SensorFactory, self).__init__(**properties)
        
        # Create a connection to our input RTSP stream and obtain the width and height
        # TODO Cyrus need to tell user in readme that they require opencv with gstreamer in order to use this
        # TODO Cyrus LD_LIBRARY_PATH and PYTHONPATH
        input_gstreamer_pipeline = "rtspsrc location={} ! decodebin ! videoconvert ! appsink max-buffers=2 drop=true".format(opt.input_rtsp_stream)
        self.cap = cv2.VideoCapture(input_gstreamer_pipeline)
        width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 

        self.number_frames = 0
        self.fps = 30
        self.duration = 1 / self.fps * Gst.SECOND  # duration of a frame in nanoseconds
        self.launch_string = 'appsrc name=source is-live=true block=true format=GST_FORMAT_TIME ' \
                             'caps=video/x-raw,format=BGR,width={},height={},framerate={}/1 ' \
                             '! videoconvert ! video/x-raw,format=I420 ' \
                             '! x264enc speed-preset=ultrafast tune=zerolatency ' \
                             '! rtph264pay config-interval=1 name=pay0 pt=96' \
                             .format(width, height, self.fps)

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

        # Load the database and collection which we previously populated from disk
        res = self.sdk.create_database_connection("my_database.db")
        if (res != tfsdk.ERRORCODE.NO_ERROR):
          print("Unable to create database connection")
          quit()

        res = self.sdk.create_load_collection("my_collection")
        if (res != tfsdk.ERRORCODE.NO_ERROR):
            print("Unable to create collection")
            quit()

        # Run a single identification query to load the model into memory before the main loop
        res = self.sdk.set_image("../../images/armstrong/armstrong1.jpg")
        if (res != tfsdk.ERRORCODE.NO_ERROR):
            print("Unable to set image 1")
            quit()

        # Extract the feature vector
        res, v1, found = self.sdk.get_largest_face_feature_vector()
        if (res != tfsdk.ERRORCODE.NO_ERROR or found == False):
            print("Unable to generate feature vector 1, no face detected")
            quit()

        print("SDK has been initialized")
        print("Ready to accept connections...")


    # Method for grabbing frames from the video capture, running face recognition, then pushing annotated images to streaming buffer
    def on_need_data(self, src, lenght):
        if self.cap.isOpened():
            # Grab a frame from the input rtsp stream
            ret, frame = self.cap.read()
            if ret:
                # Set the image with the SDK
                res = self.sdk.set_image(frame, frame.shape[1], frame.shape[0], tfsdk.COLORCODE.bgr)
                if (res == tfsdk.ERRORCODE.NO_ERROR):

                    # Run face detection
                    faceboxes = self.sdk.detect_faces()
                    for facebox in faceboxes:
                        did_find_match = False

                        # Extract a feature vector for all detected faces in the image
                        res, faceprint = self.sdk.get_face_feature_vector(facebox)
                        if (res == tfsdk.ERRORCODE.NO_ERROR):
                            
                            # Run a 1 to N identification query against our collection
                            res, match_found, candidate = self.sdk.identify_top_candidate(faceprint, threshold=0.4)

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

                data = frame.tobytes()
                buf = Gst.Buffer.new_allocate(None, len(data), None)
                buf.fill(0, data)
                buf.duration = self.duration
                timestamp = self.number_frames * self.duration
                buf.pts = buf.dts = int(timestamp)
                buf.offset = timestamp
                self.number_frames += 1
                retval = src.emit('push-buffer', buf)
                print('pushed buffer, frame {}, duration {} ns, durations {} s'.format(self.number_frames,
                                                                                       self.duration,
                                                                                       self.duration / Gst.SECOND))
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
    def __init__(self, **properties):
        super(GstServer, self).__init__(**properties)
        self.factory = SensorFactory()
        self.factory.set_shared(True)
        self.get_mount_points().add_factory(opt.stream_uri, self.factory)
        self.attach(None)

# Get thre reqired parameters
parser = argparse.ArgumentParser()
parser.add_argument("--input_rtsp_stream", required=True, help="The url for the input RTSP stream")
parser.add_argument("--stream_uri", default = "/video_stream", help="output rtsp video stream uri")
opt = parser.parse_args()

# initializing the threads and running the stream on loop.
Gst.init(None)
server = GstServer()
loop = GLib.MainLoop()
loop.run()


# rtsp://localhost:8554/trueface_stream