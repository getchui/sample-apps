import gi
import cv2
import tfsdk
import os

# import required library like Gstreamer and GstreamerRtspServer
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GObject


def draw_rectangle(frame, bounding_box):
    # Draw the rectangle on the frame
    cv2.rectangle(frame,
                  (int(bounding_box.top_left.x), int(bounding_box.top_left.y)),
                  (int(bounding_box.bottom_right.x), int(bounding_box.bottom_right.y)), (194,134,58), 3)

# Sensor Factory class which inherits the GstRtspServer base class and add
# properties to it.
class SensorFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, **properties):
        super(SensorFactory, self).__init__(**properties)
        
        # Create a connection to our input RTSP stream and obtain the width and height
        self.cap = cv2.VideoCapture("rtsp://root:admin@192.168.88.248/stream1")
        width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
        print(width)
        print(height)

        self.number_frames = 0
        self.fps = 30
        self.duration = 1 / self.fps * Gst.SECOND  # duration of a frame in nanoseconds
        self.launch_string = 'appsrc name=source is-live=true block=true format=GST_FORMAT_TIME ' \
                             'caps=video/x-raw,format=BGR,width={},height={},framerate={}/1 ' \
                             '! videoconvert ! video/x-raw,format=I420 ' \
                             '! x264enc speed-preset=ultrafast tune=zerolatency ' \
                             '! rtph264pay config-interval=1 name=pay0 pt=96' \
                             .format(width, height, self.fps)

        # Initialize the SDK
        options = tfsdk.ConfigurationOptions()
        options.smallest_face_height = 40
        options.fr_model = tfsdk.FACIALRECOGNITIONMODEL.TFV5 
        options.enable_GPU = True # Use the GPU
        
        # TODO Cyrus will need to load collection from disk
        self.sdk = tfsdk.SDK(options)

        # Set and validate the license token
        is_valid = self.sdk.set_license(os.environ['TRUEFACE_TOKEN'])
        if (is_valid == False):
            print("Invalid License Provided")
            print("Be sure to export your license token as TRUEFACE_TOKEN")
            quit()

    # Method for grabbing frames from the video capture, running face recognition, then pushing annotated images to streaming buffer
    def on_need_data(self, src, lenght):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                res = self.sdk.set_image(frame, frame.shape[1], frame.shape[0], tfsdk.COLORCODE.bgr)
                if (res == tfsdk.ERRORCODE.NO_ERROR):
                    faceboxes = self.sdk.detect_faces()
                    for facebox in faceboxes:
                        draw_rectangle(frame, facebox)                   
                else:                
                    print("Unable to set frame.")                

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
        self.get_mount_points().add_factory("/trueface_stream", self.factory)
        self.attach(None)

# initializing the threads and running the stream on loop.
GObject.threads_init()
Gst.init(None)
server = GstServer()
loop = GObject.MainLoop()
loop.run()


# rtsp://localhost:8554/trueface_stream