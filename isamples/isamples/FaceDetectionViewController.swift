//
//  FaceDetectionViewController.swift
//  isamples
//
//  Created by adel boussaken on 11/4/2023.
//

import Foundation
import UIKit
import AVFoundation

func stringFromTFObjectLabel(_ label: TFObjectLabel) -> String {
    let labelStrings = [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush"
    ]
    
    let index = label.rawValue
    if index >= 0 && index < labelStrings.count {
        return labelStrings[Int(index)]
    } else {
        return " "
    }
}

// FaceDetectionViewController is a UIViewController that initializes and manages the AVCaptureSession
// for real-time face detection using the SDK.
class FaceDetectionViewController: UIViewController {
    
    var sdk: TFSDK!
    var videoDataOutput: AVCaptureVideoDataOutput!
    var videoDataOutputQueue: DispatchQueue!
    var previewLayer: AVCaptureVideoPreviewLayer!
    var captureDevice: AVCaptureDevice!
    let session = AVCaptureSession()
    
    var onFaceNotDetected: ((_ sdk: TFSDK, _ tfImage: TFImage) -> Void)?
    var onFaceDetected: ((_ sdk: TFSDK, _ tfImage: TFImage, _ face: TFFaceBoxAndLandmarks) -> Void)?
    var faceRectLayer: CAShapeLayer?
    var frame = UIImage()
    
    // hack in to add object detection
    var objectLayers: [CAShapeLayer] = []
    let faceDetectionSwitch = UISwitch()
    let faceDetectionLabel = UILabel()
    let objectDetectionSwitch = UISwitch()
    let objectDetectionLabel = UILabel()

    // @IBOutlet weak var previewView: UIView!
    var previewView: UIView!
    
    var cameraPosition: AVCaptureDevice.Position = .back

    // Deinitializer to stop the camera.
    // import for navigation and to avoid crashes
    deinit {
        stopCamera()
    }
    
    override func viewDidLoad() {
        previewView = UIView(frame: view.bounds)
        previewView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        view.addSubview(previewView)
        view.sendSubviewToBack(previewView)
        super.viewDidLoad()
        initSDK()
        setupAVCapture()
        
        setupSwitch(faceDetectionSwitch, label: faceDetectionLabel, text: "Face Detection", yPos: 100)
        setupSwitch(objectDetectionSwitch, label: objectDetectionLabel, text: "Object Detection", yPos: 140)
    }
    
    func setupSwitch(_ switchWidget: UISwitch, label: UILabel, text: String, yPos: CGFloat) {
        // Setup label
        label.frame = CGRect(x: 20, y: yPos, width: 150, height: 31)
        label.text = text
        label.textColor = .black
        label.shadowColor = .lightGray
        label.shadowOffset = CGSize(width: 1, height: 1)
        view.addSubview(label)

        // Setup switch
        switchWidget.frame = CGRect(x: label.frame.maxX + 10, y: yPos, width: 0, height: 0)
        switchWidget.addTarget(self, action: #selector(switchChanged), for: .valueChanged)
        view.addSubview(switchWidget)
    }
    
    @objc func switchChanged(_ sender: UISwitch) {
        if !sender.isOn {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                for layer in self.objectLayers {
                    layer.removeFromSuperlayer()
                }
                self.objectLayers.removeAll()
            }
        }
    }
    
    // MARK: - SDK Initialization
    // Initialize the SDK with the SDKManager instance.
    func initSDK() {
        sdk = SDKManager.shared.sdk
    }
    
    // MARK: - AVCapture Setup
    // Configure and set up the AVCaptureSession.
    func setupAVCapture() {
        session.beginConfiguration()
        session.sessionPreset = AVCaptureSession.Preset.hd1920x1080
        configureCaptureDevice()
        beginSession()
    }
    
    // Configure the AVCaptureDevice for the capture session.
    func configureCaptureDevice() {
        guard let device = AVCaptureDevice
            .default(.builtInWideAngleCamera,
                     for: .video,
                     position: cameraPosition) else {
            return
        }
        captureDevice = device
    }
    
    // MARK: - AVCapture Session Handling
    // Begin the AVCaptureSession with the configured device input and output.
    func beginSession(){
        var deviceInput: AVCaptureDeviceInput!
        
        do {
            deviceInput = try AVCaptureDeviceInput(device: captureDevice)
            guard deviceInput != nil else {
                print("error: cant get deviceInput")
                return
            }
            
            if self.session.canAddInput(deviceInput){
                self.session.addInput(deviceInput)
            }
            
            videoDataOutput = AVCaptureVideoDataOutput()
            videoDataOutput.alwaysDiscardsLateVideoFrames=true
            videoDataOutputQueue = DispatchQueue(label: "VideoDataOutputQueue")
            videoDataOutput.setSampleBufferDelegate(self, queue:self.videoDataOutputQueue)
            
            if session.canAddOutput(self.videoDataOutput){
                session.addOutput(self.videoDataOutput)
            }
            
            videoDataOutput.connection(with: .video)?.isEnabled = true
            videoDataOutput.connection(with: AVMediaType.video)?.videoOrientation = .portrait
            
            previewLayer = AVCaptureVideoPreviewLayer(session: self.session)
            previewLayer.connection?.videoOrientation = .portrait
            previewLayer.videoGravity = AVLayerVideoGravity.resize
            let rootLayer :CALayer = self.previewView.layer
            rootLayer.masksToBounds=true
            previewLayer.frame = rootLayer.bounds
            rootLayer.addSublayer(self.previewLayer)
            session.commitConfiguration()
            
            DispatchQueue.global(qos: .userInitiated).async {
                self.session.startRunning()
            }
        } catch let error as NSError {
            deviceInput = nil
            print("error: \(error.localizedDescription)")
        }
    }
    
    // Configure the AVCaptureDeviceInput for the capture session.
    func configureDeviceInput() {
        do {
            let deviceInput = try AVCaptureDeviceInput(device: captureDevice)
            if self.session.canAddInput(deviceInput) {
                self.session.addInput(deviceInput)
            }
        } catch let error as NSError {
            print("error: \(error.localizedDescription)")
        }
    }
    
    // Configure the AVCaptureVideoDataOutput for the capture session.
    func configureVideoDataOutput() {
        videoDataOutput = AVCaptureVideoDataOutput()
        videoDataOutput.alwaysDiscardsLateVideoFrames = true
        videoDataOutputQueue = DispatchQueue(label: "VideoDataOutputQueue")
        videoDataOutput.setSampleBufferDelegate(self, queue: videoDataOutputQueue)
        if session.canAddOutput(self.videoDataOutput) {
            session.addOutput(self.videoDataOutput)
        }
        videoDataOutput.connection(with: .video)?.isEnabled = true
        videoDataOutput.connection(with: .video)?.videoOrientation = .portrait
    }
    
    // Configure the AVCaptureVideoPreviewLayer for the AVCaptureSession.
    func configurePreviewLayer() {
        previewLayer = AVCaptureVideoPreviewLayer(session: self.session)
        previewLayer.connection?.videoOrientation = .portrait
        previewLayer.videoGravity = AVLayerVideoGravity.resizeAspectFill
        let rootLayer: CALayer = self.previewView.layer
        rootLayer.masksToBounds = true
        previewLayer.frame = rootLayer.bounds
        rootLayer.addSublayer(self.previewLayer)
    }
    
    // Stop the AVCaptureSession and remove device inputs.
    func stopCamera() {
        session.stopRunning()
        for input in (self.session.inputs as! [AVCaptureDeviceInput]) {
            self.session.removeInput(input)
        }
    }
    
    // Start the AVCaptureSession.
    func startCamera() {
        if !session.isRunning {
            DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                self?.session.startRunning()
            }
        }
    }
    
    // Pause the AVCaptureSession.
    func pauseCamera() {
        session.stopRunning()
    }

    // Resume the AVCaptureSession.
    func resumeCamera() {
        DispatchQueue.global(qos: .userInitiated).async {
            self.session.startRunning()
        }
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate
// Extension for FaceDetectionViewController to handle AVCaptureVideoDataOutputSampleBufferDelegate.
extension FaceDetectionViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    // Implement the captureOutput delegate method.
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        updateVideoOrientation(for: connection)
        
        if let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
            processImageBuffer(imageBuffer)
        }
    }
    
    // Update video orientation based on device orientation.
    func updateVideoOrientation(for connection: AVCaptureConnection) {
        if connection.isVideoOrientationSupported {
            let orientation = UIDevice.current.orientation
            switch orientation {
            case .portrait:
                connection.videoOrientation = .portrait
            case .portraitUpsideDown:
                connection.videoOrientation = .portrait
            case .landscapeLeft:
                connection.videoOrientation = .portrait
            case .landscapeRight:
                connection.videoOrientation = .portrait
            default:
                break
            }
        }
    }
    
    // Process the image buffer and detect faces using the SDK.
    func processImageBuffer(_ imageBuffer: CVImageBuffer) {
        autoreleasepool {
            let ciimage = CIImage(cvPixelBuffer: imageBuffer)
            self.frame = convert(cmage: ciimage)
            let height = CVPixelBufferGetHeight(imageBuffer)
            let width = CVPixelBufferGetWidth(imageBuffer)
            
            guard let tfimage = sdk.preprocessImage(self.frame) else { return }
            
            var face: TFFaceBoxAndLandmarks?
            var objects: [Any]?
            
            var faceDetectionEnabled = false
            var objectDetectionEnabled = false
            // Check the switch states on the main thread
            DispatchQueue.main.sync {
                faceDetectionEnabled = faceDetectionSwitch.isOn
                objectDetectionEnabled = objectDetectionSwitch.isOn
            }

            if faceDetectionEnabled {
                face = sdk.detectLargestFace(tfimage)
            }

            if objectDetectionEnabled {
                objects = sdk.detectObjects(tfimage)
            }
            
            DispatchQueue.main.async {
                self.previewView.subviews.forEach({ $0.removeFromSuperview() })
                self.faceRectLayer?.removeFromSuperlayer()
                self.faceRectLayer = nil
            }
            
            if face != nil {
                DispatchQueue.main.async {
                    self.processFace(tfimage: tfimage, face: face!, width: width, height: height)
                }
            }
            
            if objects != nil {
                DispatchQueue.main.async {
                    self.processObjects(tfimage: tfimage, objects: objects!, width: width, height: height)
                }
            }
            
            defer {
                // tfimage.destroy()
            }
            
            if face != nil {
                onFaceDetected?(sdk, tfimage, face!)
            } else {
                onFaceNotDetected?(sdk, tfimage)
            }
        }
    }
    
    // Process the detected objects and draw a rectangle around it.
    func processObjects(tfimage: TFImage, objects: [Any], width: Int, height: Int) {
        for layer in objectLayers {
            layer.removeFromSuperlayer()
        }
        objectLayers.removeAll()
        
        for item in objects {
            if let boundingBox = item as? TFBoundingBox {
                if boundingBox.label == person {
                    continue
                }
                let adjustedObjectRect = transformObjectRectToLayerCoordinates(object: boundingBox, layerWidth: width, layerHeight: height, usingFrontCamera: !isUsingBackCamera())
                let objectName = stringFromTFObjectLabel(boundingBox.label)
                let objectLayer = drawRectOnObject(rect: adjustedObjectRect, label: " \(objectName)")
                objectLayers.append(objectLayer)
            }
        }
    }

    // Process the detected face and draw a rectangle around it.
    func processFace(tfimage: TFImage, face: TFFaceBoxAndLandmarks, width: Int, height: Int) {
        // let adjustedFaceRect = backCameraTransformRectToLayerCoordinates(face: face, width: width, height: height)
        let adjustedFaceRect = transformFaceRectToLayerCoordinates(face: face, layerWidth: width, layerHeight: height, usingFrontCamera: !isUsingBackCamera())
        drawRectOnFace(rect: adjustedFaceRect)
    }

    // Check if the current camera is the back camera
    func isUsingBackCamera() -> Bool {
        return captureDevice.position == .back
    }
    
    // Transform the detected object rectangle coordinates to layer coordinates.
    func transformObjectRectToLayerCoordinates(object: TFBoundingBox, layerWidth: Int, layerHeight: Int, usingFrontCamera: Bool) -> CGRect {
        let previewLayerBounds = previewLayer.bounds
        let scaleX = previewLayerBounds.width / CGFloat(layerWidth)
        let scaleY = previewLayerBounds.height / CGFloat(layerHeight)
        
        var x = CGFloat(object.topLeft.x) * scaleX
        let y = CGFloat(object.topLeft.y) * scaleY
        let rectWidth = CGFloat(object.width) * scaleX
        let rectHeight = CGFloat(object.height) * scaleY
        
        // Adjust the x coordinate for the front camera
        if usingFrontCamera {
            x = (1 - CGFloat(object.topLeft.x) / CGFloat(layerWidth)) * previewLayerBounds.width - rectWidth
        }
        
        return CGRect(x: x, y: y, width: rectWidth, height: rectHeight)
    }
    
    // Transform the detected face rectangle coordinates to layer coordinates.
    func transformFaceRectToLayerCoordinates(face: TFFaceBoxAndLandmarks, layerWidth: Int, layerHeight: Int, usingFrontCamera: Bool) -> CGRect {
        let previewLayerBounds = previewLayer.bounds
        let scaleX = previewLayerBounds.width / CGFloat(layerWidth)
        let scaleY = previewLayerBounds.height / CGFloat(layerHeight)
        
        var x = CGFloat(face.topLeft.x) * scaleX
        let y = CGFloat(face.topLeft.y) * scaleY
        let rectWidth = (CGFloat(face.bottomRight.x) - CGFloat(face.topLeft.x)) * scaleX
        let rectHeight = (CGFloat(face.bottomRight.y) - CGFloat(face.topLeft.y)) * scaleY
        
        // Adjust the x coordinate for the front camera
        if usingFrontCamera {
            x = (1 - CGFloat(face.topLeft.x) / CGFloat(layerWidth)) * previewLayerBounds.width - rectWidth
        }
        
        return CGRect(x: x, y: y, width: rectWidth, height: rectHeight)
    }


    // Convert CIImage to UIImage.
    func convert(cmage: CIImage) -> UIImage {
        let context = CIContext(options: nil)
        let cgImage = context.createCGImage(cmage, from: cmage.extent)!
        let image = UIImage(cgImage: cgImage)
        return image
    }
    
    // Draw a rectangle on the detected face.
    func drawRectOnFace(rect: CGRect) {
        if faceRectLayer == nil {
            faceRectLayer = CAShapeLayer()
            faceRectLayer?.strokeColor = UIColor.green.cgColor
            faceRectLayer?.lineWidth = 3.0
            faceRectLayer?.fillColor = UIColor.clear.cgColor
            self.previewView.layer.addSublayer(faceRectLayer!)
        }
        
        faceRectLayer?.path = UIBezierPath(rect: rect).cgPath
        faceRectLayer?.frame = self.previewView.bounds
    }

    // Draw a red rectangle on the detected object, add a label, and return the layer.
    func drawRectOnObject(rect: CGRect, label: String) -> CAShapeLayer {
        let objectLayer = CAShapeLayer()
        objectLayer.strokeColor = UIColor.red.cgColor
        objectLayer.lineWidth = 3.0
        objectLayer.fillColor = UIColor.clear.cgColor
        objectLayer.path = UIBezierPath(rect: rect).cgPath
        objectLayer.frame = self.previewView.bounds

        // Create a text layer
        let textLayer = CATextLayer()
        textLayer.string = label
        textLayer.foregroundColor = UIColor.white.cgColor
        textLayer.backgroundColor = UIColor.black.withAlphaComponent(0.5).cgColor
        textLayer.alignmentMode = .left
        textLayer.fontSize = 14
        textLayer.frame = CGRect(x: rect.origin.x, y: rect.origin.y - 20, width: rect.width, height: 24)

        objectLayer.addSublayer(textLayer)
        self.previewView.layer.addSublayer(objectLayer)

        return objectLayer
    }
    
}
