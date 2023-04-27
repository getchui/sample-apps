//
//  FaceDetectionViewController.swift
//  isamples
//
//  Created by adel boussaken on 11/4/2023.
//

import Foundation
import UIKit
import AVFoundation

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
    
    // @IBOutlet weak var previewView: UIView!
    var previewView: UIView!
    
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
                     position: .front) else {
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
                connection.videoOrientation = .portraitUpsideDown
            case .landscapeLeft:
                connection.videoOrientation = .landscapeRight
            case .landscapeRight:
                connection.videoOrientation = .landscapeLeft
            default:
                break
            }
        }
    }
    
    // Process the image buffer and detect faces using the SDK.
    func processImageBuffer(_ imageBuffer: CVImageBuffer) {
        let ciimage = CIImage(cvPixelBuffer: imageBuffer)
        self.frame = convert(cmage: ciimage)
        let height = CVPixelBufferGetHeight(imageBuffer)
        let width = CVPixelBufferGetWidth(imageBuffer)
        
        guard let tfimage = sdk.preprocessImage(self.frame) else { return }
        let face = sdk.detectLargestFace(tfimage)
        
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
        
        defer {
            // tfimage.destroy()
        }

        if face != nil {
            onFaceDetected?(sdk, tfimage, face!)
        } else {
            onFaceNotDetected?(sdk, tfimage)
        }
    }
    
    // Process the detected face and draw a rectangle around it.
    func processFace(tfimage: TFImage, face: TFFaceBoxAndLandmarks, width: Int, height: Int) {
        let adjustedFaceRect = transformRectToLayerCoordinates(face: face, width: width, height: height)
        drawRectOnFace(rect: adjustedFaceRect)
    }

    // Transform the detected face rectangle coordinates to layer coordinates.
    func transformRectToLayerCoordinates(face: TFFaceBoxAndLandmarks, width: Int, height: Int) -> CGRect {
        let previewLayerBounds = previewLayer.bounds
        
        let x = (1 - CGFloat(face.topLeft.x) / CGFloat(width)) * previewLayerBounds.width
        let y = CGFloat(face.topLeft.y) * previewLayerBounds.height / CGFloat(height)
        let rectWidth = (CGFloat(face.bottomRight.x) - CGFloat(face.topLeft.x)) * previewLayerBounds.width / CGFloat(width)
        let rectHeight = (CGFloat(face.bottomRight.y) - CGFloat(face.topLeft.y)) * previewLayerBounds.height / CGFloat(height)
        
        let adjustedFaceRect = CGRect(x: x - rectWidth, y: y, width: rectWidth, height: rectHeight)
        return adjustedFaceRect
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
    
}
