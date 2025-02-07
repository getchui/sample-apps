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
    
    // used for convert(cmage: CIImage)
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])
    
    // used properties for debugging
    private var formatDebugEnabled = true
    private var lastFrameTime = Date()
    private let frameLogInterval = 2.0
    
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

        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
            self.setupAVCapture()
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
        // session.sessionPreset = AVCaptureSession.Preset.hd1920x1080
        // configureCaptureDevice()
        
        guard configureCaptureDevice() else {
            session.commitConfiguration()
            showCameraErrorAlert()
            return
        }
        
        // Log all supported formats
        logSupportedFormats()
        
        if let format = findSupportedFormat() {
            do {
                try captureDevice.lockForConfiguration()
                captureDevice.activeFormat = format
                captureDevice.unlockForConfiguration()
                
                // Log selected format
                let selectedDimensions = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
                print("📸 Selected camera format: \(selectedDimensions.width)x\(selectedDimensions.height)")
            } catch {
                print("🔴 Format config error: \(error)")
            }
        } else {
            print("🔴 No supported format found!")
        }
        
        beginSession()
    }
    
    private func logSupportedFormats() {
        print("📷 Available camera formats:")
        for format in captureDevice.formats {
            let dimensions = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
            let ranges = format.videoSupportedFrameRateRanges
            let fps = ranges.first?.maxFrameRate ?? 0
            let colorSpaces = format.supportedColorSpaces.map { "\($0)" }.joined(separator: ", ")
            
            print(String(format: "▫️ %4dx%-4d | %2.0f FPS | %@",
                         dimensions.width,
                         dimensions.height,
                         fps,
                         colorSpaces))
        }
    }
    
    private func findSupportedFormat() -> AVCaptureDevice.Format? {
        // Prioritize formats that work across devices
        let preferredResolutions = [
            (width: 1280, height: 720),  // 720p
            (width: 1920, height: 1080), // 1080p
            (width: 640, height: 480)    // 480p
        ]
        
        return captureDevice.formats.first { format in
            let dimensions = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
            return preferredResolutions.contains {
                $0.width == Int(dimensions.width) && $0.height == Int(dimensions.height)
            }
        }
    }

    private func showCameraErrorAlert() {
        DispatchQueue.main.async {
            let alert = UIAlertController(
                title: "Camera Error",
                message: "Front camera not available",
                preferredStyle: .alert
            )
            alert.addAction(UIAlertAction(title: "OK", style: .default))
            self.present(alert, animated: true)
        }
    }
    
    // Configure the AVCaptureDevice for the capture session.
    func configureCaptureDevice() -> Bool {
        guard let device = AVCaptureDevice.default(
            .builtInWideAngleCamera,
            for: .video,
            position: .front
        ) else {
            return false
        }
        captureDevice = device
        return true
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
            previewLayer.addObserver(self,
                                     forKeyPath: "bounds",
                                     options: [.new, .initial],
                                     context: nil)
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
    
    override func observeValue(forKeyPath keyPath: String?,
                               of object: Any?,
                               change: [NSKeyValueChangeKey : Any]?,
                               context: UnsafeMutableRawPointer?) {
        if keyPath == "bounds", let layer = object as? CALayer {
            print("🖼️ Preview layer size updated: \(layer.bounds.size)")
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
                // connection.videoOrientation = .portraitUpsideDown
                connection.videoOrientation = .portrait
            case .landscapeLeft:
                // connection.videoOrientation = .landscapeRight
                connection.videoOrientation = .portrait
            case .landscapeRight:
                // connection.videoOrientation = .landscapeLeft
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
            let face = sdk.detectLargestFace(tfimage)
            
            DispatchQueue.main.async {
                self.previewView.subviews.forEach({ $0.removeFromSuperview() })
                self.faceRectLayer?.removeFromSuperlayer()
                self.faceRectLayer = nil
            }
            
            if face != nil {
                DispatchQueue.main.async {
                    self.processFace(tfimage: tfimage, face: face!.faceBoxAndLandmarks, width: width, height: height)
                }
            }
            
            defer {
                // tfimage.destroy()
            }
            
            if face != nil && face!.faceBoxAndLandmarks.score > 0.9 &&
              face!.faceBoxAndLandmarks.topLeft.x > 0 && face!.faceBoxAndLandmarks.topLeft.y > 0 &&
              face!.faceBoxAndLandmarks.bottomRight.x > face!.faceBoxAndLandmarks.topLeft.x &&
              face!.faceBoxAndLandmarks.bottomRight.y > face!.faceBoxAndLandmarks.topLeft.y {
               onFaceDetected?(sdk, tfimage, face!.faceBoxAndLandmarks)
            } else {
               onFaceNotDetected?(sdk, tfimage)
            }
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
        guard let cgImage = ciContext.createCGImage(cmage, from: cmage.extent) else {
            return UIImage()
        }
        return UIImage(cgImage: cgImage)
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
