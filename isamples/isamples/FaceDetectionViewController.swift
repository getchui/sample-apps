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
    
    var previewView: UIView!
    
    // used for convert(cmage: CIImage)
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])
    
    // used properties for debugging
    private var formatDebugEnabled = true
    private var lastFrameTime = Date()
    private let frameLogInterval = 2.0
    
    // --- Frequency controlled logging ---
    var logFrequencyInterval: TimeInterval = 10.0 // Parameter to control log frequency, default 10 seconds
    private var lastPreviewLayerLogTime: Date?
    // --- End frequency controlled logging ---
    
    
    // Deinitializer to stop the camera.
    // import for navigation and to avoid crashes
    deinit {
        print("🗑️ FaceDetectionViewController deinit - Stopping camera and removing observer")
        stopCamera()
        print("Camera stopped in deinit")
        previewLayer?.removeObserver(self, forKeyPath: "bounds")
        print("👁️ Observer for previewLayer bounds removed")
        
        UIDevice.current.endGeneratingDeviceOrientationNotifications()
        NotificationCenter.default.removeObserver(self)
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        print("viewDidLoad - Device orientation at start: \(UIDevice.current.orientation.rawValue)") // Log initial device orientation
        print("viewDidLoad - Interface orientation at start: \(self.view.window?.windowScene?.interfaceOrientation.rawValue ?? -1)") // Log initial interface orientation
        
        previewView = UIView(frame: view.bounds)
        previewView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        view.addSubview(previewView)
        view.sendSubviewToBack(previewView)
        super.viewDidLoad()
        print("🎬 FaceDetectionViewController viewDidLoad - Starting SDK and AVCapture setup")
        initSDK()
        
        // Add device orientation notification observer
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(orientationChanged),
            name: UIDevice.orientationDidChangeNotification,
            object: nil
        )
        // Enable device orientation notifications
        UIDevice.current.beginGeneratingDeviceOrientationNotifications()
        
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in // Short delay, mainly for UI setup
            guard let self = self else { return }
            print("⏳ Delayed AVCapture setup started")
            self.setupAVCapture()
            
            // Force initial orientation update based on interface orientation
            let initialInterfaceOrientation = self.view.window?.windowScene?.interfaceOrientation ?? .portrait
            print("📱 Initial interface orientation: \(initialInterfaceOrientation.rawValue)")
            let deviceOrientation: UIDeviceOrientation // Map interface to device orientation for initial setup
            switch initialInterfaceOrientation {
            case .landscapeLeft:
                deviceOrientation = .landscapeRight // Map landscapeLeft interface to landscapeRight device for video
            case .landscapeRight:
                deviceOrientation = .landscapeLeft  // Map landscapeRight interface to landscapeLeft device for video
            case .portraitUpsideDown:
                deviceOrientation = .portraitUpsideDown
            default:
                deviceOrientation = .portrait // Default for portrait and unknown
            }
            
            // Update initial orientation based on interface orientation mapping
            DispatchQueue.main.async {
                UIDevice.current.setValue(deviceOrientation.rawValue, forKey: "orientation") // Still try setting, though likely ignored
                self.updateOrientation() // Initial orientation update based on derived device orientation
            }
        }
    }
    
    // MARK: - SDK Initialization
    // Initialize the SDK with the SDKManager instance.
    func initSDK() {
        print("🎛️ Initializing SDK in FaceDetectionViewController")
        sdk = SDKManager.shared.sdk
        print("✅ SDK initialized from SDKManager")
    }
    
    // MARK: - AVCapture Setup
    // Configure and set up the AVCaptureSession.
    func setupAVCapture() {
        print("⚙️ Setting up AVCaptureSession")
        print("📸 Device types: \(AVCaptureDevice.DeviceType.builtInWideAngleCamera.rawValue)")
        
        let discoverySession = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInWideAngleCamera],
            mediaType: .video,
            position: .front
        )
        print("📸 Available devices: \(discoverySession.devices.map { $0.localizedName })")
        
        session.beginConfiguration()
        
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
        print("🔍 Finding best supported camera format")
        // Prioritize formats that work across devices
        let preferredResolutions = [
            (width: 1280, height: 720),  // 720p
            (width: 1920, height: 1080), // 1080p
            (width: 640, height: 480)    // 480p
        ]
        
        if let format = captureDevice.formats.first(where: { format in
            let dimensions = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
            return preferredResolutions.contains {
                $0.width == Int(dimensions.width) && $0.height == Int(dimensions.height)
            }
        }) {
            let dimensions = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
            print("✅ Found supported format: \(dimensions.width)x\(dimensions.height)")
            return format
        } else {
            print("⚠️ No preferred format found, using default if available")
            return captureDevice.formats.first // Fallback to any format if preferred not found
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
            print("🚨 Showing camera error alert: Front camera not available")
        }
    }
    
    // Configure the AVCaptureDevice for the capture session.
    func configureCaptureDevice() -> Bool {
        print("📲 Configuring capture device")
        var cameraAccessGranted = false
        let semaphore = DispatchSemaphore(value: 0) // Use semaphore to wait for camera access response
        
        AVCaptureDevice.requestAccess(for: .video) { granted in
            print("📸 Camera access granted: \(granted)")
            cameraAccessGranted = granted
            semaphore.signal() // Signal that camera access response is received
        }
        semaphore.wait() // Wait until camera access response is received
        
        if !cameraAccessGranted {
            print("🔴 Camera access NOT granted, failing configuration")
            return false
        }
        
        
        let discoverySession = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInWideAngleCamera, .builtInDualCamera, .builtInTrueDepthCamera],
            mediaType: .video,
            position: .front
        )
        
        guard let device = discoverySession.devices.first else {
            print("🔴 No front camera discovered")
            return false
        }
        
        captureDevice = device
        print("✅ Capture device configured: \(captureDevice.localizedName)")
        return true
    }
    
    // MARK: - AVCapture Session Handling
    // Begin the AVCaptureSession with the configured device input and output.
    func beginSession(){
        print("🎬 Beginning capture session")
        var deviceInput: AVCaptureDeviceInput!
        
        do {
            deviceInput = try AVCaptureDeviceInput(device: captureDevice)
            guard deviceInput != nil else {
                print("🔴 Error: cant get deviceInput")
                return
            }
            
            if self.session.canAddInput(deviceInput){
                self.session.addInput(deviceInput)
                print("✅ Device input added to session")
            }
            
            videoDataOutput = AVCaptureVideoDataOutput()
            videoDataOutput.alwaysDiscardsLateVideoFrames=true
            videoDataOutputQueue = DispatchQueue(label: "VideoDataOutputQueue")
            videoDataOutput.setSampleBufferDelegate(self, queue:self.videoDataOutputQueue)
            
            if session.canAddOutput(self.videoDataOutput){
                session.addOutput(self.videoDataOutput)
                print("✅ Video data output added to session")
            }
            
            // Get initial orientation
            let initialOrientation = UIDevice.current.orientation
            let videoOrientation = videoOrientation(from: initialOrientation)
            
            // print("📱 Getting initial orientation in beginSession")
            // print("📱 Device orientation: \(initialOrientation.rawValue)")
            // print("🎥 Calculated video orientation: \(videoOrientation.rawValue)")
            
            videoDataOutput.connection(with: .video)?.isEnabled = true
            if let videoConnection = videoDataOutput.connection(with: .video) {
                if videoConnection.isVideoOrientationSupported {
                    videoConnection.videoOrientation = videoOrientation
                }
            }
            
            previewLayer = AVCaptureVideoPreviewLayer(session: self.session)
            previewLayer.addObserver(self,
                                     forKeyPath: "bounds",
                                     options: [.new, .initial],
                                     context: nil)
            
            // Set initial preview layer orientation
            if let connection = previewLayer.connection, connection.isVideoOrientationSupported {
                connection.videoOrientation = videoOrientation
            }
            
            previewLayer.videoGravity = AVLayerVideoGravity.resize
            let rootLayer :CALayer = self.previewView.layer
            rootLayer.masksToBounds=true
            previewLayer.frame = rootLayer.bounds
            rootLayer.addSublayer(self.previewLayer)
            session.commitConfiguration()
            print("✅ Session configuration committed")
            
            DispatchQueue.global(qos: .userInitiated).async {
                print("🚀 Starting session run asynchronously")
                self.session.startRunning()
                print("✅ Session started and running")
            }
        } catch let error as NSError {
            deviceInput = nil
            print("🔴 Error creating device input: \(error.localizedDescription)")
        }
    }
    
    override func observeValue(forKeyPath keyPath: String?,
                               of object: Any?,
                               change: [NSKeyValueChangeKey : Any]?,
                               context: UnsafeMutableRawPointer?) {
        if keyPath == "bounds", let layer = object as? CALayer {
            // Frequency controlled log for preview layer bounds update
            if shouldLogBasedOnFrequency(lastLogTime: &lastPreviewLayerLogTime) {
                print("🖼️ Preview layer size updated: \(layer.bounds.size)")
            }
        }
    }
    
    // --- Frequency controlled logging helper function ---
    private func shouldLogBasedOnFrequency(lastLogTime: inout Date?) -> Bool {
        let currentTime = Date()
        if let lastTime = lastLogTime {
            if currentTime.timeIntervalSince(lastTime) >= logFrequencyInterval {
                lastLogTime = currentTime
                return true
            } else {
                return false
            }
        } else {
            lastLogTime = currentTime // First log, so log it and set last log time
            return true
        }
    }
    // --- End frequency controlled logging helper function ---
    
    
    // Configure the AVCaptureDeviceInput for the capture session.
    func configureDeviceInput() {
        print("⚙️ Configuring device input (configureDeviceInput - manual)")
        do {
            let deviceInput = try AVCaptureDeviceInput(device: captureDevice)
            if self.session.canAddInput(deviceInput) {
                self.session.addInput(deviceInput)
                print("✅ Device input added")
            }
        } catch let error as NSError {
            print("🔴 Error configuring device input: \(error.localizedDescription)")
        }
    }
    
    // Configure the AVCaptureVideoDataOutput for the capture session.
    func configureVideoDataOutput() {
        print("⚙️ Configuring video data output (configureVideoDataOutput - manual)")
        videoDataOutput = AVCaptureVideoDataOutput()
        videoDataOutput.alwaysDiscardsLateVideoFrames = true
        videoDataOutputQueue = DispatchQueue(label: "VideoDataOutputQueue")
        videoDataOutput.setSampleBufferDelegate(self, queue: videoDataOutputQueue)
        if session.canAddOutput(self.videoDataOutput) {
            session.addOutput(self.videoDataOutput)
            print("✅ Video data output added")
        }
        videoDataOutput.connection(with: .video)?.isEnabled = true
        videoDataOutput.connection(with: .video)?.videoOrientation = .portrait
        print("✅ Video data output configured")
    }
    
    // Configure the AVCaptureVideoPreviewLayer for the AVCaptureSession.
    func configurePreviewLayer() {
        print("⚙️ Configuring preview layer (configurePreviewLayer - manual)")
        previewLayer = AVCaptureVideoPreviewLayer(session: self.session)
        previewLayer.connection?.videoOrientation = .portrait
        previewLayer.videoGravity = AVLayerVideoGravity.resizeAspectFill
        let rootLayer: CALayer = self.previewView.layer
        rootLayer.masksToBounds = true
        previewLayer.frame = rootLayer.bounds
        rootLayer.addSublayer(self.previewLayer)
        print("✅ Preview layer configured")
    }
    
    // Stop the AVCaptureSession and remove device inputs.
    func stopCamera() {
        if session.isRunning {
            print("⏹️ Stopping camera session")
            session.stopRunning()
            print("✅ Camera session stopped")
        } else {
            print("⚠️ Camera session is already stopped")
        }
        for input in (self.session.inputs as! [AVCaptureDeviceInput]) {
            session.removeInput(input)
            print("➖ Input removed from session")
        }
    }
    
    // Start the AVCaptureSession.
    func startCamera() {
        if !session.isRunning {
            print("▶️ Starting camera session")
            DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                self?.session.startRunning()
                print("✅ Camera session started")
            }
        } else {
            print("⚠️ Camera session is already running")
        }
    }
    
    // Pause the AVCaptureSession.
    func pauseCamera() {
        if session.isRunning {
            print("⏸️ Pausing camera session")
            session.stopRunning()
            print("✅ Camera session paused")
        } else {
            print("⚠️ Camera session is already paused/stopped")
        }
    }
    
    // Resume the AVCaptureSession.
    func resumeCamera() {
        if !session.isRunning {
            print("⏯️ Resuming camera session")
            DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                guard let self = self else {
                    print("⚠️ Warning: self is nil in resumeCamera, cannot resume session.")
                    return // Exit if self is nil
                }
                self.session.startRunning()
                print("✅ Camera session resumed")
            }
        } else {
            print("⚠️ Camera session is already running, resume has no effect")
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
            connection.videoOrientation = videoOrientation(from: orientation)
        }
    }
    
    // Process the image buffer and detect faces using the SDK.
    func processImageBuffer(_ imageBuffer: CVImageBuffer) {
        autoreleasepool {
            let ciimage = CIImage(cvPixelBuffer: imageBuffer)
            self.frame = convert(cmage: ciimage)
            let height = CVPixelBufferGetHeight(imageBuffer)
            let width = CVPixelBufferGetWidth(imageBuffer)
            
            guard let tfimage = sdk.preprocessImage(self.frame) else {
                print("⚠️ Preprocess image failed, skipping frame")
                return
            }
            let face = sdk.detectLargestFace(tfimage)
            
            DispatchQueue.main.async {
                // Clear previous frame drawings, but keep the layer itself
                self.previewView.subviews.forEach({ $0.removeFromSuperview() })
                // Revert: DO NOT remove faceRectLayer from superlayer here - this was breaking the display!
                // self.faceRectLayer?.removeFromSuperlayer()
            }
            
            if face != nil {
                DispatchQueue.main.async {
                    self.processFace(tfimage: tfimage, face: face!.faceBoxAndLandmarks, width: width, height: height)
                }
            }
            
            defer {
                // Note: While ARC usually handles memory management, during high-load operations
                // you may need to explicitly call tfimage.destroy() to prevent memory growth and crashes.
                // Monitor memory usage and uncomment if needed.
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
            print("🔴 CIImage to CGImage conversion failed")
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
            self.previewView.layer.addSublayer(faceRectLayer!) // Ensure layer is added as sublayer
            print("🎨 Face rect layer created and added") // Log only once when layer is created
        } else if faceRectLayer?.superlayer == nil {
            // Re-add the layer if it somehow gets removed from superlayer (unlikely in this code, but for robustness)
            self.previewView.layer.addSublayer(faceRectLayer!)
        }
        
        
        faceRectLayer?.path = UIBezierPath(rect: rect).cgPath
        faceRectLayer?.frame = self.previewView.bounds
    }
    
    @objc func orientationChanged() {
        DispatchQueue.main.async { [weak self] in
            self?.updateOrientation()
        }
    }
    
    override func viewWillLayoutSubviews() {
        super.viewWillLayoutSubviews()
        print("viewWillLayoutSubviews - Device orientation: \(UIDevice.current.orientation.rawValue)")
        updateOrientation()
    }
}

extension FaceDetectionViewController {
    // Convert device orientation to video orientation
    func videoOrientation(from deviceOrientation: UIDeviceOrientation) -> AVCaptureVideoOrientation {
        let result: AVCaptureVideoOrientation
        let currentInterfaceOrientation = self.view.window?.windowScene?.interfaceOrientation ?? .portrait

        switch currentInterfaceOrientation {
        case .portrait:
            result = .portrait
        case .portraitUpsideDown:
            result = .portraitUpsideDown
        case .landscapeLeft:
            result = .landscapeLeft
        case .landscapeRight:
            result = .landscapeRight
        default:
            result = .landscapeRight
        }
        return result
    }

    // Update orientation for both preview layer and video connection
    func updateOrientation() {
        let deviceOrientation = UIDevice.current.orientation
        let interfaceOrientation = self.view.window?.windowScene?.interfaceOrientation ?? .portrait // Get fresh interface orientation
        print("🔄 updateOrientation - Device orientation: \(deviceOrientation.rawValue)")
        print("🔄 updateOrientation - Interface orientation: \(interfaceOrientation.rawValue)")

        let videoOrientation = videoOrientation(from: deviceOrientation) // Recalculate video orientation
        print("🔄 updateOrientation - Calculated videoOrientation: \(videoOrientation.rawValue)")

        // Update preview layer orientation
        if let connection = previewLayer?.connection, connection.isVideoOrientationSupported {
            let oldPreviewOrientation = connection.videoOrientation.rawValue
            print("🖼️ Preview layer orientation before update: \(oldPreviewOrientation)")
            connection.videoOrientation = videoOrientation
            let newPreviewOrientation = connection.videoOrientation.rawValue
            print("✅ Updated preview layer orientation to: \(newPreviewOrientation)")
        } else {
            print("⚠️ No preview layer connection or orientation not supported")
        }

        // Update video data output orientation
        if let connection = videoDataOutput?.connection(with: .video), connection.isVideoOrientationSupported {
            let oldOutputOrientation = connection.videoOrientation.rawValue
            print("📹 Video data output orientation before update: \(oldOutputOrientation)")
            connection.videoOrientation = videoOrientation
            let newOutputOrientation = connection.videoOrientation.rawValue
            print("✅ Updated video data output orientation to: \(newOutputOrientation)")
        } else {
            print("⚠️ No video data output connection or orientation not supported")
        }

        // Update preview layer frame (keep this)
        if let previewLayer = previewLayer {
            let oldFrame = previewLayer.frame
            previewLayer.frame = previewView.bounds
            print("🖼️ Updated preview layer frame from: \(oldFrame) to: \(previewLayer.frame)")
        } else {
            print("⚠️ No preview layer available for frame update")
        }
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        print("WillAppear: Forcing orientation update in viewWillAppear")
        updateOrientation() // Force orientation update every time view appears
    }
}
