import SwiftUI
#if canImport(VisionKit)
import VisionKit
#endif

/// Wraps `VNDocumentCameraViewController` for SwiftUI. Hands the captured
/// pages back via `onComplete`; errors surface through `onError` so the
/// caller can populate an alert.
#if canImport(UIKit) && canImport(VisionKit)
public struct ScannerSheet: UIViewControllerRepresentable {
    public let onComplete: ([UIImage]) -> Void
    public let onCancel: () -> Void
    public let onError: (Error) -> Void

    public init(
        onComplete: @escaping ([UIImage]) -> Void,
        onCancel: @escaping () -> Void,
        onError: @escaping (Error) -> Void
    ) {
        self.onComplete = onComplete
        self.onCancel = onCancel
        self.onError = onError
    }

    public func makeCoordinator() -> Coordinator { Coordinator(self) }

    public func makeUIViewController(context: Context) -> VNDocumentCameraViewController {
        let controller = VNDocumentCameraViewController()
        controller.delegate = context.coordinator
        return controller
    }

    public func updateUIViewController(_ uiViewController: VNDocumentCameraViewController, context: Context) {}

    public final class Coordinator: NSObject, VNDocumentCameraViewControllerDelegate {
        private let parent: ScannerSheet
        init(_ parent: ScannerSheet) { self.parent = parent }

        public func documentCameraViewControllerDidCancel(_ controller: VNDocumentCameraViewController) {
            parent.onCancel()
        }
        public func documentCameraViewController(_ controller: VNDocumentCameraViewController, didFailWithError error: Error) {
            parent.onError(error)
        }
        public func documentCameraViewController(_ controller: VNDocumentCameraViewController, didFinishWith scan: VNDocumentCameraScan) {
            let pages = (0..<scan.pageCount).map { scan.imageOfPage(at: $0) }
            parent.onComplete(pages)
        }
    }
}
#else
public struct ScannerSheet: View {
    public let onComplete: ([Any]) -> Void
    public let onCancel: () -> Void
    public let onError: (Error) -> Void

    public init(onComplete: @escaping ([Any]) -> Void, onCancel: @escaping () -> Void, onError: @escaping (Error) -> Void) {
        self.onComplete = onComplete
        self.onCancel = onCancel
        self.onError = onError
    }

    public var body: some View {
        VStack(spacing: 12) {
            Text("Document scanning is only available on iOS devices with a camera.")
                .multilineTextAlignment(.center)
            Button("Close", action: onCancel).buttonStyle(.omSecondary(.sm))
        }
        .padding(24)
    }
}
#endif

/// Thin runtime check so callers can disable the scan CTA on simulators.
public enum ScannerSupport {
    public static var isSupported: Bool {
        #if canImport(VisionKit)
        return VNDocumentCameraViewController.isSupported
        #else
        return false
        #endif
    }
}
