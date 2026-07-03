package org.openmed.scan

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview as ComposePreview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.google.mlkit.vision.common.InputImage
import kotlinx.coroutines.launch

@Composable
fun OpenMedScanDemoApp(
    viewModel: ScanFlowViewModel = remember { ScanFlowViewModel() },
) {
    val state = viewModel.uiState
    MaterialTheme(
        colorScheme = lightColorScheme(
            primary = Color(0xFF0F766E),
            secondary = Color(0xFF1D4ED8),
            tertiary = Color(0xFFB45309),
            background = Color(0xFFF8FAFC),
            surface = Color.White,
        ),
    ) {
        Surface(
            modifier = Modifier.fillMaxSize(),
            color = MaterialTheme.colorScheme.background,
        ) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .verticalScroll(rememberScrollState())
                    .padding(20.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
            ) {
                Header()
                StageIndicator(state)
                if (state.isWorking) {
                    WorkingPanel(state)
                }
                state.errorMessage?.let { message ->
                    ErrorPanel(message)
                }
                if (state.stage == ScanStage.RESULT && state.result != null) {
                    ResultScreen(
                        result = state.result,
                        source = state.source,
                        onCaptureAnother = viewModel::restart,
                    )
                } else {
                    CameraCaptureScreen(viewModel)
                }
            }
        }
    }
}

@Composable
fun CameraCaptureScreen(viewModel: ScanFlowViewModel) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED,
        )
    }
    var imageCapture by remember { mutableStateOf<ImageCapture?>(null) }
    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted ->
        hasCameraPermission = granted
    }

    LaunchedEffect(Unit) {
        if (!hasCameraPermission) {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    Card(
        modifier = Modifier.widthIn(max = 860.dp),
        shape = RoundedCornerShape(8.dp),
        colors = CardDefaults.cardColors(containerColor = Color.White),
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            Text(
                text = "Document Capture",
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.SemiBold,
            )
            if (hasCameraPermission) {
                CameraPreview(
                    onImageCaptureReady = { imageCapture = it },
                    onCameraUnavailable = {
                        viewModel.showError("Camera preview is unavailable. Use the sample image fallback.")
                    },
                )
            } else {
                PermissionFallback()
            }
            Row(
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Button(
                    enabled = hasCameraPermission && imageCapture != null && !viewModel.uiState.isWorking,
                    onClick = {
                        imageCapture?.let { capture ->
                            captureStill(
                                context = context,
                                imageCapture = capture,
                                onImage = { imageProxy ->
                                    val mediaImage = imageProxy.image
                                    if (mediaImage == null) {
                                        imageProxy.close()
                                        viewModel.showError("Camera frame was unavailable. Use the sample image fallback.")
                                        return@captureStill
                                    }
                                    val image = InputImage.fromMediaImage(
                                        mediaImage,
                                        imageProxy.imageInfo.rotationDegrees,
                                    )
                                    scope.launch {
                                        try {
                                            viewModel.processImage(image, ScanSource.CAMERA)
                                        } finally {
                                            imageProxy.close()
                                        }
                                    }
                                },
                                onError = {
                                    viewModel.showError(
                                        "Camera capture failed. Use the sample image fallback.",
                                    )
                                },
                            )
                        }
                    },
                ) {
                    Text("Capture")
                }
                OutlinedButton(
                    enabled = !viewModel.uiState.isWorking,
                    onClick = {
                        scope.launch {
                            viewModel.processSampleDocument(context)
                        }
                    },
                ) {
                    Text("Use Sample")
                }
            }
        }
    }
}

@Composable
private fun CameraPreview(
    onImageCaptureReady: (ImageCapture) -> Unit,
    onCameraUnavailable: () -> Unit,
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    AndroidView(
        modifier = Modifier
            .fillMaxWidth()
            .height(360.dp)
            .background(Color(0xFFE2E8F0), RoundedCornerShape(8.dp)),
        factory = { viewContext ->
            PreviewView(viewContext).apply {
                scaleType = PreviewView.ScaleType.FILL_CENTER
            }
        },
        update = { previewView ->
            val providerFuture = ProcessCameraProvider.getInstance(context)
            providerFuture.addListener(
                {
                    try {
                        val provider = providerFuture.get()
                        val preview = Preview.Builder().build().also {
                            it.setSurfaceProvider(previewView.surfaceProvider)
                        }
                        val capture = ImageCapture.Builder()
                            .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                            .build()
                        provider.unbindAll()
                        provider.bindToLifecycle(
                            lifecycleOwner,
                            CameraSelector.DEFAULT_BACK_CAMERA,
                            preview,
                            capture,
                        )
                        onImageCaptureReady(capture)
                    } catch (_: Exception) {
                        onCameraUnavailable()
                    }
                },
                ContextCompat.getMainExecutor(context),
            )
        },
    )
}

private fun captureStill(
    context: Context,
    imageCapture: ImageCapture,
    onImage: (ImageProxy) -> Unit,
    onError: (ImageCaptureException) -> Unit,
) {
    imageCapture.takePicture(
        ContextCompat.getMainExecutor(context),
        object : ImageCapture.OnImageCapturedCallback() {
            override fun onCaptureSuccess(image: ImageProxy) {
                onImage(image)
            }

            override fun onError(exception: ImageCaptureException) {
                onError(exception)
            }
        },
    )
}

@Composable
private fun Header() {
    Column(
        modifier = Modifier.widthIn(max = 860.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Text(
            text = "OpenMed Scan",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold,
        )
        Text(
            text = "Capture, OCR, and redaction stay on device. Not a medical device; review output before clinical use.",
            style = MaterialTheme.typography.bodyMedium,
            color = Color(0xFF475569),
        )
    }
}

@Composable
private fun StageIndicator(state: ScanUiState) {
    Row(
        modifier = Modifier.widthIn(max = 860.dp),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        ScanStage.values().forEach { stage ->
            val active = stage == state.stage
            Surface(
                modifier = Modifier.weight(1f),
                color = if (active) MaterialTheme.colorScheme.primary else Color(0xFFE2E8F0),
                contentColor = if (active) Color.White else Color(0xFF334155),
                shape = RoundedCornerShape(8.dp),
            ) {
                Text(
                    text = stage.title,
                    modifier = Modifier.padding(horizontal = 8.dp, vertical = 10.dp),
                    style = MaterialTheme.typography.labelLarge,
                    fontWeight = FontWeight.SemiBold,
                )
            }
        }
    }
}

@Composable
private fun WorkingPanel(state: ScanUiState) {
    val phase = state.phase ?: return
    Card(
        modifier = Modifier.widthIn(max = 860.dp),
        shape = RoundedCornerShape(8.dp),
        colors = CardDefaults.cardColors(containerColor = Color(0xFFECFEFF)),
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                CircularProgressIndicator(modifier = Modifier.size(20.dp), strokeWidth = 2.dp)
                Text(
                    text = "${phase.title}: ${phase.detail}",
                    modifier = Modifier.weight(1f),
                    style = MaterialTheme.typography.bodyMedium,
                    fontWeight = FontWeight.Medium,
                )
            }
            LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
        }
    }
}

@Composable
private fun PermissionFallback() {
    Surface(
        modifier = Modifier
            .fillMaxWidth()
            .height(220.dp),
        color = Color(0xFFF1F5F9),
        shape = RoundedCornerShape(8.dp),
    ) {
        Column(
            modifier = Modifier.padding(18.dp),
            verticalArrangement = Arrangement.Center,
        ) {
            Text(
                text = "Camera permission is unavailable.",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold,
            )
            Text(
                text = "The bundled synthetic sample runs the same OCR-text to de-identification stages.",
                style = MaterialTheme.typography.bodyMedium,
                color = Color(0xFF475569),
            )
        }
    }
}

@Composable
private fun ErrorPanel(message: String) {
    Surface(
        modifier = Modifier.widthIn(max = 860.dp),
        color = Color(0xFFFFF1F2),
        contentColor = Color(0xFF9F1239),
        shape = RoundedCornerShape(8.dp),
    ) {
        Text(
            text = message,
            modifier = Modifier.padding(14.dp),
            style = MaterialTheme.typography.bodyMedium,
            fontWeight = FontWeight.Medium,
        )
    }
}

@ComposePreview(showBackground = true)
@Composable
private fun OpenMedScanDemoPreview() {
    OpenMedScanDemoApp(
        viewModel = ScanFlowViewModel(
            pipeline = ScanPipeline(
                ocrAdapter = com.openmed.openmedkit.ocr.FakeOcrAdapter(
                    SampleClinicalDocument.ocrResult(),
                ),
                deidentifier = OnDeviceDeidentifier(),
            ),
        ),
    )
}
