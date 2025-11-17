package com.surendramaran.yolov8tflite

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Build
import android.os.Bundle
import android.os.VibrationEffect
import android.os.Vibrator
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.RequiresPermission
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.AspectRatio
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.surendramaran.yolov8tflite.Constants.LABELS_PATH
import com.surendramaran.yolov8tflite.Constants.MODEL_PATH
import com.surendramaran.yolov8tflite.databinding.ActivityMainBinding
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import com.aless.driver_distraction_library.DriverDetector
import com.aless.driver_distraction_library.DriverState

class MainActivity : AppCompatActivity(), Detector.DetectorListener {
    enum class DisplayMode { BOX, ALERT }

    private var displayMode = DisplayMode.BOX

    // Sound
    private var soundPool: android.media.SoundPool? = null
    private var alertSoundId: Int = 0
    private var lastAlertAt = 0L
    private val ALERT_COOLDOWN_MS = 1000L  // suono max 1 volta al secondo
    private lateinit var binding: ActivityMainBinding
    private val isFrontCamera = true

    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null

    // --- THROTTLE + HOLD ---
    private var lastProcessedAt = 0L
    private val MIN_INTERVAL_MS = 120L   // processa max ~8 fps (alza/abbassa a piacere)

    private val HOLD_MS = 800L           // mantieni le box per 0.4s dopo l’ultimo hit
    private val uiHandler = android.os.Handler(android.os.Looper.getMainLooper())
    private val clearRunnable = Runnable { binding.overlay.clearDetections() }

    // --- SMOOTHING ---
    private var lastBoxes: List<BoundingBox>? = null
    private val SMOOTH_ALPHA = 0.6f      // 0.0 = tutto passato, 1.0 = tutto nuovo

    //private lateinit var detector: Detector

    private lateinit var detector: DriverDetector

    private lateinit var cameraExecutor: ExecutorService

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Init APP DETECTOR
        //detector = Detector(baseContext, MODEL_PATH, LABELS_PATH, this)
        //detector.setup()

        // Init LIBRARY DETECTOR
        detector = DriverDetector(
            context = this,
            modelPath = "model.tflite",
            labelPath = "labels.txt",
            distractedLabels = setOf("phone", "bottle")
        )
        detector.setup()

        // Spinner modalità
        binding.spinnerMode.onItemSelectedListener = object : android.widget.AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: android.widget.AdapterView<*>, view: android.view.View?, position: Int, id: Long) {
                displayMode = if (position == 0) DisplayMode.BOX else DisplayMode.ALERT
                applyDisplayModeIdleState()
            }
            override fun onNothingSelected(parent: android.widget.AdapterView<*>) {}
        }

        // SoundPool semplice
        soundPool = android.media.SoundPool.Builder().setMaxStreams(1).build()
        alertSoundId = soundPool!!.load(this, R.raw.alert, 1)

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }

        cameraExecutor = Executors.newSingleThreadExecutor()
    }

    private fun applyDisplayModeIdleState() {
        // quando cambi modalità e non c’è detection attiva, pulisci UI coerentemente
        when (displayMode) {
            DisplayMode.BOX -> {
                binding.overlay.visibility = android.view.View.VISIBLE
                binding.alertBanner.visibility = android.view.View.GONE
                binding.overlay.clearDetections()
            }
            DisplayMode.ALERT -> {
                binding.overlay.clearDetections()
                binding.overlay.visibility = android.view.View.GONE
                binding.alertBanner.visibility = android.view.View.GONE
            }
        }
    }

    private fun showAlertBanner(text: String) {
        if (binding.alertBanner.visibility != View.VISIBLE) {
            binding.alertBanner.alpha = 0f
            binding.alertBanner.visibility = View.VISIBLE
            binding.alertBanner.animate().alpha(1f).setDuration(150).start()
        }
        binding.alertText.text = text

        val now = System.currentTimeMillis()
        if (now - lastAlertAt > ALERT_COOLDOWN_MS) {
            lastAlertAt = now
            soundPool?.play(alertSoundId, 1f, 1f, 1, 0, 1f)
            // vibrazione
//            val v = getSystemService(VIBRATOR_SERVICE) as Vibrator
//             if (Build.VERSION.SDK_INT >= 26)
//                 v.vibrate(VibrationEffect.createOneShot(120, VibrationEffect.DEFAULT_AMPLITUDE))
//             else v.vibrate(120)
        }
    }

    private fun hideAlertBanner() {
        if (binding.alertBanner.visibility == android.view.View.VISIBLE) {
            binding.alertBanner.animate().alpha(0f).setDuration(120).withEndAction {
                binding.alertBanner.visibility = android.view.View.GONE
            }.start()
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            cameraProvider  = cameraProviderFuture.get()
            bindCameraUseCases()
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindCameraUseCases() {
        val cameraProvider = cameraProvider ?: throw IllegalStateException("Camera initialization failed.")

        val rotation = binding.viewFinder.display.rotation

        val cameraSelector = CameraSelector
            .Builder()
            .requireLensFacing(CameraSelector.LENS_FACING_FRONT)
            .build()

        preview =  Preview.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(rotation)
            .build()

        imageAnalyzer = ImageAnalysis.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setTargetRotation(binding.viewFinder.display.rotation)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()

        imageAnalyzer?.setAnalyzer(cameraExecutor) { imageProxy ->
            val now = System.currentTimeMillis()
            if (now - lastProcessedAt < MIN_INTERVAL_MS) {
                imageProxy.close()
                return@setAnalyzer
            }
            lastProcessedAt = now

            val bitmapBuffer =
                Bitmap.createBitmap(
                    imageProxy.width,
                    imageProxy.height,
                    Bitmap.Config.ARGB_8888
                )
            imageProxy.use { bitmapBuffer.copyPixelsFromBuffer(imageProxy.planes[0].buffer) }
            imageProxy.close()

            val matrix = Matrix().apply {
                postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())

                if (isFrontCamera) {
                    postScale(
                        -1f,
                        1f,
                        imageProxy.width.toFloat(),
                        imageProxy.height.toFloat()
                    )
                }
            }

            val rotatedBitmap = Bitmap.createBitmap(
                bitmapBuffer, 0, 0, bitmapBuffer.width, bitmapBuffer.height,
                matrix, true
            )

            // Detect APP DETECTOR
            //detector.detect(rotatedBitmap)

            // Detect LIBRARY DETECTOR
            val result = detector.detectState(rotatedBitmap)

            Log.d("TestLibrary", "State = ${result.state}, conf = ${result.confidence}")
            Log.d("TestLibrary", "Boxes = ${result.boxes.size}")

            runOnUiThread {
                when (result.state) {
                    DriverState.ATTENTIVE -> {
                        Toast.makeText(this, "ATTENTIVE (${result.confidence})", Toast.LENGTH_SHORT).show()
                    }
                    DriverState.DISTRACTED -> {
                        Toast.makeText(this, "DISTRACTED (${result.confidence})", Toast.LENGTH_SHORT).show()
                    }
                }
            }
        }

        cameraProvider.unbindAll()

        try {
            camera = cameraProvider.bindToLifecycle(
                this,
                cameraSelector,
                preview,
                imageAnalyzer
            )

            preview?.setSurfaceProvider(binding.viewFinder.surfaceProvider)
        } catch(exc: Exception) {
            Log.e(TAG, "Use case binding failed", exc)
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()) {
        if (it[Manifest.permission.CAMERA] == true) { startCamera() }
    }

    override fun onDestroy() {
        super.onDestroy()
        detector.clear()
        cameraExecutor.shutdown()
        soundPool?.release()
        soundPool = null
    }

    override fun onResume() {
        super.onResume()
        if (allPermissionsGranted()){
            startCamera()
        } else {
            requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
        }
    }

    companion object {
        private const val TAG = "Camera"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = mutableListOf (
            Manifest.permission.CAMERA
        ).toTypedArray()
    }

    private fun iou(a: BoundingBox, b: BoundingBox): Float {
        val x1 = maxOf(a.x1, b.x1)
        val y1 = maxOf(a.y1, b.y1)
        val x2 = minOf(a.x2, b.x2)
        val y2 = minOf(a.y2, b.y2)
        val inter = maxOf(0f, x2 - x1) * maxOf(0f, y2 - y1)
        val areaA = a.w * a.h
        val areaB = b.w * b.h
        val denom = areaA + areaB - inter
        return if (denom <= 0f) 0f else inter / denom
    }

    private fun lerp(a: Float, b: Float, t: Float) = a + (b - a) * t

    private fun smoothBox(prev: BoundingBox, cur: BoundingBox, alpha: Float): BoundingBox =
        BoundingBox(
            x1 = lerp(prev.x1, cur.x1, alpha),
            y1 = lerp(prev.y1, cur.y1, alpha),
            x2 = lerp(prev.x2, cur.x2, alpha),
            y2 = lerp(prev.y2, cur.y2, alpha),
            cx = lerp(prev.cx, cur.cx, alpha),
            cy = lerp(prev.cy, cur.cy, alpha),
            w  = lerp(prev.w,  cur.w,  alpha),
            h  = lerp(prev.h,  cur.h,  alpha),
            cnf = lerp(prev.cnf, cur.cnf, alpha),
            cls = cur.cls,
            clsName = cur.clsName
        )

    private fun smoothWithPrev(
        prevList: List<BoundingBox>?,
        curList: List<BoundingBox>,
        alpha: Float,
        matchIou: Float = 0.3f
    ): List<BoundingBox> {
        if (prevList == null || prevList.isEmpty()) return curList
        val usedPrev = BooleanArray(prevList.size)
        val result = mutableListOf<BoundingBox>()

        for (cur in curList) {
            var best = -1
            var bestIou = 0f
            for (i in prevList.indices) {
                if (usedPrev[i]) continue
                val p = prevList[i]
                if (p.cls != cur.cls) continue
                val iouVal = iou(p, cur)
                if (iouVal > bestIou) { bestIou = iouVal; best = i }
            }
            if (best >= 0 && bestIou >= matchIou) {
                usedPrev[best] = true
                result.add(smoothBox(prevList[best], cur, alpha))
            } else {
                result.add(cur) // nuovo box: niente smoothing
            }
        }
        return result
    }


    override fun onEmptyDetect() {
        runOnUiThread {
            when (displayMode) {
                DisplayMode.BOX -> {
                    binding.overlay.clearDetections()
                    // overlay resta visibile in BOX
                }
                DisplayMode.ALERT -> {
                    hideAlertBanner()
                    binding.overlay.visibility = android.view.View.GONE
                }
            }
        }
    }

    override fun onDetect(boundingBoxes: List<BoundingBox>, inferenceTime: Long) {
        runOnUiThread {
            binding.inferenceTime.text = "${inferenceTime}ms"

            when (displayMode) {
                DisplayMode.BOX -> {
                    // Mostra le bbox e nascondi banner
                    binding.overlay.visibility = android.view.View.VISIBLE
                    binding.overlay.setResults(boundingBoxes)
                    hideAlertBanner()
                }
                DisplayMode.ALERT -> {
                    // Nascondi bbox e mostra banner con testo
                    binding.overlay.clearDetections()
                    binding.overlay.visibility = android.view.View.GONE

                    // Costruisci un messaggio (ad es. prima classe con conf max)
                    val top = boundingBoxes.maxByOrNull { it.cnf }
                    val label = top?.clsName ?: "Object"
                    val conf = if (top != null) String.format("%.0f%%", top.cnf * 100) else ""
                    showAlertBanner("Alert: $label detected $conf")
                }
            }

            // 1) smoothing con EMA sui box (stessa classe e IoU alto)
            val smoothed = smoothWithPrev(lastBoxes, boundingBoxes, SMOOTH_ALPHA)
            lastBoxes = smoothed

            // 2) mostra box
            binding.overlay.setResults(smoothed)

            // 3) “sticky”: ogni hit rinvia il clear
            uiHandler.removeCallbacks(clearRunnable)
            uiHandler.postDelayed(clearRunnable, HOLD_MS)
        }
    }
}
