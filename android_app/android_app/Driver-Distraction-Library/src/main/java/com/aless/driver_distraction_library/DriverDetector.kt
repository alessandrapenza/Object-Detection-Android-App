package com.aless.driver_distraction_library

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStream
import java.io.InputStreamReader

class DriverDetector(
    private val context: Context,
    private val modelPath: String = "model.tflite",
    private val labelPath: String = "labels.txt",
    private val distractedLabels: Set<String> = setOf("phone", "bottle") //classi che definisono lo stato DISTRACTED
) {

    private var interpreter: Interpreter? = null
    private val labels = mutableListOf<String>()

    private var tensorWidth = 0
    private var tensorHeight = 0
    private var numChannel = 0
    private var numElements = 0

    private val imageProcessor = ImageProcessor.Builder()
        .add(NormalizeOp(INPUT_MEAN, INPUT_STANDARD_DEVIATION))
        .add(CastOp(INPUT_IMAGE_TYPE))
        .build()

    fun setup() {
        val model = FileUtil.loadMappedFile(context, modelPath)
        val options = Interpreter.Options().apply { numThreads = 4 }
        interpreter = Interpreter(model, options)

        val inShape = interpreter!!.getInputTensor(0).shape()
        val outShape = interpreter!!.getOutputTensor(0).shape()

        tensorWidth = inShape[1]
        tensorHeight = inShape[2]
        numChannel = outShape[1]
        numElements = outShape[2]

        try {
            val inputStream: InputStream = context.assets.open(labelPath)
            val reader = BufferedReader(InputStreamReader(inputStream))
            var line: String? = reader.readLine()
            while (!line.isNullOrEmpty()) {
                labels.add(line)
                line = reader.readLine()
            }
            reader.close()
            inputStream.close()
            Log.d(TAG, "Labels loaded: ${labels.size}")
        } catch (e: IOException) {
            Log.e(TAG, "Error loading labels: ${e.message}", e)
        }

        Log.d(TAG, "Model setup complete: in=${inShape.joinToString()}, out=${outShape.joinToString()}")
    }

    fun clear() {
        interpreter?.close()
        interpreter = null
    }

    /**
     * Livello basso: restituisce solo le box.
     * Se non trova nulla -> lista vuota.
     */
    fun detect(frame: Bitmap): List<BoundingBox> {
        val intr = interpreter ?: throw IllegalStateException("Call setup() before detect()")
        if (tensorWidth == 0 || tensorHeight == 0 || numChannel == 0 || numElements == 0) {
            throw IllegalStateException("Tensor dimensions not initialized correctly.")
        }

        val start = SystemClock.uptimeMillis()

        val resizedBitmap = Bitmap.createScaledBitmap(frame, tensorWidth, tensorHeight, false)

        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(resizedBitmap)
        val processedImage = imageProcessor.process(tensorImage)
        val imageBuffer = processedImage.buffer

        val output = TensorBuffer.createFixedSize(
            intArrayOf(1, numChannel, numElements),
            OUTPUT_IMAGE_TYPE
        )

        intr.run(imageBuffer, output.buffer)

        val boxes = bestBox(output.floatArray) ?: emptyList()
        val inferenceTime = SystemClock.uptimeMillis() - start
        Log.d(TAG, "Inference time = ${inferenceTime}ms, boxes=${boxes.size}")

        return boxes
    }

    /**
     * Livello alto: restituisce direttamente lo stato del guidatore.
     */
    fun detectState(frame: Bitmap): DetectionResult {
        val boxes = detect(frame)
        if (boxes.isEmpty()) {
            return DetectionResult(
                state = DriverState.ATTENTIVE,
                confidence = 0f,
                boxes = emptyList()
            )
        }

        // prendo la box con confidenza massima
        val best = boxes.maxByOrNull { it.cnf }!!

        val state = if (best.clsName in distractedLabels) {
            DriverState.DISTRACTED
        } else {
            DriverState.ATTENTIVE
        }

        return DetectionResult(
            state = state,
            confidence = best.cnf,
            boxes = boxes
        )
    }

    // ---- POST-PROCESSING ----

    private fun bestBox(array: FloatArray): List<BoundingBox>? {
        val boundingBoxes = mutableListOf<BoundingBox>()

        for (c in 0 until numElements) {
            var maxConf = -1.0f
            var maxIdx = -1
            var j = 4
            var arrayIdx = c + numElements * j
            while (j < numChannel) {
                if (array[arrayIdx] > maxConf) {
                    maxConf = array[arrayIdx]
                    maxIdx = j - 4
                }
                j++
                arrayIdx += numElements
            }

            if (maxConf > CONFIDENCE_THRESHOLD) {
                val clsName = labels[maxIdx]
                val cx = array[c]
                val cy = array[c + numElements]
                val w = array[c + numElements * 2]
                val h = array[c + numElements * 3]
                val x1 = cx - (w / 2F)
                val y1 = cy - (h / 2F)
                val x2 = cx + (w / 2F)
                val y2 = cy + (h / 2F)

                if (x1 < 0F || x1 > 1F || y1 < 0F || y1 > 1F ||
                    x2 < 0F || x2 > 1F || y2 < 0F || y2 > 1F
                ) {
                    continue
                }

                boundingBoxes.add(
                    BoundingBox(
                        x1 = x1, y1 = y1, x2 = x2, y2 = y2,
                        cx = cx, cy = cy, w = w, h = h,
                        cnf = maxConf, cls = maxIdx, clsName = clsName
                    )
                )
            }
        }

        if (boundingBoxes.isEmpty()) return null
        return applyNMS(boundingBoxes)
    }

    private fun applyNMS(boxes: List<BoundingBox>): MutableList<BoundingBox> {
        val sortedBoxes = boxes.sortedByDescending { it.cnf }.toMutableList()
        val selectedBoxes = mutableListOf<BoundingBox>()

        while (sortedBoxes.isNotEmpty()) {
            val first = sortedBoxes.first()
            selectedBoxes.add(first)
            sortedBoxes.remove(first)

            val iterator = sortedBoxes.iterator()
            while (iterator.hasNext()) {
                val nextBox = iterator.next()
                val iou = calculateIoU(first, nextBox)
                if (iou >= IOU_THRESHOLD) {
                    iterator.remove()
                }
            }
        }
        return selectedBoxes
    }

    private fun calculateIoU(box1: BoundingBox, box2: BoundingBox): Float {
        val x1 = maxOf(box1.x1, box2.x1)
        val y1 = maxOf(box1.y1, box2.y1)
        val x2 = minOf(box1.x2, box2.x2)
        val y2 = minOf(box1.y2, box2.y2)
        val intersectionArea = maxOf(0F, x2 - x1) * maxOf(0F, y2 - y1)
        val box1Area = box1.w * box1.h
        val box2Area = box2.w * box2.h
        return intersectionArea / (box1Area + box2Area - intersectionArea)
    }

    companion object {
        private const val TAG = "DriverDetector"

        private const val INPUT_MEAN = 0f
        private const val INPUT_STANDARD_DEVIATION = 255f
        private val INPUT_IMAGE_TYPE = DataType.FLOAT32
        private val OUTPUT_IMAGE_TYPE = DataType.FLOAT32

        private const val CONFIDENCE_THRESHOLD = 0.4F
        private const val IOU_THRESHOLD = 0.5F
    }
}
