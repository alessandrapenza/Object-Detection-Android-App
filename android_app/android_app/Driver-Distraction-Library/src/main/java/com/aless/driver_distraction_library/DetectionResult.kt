package com.aless.driver_distraction_library

data class DetectionResult(
    val state: DriverState,
    val confidence: Float,
    val boxes: List<BoundingBox> = emptyList()
)