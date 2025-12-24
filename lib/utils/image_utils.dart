import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;

class ImageUtils {
  /// Converts a [List<Offset>] (drawing points) to a 28x28 grayscale pixel array.
  /// 
  /// The process involves:
  /// 1. Creating a blank canvas.
  /// 2. Drawing the strokes.
  /// 3. Capturing the image.
  /// 4. Resizing to 28x28.
  /// 5. Converting to grayscale and normalizing to [0, 1].
  static Future<List<double>> getPixelArray(
      List<Offset?> points, double strokeWidth) async {
    
    // 1. Define canvas size (large enough to draw comfortably)
    const double canvasSize = 300.0;
    
    // 2. Prepare Recorder
    final recorder = ui.PictureRecorder();
    final canvas = Canvas(recorder,
        Rect.fromPoints(const Offset(0, 0), const Offset(canvasSize, canvasSize)));
    
    // 3. Draw black background
    final bgPaint = Paint()..color = Colors.black;
    canvas.drawRect(
        const Rect.fromLTWH(0, 0, canvasSize, canvasSize), bgPaint);

    // 4. Draw white strokes
    final paint = Paint()
      ..color = Colors.white
      ..strokeCap = StrokeCap.round
      ..strokeWidth = strokeWidth;

    for (int i = 0; i < points.length - 1; i++) {
      if (points[i] != null && points[i + 1] != null) {
        canvas.drawLine(points[i]!, points[i + 1]!, paint);
      }
    }

    // 5. Convert to Image
    final picture = recorder.endRecording();
    final ui.Image image = await picture.toImage(canvasSize.toInt(), canvasSize.toInt());
    final ByteData? byteData = await image.toByteData(format: ui.ImageByteFormat.png);
    
    if (byteData == null) {
      throw Exception("Failed to capture image");
    }

    // 6. Decode with 'image' package for resizing and pixel access
    img.Image? originalImage = img.decodePng(byteData.buffer.asUint8List());

    if (originalImage == null) {
      throw Exception("Failed to decode image");
    }

    // 7. Resize to 28x28
    // Use cubic interpolation for better quality
    img.Image resized = img.copyResize(originalImage, width: 28, height: 28);
    
    // 8. Convert to Grayscale & Normalize
    // We expect a flattened array of 28*28 = 784 float values
    List<double> pixelValues = [];

    for (int y = 0; y < 28; y++) {
      for (int x = 0; x < 28; x++) {
        // Get pixel info
        img.Pixel pixel = resized.getPixel(x, y);
        // Get luminance (grayscale)
        double luminance = pixel.luminanceNormalized.toDouble(); // 0.0 to 1.0
        
        pixelValues.add(luminance);
      }
    }
    
    return pixelValues;
  }
}
