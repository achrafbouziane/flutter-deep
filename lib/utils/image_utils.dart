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

    // 6. Decode with 'image' package
    img.Image? originalImage = img.decodePng(byteData.buffer.asUint8List());

    if (originalImage == null) {
      throw Exception("Failed to decode image");
    }

    // --- MNIST STANDARD PREPROCESSING (Center-and-Scale) ---
    // This forces the input to be a 20x20 digit centered in a 28x28 box.
    
    // Step A: Find Bounding Box
    int minX = originalImage.width;
    int minY = originalImage.height;
    int maxX = 0;
    int maxY = 0;
    bool foundContent = false;

    for (int y = 0; y < originalImage.height; y++) {
      for (int x = 0; x < originalImage.width; x++) {
        img.Pixel pixel = originalImage.getPixel(x, y);
        if (pixel.luminance > 0) {
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
          foundContent = true;
        }
      }
    }

    if (!foundContent) {
      // Return zeroed array if empty
      return List.filled(28 * 28, 0.0);
    }

    // Step B: Crop to bounding box
    int w = maxX - minX + 1;
    int h = maxY - minY + 1;
    img.Image cropped = img.copyCrop(originalImage, x: minX, y: minY, width: w, height: h);

    // Step C: Square the aspect ratio and Center
    int maxDim = w > h ? w : h;
    img.Image squareImg = img.Image(width: maxDim, height: maxDim);
    img.fill(squareImg, color: img.ColorRgb8(0, 0, 0));
    
    // Center cropped on square canvas
    int dstX = (maxDim - w) ~/ 2;
    int dstY = (maxDim - h) ~/ 2;
    img.compositeImage(squareImg, cropped, dstX: dstX, dstY: dstY);

    // Step D: Resize content to 20x20
    img.Image scaledContent = img.copyResize(squareImg, width: 20, height: 20, interpolation: img.Interpolation.cubic);

    // Step E: Paste into 28x28 black canvas (centered -> 4px margin)
    img.Image finalImage = img.Image(width: 28, height: 28);
    img.fill(finalImage, color: img.ColorRgb8(0, 0, 0));
    img.compositeImage(finalImage, scaledContent, dstX: 4, dstY: 4);

    // Step F: Normalize
    List<double> pixelValues = [];
    for (int y = 0; y < 28; y++) {
      for (int x = 0; x < 28; x++) {
        img.Pixel pixel = finalImage.getPixel(x, y);
        pixelValues.add(pixel.luminanceNormalized.toDouble());
      }
    }
    
    return pixelValues;
  }
}
