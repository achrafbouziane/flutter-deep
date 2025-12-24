import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

class Classifier {
  Interpreter? _interpreter;
  List<String> _labels = [];

  bool get isLoaded => _interpreter != null;

  /// Loads the interpreter and labels.
  /// [modelPath] like 'assets/digits.tflite'
  /// [labelsPath] like 'assets/digits.txt'
  Future<void> loadModel(String modelPath, String labelsPath) async {
    try {
      _interpreter = await Interpreter.fromAsset(modelPath);
      print('Loaded model: $modelPath');

      final labelsData = await rootBundle.loadString(labelsPath);
      _labels = labelsData.split('\n').where((l) => l.isNotEmpty).toList();
      print('Loaded labels: ${_labels.length}');
    } catch (e) {
      print('Error loading model: $e');
    }
  }

  /// Runs inference on the input image (flattened 28x28 = 784 float values).
  /// Returns a map of {label: confidence}.
  Future<Map<String, double>> predict(List<double> input) async {
    if (_interpreter == null) {
      throw Exception('Interpreter not loaded');
    }

    // Input shape: [1, 28, 28, 1]
    // The input list is 784 long.
    // Reshape to [1, 28, 28, 1] for the model.
    // Ensure input is float32
    var inputBuffer = Float32List.fromList(input).reshape([1, 28, 28, 1]);

    // Output shape: [1, num_classes]
    var outputBuffer = List.filled(1 * _labels.length, 0.0).reshape([1, _labels.length]);

    _interpreter!.run(inputBuffer, outputBuffer);

    // Parse result
    List<double> output = List<double>.from(outputBuffer[0]);
    Map<String, double> results = {};

    for (int i = 0; i < output.length; i++) {
      if (i < _labels.length) {
        results[_labels[i]] = output[i];
      }
    }

    return results;
  }

  void close() {
    _interpreter?.close();
  }
}
