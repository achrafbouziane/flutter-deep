import 'package:flutter/material.dart';
import '../canvas/drawing_canvas.dart';
import '../ml/classifier.dart';
import '../utils/image_utils.dart';

class RecognitionScreen extends StatefulWidget {
  final String mode;
  final String title;

  const RecognitionScreen({
    super.key,
    required this.mode,
    required this.title,
  });

  @override
  State<RecognitionScreen> createState() => _RecognitionScreenState();
}

class _RecognitionScreenState extends State<RecognitionScreen> {
  final Classifier _classifier = Classifier();
  final List<Offset?> _points = [];
  double _strokeWidth = 30.0;
  String _result = "Draw something!";
  bool _isLoading = false;

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    setState(() => _isLoading = true);
    // assets/digits.tflite, assets/digits.txt
    String modelPath = 'assets/${widget.mode}.tflite';
    String labelsPath = 'assets/${widget.mode}.txt';
    
    await _classifier.loadModel(modelPath, labelsPath);
    setState(() {
      _isLoading = false;
      _result = _classifier.isLoaded 
          ? "Model Loaded. Draw & Predict!" 
          : "Failed to load model. Run training scripts.";
    });
  }

  @override
  void dispose() {
    _classifier.close();
    super.dispose();
  }

  void _addPoint(Offset? point) {
    setState(() {
      _points.add(point);
    });
  }

  void _clearCanvas() {
    setState(() {
      _points.clear();
      _result = "Canvas Cleared";
    });
  }

  Future<void> _predict() async {
    if (_points.isEmpty) {
      setState(() => _result = "Canvas is empty!");
      return;
    }
    
    if (!_classifier.isLoaded) {
      await _loadModel();
      if (!_classifier.isLoaded) return;
    }

    try {
      // Get pixels
      List<double> pixels = await ImageUtils.getPixelArray(_points, _strokeWidth);
      
      // Predict
      final predictions = await _classifier.predict(pixels);
      
      if (predictions.isEmpty) {
        setState(() => _result = "No prediction");
        return;
      }

      // Sort by confidence
      var sortedKeys = predictions.keys.toList(growable: false)
        ..sort((k1, k2) => predictions[k2]!.compareTo(predictions[k1]!));
        
      String topLabel = sortedKeys.first;
      double topConf = predictions[topLabel]!;

      setState(() {
        _result = "Prediction: $topLabel\nConfidence: ${(topConf * 100).toStringAsFixed(1)}%";
      });
      
    } catch (e) {
      setState(() => _result = "Error: $e");
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.title),
        backgroundColor: Colors.deepPurple,
        foregroundColor: Colors.white,
        actions: [
          IconButton(
            icon: const Icon(Icons.delete),
            onPressed: _clearCanvas,
            tooltip: 'Clear Canvas',
          ),
        ],
      ),
      body: Column(
        children: [
          // Upper area for result
          Container(
            height: 100,
            width: double.infinity,
            color: Colors.deepPurple.shade50,
            alignment: Alignment.center,
            child: _isLoading 
                ? const CircularProgressIndicator()
                : Text(
                    _result,
                    textAlign: TextAlign.center,
                    style: const TextStyle(
                      fontSize: 20, 
                      fontWeight: FontWeight.bold,
                      color: Colors.deepPurple
                    ),
                  ),
          ),
          
          // Canvas Area
          Expanded(
            child: Container(
              margin: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                border: Border.all(color: Colors.deepPurple, width: 3),
                color: Colors.black,
              ),
              child: DrawingCanvas(
                points: _points,
                onUpdate: _addPoint,
                strokeWidth: _strokeWidth,
              ),
            ),
          ),

          // Controls
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Row(
              children: [
                const Text("Stroke: "),
                Expanded(
                  child: Slider(
                    value: _strokeWidth,
                    min: 10.0,
                    max: 50.0,
                    onChanged: (val) => setState(() => _strokeWidth = val),
                  ),
                ),
                ElevatedButton.icon(
                  onPressed: _predict,
                  icon: const Icon(Icons.psychology),
                  label: const Text("PREDICT"),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.deepPurple,
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
                  ),
                )
              ],
            ),
          ),
        ],
      ),
    );
  }
}
