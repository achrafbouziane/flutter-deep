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

  // ... (omitted)

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
