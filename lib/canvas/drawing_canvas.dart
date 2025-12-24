import 'package:flutter/material.dart';

class DrawingCanvas extends StatefulWidget {
  final List<Offset?> points;
  final Function(Offset?) onUpdate;
  final double strokeWidth;

  const DrawingCanvas({
    super.key,
    required this.points,
    required this.onUpdate,
    required this.strokeWidth,
  });

  @override
  State<DrawingCanvas> createState() => _DrawingCanvasState();
}

class _DrawingCanvasState extends State<DrawingCanvas> {
  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onPanUpdate: (details) {
        RenderBox? renderBox = context.findRenderObject() as RenderBox?;
        if (renderBox != null) {
          Offset localPosition =
              renderBox.globalToLocal(details.globalPosition);
          // Ensure we don't draw outside
          if (localPosition.dx >= 0 &&
              localPosition.dy >= 0 &&
              localPosition.dx <= renderBox.size.width &&
              localPosition.dy <= renderBox.size.height) {
            widget.onUpdate(localPosition);
          }
        }
      },
      onPanEnd: (details) {
        widget.onUpdate(null); // End of stroke
      },
      child: CustomPaint(
        painter: DrawingPainter(widget.points, widget.strokeWidth),
        size: Size.infinite,
      ),
    );
  }
}

class DrawingPainter extends CustomPainter {
  final List<Offset?> points;
  final double strokeWidth;

  DrawingPainter(this.points, this.strokeWidth);

  @override
  void paint(Canvas canvas, Size size) {
    // 1. Draw Background
    final bgPaint = Paint()..color = Colors.black;
    canvas.drawRect(Rect.fromLTWH(0, 0, size.width, size.height), bgPaint);

    // 2. Draw Strokes
    final paint = Paint()
      ..color = Colors.white
      ..strokeCap = StrokeCap.round
      ..strokeWidth = strokeWidth;

    for (int i = 0; i < points.length - 1; i++) {
      if (points[i] != null && points[i + 1] != null) {
        canvas.drawLine(points[i]!, points[i + 1]!, paint);
      }
    }
  }

  @override
  bool shouldRepaint(covariant DrawingPainter oldDelegate) {
    return true; // Always repaint when points change
  }
}
