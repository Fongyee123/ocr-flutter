import 'package:flutter/material.dart';
import 'paddle_ocr_screen.dart'; // Your OCRScreen
import 'easy_ocr_screen.dart';   // Placeholder
import 'mmocr_screen.dart';      // Placeholder

void main() => runApp(OCRApp());

class OCRApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'OCR Comparison Hub',
      home: MainMenu(),
    );
  }
}

class MainMenu extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("OCR Comparison Hub")),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text("Select OCR Method", style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
              SizedBox(height: 30),
              ElevatedButton(
                onPressed: () {
                  Navigator.push(context, MaterialPageRoute(builder: (_) => EasyOCRScreen()));
                },
                child: Text("OpenCV + EasyOCR"),
              ),
              SizedBox(height: 20),
              ElevatedButton(
                onPressed: () {
                  Navigator.push(context, MaterialPageRoute(builder: (_) => PaddleOCRScreen()));
                },
                child: Text("OpenCV + PaddleOCR"),
              ),
              SizedBox(height: 20),
              ElevatedButton(
                onPressed: () {
                  Navigator.push(context, MaterialPageRoute(builder: (_) => MMOCRScreen()));
                },
                child: Text("OpenCV + MMOCR"),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
