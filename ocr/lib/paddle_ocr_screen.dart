// r/Handwriting The quick brown fox jumped over the lazy dog. How's my handwriting?
// every day is a fresh START
// Static 1)Standard Static Route 2)Fully Specified Static Route 3)Default Static Route 4)Floating Static Route 5)Summary Route

import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:flutter/services.dart';

class PaddleOCRScreen extends StatefulWidget {
  @override
  _PaddleOCRScreenState createState() => _PaddleOCRScreenState();
}

class _PaddleOCRScreenState extends State<PaddleOCRScreen> {
  File? _image;
  String _ocrText = '';
  List<dynamic> _words = [];
  List<dynamic> _wordsBefore = [];
  bool _isLoading = false;

  String _groundTruth = '';
  TextEditingController _groundTruthController = TextEditingController();
  Map<String, dynamic>? _beforePreprocessing;
  Map<String, dynamic>? _afterPreprocessing;
  double? _accuracyImprovement;
  double? _charAccuracyImprovement;

  Future<void> _pickImage(ImageSource source) async {
    final pickedFile = await ImagePicker().pickImage(source: source);
    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
        _ocrText = '';
        _words = [];
        _wordsBefore = [];
        _beforePreprocessing = null;
        _afterPreprocessing = null;
        _accuracyImprovement = null;
        _charAccuracyImprovement = null;
      });
    }
  }

  Future<void> _uploadImage() async {
    _groundTruth = _groundTruthController.text.trim();

    if (_image == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Please select an image first.')),
      );
      return;
    }

    if (_groundTruth.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Please enter the ground truth text for evaluation.')),
      );
      return;
    }

    setState(() { _isLoading = true; });

    try {
      final uri = Uri.parse('http://192.168.1.16:5002/paddle-ocr');
      final request = http.MultipartRequest('POST', uri)
        ..files.add(await http.MultipartFile.fromPath('image', _image!.path))
        ..fields['test_plan'] = 'test3'
        ..fields['ground_truth'] = _groundTruth;

      final response = await request.send();
      final responseBody = await response.stream.bytesToString();
      final responseStatusCode = response.statusCode;

      if (mounted) {
        if (responseStatusCode == 200) {
          final result = json.decode(responseBody);
          setState(() {
            _groundTruth = result['ground_truth'] ?? _groundTruth;
            if (_groundTruthController.text != _groundTruth) {
              _groundTruthController.text = _groundTruth;
            }

            _beforePreprocessing = result['before_preprocessing'] != null
                ? Map<String, dynamic>.from(result['before_preprocessing'])
                : null;
            _afterPreprocessing = result['after_preprocessing'] != null
                ? Map<String, dynamic>.from(result['after_preprocessing'])
                : null;
            _accuracyImprovement = (result['word_accuracy_improvement'] as num?)?.toDouble();
            _charAccuracyImprovement = (result['char_accuracy_improvement'] as num?)?.toDouble();

            _wordsBefore = _beforePreprocessing?['words'] != null
                ? List.from(_beforePreprocessing!['words']) : [];
            _ocrText = _afterPreprocessing?['ocr_text'] ?? 'No text extracted';
            _words = _afterPreprocessing?['words'] != null
                ? List.from(_afterPreprocessing!['words']) : [];
          });
        } else {
          String errorMessage = 'Error: $responseStatusCode';
          try {
            final errorResult = json.decode(responseBody);
            errorMessage = errorResult['error'] ?? errorMessage;
          } catch (e) { /* Use default */ }
          ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(errorMessage)));
        }
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Error connecting: $e')));
      }
    } finally {
      if (mounted) { setState(() { _isLoading = false; }); }
    }
  }

  Widget _buildMetricCard(String title, String value, [Color? valueColor, Color? titleColor]) {
    if (value.isEmpty || value == 'null') return SizedBox.shrink();

    return Card(
      elevation: 1,
      margin: EdgeInsets.symmetric(vertical: 3, horizontal: 0),
      color: Colors.grey[50],
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(6)),
      child: Padding(
        padding: EdgeInsets.symmetric(vertical: 8, horizontal: 12),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Flexible(
              flex: 3,
              child: Text(
                '$title:',
                style: TextStyle(
                  color: titleColor ?? Colors.grey[700],
                  fontSize: 13,
                ),
              ),
            ),
            SizedBox(width: 8),
            Flexible(
              flex: 2,
              child: Text(
                value,
                textAlign: TextAlign.end,
                style: TextStyle(
                  fontWeight: FontWeight.w600,
                  color: valueColor ?? Theme.of(context).primaryColorDark,
                  fontSize: 14,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSectionHeader(String title) {
    return Padding(
      padding: const EdgeInsets.only(top: 16.0, bottom: 8.0),
      child: Text(
        title,
        style: TextStyle(
          fontSize: 16,
          fontWeight: FontWeight.bold,
          color: Colors.grey[600],
        ),
      ),
    );
  }

  Widget _buildMetricsSection(Map<String, dynamic>? data, String label) {
    final metrics = data?['metrics'] != null ? Map<String, dynamic>.from(data!['metrics']) : null;

    if (metrics == null || metrics.values.every((v) => v == null)) {
      return SizedBox.shrink();
    }

    String getMetric(String key) {
      final value = metrics[key];
      if (value == null) return '';
      if (value is num) {
        if (key.contains('_percent') || key == 'precision' || key == 'recall' ||
            key == 'f1_score' || key == 'precision_c' || key == 'recall_c' || key == 'f1_score_c') {
          return '${value.toStringAsFixed(2)}%';
        } else {
          return value.toInt().toString();
        }
      }
      return value.toString();
    }

    return Card(
      elevation: 3,
      margin: EdgeInsets.only(bottom: 16),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              '$label Metrics',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
                color: Theme.of(context).primaryColor,
              ),
            ),
            Divider(thickness: 1, height: 24),

            _buildSectionHeader("Counts"),
            _buildMetricCard('Total Ground Truth Words', getMetric('total_gt_words')),
            _buildMetricCard('Total Ground Truth Chars', getMetric('total_gt_chars')),
            _buildMetricCard('Total Detected Words', getMetric('total_ocr_words')),
            _buildMetricCard('Total Detected Chars', getMetric('total_ocr_chars')),

            _buildSectionHeader("Matches & Unmatches (vs Ground Truth)"),
            _buildMetricCard('Matched Words', getMetric('matched_words'), Colors.green[700]),
            _buildMetricCard('Matched Chars', getMetric('matched_chars'), Colors.green[700]),
            _buildMetricCard('GT Unmatched Words (FN)', getMetric('gt_unmatched_words'), Colors.orange[800]),
            _buildMetricCard('GT Unmatched Chars (FN)', getMetric('gt_unmatched_chars'), Colors.orange[800]),
            _buildMetricCard('OCR Extra/Wrong Words (FP)', getMetric('ocr_unmatched_words'), Colors.red[600]),
            _buildMetricCard('OCR Extra/Wrong Chars (FP)', getMetric('ocr_unmatched_chars'), Colors.red[600]),
            _buildMetricCard('Word Errors', getMetric('word_errors'), Colors.red[400]),
            _buildMetricCard('Char Errors', getMetric('char_errors'), Colors.red[400]),

            _buildSectionHeader("Accuracy"),
            _buildMetricCard('Word Accuracy (Ratio)', getMetric('word_accuracy_ratio_percent')),
            _buildMetricCard('Char Accuracy (Ratio)', getMetric('char_accuracy_ratio_percent')),

            if (metrics.containsKey('precision') || metrics.containsKey('recall') || metrics.containsKey('f1_score')) ...[
              if (metrics['precision'] != null || metrics['recall'] != null || metrics['f1_score'] != null) ...[
                _buildSectionHeader("Classification Metrics (Word Level)"),
                _buildMetricCard('Precision', getMetric('precision')),
                _buildMetricCard('Recall', getMetric('recall')),
                _buildMetricCard('F1 Score', getMetric('f1_score')),
              ]
            ],

            if (metrics.containsKey('precision_c') || metrics.containsKey('recall_c') || metrics.containsKey('f1_score_c')) ...[
              if (metrics['precision_c'] != null || metrics['recall_c'] != null || metrics['f1_score_c'] != null) ...[
                _buildSectionHeader("Classification Metrics (Character Level)"),
                _buildMetricCard('Precision (Char)', getMetric('precision_c')),
                _buildMetricCard('Recall (Char)', getMetric('recall_c')),
                _buildMetricCard('F1 Score (Char)', getMetric('f1_score_c')),
              ]
            ]
          ],
        ),
      ),
    );
  }

  Widget _buildWordChipsForList(List<dynamic> wordsList) {
    if (wordsList.isEmpty) return SizedBox.shrink();

    return Wrap(
      spacing: 8.0,
      runSpacing: 4.0,
      children: wordsList.map((wordData) {
        final text = wordData['text'] as String? ?? '';
        final confidence = (wordData['confidence'] as num?)?.toDouble() ?? 0.0;

        Color chipColor;
        if (confidence >= 90) {
          chipColor = Colors.green.shade400;
        } else if (confidence >= 75) {
          chipColor = Colors.orange.shade400;
        } else {
          chipColor = Colors.red.shade300;
        }

        return Chip(
          label: Text(
              text.isNotEmpty ? '$text (${confidence.toStringAsFixed(1)}%)' : '(?)',
              style: TextStyle(color: Colors.white, fontSize: 12)
          ),
          backgroundColor: chipColor,
          padding: EdgeInsets.symmetric(horizontal: 8, vertical: 4),
          materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
          visualDensity: VisualDensity.compact,
          elevation: 1,
        );
      }).toList(),
    );
  }

  @override
  Widget build(BuildContext context) {
    final hasResults = !_isLoading && (_beforePreprocessing != null || _afterPreprocessing != null);

    return Scaffold(
      appBar: AppBar(
        title: Text('OCR Evaluation Tool'),
        centerTitle: true,
        elevation: 2,
        backgroundColor: Theme.of(context).primaryColor,
      ),
      body: GestureDetector(
        onTap: () => FocusScope.of(context).unfocus(),
        child: SingleChildScrollView(
          padding: EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Card(
                elevation: 4,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                child: Padding(
                  padding: EdgeInsets.all(16),
                  child: Column(
                    children: [
                      Text(
                        '1. Select Image',
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                          color: Theme.of(context).primaryColorDark,
                        ),
                      ),
                      SizedBox(height: 12),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                        children: [
                          ElevatedButton.icon(
                            onPressed: _isLoading ? null : () => _pickImage(ImageSource.gallery),
                            icon: Icon(Icons.photo_library),
                            label: Text('Gallery'),
                            style: ElevatedButton.styleFrom(
                              padding: EdgeInsets.symmetric(horizontal: 20, vertical: 12),
                              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                            ),
                          ),
                          ElevatedButton.icon(
                            onPressed: _isLoading ? null : () => _pickImage(ImageSource.camera),
                            icon: Icon(Icons.camera_alt),
                            label: Text('Camera'),
                            style: ElevatedButton.styleFrom(
                              padding: EdgeInsets.symmetric(horizontal: 20, vertical: 12),
                              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                            ),
                          ),

                        ],
                      ),
                    ],
                  ),
                ),
              ),

              if (_image != null) ...[
                SizedBox(height: 16),
                Card(
                  elevation: 4,
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                  child: Padding(
                    padding: EdgeInsets.all(16),
                    child: Column(
                      children: [
                        Text(
                          '2. Preview & Enter Ground Truth',
                          style: TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                            color: Theme.of(context).primaryColorDark,
                          ),
                        ),
                        SizedBox(height: 12),
                        ClipRRect(
                          borderRadius: BorderRadius.circular(8),
                          child: Image.file(_image!, height: 200, fit: BoxFit.contain),
                        ),
                        SizedBox(height: 16),
                        TextField(
                          controller: _groundTruthController,
                          decoration: InputDecoration(
                            labelText: 'Ground Truth Text',
                            hintText: 'Enter the exact text visible in the image...',
                            border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
                            filled: true,
                            fillColor: Colors.grey[100],
                          ),
                          maxLines: 3,
                          textInputAction: TextInputAction.done,
                        ),
                        SizedBox(height: 16),
                        SizedBox(
                          width: double.infinity,
                          child: ElevatedButton.icon(
                            icon: Icon(Icons.compare_arrows),
                            label: Text('Process & Evaluate'),
                            onPressed: _isLoading ? null : _uploadImage,
                            style: ElevatedButton.styleFrom(
                              padding: EdgeInsets.symmetric(vertical: 14),
                              textStyle: TextStyle(fontSize: 16),
                              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                            ),
                          ),
                        ),
                        if (_isLoading) ...[
                          SizedBox(height: 12),
                          Center(child: CircularProgressIndicator()),
                          SizedBox(height: 4),
                          Center(child: Text("Processing...", style: TextStyle(color: Colors.grey[600])))
                        ],
                      ],
                    ),
                  ),
                ),
              ],

              if (hasResults) ...[
                SizedBox(height: 16),

                if (_beforePreprocessing?['ocr_text'] != null || (_beforePreprocessing?['words'] as List?)?.isNotEmpty == true) ...[
                  Card(
                    elevation: 4,
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                    child: Padding(
                      padding: EdgeInsets.all(16),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            mainAxisAlignment: MainAxisAlignment.spaceBetween,
                            children: [
                              Flexible(
                                child: Text(
                                  'Extracted Text (Before Processing)',
                                  style: TextStyle(
                                    fontSize: 16,
                                    fontWeight: FontWeight.bold,
                                    color: Theme.of(context).primaryColorDark,
                                  ),
                                ),
                              ),
                              IconButton(
                                icon: Icon(Icons.copy, size: 20, color: Colors.grey[600]),
                                tooltip: 'Copy Text (Before)',
                                onPressed: () {
                                  final textToCopy = _beforePreprocessing?['ocr_text'] as String? ?? '';
                                  if (textToCopy.isNotEmpty) {
                                    Clipboard.setData(ClipboardData(text: textToCopy));
                                    ScaffoldMessenger.of(context).showSnackBar(
                                      SnackBar(
                                        content: Text('Original extracted text copied'),
                                        behavior: SnackBarBehavior.floating,
                                        duration: Duration(seconds: 2),
                                      ),
                                    );
                                  }
                                },
                              ),
                            ],
                          ),
                          Divider(height: 16),
                          Text(
                            (_beforePreprocessing?['ocr_text'] as String? ?? '').isNotEmpty
                                ? (_beforePreprocessing!['ocr_text'] as String)
                                : '(No text detected)',
                            style: TextStyle(fontSize: 14, height: 1.4),
                          ),
                          SizedBox(height: 16),
                          Text(
                            'Word Confidence (Before):',
                            style: TextStyle(
                              fontWeight: FontWeight.bold,
                              fontSize: 14,
                              color: Colors.grey[700],
                            ),
                          ),
                          SizedBox(height: 8),
                          _buildWordChipsForList(_wordsBefore),
                        ],
                      ),
                    ),
                  ),
                  SizedBox(height: 16),
                ],

                _buildMetricsSection(_beforePreprocessing, 'Before Preprocessing'),

                if (_afterPreprocessing?['ocr_text'] != null || (_afterPreprocessing?['words'] as List?)?.isNotEmpty == true) ...[
                  Card(
                    elevation: 4,
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                    child: Padding(
                      padding: EdgeInsets.all(16),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            mainAxisAlignment: MainAxisAlignment.spaceBetween,
                            children: [
                              Flexible(
                                child: Text(
                                  'Extracted Text (After Processing)',
                                  style: TextStyle(
                                    fontSize: 16,
                                    fontWeight: FontWeight.bold,
                                    color: Theme.of(context).primaryColorDark,
                                  ),
                                ),
                              ),
                              IconButton(
                                icon: Icon(Icons.copy, size: 20, color: Colors.grey[600]),
                                tooltip: 'Copy Text (After)',
                                onPressed: () {
                                  final textToCopy = _afterPreprocessing?['ocr_text'] as String? ?? '';
                                  if (textToCopy.isNotEmpty) {
                                    Clipboard.setData(ClipboardData(text: textToCopy));
                                    ScaffoldMessenger.of(context).showSnackBar(
                                      SnackBar(
                                        content: Text('Processed extracted text copied'),
                                        behavior: SnackBarBehavior.floating,
                                        duration: Duration(seconds: 2),
                                      ),
                                    );
                                  }
                                },
                              ),
                            ],
                          ),
                          Divider(height: 16),
                          Text(
                            (_afterPreprocessing?['ocr_text'] as String? ?? '').isNotEmpty
                                ? (_afterPreprocessing!['ocr_text'] as String)
                                : '(No text detected)',
                            style: TextStyle(fontSize: 14, height: 1.4),
                          ),
                          SizedBox(height: 16),
                          Text(
                            'Word Confidence (After):',
                            style: TextStyle(
                              fontWeight: FontWeight.bold,
                              fontSize: 14,
                              color: Colors.grey[700],
                            ),
                          ),
                          SizedBox(height: 8),
                          _buildWordChipsForList(_words),
                        ],
                      ),
                    ),
                  ),
                  SizedBox(height: 16),
                ],

                _buildMetricsSection(_afterPreprocessing, 'After Processing'),

                if (_accuracyImprovement != null) ...[
                  SizedBox(height: 8),
                  Card(
                    elevation: 3,
                    color: _accuracyImprovement! >= 0 ? Colors.green[50] : Colors.red[50],
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                      side: BorderSide(
                        color: _accuracyImprovement! >= 0 ? Colors.green.shade100 : Colors.red.shade100,
                        width: 1,
                      ),
                    ),
                    child: Padding(
                      padding: EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                      child: Row(
                        children: [
                          Icon(
                            _accuracyImprovement! >= 0 ? Icons.trending_up : Icons.trending_down,
                            color: _accuracyImprovement! >= 0 ? Colors.green[700] : Colors.red[700],
                            size: 28,
                          ),
                          SizedBox(width: 12),
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  'Word Accuracy Improvement',
                                  style: TextStyle(
                                    fontWeight: FontWeight.bold,
                                    color: Colors.grey[800],
                                    fontSize: 15,
                                  ),
                                ),
                                SizedBox(height: 2),
                                Text(
                                  '${_accuracyImprovement! >= 0 ? '+' : ''}${_accuracyImprovement!.toStringAsFixed(2)}%',
                                  style: TextStyle(
                                    fontSize: 20,
                                    fontWeight: FontWeight.bold,
                                    color: _accuracyImprovement! >= 0 ? Colors.green[800] : Colors.red[800],
                                  ),
                                ),
                              ],
                            ),
                          ),
                          Text(
                            '(Compared to original)',
                            style: TextStyle(
                              fontSize: 12,
                              color: Colors.grey[600],
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                  SizedBox(height: 16),
                ],

                if (_charAccuracyImprovement != null) ...[
                  SizedBox(height: 8),
                  Card(
                    elevation: 3,
                    color: _charAccuracyImprovement! >= 0 ? Colors.green[50] : Colors.red[50],
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                      side: BorderSide(
                        color: _charAccuracyImprovement! >= 0 ? Colors.green.shade100 : Colors.red.shade100,
                        width: 1,
                      ),
                    ),
                    child: Padding(
                      padding: EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                      child: Row(
                        children: [
                          Icon(
                            _charAccuracyImprovement! >= 0 ? Icons.trending_up : Icons.trending_down,
                            color: _charAccuracyImprovement! >= 0 ? Colors.green[700] : Colors.red[700],
                            size: 28,
                          ),
                          SizedBox(width: 12),
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  'Character Accuracy Improvement',
                                  style: TextStyle(
                                    fontWeight: FontWeight.bold,
                                    color: Colors.grey[800],
                                    fontSize: 15,
                                  ),
                                ),
                                SizedBox(height: 2),
                                Text(
                                  '${_charAccuracyImprovement! >= 0 ? '+' : ''}${_charAccuracyImprovement!.toStringAsFixed(2)}%',
                                  style: TextStyle(
                                    fontSize: 20,
                                    fontWeight: FontWeight.bold,
                                    color: _charAccuracyImprovement! >= 0 ? Colors.green[800] : Colors.red[800],
                                  ),
                                ),
                              ],
                            ),
                          ),
                          Text(
                            '(Compared to original)',
                            style: TextStyle(
                              fontSize: 12,
                              color: Colors.grey[600],
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                  SizedBox(height: 16),
                ],
              ],
            ],
          ),
        ),
      ),
    );
  }
}