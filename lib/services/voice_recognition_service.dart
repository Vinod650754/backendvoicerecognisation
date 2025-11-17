import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:io';
import 'package:record/record.dart';

/// Voice recognition service for on-device model inference
class VoiceRecognitionService {
  final String backendUrl;
  final AudioRecorder _recorder = AudioRecorder();

  bool _isRecording = false;
  String? _lastRecordingPath;

  VoiceRecognitionService({this.backendUrl = 'http://10.0.2.2:8000'});

  /// Start recording voice audio
  Future<bool> startRecording() async {
    try {
      if (_isRecording) return false;

      final String tempPath =
          '/data/data/com.example.home_app/cache/temp_audio.wav';

      await _recorder.start(
        const RecordConfig(
          encoder: AudioEncoder.wav,
          sampleRate: 16000,
          numChannels: 1,
          bitRate: 128000,
        ),
        path: tempPath,
      );

      _isRecording = true;
      _lastRecordingPath = tempPath;
      return true;
    } catch (e) {
      debugPrint('Error starting recording: $e');
      return false;
    }
  }

  /// Stop recording and return path to audio file
  Future<String?> stopRecording() async {
    try {
      final path = await _recorder.stop();
      _isRecording = false;
      return path;
    } catch (e) {
      debugPrint('Error stopping recording: $e');
      return null;
    }
  }

  /// Detect wakeword in audio
  Future<Map<String, dynamic>> detectWakeword(String audioPath) async {
    try {
      final file = File(audioPath);
      if (!file.existsSync()) {
        return {'error': 'Audio file not found', 'wakeword_detected': false};
      }

      final request = http.MultipartRequest(
        'POST',
        Uri.parse('$backendUrl/detect_wakeword'),
      );

      request.files.add(await http.MultipartFile.fromPath('file', audioPath));

      final response = await request.send().timeout(
        const Duration(seconds: 10),
      );
      final responseData = await response.stream.bytesToString();

      if (response.statusCode == 200) {
        return jsonDecode(responseData) as Map<String, dynamic>;
      } else {
        return {
          'error': 'Backend error: ${response.statusCode}',
          'wakeword_detected': false,
        };
      }
    } catch (e) {
      debugPrint('Error detecting wakeword: $e');
      return {'error': e.toString(), 'wakeword_detected': false};
    }
  }

  /// Verify voice biometric
  Future<Map<String, dynamic>> verifyVoice(String audioPath) async {
    try {
      final file = File(audioPath);
      if (!file.existsSync()) {
        return {'error': 'Audio file not found', 'verified': false};
      }

      final request = http.MultipartRequest(
        'POST',
        Uri.parse('$backendUrl/verify_voice'),
      );

      request.files.add(await http.MultipartFile.fromPath('file', audioPath));

      final response = await request.send().timeout(
        const Duration(seconds: 10),
      );
      final responseData = await response.stream.bytesToString();

      if (response.statusCode == 200) {
        return jsonDecode(responseData) as Map<String, dynamic>;
      } else {
        return {
          'error': 'Backend error: ${response.statusCode}',
          'verified': false,
        };
      }
    } catch (e) {
      debugPrint('Error verifying voice: $e');
      return {'error': e.toString(), 'verified': false};
    }
  }

  /// Detect command in audio
  Future<Map<String, dynamic>> detectCommand(String audioPath) async {
    try {
      final file = File(audioPath);
      if (!file.existsSync()) {
        return {'error': 'Audio file not found', 'intent': null};
      }

      final request = http.MultipartRequest(
        'POST',
        Uri.parse('$backendUrl/detect_command'),
      );

      request.files.add(await http.MultipartFile.fromPath('file', audioPath));

      final response = await request.send().timeout(
        const Duration(seconds: 10),
      );
      final responseData = await response.stream.bytesToString();

      if (response.statusCode == 200) {
        return jsonDecode(responseData) as Map<String, dynamic>;
      } else {
        return {
          'error': 'Backend error: ${response.statusCode}',
          'intent': null,
        };
      }
    } catch (e) {
      debugPrint('Error detecting command: $e');
      return {'error': e.toString(), 'intent': null};
    }
  }

  /// Upload audio sample for training
  Future<bool> uploadSample(String audioPath, String label) async {
    try {
      final file = File(audioPath);
      if (!file.existsSync()) {
        debugPrint('Audio file not found: $audioPath');
        return false;
      }

      final request = http.MultipartRequest(
        'POST',
        Uri.parse('$backendUrl/collect_sample'),
      );

      request.fields['label'] = label; // 'wakeword', 'open_door', 'close_door'
      request.files.add(await http.MultipartFile.fromPath('file', audioPath));

      final response = await request.send().timeout(
        const Duration(seconds: 15),
      );

      if (response.statusCode == 200) {
        debugPrint('Sample uploaded successfully: $label');
        return true;
      } else {
        debugPrint('Upload failed: ${response.statusCode}');
        return false;
      }
    } catch (e) {
      debugPrint('Error uploading sample: $e');
      return false;
    }
  }

  /// Dispose recorder resources
  Future<void> dispose() async {
    await _recorder.dispose();
  }

  bool get isRecording => _isRecording;
  String? get lastRecordingPath => _lastRecordingPath;
}
