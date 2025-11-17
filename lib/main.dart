import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:record/record.dart';
import 'package:http/http.dart' as http;
import 'dart:io';
import 'dart:convert';
import 'dart:async';
import 'package:permission_handler/permission_handler.dart';
import 'package:vibration/vibration.dart';

void main() {
  runApp(const ProviderScope(child: MyApp()));
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Smart Door Lock',
      theme: ThemeData(primarySwatch: Colors.blue, useMaterial3: true),
      home: const MyHomePage(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key});

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> with TickerProviderStateMixin {
  late TabController _tabController;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 2, vsync: this);
    _requestPermissions();
  }

  @override
  void dispose() {
    _tabController.dispose();
    super.dispose();
  }

  void _requestPermissions() async {
    await Permission.microphone.request();
    await Permission.storage.request();
    await Permission.bluetooth.request();
    await Permission.bluetoothScan.request();
    await Permission.bluetoothConnect.request();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('🔐 Smart Door Lock System'),
        centerTitle: true,
        elevation: 0,
        bottom: TabBar(
          controller: _tabController,
          indicatorColor: Colors.white,
          tabs: const [
            Tab(icon: Icon(Icons.lock), text: 'Voice Auth & Control'),
            Tab(icon: Icon(Icons.mic), text: 'Training Samples'),
          ],
        ),
      ),
      body: TabBarView(
        controller: _tabController,
        children: const [VoiceAuthTab(), TrainingSamplesTab()],
      ),
    );
  }
}

// ===== TAB 1: VOICE AUTHENTICATION & DOOR CONTROL =====
class VoiceAuthTab extends StatefulWidget {
  const VoiceAuthTab({super.key});

  @override
  State<VoiceAuthTab> createState() => _VoiceAuthTabState();
}

class _VoiceAuthTabState extends State<VoiceAuthTab>
    with SingleTickerProviderStateMixin {
  final AudioRecorder _recorder = AudioRecorder();
  late AnimationController _animController;
  late Animation<double> _pulseAnimation;

  bool _isListening = false;
  bool _doorOpen = false;
  String _status = 'Ready';
  Color _statusColor = Colors.grey;
  String _authStatus = 'Not Authenticated';
  Color _authColor = Colors.red;

  static const String backendUrl = 'http://10.0.2.2:8000';

  @override
  void initState() {
    super.initState();
    _animController = AnimationController(
      duration: const Duration(milliseconds: 1000),
      vsync: this,
    )..repeat();

    _pulseAnimation = Tween<double>(begin: 1, end: 1.2).animate(
      CurvedAnimation(parent: _animController, curve: Curves.easeInOut),
    );

    _checkBackendHealth();
  }

  @override
  void dispose() {
    _animController.dispose();
    _recorder.dispose();
    super.dispose();
  }

  Future<void> _checkBackendHealth() async {
    try {
      final response = await http
          .get(Uri.parse('$backendUrl/health'))
          .timeout(const Duration(seconds: 5));
      if (response.statusCode == 200) {
        if (!mounted) return;
        setState(() {
          _status = 'Backend: OK ✓';
          _statusColor = Colors.green;
        });
      }
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _status = 'Backend: Offline ✗';
        _statusColor = Colors.red;
      });
    }
  }

  Future<void> _startVoiceAuth() async {
    if (_isListening) {
      setState(() {
        _isListening = false;
        _status = 'Stopped';
        _statusColor = Colors.grey;
      });
      return;
    }

    setState(() {
      _isListening = true;
      _status = 'Listening for wakeword...';
      _statusColor = Colors.orange;
      _authStatus = 'Listening...';
      _authColor = Colors.orange;
    });

    _runVoiceAuthPipeline();
  }

  Future<void> _runVoiceAuthPipeline() async {
    while (_isListening && mounted) {
      try {
        // Record 1.2s wakeword detection clip
        final audioPath = await _recordAudio(
          duration: const Duration(milliseconds: 1200),
        );
        if (audioPath == null || !_isListening) break;

        // Detect wakeword
        final wakewordResult = await _detectWakeword(audioPath);
        if (!_isListening) break;

        if (wakewordResult['wakeword_detected'] == true) {
          if (!mounted) break;
          setState(() {
            _status = 'Wakeword detected! Recording command...';
            _statusColor = Colors.blue;
            _authStatus = 'Wakeword OK ✓';
            _authColor = Colors.blue;
          });

          await Future.delayed(const Duration(milliseconds: 500));

          // Record command
          final commandPath = await _recordAudio(
            duration: const Duration(milliseconds: 2500),
          );
          if (commandPath == null || !_isListening) break;

          // Verify voice biometric
          final verifyResult = await _verifyVoice(commandPath);
          if (!_isListening) break;

          if (verifyResult['verified'] == true) {
            if (!mounted) break;
            setState(() {
              _status = 'Voice verified! ✓';
              _statusColor = Colors.green;
              _authStatus = 'AUTHENTICATED ✓';
              _authColor = Colors.green;
            });

            // Detect command
            final commandResult = await _detectCommand(commandPath);
            final intent = commandResult['intent'];

            if (intent == 'lock') {
              await _closeDoor();
            } else if (intent == 'unlock' || intent == 'open_door') {
              await _openDoor();
            }

            await Vibration.vibrate(duration: 200);
            await Future.delayed(const Duration(seconds: 3));
          } else {
            if (!mounted) break;
            setState(() {
              _status = 'Voice verification failed';
              _statusColor = Colors.red;
              _authStatus = 'AUTH FAILED ✗';
              _authColor = Colors.red;
            });
            await Vibration.vibrate(duration: 300);
            await Future.delayed(const Duration(seconds: 2));
          }

          if (!mounted) break;
          setState(() {
            _status = 'Listening for wakeword...';
            _statusColor = Colors.orange;
            _authStatus = 'Listening...';
            _authColor = Colors.orange;
          });
        }

        // Wait before next check
        await Future.delayed(const Duration(milliseconds: 1500));
      } catch (e) {
        debugPrint('Pipeline error: $e');
        await Future.delayed(const Duration(milliseconds: 500));
      }
    }

    if (mounted) {
      setState(() {
        _isListening = false;
        _status = 'Stopped';
        _statusColor = Colors.grey;
      });
    }
  }

  Future<String?> _recordAudio({required Duration duration}) async {
    try {
      final tempDir = Directory.systemTemp;
      final timestamp = DateTime.now().millisecondsSinceEpoch;
      final audioPath = '${tempDir.path}/audio_$timestamp.wav';

      await _recorder.start(
        const RecordConfig(
          encoder: AudioEncoder.wav,
          sampleRate: 16000,
          numChannels: 1,
        ),
        path: audioPath,
      );

      await Future.delayed(duration);
      final path = await _recorder.stop();
      return path;
    } catch (e) {
      debugPrint('Recording error: $e');
      return null;
    }
  }

  Future<Map<String, dynamic>> _detectWakeword(String audioPath) async {
    try {
      final file = File(audioPath);
      if (!file.existsSync()) return {'wakeword_detected': false};

      final request = http.MultipartRequest(
        'POST',
        Uri.parse('$backendUrl/detect_wakeword'),
      );
      request.files.add(await http.MultipartFile.fromPath('file', audioPath));

      final response = await request.send().timeout(const Duration(seconds: 5));

      if (response.statusCode == 200) {
        final data =
            jsonDecode(await response.stream.bytesToString())
                as Map<String, dynamic>;
        return data;
      }
      return {'wakeword_detected': false};
    } catch (e) {
      debugPrint('Wakeword detect error: $e');
      return {'wakeword_detected': false};
    }
  }

  Future<Map<String, dynamic>> _verifyVoice(String audioPath) async {
    try {
      final file = File(audioPath);
      if (!file.existsSync()) return {'verified': false};

      final request = http.MultipartRequest(
        'POST',
        Uri.parse('$backendUrl/verify_voice'),
      );
      request.files.add(await http.MultipartFile.fromPath('file', audioPath));

      final response = await request.send().timeout(const Duration(seconds: 5));

      if (response.statusCode == 200) {
        final data =
            jsonDecode(await response.stream.bytesToString())
                as Map<String, dynamic>;
        return data;
      }
      return {'verified': false};
    } catch (e) {
      debugPrint('Voice verify error: $e');
      return {'verified': false};
    }
  }

  Future<Map<String, dynamic>> _detectCommand(String audioPath) async {
    try {
      final file = File(audioPath);
      if (!file.existsSync()) return {'intent': null};

      final request = http.MultipartRequest(
        'POST',
        Uri.parse('$backendUrl/detect_command'),
      );
      request.files.add(await http.MultipartFile.fromPath('file', audioPath));

      final response = await request.send().timeout(const Duration(seconds: 5));

      if (response.statusCode == 200) {
        final data =
            jsonDecode(await response.stream.bytesToString())
                as Map<String, dynamic>;
        return data;
      }
      return {'intent': null};
    } catch (e) {
      debugPrint('Command detect error: $e');
      return {'intent': null};
    }
  }

  Future<void> _openDoor() async {
    setState(() => _doorOpen = true);
    await Vibration.vibrate(duration: 100);
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('🔓 Door opened successfully!'),
        duration: Duration(seconds: 2),
      ),
    );
  }

  Future<void> _closeDoor() async {
    setState(() => _doorOpen = false);
    await Vibration.vibrate(duration: 100);
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('🔐 Door closed successfully!'),
        duration: Duration(seconds: 2),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(20),
      child: Column(
        children: [
          // Door status display
          ScaleTransition(
            scale: _pulseAnimation,
            child: Container(
              width: 150,
              height: 150,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: _doorOpen ? Colors.red[100] : Colors.green[100],
                border: Border.all(
                  color: _doorOpen ? Colors.red : Colors.green,
                  width: 3,
                ),
              ),
              child: Center(
                child: Icon(
                  _doorOpen ? Icons.lock_open : Icons.lock,
                  size: 80,
                  color: _doorOpen ? Colors.red : Colors.green,
                ),
              ),
            ),
          ),
          const SizedBox(height: 20),
          Text(
            _doorOpen ? 'Door OPEN' : 'Door LOCKED',
            style: TextStyle(
              fontSize: 24,
              fontWeight: FontWeight.bold,
              color: _doorOpen ? Colors.red : Colors.green,
            ),
          ),
          const SizedBox(height: 30),

          // Auth status
          Container(
            padding: const EdgeInsets.all(15),
            decoration: BoxDecoration(
              color: _authColor.withOpacity(0.2),
              borderRadius: BorderRadius.circular(10),
              border: Border.all(color: _authColor, width: 2),
            ),
            child: Column(
              children: [
                Text(
                  'Authentication Status',
                  style: const TextStyle(fontSize: 14, color: Colors.grey),
                ),
                const SizedBox(height: 5),
                Text(
                  _authStatus,
                  style: TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                    color: _authColor,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 20),

          // Backend status
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: _statusColor.withOpacity(0.1),
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: _statusColor),
            ),
            child: Row(
              children: [
                Icon(Icons.cloud, color: _statusColor),
                const SizedBox(width: 10),
                Expanded(
                  child: Text(
                    _status,
                    style: TextStyle(
                      color: _statusColor,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 30),

          // Voice listening button
          ElevatedButton.icon(
            onPressed: _startVoiceAuth,
            icon: Icon(_isListening ? Icons.stop : Icons.mic),
            label: Text(_isListening ? 'Stop Listening' : 'Start Voice Auth'),
            style: ElevatedButton.styleFrom(
              backgroundColor: _isListening ? Colors.red : Colors.blue,
              padding: const EdgeInsets.symmetric(horizontal: 30, vertical: 15),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(10),
              ),
            ),
          ),
          const SizedBox(height: 20),

          // Manual controls
          Row(
            children: [
              Expanded(
                child: ElevatedButton.icon(
                  onPressed: _openDoor,
                  icon: const Icon(Icons.lock_open),
                  label: const Text('Open Door'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.red,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(10),
                    ),
                  ),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: ElevatedButton.icon(
                  onPressed: _closeDoor,
                  icon: const Icon(Icons.lock),
                  label: const Text('Close Door'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.green,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(10),
                    ),
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 20),

          // Instructions
          Container(
            padding: const EdgeInsets.all(15),
            decoration: BoxDecoration(
              color: Colors.blue[50],
              borderRadius: BorderRadius.circular(10),
            ),
            child: const Text(
              'Voice Authentication Pipeline:\n'
              '1️⃣ Listen for wakeword detection\n'
              '2️⃣ Upon detection, record command\n'
              '3️⃣ Verify your voice biometric\n'
              '4️⃣ Execute command (open/close door)\n\n'
              'Manual Controls: Use buttons to open/close anytime',
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 12),
            ),
          ),
        ],
      ),
    );
  }
}

// ===== TAB 2: TRAINING SAMPLES =====
class TrainingSamplesTab extends StatefulWidget {
  const TrainingSamplesTab({super.key});

  @override
  State<TrainingSamplesTab> createState() => _TrainingSamplesTabState();
}

class _TrainingSamplesTabState extends State<TrainingSamplesTab> {
  final AudioRecorder _recorder = AudioRecorder();
  bool _isRecording = false;
  String _recordingType = 'wakeword';
  String _status = 'Ready';
  Color _statusColor = Colors.grey;
  int _wakewordCount = 0;
  int _openDoorCount = 0;
  int _closeDoorCount = 0;

  static const String backendUrl = 'http://10.0.2.2:8000';

  @override
  void dispose() {
    _recorder.dispose();
    super.dispose();
  }

  Future<void> _recordAndUpload({required String label}) async {
    if (_isRecording) {
      final path = await _recorder.stop();
      _isRecording = false;

      if (path != null) {
        if (!mounted) return;
        setState(() {
          _status = 'Uploading...';
          _statusColor = Colors.orange;
        });

        await _uploadSample(path, label);
      }
      return;
    }

    if (!mounted) return;
    setState(() {
      _isRecording = true;
      _recordingType = label;
      _status = 'Recording 2 seconds...';
      _statusColor = Colors.blue;
    });

    try {
      final tempDir = Directory.systemTemp;
      final timestamp = DateTime.now().millisecondsSinceEpoch;
      final audioPath = '${tempDir.path}/sample_${label}_$timestamp.wav';

      await _recorder.start(
        const RecordConfig(
          encoder: AudioEncoder.wav,
          sampleRate: 16000,
          numChannels: 1,
        ),
        path: audioPath,
      );

      await Future.delayed(const Duration(seconds: 2));

      if (_isRecording) {
        final path = await _recorder.stop();
        _isRecording = false;

        if (path != null) {
          if (!mounted) return;
          setState(() {
            _status = 'Uploading...';
            _statusColor = Colors.orange;
          });
          await _uploadSample(path, label);
        }
      }
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _status = 'Error: $e';
        _statusColor = Colors.red;
        _isRecording = false;
      });
    }
  }

  Future<void> _uploadSample(String audioPath, String label) async {
    try {
      final file = File(audioPath);
      if (!file.existsSync()) {
        if (!mounted) return;
        setState(() {
          _status = 'File not found';
          _statusColor = Colors.red;
        });
        return;
      }

      final request = http.MultipartRequest(
        'POST',
        Uri.parse('$backendUrl/collect_sample'),
      );
      request.fields['label'] = label;
      request.files.add(await http.MultipartFile.fromPath('file', audioPath));

      final response = await request.send().timeout(
        const Duration(seconds: 10),
      );

      if (response.statusCode == 200) {
        if (!mounted) return;
        setState(() {
          if (label == 'wakeword') {
            _wakewordCount++;
          } else if (label == 'open_door') {
            _openDoorCount++;
          } else if (label == 'close_door') {
            _closeDoorCount++;
          }
          _status = '✓ Uploaded to backend/data/$label/';
          _statusColor = Colors.green;
        });

        await Vibration.vibrate(duration: 100);
        await Future.delayed(const Duration(seconds: 2));

        if (!mounted) return;
        setState(() {
          _status = 'Ready';
          _statusColor = Colors.grey;
        });
      } else {
        if (!mounted) return;
        setState(() {
          _status = 'Upload failed: ${response.statusCode}';
          _statusColor = Colors.red;
        });
      }

      try {
        await file.delete();
      } catch (_) {}
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _status = 'Error: $e';
        _statusColor = Colors.red;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(20),
      child: Column(
        children: [
          const Text(
            'Collect Training Samples',
            style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 30),

          _sampleCountCard('Wakeword', _wakewordCount, Colors.blue),
          const SizedBox(height: 15),
          _recordButton('Wakeword', 'wakeword'),
          const SizedBox(height: 30),

          _sampleCountCard('Open Door Command', _openDoorCount, Colors.orange),
          const SizedBox(height: 15),
          _recordButton('Open Door', 'open_door'),
          const SizedBox(height: 30),

          _sampleCountCard('Close Door Command', _closeDoorCount, Colors.green),
          const SizedBox(height: 15),
          _recordButton('Close Door', 'close_door'),
          const SizedBox(height: 30),

          // Status
          Container(
            padding: const EdgeInsets.all(15),
            decoration: BoxDecoration(
              color: _statusColor.withOpacity(0.1),
              borderRadius: BorderRadius.circular(10),
              border: Border.all(color: _statusColor),
            ),
            child: Text(
              _status,
              textAlign: TextAlign.center,
              style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.bold,
                color: _statusColor,
              ),
            ),
          ),
          const SizedBox(height: 20),

          Container(
            padding: const EdgeInsets.all(15),
            decoration: BoxDecoration(
              color: Colors.grey[100],
              borderRadius: BorderRadius.circular(10),
            ),
            child: const Text(
              'Training Tips:\n'
              '• Record 20-50 samples per category\n'
              '• Speak clearly at normal volume\n'
              '• Vary your tone and speaking speed\n'
              '• Record in different environments\n\n'
              'Samples are stored in:\n'
              'backend/data/wakeword/\n'
              'backend/data/open_door/\n'
              'backend/data/close_door/',
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 12),
            ),
          ),
        ],
      ),
    );
  }

  Widget _sampleCountCard(String title, int count, Color color) {
    return Container(
      padding: const EdgeInsets.all(15),
      decoration: BoxDecoration(
        color: color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: color, width: 2),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            title,
            style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
          ),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 15, vertical: 8),
            decoration: BoxDecoration(
              color: color,
              borderRadius: BorderRadius.circular(20),
            ),
            child: Text(
              '$count samples',
              style: const TextStyle(
                color: Colors.white,
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _recordButton(String label, String type) {
    return ElevatedButton.icon(
      onPressed: () => _recordAndUpload(label: type),
      icon: Icon(
        _isRecording && _recordingType == type ? Icons.stop : Icons.mic,
      ),
      label: Text(
        _isRecording && _recordingType == type
            ? 'Stop Recording'
            : 'Record $label',
      ),
      style: ElevatedButton.styleFrom(
        minimumSize: const Size(double.infinity, 50),
        backgroundColor: _isRecording && _recordingType == type
            ? Colors.red
            : Colors.blue,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
      ),
    );
  }
}
