/// Door lock models and state
class DoorStatus {
  final bool isOpen;
  final DateTime lastUpdated;
  final String source; // 'sensor', 'command', 'error'
  final bool isForcefullyOpened; // Detected from sensor anomalies

  DoorStatus({
    required this.isOpen,
    required this.lastUpdated,
    this.source = 'sensor',
    this.isForcefullyOpened = false,
  });

  DoorStatus copyWith({
    bool? isOpen,
    DateTime? lastUpdated,
    String? source,
    bool? isForcefullyOpened,
  }) {
    return DoorStatus(
      isOpen: isOpen ?? this.isOpen,
      lastUpdated: lastUpdated ?? this.lastUpdated,
      source: source ?? this.source,
      isForcefullyOpened: isForcefullyOpened ?? this.isForcefullyOpened,
    );
  }

  @override
  String toString() =>
      'DoorStatus(isOpen: $isOpen, source: $source, forceful: $isForcefullyOpened)';
}

class VoiceAuthState {
  final bool isListening;
  final bool wakewordDetected;
  final bool biometricVerified;
  final String? lastCommand; // 'open_door', 'close_door', or null
  final double wakewordConfidence;
  final double biometricConfidence;
  final double commandConfidence;
  final String status; // 'idle', 'listening', 'processing', 'verified'

  VoiceAuthState({
    this.isListening = false,
    this.wakewordDetected = false,
    this.biometricVerified = false,
    this.lastCommand,
    this.wakewordConfidence = 0.0,
    this.biometricConfidence = 0.0,
    this.commandConfidence = 0.0,
    this.status = 'idle',
  });

  VoiceAuthState copyWith({
    bool? isListening,
    bool? wakewordDetected,
    bool? biometricVerified,
    String? lastCommand,
    double? wakewordConfidence,
    double? biometricConfidence,
    double? commandConfidence,
    String? status,
  }) {
    return VoiceAuthState(
      isListening: isListening ?? this.isListening,
      wakewordDetected: wakewordDetected ?? this.wakewordDetected,
      biometricVerified: biometricVerified ?? this.biometricVerified,
      lastCommand: lastCommand ?? this.lastCommand,
      wakewordConfidence: wakewordConfidence ?? this.wakewordConfidence,
      biometricConfidence: biometricConfidence ?? this.biometricConfidence,
      commandConfidence: commandConfidence ?? this.commandConfidence,
      status: status ?? this.status,
    );
  }
}

class BleConnectionState {
  final bool isConnected;
  final bool isScanning;
  final String? deviceName;
  final String? deviceId;
  final int signalStrength; // RSSI (-100 to 0)
  final String? errorMessage;

  BleConnectionState({
    this.isConnected = false,
    this.isScanning = false,
    this.deviceName,
    this.deviceId,
    this.signalStrength = 0,
    this.errorMessage,
  });

  BleConnectionState copyWith({
    bool? isConnected,
    bool? isScanning,
    String? deviceName,
    String? deviceId,
    int? signalStrength,
    String? errorMessage,
  }) {
    return BleConnectionState(
      isConnected: isConnected ?? this.isConnected,
      isScanning: isScanning ?? this.isScanning,
      deviceName: deviceName ?? this.deviceName,
      deviceId: deviceId ?? this.deviceId,
      signalStrength: signalStrength ?? this.signalStrength,
      errorMessage: errorMessage ?? this.errorMessage,
    );
  }
}

class AlertEvent {
  final String title;
  final String message;
  final DateTime timestamp;
  final String type; // 'forceful_open', 'connection_lost', 'auth_failed'
  final String? recipientEmail;

  AlertEvent({
    required this.title,
    required this.message,
    required this.timestamp,
    this.type = 'security_alert',
    this.recipientEmail,
  });
}
