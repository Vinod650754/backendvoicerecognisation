import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:home_app/models/door_models.dart';
import 'package:home_app/services/bluetooth_service.dart';
import 'package:home_app/services/voice_recognition_service.dart';
import 'package:home_app/services/email_alert_service.dart';
import 'package:vibration/vibration.dart';

// ===== STATE PROVIDERS =====

/// Door status state provider
final doorStatusProvider =
    StateNotifierProvider<DoorStatusNotifier, DoorStatus>((ref) {
      return DoorStatusNotifier();
    });

class DoorStatusNotifier extends StateNotifier<DoorStatus> {
  DoorStatusNotifier()
    : super(
        DoorStatus(isOpen: false, lastUpdated: DateTime.now(), source: 'idle'),
      );

  void updateStatus(DoorStatus status) {
    state = status;
  }

  void setOpen(bool isOpen) {
    state = state.copyWith(
      isOpen: isOpen,
      lastUpdated: DateTime.now(),
      source: 'command',
    );
  }

  void setForcefullyOpened(bool forceful) {
    state = state.copyWith(
      isForcefullyOpened: forceful,
      lastUpdated: DateTime.now(),
    );
    if (forceful) {
      Vibration.vibrate(duration: 500);
    }
  }
}

/// Voice authentication state provider
final voiceAuthStateProvider =
    StateNotifierProvider<VoiceAuthNotifier, VoiceAuthState>((ref) {
      return VoiceAuthNotifier();
    });

class VoiceAuthNotifier extends StateNotifier<VoiceAuthState> {
  VoiceAuthNotifier() : super(VoiceAuthState());

  void startListening() {
    state = state.copyWith(isListening: true, status: 'listening');
  }

  void stopListening() {
    state = state.copyWith(isListening: false, status: 'idle');
  }

  void setWakewordDetected(bool detected, double confidence) {
    state = state.copyWith(
      wakewordDetected: detected,
      wakewordConfidence: confidence,
      status: detected ? 'processing' : 'listening',
    );
  }

  void setBiometricVerified(bool verified, double confidence) {
    state = state.copyWith(
      biometricVerified: verified,
      biometricConfidence: confidence,
      status: verified ? 'verified' : 'processing',
    );
  }

  void setCommand(String? command, double confidence) {
    state = state.copyWith(lastCommand: command, commandConfidence: confidence);
  }

  void reset() {
    state = VoiceAuthState();
  }
}

/// Bluetooth connection state provider
final bleConnectionProvider =
    StateNotifierProvider<BleConnectionNotifier, BleConnectionState>((ref) {
      return BleConnectionNotifier();
    });

class BleConnectionNotifier extends StateNotifier<BleConnectionState> {
  BleConnectionNotifier() : super(BleConnectionState());

  void setConnected(bool connected, String? deviceName, String? deviceId) {
    state = state.copyWith(
      isConnected: connected,
      deviceName: deviceName,
      deviceId: deviceId,
      errorMessage: null,
    );
  }

  void setScanning(bool scanning) {
    state = state.copyWith(isScanning: scanning);
  }

  void setError(String error) {
    state = state.copyWith(errorMessage: error);
  }

  void setSignalStrength(int rssi) {
    state = state.copyWith(signalStrength: rssi);
  }
}

/// Alert events stream provider
final alertEventsProvider = StateProvider<List<AlertEvent>>((ref) {
  return [];
});

/// Service providers (singletons)
final bluetoothServiceProvider = Provider<BluetoothService>((ref) {
  return BluetoothService();
});

final voiceRecognitionServiceProvider = Provider<VoiceRecognitionService>((
  ref,
) {
  return VoiceRecognitionService();
});

final emailAlertServiceProvider = Provider<EmailAlertService>((ref) {
  return EmailAlertService();
});
