import 'package:flutter/foundation.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart' as fbp;
import 'package:home_app/models/door_models.dart';

/// Bluetooth service for ESP32 communication
class BluetoothService {
  static const String esp32DevicePrefix = 'ESP32';
  static const String doorStatusCharacteristic = '180A';
  static const String doorCommandCharacteristic = '180B';

  fbp.BluetoothDevice? _device;
  fbp.BluetoothCharacteristic? _statusChar;
  fbp.BluetoothCharacteristic? _commandChar;
  bool _isConnected = false;

  // Getters
  bool get isConnected => _isConnected;
  fbp.BluetoothDevice? get device => _device;

  /// Scan for ESP32 BLE devices
  Future<List<fbp.BluetoothDevice>> scanForDevices({
    Duration timeout = const Duration(seconds: 5),
  }) async {
    try {
      List<fbp.BluetoothDevice> devices = [];
      fbp.FlutterBluePlus.startScan(timeout: timeout);

      fbp.FlutterBluePlus.scanResults.listen((results) {
        for (fbp.ScanResult r in results) {
          if (r.device.platformName.contains(esp32DevicePrefix)) {
            if (!devices.any((d) => d.id == r.device.id)) {
              devices.add(r.device);
            }
          }
        }
      });

      await Future.delayed(timeout);
      fbp.FlutterBluePlus.stopScan();
      return devices;
    } catch (e) {
      debugPrint('Error scanning for devices: $e');
      return [];
    }
  }

  /// Connect to ESP32 device
  Future<bool> connectToDevice(fbp.BluetoothDevice device) async {
    try {
      _device = device;
      await device.connect(
        timeout: const Duration(seconds: 10),
        license: fbp.License.free,
      );
      _isConnected = true;

      // Discover services
      await _discoverServices(device);
      return true;
    } catch (e) {
      debugPrint('Error connecting to device: $e');
      _isConnected = false;
      return false;
    }
  }

  /// Discover GATT services and characteristics
  Future<void> _discoverServices(fbp.BluetoothDevice device) async {
    try {
      List<fbp.BluetoothService> services = await device.discoverServices();

      for (var service in services) {
        for (var char in service.characteristics) {
          if (char.uuid.toString().startsWith(doorStatusCharacteristic)) {
            _statusChar = char;
          } else if (char.uuid.toString().startsWith(
            doorCommandCharacteristic,
          )) {
            _commandChar = char;
          }
        }
      }
    } catch (e) {
      debugPrint('Error discovering services: $e');
    }
  }

  /// Send command to open door
  Future<bool> sendOpenDoorCommand() async {
    return _sendCommand('OPEN_DOOR');
  }

  /// Send command to close door
  Future<bool> sendCloseDoorCommand() async {
    return _sendCommand('CLOSE_DOOR');
  }

  /// Send generic command
  Future<bool> _sendCommand(String command) async {
    if (_commandChar == null || !_isConnected) {
      debugPrint(
        'Cannot send command: device not connected or characteristic not found',
      );
      return false;
    }

    try {
      await _commandChar!.write(command.codeUnits);
      debugPrint('Sent command: $command');
      return true;
    } catch (e) {
      debugPrint('Error sending command: $e');
      return false;
    }
  }

  /// Read door status from sensor
  Future<DoorStatus?> readDoorStatus() async {
    if (_statusChar == null) return null;

    try {
      List<int> value = await _statusChar!.read();
      String statusStr = String.fromCharCodes(value);

      bool isOpen = statusStr.contains('OPEN');
      bool isForceful = statusStr.contains('FORCEFUL');

      return DoorStatus(
        isOpen: isOpen,
        lastUpdated: DateTime.now(),
        source: 'sensor',
        isForcefullyOpened: isForceful,
      );
    } catch (e) {
      debugPrint('Error reading door status: $e');
      return null;
    }
  }

  /// Listen for door status updates via notifications
  Stream<DoorStatus> getDoorStatusStream() async* {
    if (_statusChar == null) return;

    try {
      await _statusChar!.setNotifyValue(true);

      await for (final value in _statusChar!.onValueReceived) {
        String statusStr = String.fromCharCodes(value);
        bool isOpen = statusStr.contains('OPEN');
        bool isForceful = statusStr.contains('FORCEFUL');

        yield DoorStatus(
          isOpen: isOpen,
          lastUpdated: DateTime.now(),
          source: 'sensor',
          isForcefullyOpened: isForceful,
        );
      }
    } catch (e) {
      debugPrint('Error in door status stream: $e');
    }
  }

  /// Disconnect from device
  Future<void> disconnect() async {
    if (_device != null) {
      try {
        await _device!.disconnect();
        _isConnected = false;
        _device = null;
        _statusChar = null;
        _commandChar = null;
      } catch (e) {
        debugPrint('Error disconnecting: $e');
      }
    }
  }

  /// Check if device is currently connected
  Future<bool> isDeviceConnected(fbp.BluetoothDevice device) async {
    try {
      final state = await device.state.first;
      return state == fbp.BluetoothConnectionState.connected;
    } catch (e) {
      return false;
    }
  }
}
