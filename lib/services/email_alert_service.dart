import 'package:flutter/foundation.dart';
import 'package:mailer/mailer.dart';
import 'package:mailer/smtp_server.dart';
import 'package:home_app/models/door_models.dart';

/// Email alert service for security notifications
class EmailAlertService {
  static const String smtpServer = 'smtp.gmail.com';
  static const int smtpPort = 465;

  String? _senderEmail;
  String? _senderPassword;
  String? _recipientEmail;

  /// Initialize email service with sender credentials
  void initialize({
    required String senderEmail,
    required String senderPassword,
    required String recipientEmail,
  }) {
    _senderEmail = senderEmail;
    _senderPassword = senderPassword;
    _recipientEmail = recipientEmail;
  }

  /// Send security alert email
  Future<bool> sendSecurityAlert(AlertEvent alert) async {
    if (_senderEmail == null ||
        _senderPassword == null ||
        _recipientEmail == null) {
      debugPrint('Email service not initialized');
      return false;
    }

    try {
      final smtpServer_ = gmail(_senderEmail!, _senderPassword!);

      final message = Message()
        ..from = Address(_senderEmail!, 'Smart Door Lock')
        ..recipients.add(_recipientEmail!)
        ..subject = '🚨 ${alert.title}'
        ..text =
            '''
Security Alert from Your Smart Door Lock

Type: ${alert.type}
Time: ${alert.timestamp}

Message: ${alert.message}

Please review your door lock status immediately.

---
Smart Door Lock System
''';

      await send(message, smtpServer_);
      debugPrint('Security alert email sent successfully');
      return true;
    } catch (e) {
      debugPrint('Error sending email: $e');
      return false;
    }
  }

  /// Send forceful door opening alert
  Future<bool> sendForcefulOpeningAlert(DoorStatus doorStatus) async {
    final alert = AlertEvent(
      title: 'Forceful Door Opening Detected!',
      message:
          'Your door was forcefully opened or unusual sensor activity was detected.',
      timestamp: doorStatus.lastUpdated,
      type: 'forceful_open',
      recipientEmail: _recipientEmail,
    );

    return sendSecurityAlert(alert);
  }

  /// Send connection lost alert
  Future<bool> sendConnectionLostAlert() async {
    final alert = AlertEvent(
      title: 'Smart Door Lock - Connection Lost',
      message:
          'Your mobile device lost connection with the door lock. Please reconnect.',
      timestamp: DateTime.now(),
      type: 'connection_lost',
      recipientEmail: _recipientEmail,
    );

    return sendSecurityAlert(alert);
  }

  /// Send authentication failure alert
  Future<bool> sendAuthFailureAlert(int failureCount) async {
    final alert = AlertEvent(
      title: 'Door Lock - Authentication Failed',
      message:
          'Failed authentication attempt #$failureCount. Please try again or use alternative access method.',
      timestamp: DateTime.now(),
      type: 'auth_failed',
      recipientEmail: _recipientEmail,
    );

    return sendSecurityAlert(alert);
  }
}
