import 'package:flutter/material.dart';
import 'package:openmedkit_flutter/openmedkit.dart';

/// Small offline example that redacts a fully synthetic clinical note.
class RedactSyntheticNote extends StatefulWidget {
  const RedactSyntheticNote({Key? key, required this.openMedKit})
    : super(key: key);

  final OpenMedKit openMedKit;

  @override
  State<RedactSyntheticNote> createState() => _RedactSyntheticNoteState();
}

class _RedactSyntheticNoteState extends State<RedactSyntheticNote> {
  static const String _syntheticNote =
      'Patient Alice Nguyen was born on 1979-04-12. '
      'Email alice@example.org.';

  String _displayText = _syntheticNote;
  bool _working = false;

  Future<void> _redact() async {
    setState(() => _working = true);
    try {
      final OpenMedDeidentificationResult result = await widget.openMedKit
          .deidentify(_syntheticNote);
      if (mounted) {
        setState(() => _displayText = result.deidentifiedText);
      }
    } finally {
      if (mounted) {
        setState(() => _working = false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: <Widget>[
        const Text('Synthetic note'),
        const SizedBox(height: 8),
        SelectableText(_displayText),
        const SizedBox(height: 12),
        ElevatedButton(
          onPressed: _working ? null : _redact,
          child: Text(_working ? 'Redacting locally…' : 'Redact locally'),
        ),
      ],
    );
  }
}
