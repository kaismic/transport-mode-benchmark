import 'dart:convert';
import 'dart:io';
import 'package:flutter_test/flutter_test.dart';
import 'package:transport_mode_benchmark/replay.dart';

void main() {
  List<List<num?>> simple() =>
      List.generate(50, (i) => <num?>[i * 20, 3, 4, 0, 0, 0, 2, 1, 0, 0]);
  test('duplicates, half-open endpoints and insufficient samples', () {
    final window = SensorWindow();
    final rows = simple();
    window.add(rows.first);
    window.add([0, ...List<num>.filled(9, 100)]);
    for (final row in rows.skip(1)) {
      window.add(row);
    }
    window.add([1000, ...List<num>.filled(9, 100)]);
    expect(window.features(0, 1), [5, 0, 0, 5, 2, 0, 0, 2, 0, 0, 1]);
    final short = SensorWindow();
    for (final row in rows.take(20)) {
      short.add(row);
    }
    expect(short.features(0, 1), isNull);
    final gap = SensorWindow();
    for (final row in rows.skip(30)) {
      gap.add(row);
    }
    expect(gap.features(0, 1), isNull);
    window.discardBefore(500);
    expect(window.rows.first[0], 500);
    expect(
      () => window.add([1, ...List<num>.filled(9, 0)]),
      throwsFormatException,
    );
  });

  test('streamed replay matches Python windows with bounded buffers', () {
    final manifestFile = File('assets/benchmark/manifest.json');
    if (!manifestFile.existsSync()) return;
    final manifest =
        jsonDecode(manifestFile.readAsStringSync()) as Map<String, dynamic>;
    for (final session in manifest['replay'] as List) {
      final golden =
          jsonDecode(
                File(
                  'assets/benchmark/${session['golden_asset']}',
                ).readAsStringSync(),
              )
              as Map<String, dynamic>;
      final rows = File('assets/benchmark/${session['asset']}')
          .readAsLinesSync()
          .map((line) => (jsonDecode(line) as List).cast<num?>())
          .toList();
      for (final seconds in [100, 120, 140]) {
        final buffer = SensorWindow();
        var start = session['start_ms'] as int;
        var index = 0;
        void emit() {
          final actual = buffer.features(start, seconds);
          final expected = (golden['$seconds'] as List)[index++];
          expect(start, expected['start_ms']);
          expect(actual == null, expected['features'] == null);
          if (actual != null) assertClose(actual, expected['features'] as List);
          start += seconds * 500;
          buffer.discardBefore(start);
        }

        for (final row in rows) {
          while (start + seconds * 1000 <= row[0]! &&
              start + seconds * 1000 <= (session['end_ms'] as int)) {
            emit();
          }
          buffer.add(row);
        }
        while (start + seconds * 1000 <= (session['end_ms'] as int)) {
          emit();
        }
        expect(index, (golden['$seconds'] as List).length);
        // Historic sessions predate the collector's 60 Hz cap. Bound time,
        // rather than assuming a rate or dropping original duplicate rows.
        expect(buffer.maxSpanMs, lessThan(seconds * 1000));
        expect(buffer.maxRows, lessThan(rows.length));
      }
    }
  });
}
