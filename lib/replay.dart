import 'dart:collection';
import 'dart:math';
import 'dart:typed_data';

const sydneyFeatures = [
  'accelerometer#mean',
  'accelerometer#standard_deviation',
  'accelerometer#range',
  'accelerometer#minimum',
  'gyroscope#mean',
  'gyroscope#standard_deviation',
  'gyroscope#range',
  'gyroscope#minimum',
  'magnetometer#standard_deviation',
  'magnetometer#range',
  'magnetometer#minimum',
];

/// Quality counts duplicate rows; statistics use the first finite vector at
/// each timestamp per sensor. Buffer eviction preserves the next window.
class SensorWindow {
  final rows = ListQueue<List<num?>>();
  int maxRows = 0;
  int maxSpanMs = 0;

  void add(List<num?> row) {
    if (row.length != 10 ||
        row.first == null ||
        (rows.isNotEmpty && row.first! < rows.last.first!)) {
      throw FormatException('Expected timestamp-ordered ts + nine axes');
    }
    rows.addLast(row);
    maxRows = max(maxRows, rows.length);
    maxSpanMs = max(maxSpanMs, row[0]!.toInt() - rows.first[0]!.toInt());
  }

  void discardBefore(int timestamp) {
    while (rows.isNotEmpty && rows.first.first! < timestamp) {
      rows.removeFirst();
    }
  }

  Float32List? features(int start, int seconds) {
    final end = start + seconds * 1000;
    final result = <double>[];
    for (var sensor = 0; sensor < 3; sensor++) {
      var count = 0, previous = start, maxGap = 0;
      int? lastUnique;
      final values = <double>[];
      for (final row in rows) {
        final ts = row[0]!.toInt();
        if (ts < start || ts >= end) continue;
        final v = row.sublist(1 + sensor * 3, 4 + sensor * 3);
        if (v.any((x) => x == null || !x.isFinite)) continue;
        count++;
        maxGap = max(maxGap, ts - previous);
        previous = ts;
        if (lastUnique == ts) continue;
        lastUnique = ts;
        values.add(sqrt(v.fold<double>(0, (sum, x) => sum + x! * x)));
      }
      maxGap = max(maxGap, end - previous);
      if (count < [30, 4, 2][sensor] * seconds || maxGap > 500) return null;
      final mean = values.reduce((a, b) => a + b) / values.length;
      final std = sqrt(
        values.fold<double>(0, (s, x) => s + pow(x - mean, 2)) / values.length,
      );
      final low = values.reduce(min), high = values.reduce(max);
      if (sensor != 2) result.add(mean);
      result.addAll([std, high - low, low]);
    }
    return Float32List.fromList(result);
  }
}

void assertClose(List<num> actual, List<dynamic> expected) {
  if (actual.length != expected.length) {
    throw StateError('Parity shape mismatch');
  }
  for (var i = 0; i < actual.length; i++) {
    final target = (expected[i] as num).toDouble();
    if (!actual[i].isFinite ||
        (actual[i] - target).abs() > 1e-6 + 1e-5 * target.abs()) {
      throw StateError('Parity failure at feature $i: ${actual[i]} vs $target');
    }
  }
}
