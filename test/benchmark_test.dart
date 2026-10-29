import 'dart:io';
import 'package:flutter_test/flutter_test.dart';
import 'package:transport_mode_benchmark/benchmark.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  test(
    'cancellation closes native session and skips timed samples',
    () async {
      final calls = <String>[];
      final runner = BenchmarkRunner((_) {});
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(native, (call) async {
            calls.add(call.method);
            switch (call.method) {
              case 'initialize':
                return {'runtime_version': '1.24.3'};
              case 'load':
                return {'load_us': 1.0};
              case 'thermal':
                return 0;
              case 'predict':
                runner.cancelled = true;
                return {
                  'native_us': 1.0,
                  'label': 0,
                  'probabilities': [1.0, 0.0, 0.0],
                };
              default:
                return null;
            }
          });
      try {
        final manifest = await runner.initialize();
        await expectLater(
          runner.inference(
            (manifest['models'] as List).first as Json,
            memoryPass: false,
          ),
          throwsStateError,
        );
        expect(calls.last, 'close');
        expect(runner.samples.where((x) => x['kind'] == 'warm'), isEmpty);
      } finally {
        TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
            .setMockMethodCallHandler(native, null);
      }
    },
    skip: !File('assets/benchmark/manifest.json').existsSync(),
  );
}
