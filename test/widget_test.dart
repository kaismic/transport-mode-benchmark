import 'package:flutter_test/flutter_test.dart';
import 'package:transport_mode_benchmark/benchmark.dart';
import 'package:transport_mode_benchmark/main.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  testWidgets('benchmark selectors and run actions initialize', (tester) async {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(native, (call) async {
          if (call.method == 'initialize') return {'runtime_version': '1.24.3'};
          if (call.method == 'launchConfig') return <String, dynamic>{};
          return null;
        });
    await tester.pumpWidget(const BenchmarkApp());
    await tester.runAsync(
      () => Future<void>.delayed(const Duration(milliseconds: 500)),
    );
    await tester.pump();
    expect(find.text('TMD Benchmark'), findsOneWidget);
    expect(find.text('Run selected'), findsOneWidget);
    expect(find.text('Run suite'), findsOneWidget);
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(native, null);
  });
}
