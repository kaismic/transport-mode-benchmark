import 'dart:async';
import 'dart:convert';
import 'dart:math';
import 'dart:typed_data';
import 'package:crypto/crypto.dart';
import 'package:flutter/services.dart';
import 'replay.dart';

typedef Json = Map<String, dynamic>;
const assetRoot = 'assets/benchmark/';
const native = MethodChannel('au.edu.tcct/benchmark');

class BenchmarkRunner {
  BenchmarkRunner(this.onProgress);
  final void Function(String) onProgress;
  bool cancelled = false;
  final samples = <Json>[];
  final memory = <Json>[];
  final interruptions = <Json>[];
  late Json manifest;
  String manifestHash = '';
  Json? _fixture;
  String? _fixtureId;
  int _lastThermalCheckUs = -10000000;
  final _clock = Stopwatch()..start();

  Future<Json> initialize() async {
    final bytes = await rootBundle.load('${assetRoot}manifest.json');
    final content = bytes.buffer.asUint8List(
      bytes.offsetInBytes,
      bytes.lengthInBytes,
    );
    manifestHash = sha256.convert(content).toString();
    manifest = jsonDecode(utf8.decode(content)) as Json;
    if (manifest['schema_version'] != 1) {
      throw StateError('Unsupported manifest schema');
    }
    const expectedRuntime = {
      'provider': 'CPUExecutionProvider',
      'intra_op_threads': 1,
      'inter_op_threads': 1,
      'execution_mode': 'sequential',
      'graph_optimization': 'all',
      'batch_size': 1,
      'spinning': false,
      'android_onnxruntime': '1.24.3',
    };
    final runtime = manifest['runtime'] as Json;
    if (expectedRuntime.entries.any((e) => runtime[e.key] != e.value)) {
      throw StateError('Manifest requires unsupported runtime settings');
    }
    final actual = await native.invokeMapMethod<String, dynamic>('initialize');
    if (actual?['runtime_version'] != runtime['android_onnxruntime']) {
      throw StateError('Native ONNX Runtime version does not match manifest');
    }
    return manifest;
  }

  void checkCancelled() {
    if (cancelled) throw StateError('Cancelled');
  }

  Future<void> coolDown({bool force = false}) async {
    checkCancelled();
    if (!force && _clock.elapsedMicroseconds - _lastThermalCheckUs < 5000000) {
      return;
    }
    _lastThermalCheckUs = _clock.elapsedMicroseconds;
    var status = await native.invokeMethod<int>('thermal') ?? -1;
    if (status < 2) return;
    final began = DateTime.now().toUtc().toIso8601String();
    while (status >= 2) {
      onProgress('Cooling down (thermal status $status)…');
      await Future<void>.delayed(const Duration(seconds: 1));
      checkCancelled();
      status = await native.invokeMethod<int>('thermal') ?? -1;
    }
    interruptions.add({
      'reason': 'thermal',
      'started_utc': began,
      'ended_utc': DateTime.now().toUtc().toIso8601String(),
    });
  }

  Future<Json> fixture(Json model) async {
    if (_fixtureId != model['id']) {
      _fixture =
          jsonDecode(
                await rootBundle.loadString(
                  '$assetRoot${model['fixture_asset']}',
                ),
              )
              as Json;
      _fixtureId = model['id'] as String;
    }
    return _fixture!;
  }

  Future<void> load(Json model, {bool captureMemory = false}) async {
    checkCancelled();
    if ((model['ordered_features'] as List).length !=
            model['input']['shape'][1] ||
        model['input']['dtype'] != 'float32' ||
        model['input']['shape'][0] != 1) {
      throw StateError('Unsupported input contract');
    }
    await fixture(model); // Same input/expected corpus present at baseline.
    await native.invokeMethod('verifyAssets', {
      'assets': {
        '$assetRoot${model['asset']}': model['sha256'],
        '$assetRoot${model['fixture_asset']}': model['fixture_sha256'],
      },
    });
    await native.invokeMethod('close');
    if (captureMemory) await snapshot(model, 'baseline');
    await coolDown(force: true);
    final watch = Stopwatch()..start();
    final loaded = await native.invokeMapMethod<String, dynamic>('load', {
      'asset': '$assetRoot${model['asset']}',
      'input': model['input']['name'],
      'width': model['input']['shape'][1],
      'label': model['outputs']['label'],
      'probabilities': model['outputs']['probabilities'],
    });
    watch.stop();
    samples.add({
      'model_id': model['id'],
      'kind': 'load',
      'native_us': loaded!['load_us'],
      'app_us': watch.elapsedMicroseconds,
    });
    if (captureMemory) await snapshot(model, 'loaded');
  }

  Future<Json> predict(Float32List values) async {
    final watch = Stopwatch()..start();
    final result = (await native.invokeMapMethod<String, dynamic>('predict', {
      'features': values,
    }))!;
    watch.stop();
    return {...result, 'app_us': watch.elapsedMicroseconds};
  }

  Future<void> validate(Json model) async {
    onProgress('Checking Android parity: ${model['id']}');
    final rows = (await fixture(model))['rows'] as List;
    for (var offset = 0; offset < rows.length; offset += 128) {
      checkCancelled();
      await native.invokeMethod('validate', {
        'rows': rows.skip(offset).take(128).toList(),
      });
    }
  }

  Future<void> snapshot(Json model, String phase, {String? session}) async {
    final value = (await native.invokeMapMethod<String, dynamic>('memory'))!;
    memory.add({
      'model_id': model['id'],
      'phase': phase,
      'session_id': ?session,
      'utc': DateTime.now().toUtc().toIso8601String(),
      ...value,
    });
  }

  Future<void> inference(
    Json model, {
    required bool memoryPass,
    int warmup = 100,
    int iterations = 1000,
  }) async {
    try {
      await load(model, captureMemory: memoryPass);
      final data = await fixture(model);
      final rows = data['rows'] as List;
      final count = data['source_rows'] as int;
      final order = List<int>.generate(count, (i) => i)..shuffle(Random(42));
      Float32List vector(int index) => Float32List.fromList(
        ((rows[order[index % count]] as Json)['input'] as List)
            .map((x) => x == null ? double.nan : (x as num).toDouble())
            .toList(),
      );
      final first = await predict(vector(0));
      samples.add({'model_id': model['id'], 'kind': 'first', ...first});
      await validate(model);
      for (var i = 0; i < warmup; i++) {
        checkCancelled();
        await predict(vector(i));
      }
      if (memoryPass) await snapshot(model, 'warmed');
      for (var i = 0; i < iterations; i++) {
        checkCancelled();
        await coolDown();
        final value = await predict(vector(i + warmup));
        if (!memoryPass) {
          samples.add({
            'model_id': model['id'],
            'kind': 'warm',
            'iteration': i,
            ...value,
          });
        }
        if (memoryPass && i % 100 == 0) {
          await snapshot(model, 'inference_observed');
        }
        if (i % 100 == 0) {
          onProgress(
            '${model['id']}: ${memoryPass ? 'memory' : 'latency'} $i/$iterations',
          );
        }
      }
      if (memoryPass) await snapshot(model, 'inference_end');
    } finally {
      await native.invokeMethod('close');
    }
  }

  Future<void> replay(
    Json model, {
    required bool realTime,
    required bool memoryPass,
  }) async {
    if (jsonEncode(model['ordered_features']) != jsonEncode(sydneyFeatures) ||
        manifest['replay_contract']['schema'] != 'sydney-sensor-v1') {
      throw StateError('This model needs another replay feature adapter');
    }
    try {
      await load(model, captureMemory: memoryPass);
      await validate(model);
      if (memoryPass) await snapshot(model, 'warmed');
      for (final item in manifest['replay'] as List) {
        checkCancelled();
        final session = item as Json;
        await native.invokeMethod('verifyAssets', {
          'assets': {
            '$assetRoot${session['asset']}': session['sha256'],
            '$assetRoot${session['golden_asset']}': session['golden_sha256'],
          },
        });
        final golden =
            (jsonDecode(
                      await rootBundle.loadString(
                        '$assetRoot${session['golden_asset']}',
                      ),
                    )
                    as Json)['${model['window_seconds']}']
                as List;
        final buffer = SensorWindow();
        final window = model['window_seconds'] as int,
            step = model['step_seconds'] as int;
        var start = session['start_ms'] as int;
        final end = session['end_ms'] as int;
        final playback = Stopwatch()..start();
        var decodeUs = 0, bufferUs = 0, index = 0, lastMemoryUs = -1000000;
        final interruptionCount = interruptions.length;
        Future<void> emit() async {
          final watch = Stopwatch()..start();
          final features = buffer.features(start, window);
          watch.stop();
          final expected = golden[index++] as Json;
          if (expected['start_ms'] != start ||
              (features == null) != (expected['features'] == null)) {
            throw StateError('Replay window acceptance parity failed');
          }
          Json? prediction;
          if (features != null) {
            assertClose(features, expected['features'] as List);
            prediction = await predict(features);
          }
          samples.add({
            'model_id': model['id'],
            'kind': 'replay',
            'session_id': session['session_id'],
            'real_time': realTime,
            'memory_pass': memoryPass,
            'window_start_ms': start,
            'window_end_ms': start + window * 1000,
            'accepted': features != null,
            'decode_us': decodeUs,
            'buffer_us': bufferUs,
            'feature_us': watch.elapsedMicroseconds,
            'processing_us':
                bufferUs +
                watch.elapsedMicroseconds +
                (prediction?['app_us'] as int? ?? 0),
            'schedule_lag_us': realTime
                ? max(
                    0,
                    playback.elapsedMicroseconds -
                        (start + window * 1000 - (session['start_ms'] as int)) *
                            1000,
                  )
                : null,
            'max_buffer_rows': buffer.maxRows,
            'prediction': prediction,
          });
          start += step * 1000;
          final trim = Stopwatch()..start();
          buffer.discardBefore(start);
          trim.stop();
          bufferUs = trim.elapsedMicroseconds;
          decodeUs = 0;
        }

        onProgress(
          '${model['id']}: ${realTime ? 'real-time' : 'accelerated'} ${session['mode']}',
        );
        await native.invokeMethod('openReplay', {
          'asset': '$assetRoot${session['asset']}',
        });
        while (true) {
          checkCancelled();
          await coolDown();
          if (interruptionCount != interruptions.length) {
            throw StateError(
              'Thermal interruption during replay; retry after cooling',
            );
          }
          final readWatch = Stopwatch()..start();
          final lines = await native.invokeListMethod<String>('readReplay');
          final decoded = lines!
              .map((line) => (jsonDecode(line) as List).cast<num?>())
              .toList();
          readWatch.stop();
          decodeUs += readWatch.elapsedMicroseconds;
          if (decoded.isEmpty) break;
          for (final row in decoded) {
            checkCancelled();
            if (realTime) {
              final dueUs =
                  (row[0]!.toInt() - (session['start_ms'] as int)) * 1000;
              while (dueUs > playback.elapsedMicroseconds) {
                await Future<void>.delayed(
                  Duration(
                    microseconds: min(
                      100000,
                      dueUs - playback.elapsedMicroseconds,
                    ),
                  ),
                );
                checkCancelled();
              }
            }
            while (start + window * 1000 <= row[0]! &&
                start + window * 1000 <= end) {
              await emit();
            }
            final update = Stopwatch()..start();
            buffer.add(row);
            update.stop();
            bufferUs += update.elapsedMicroseconds;
            if (memoryPass &&
                playback.elapsedMicroseconds - lastMemoryUs >= 1000000) {
              await snapshot(
                model,
                'replay_observed',
                session: session['session_id'] as String,
              );
              lastMemoryUs = playback.elapsedMicroseconds;
            }
          }
        }
        if (realTime) {
          while (playback.elapsedMicroseconds <
              (end - (session['start_ms'] as int)) * 1000) {
            await Future<void>.delayed(const Duration(milliseconds: 100));
            checkCancelled();
          }
        }
        while (start + window * 1000 <= end) {
          await emit();
        }
        if (index != golden.length) {
          throw StateError('Replay window count mismatch');
        }
        await native.invokeMethod('closeReplay');
        if (memoryPass) {
          await snapshot(
            model,
            'replay_end',
            session: session['session_id'] as String,
          );
        }
      }
    } finally {
      await native.invokeMethod('closeReplay');
      await native.invokeMethod('close');
    }
  }

  Future<Json> result(
    String status,
    String mode,
    List<String> ids, {
    String? error,
  }) async {
    final metadata = (await native.invokeMapMethod<String, dynamic>(
      'metadata',
    ))!;
    return {
      'schema_version': 1,
      'run_id': DateTime.now().toUtc().toIso8601String(),
      'status': status,
      'error': ?error,
      'mode': mode,
      'manifest_sha256': manifestHash,
      'models': (manifest['models'] as List)
          .where((m) => ids.contains(m['id']))
          .toList(),
      'runtime': manifest['runtime'],
      'device': metadata,
      'expected_model_ids': (manifest['models'] as List)
          .map((m) => m['id'])
          .toList(),
      'app_commit': const String.fromEnvironment(
        'APP_COMMIT',
        defaultValue: 'unrecorded',
      ),
      'flutter_version': const String.fromEnvironment(
        'FLUTTER_VERSION',
        defaultValue: 'unrecorded',
      ),
      'warmup': 100,
      'iterations': 1000,
      'samples': samples,
      'memory': memory,
      'interruptions': interruptions,
      'replay': manifest['replay'],
      'fresh_process': metadata['models_loaded'] == 1,
      'independent_test_cohort': false,
    };
  }
}
