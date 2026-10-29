import 'dart:convert';
import 'package:flutter/material.dart';
import 'benchmark.dart';

void main() => runApp(const BenchmarkApp());

class BenchmarkApp extends StatelessWidget {
  const BenchmarkApp({super.key});
  @override
  Widget build(BuildContext context) => MaterialApp(
    title: 'TMD Benchmark',
    debugShowCheckedModeBanner: false,
    theme: ThemeData(
      colorSchemeSeed: const Color(0xff146b62),
      useMaterial3: true,
    ),
    home: const BenchmarkPage(),
  );
}

class BenchmarkPage extends StatefulWidget {
  const BenchmarkPage({super.key});
  @override
  State<BenchmarkPage> createState() => _BenchmarkPageState();
}

class _BenchmarkPageState extends State<BenchmarkPage>
    with WidgetsBindingObserver {
  Json? manifest;
  String? selected;
  String mode = 'latency';
  String message = 'Loading benchmark bundle…';
  String? exportPath, exportToken;
  bool busy = false;
  BenchmarkRunner? runner;
  static const modes = {
    'latency': 'Inference latency',
    'memory': 'Inference memory',
    'replay_fast': 'Accelerated replay',
    'replay_realtime': 'Real-time replay',
    'replay_fast_memory': 'Accelerated replay memory',
    'replay_realtime_memory': 'Real-time replay memory',
  };

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    initialize();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    runner?.cancelled = true;
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (busy &&
        state != AppLifecycleState.resumed &&
        state != AppLifecycleState.inactive) {
      runner?.cancelled = true;
    }
  }

  void progress(String value) {
    if (mounted) setState(() => message = value);
  }

  Future<void> initialize() async {
    try {
      final initial = BenchmarkRunner(progress);
      final bundle = await initial.initialize();
      if (!mounted) return;
      setState(() {
        manifest = bundle;
        selected = (bundle['models'] as List).first['id'] as String;
        message =
            '${(bundle['models'] as List).length} validated exports ready. CPU · batch 1 · one thread';
      });
      final launch = await native.invokeMapMethod<String, dynamic>(
        'launchConfig',
      );
      if (launch?['model'] != null) {
        final id = launch!['model'] as String;
        if (!(bundle['models'] as List).any((m) => m['id'] == id)) {
          throw StateError('Unknown model $id');
        }
        final launchMode = launch['mode'] as String? ?? 'latency';
        if (!modes.containsKey(launchMode)) {
          throw StateError('Unknown mode $launchMode');
        }
        setState(() {
          selected = id;
          mode = launchMode;
        });
        await run(false, token: launch['token'] as String?);
      }
    } catch (error) {
      progress(
        'Cannot start: $error\nPrepare the bundle using tools/prepare.py, then rebuild the APK.',
      );
    }
  }

  Future<void> run(bool suite, {String? token}) async {
    if (busy || manifest == null) return;
    final ids = suite
        ? (manifest!['models'] as List).map((m) => m['id'] as String).toList()
        : [selected!];
    if (suite) ids.shuffle();
    final active = BenchmarkRunner(progress);
    runner = active;
    setState(() {
      busy = true;
      exportPath = null;
      exportToken = null;
    });
    var status = 'complete';
    String? failure;
    try {
      await active.initialize();
      final startDevice = (await native.invokeMapMethod<String, dynamic>(
        'metadata',
      ))!;
      for (final id in ids) {
        final model =
            (manifest!['models'] as List).firstWhere((m) => m['id'] == id)
                as Json;
        if (mode.startsWith('replay')) {
          await active.replay(
            model,
            realTime: mode.contains('realtime'),
            memoryPass: mode.endsWith('memory'),
          );
        } else {
          await active.inference(model, memoryPass: mode == 'memory');
        }
      }
      final result = await active.result(status, mode, ids);
      result['device_start'] = startDevice;
      await save(result, token);
      progress(
        'Complete. Results saved. ${suite ? 'Suite used one process; use the desktop runner for independent repetitions.' : ''}',
      );
    } catch (error) {
      status = active.cancelled ? 'cancelled' : 'failed';
      failure = error.toString();
      try {
        final result = await active.result(status, mode, ids, error: failure);
        await save(result, token);
      } catch (_) {
        /* A bundle/bridge failure can prevent a structured export. */
      }
      progress('$status: $failure');
    } finally {
      await native.invokeMethod('closeReplay');
      await native.invokeMethod('close');
      if (mounted) setState(() => busy = false);
    }
  }

  Future<void> save(Json result, String? token) async {
    final name = token ?? 'manual_${DateTime.now().millisecondsSinceEpoch}';
    final path = await native.invokeMethod<String>('export', {
      'json': jsonEncode(result),
      'token': name,
    });
    if (mounted) {
      setState(() {
        exportPath = path;
        exportToken = name;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final models = manifest?['models'] as List? ?? [];
    final chosen = models.where((m) => m['id'] == selected).firstOrNull;
    return Scaffold(
      appBar: AppBar(title: const Text('TMD Benchmark')),
      body: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 720),
          child: ListView(
            padding: const EdgeInsets.all(24),
            children: [
              Text(
                'Measure on this phone',
                style: Theme.of(context).textTheme.headlineMedium,
              ),
              const SizedBox(height: 8),
              const Text(
                'Compare model execution, Flutter call cost and recorded sensor processing. No live sensors or network connection required.',
              ),
              const SizedBox(height: 24),
              DropdownButtonFormField<String>(
                initialValue: selected,
                key: ValueKey('model-$selected'),
                decoration: const InputDecoration(labelText: 'Model'),
                items: models
                    .map(
                      (m) => DropdownMenuItem(
                        value: m['id'] as String,
                        child: Text(
                          '${m['window_seconds']} s · seed ${m['seed']}',
                        ),
                      ),
                    )
                    .toList(),
                onChanged: busy
                    ? null
                    : (value) => setState(() => selected = value),
              ),
              const SizedBox(height: 16),
              DropdownButtonFormField<String>(
                initialValue: mode,
                key: ValueKey('mode-$mode'),
                decoration: const InputDecoration(labelText: 'Measurement'),
                items: modes.entries
                    .map(
                      (e) =>
                          DropdownMenuItem(value: e.key, child: Text(e.value)),
                    )
                    .toList(),
                onChanged: busy
                    ? null
                    : (value) => setState(() => mode = value!),
              ),
              if (chosen != null)
                Padding(
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  child: Text(
                    '${((chosen['bytes'] as int) / 1024).toStringAsFixed(1)} KiB deployed model · '
                    '${chosen['window_seconds']} s observation · ${chosen['step_seconds']} s prediction interval',
                  ),
                ),
              Wrap(
                spacing: 12,
                runSpacing: 12,
                children: [
                  FilledButton.icon(
                    onPressed: busy || manifest == null
                        ? null
                        : () => run(false),
                    icon: const Icon(Icons.play_arrow),
                    label: const Text('Run selected'),
                  ),
                  OutlinedButton(
                    onPressed: busy || manifest == null
                        ? null
                        : () => run(true),
                    child: const Text('Run suite'),
                  ),
                  if (busy)
                    OutlinedButton(
                      onPressed: () {
                        runner?.cancelled = true;
                        progress('Cancelling…');
                      },
                      child: const Text('Cancel'),
                    ),
                ],
              ),
              const SizedBox(height: 24),
              if (busy) const LinearProgressIndicator(),
              const SizedBox(height: 12),
              SelectableText(message),
              const SizedBox(height: 16),
              const Text(
                'Latency: 100 warmups + 1,000 calls. Memory runs are separate. Real-time replay takes 15 minutes per model with the default bundle. Keep the app in the foreground.',
              ),
              const SizedBox(height: 8),
              const Text(
                'For five independent process repetitions, use tools/device.py. The in-app suite is an exploratory comparison in one process.',
              ),
              if (exportPath != null) ...[
                const Divider(height: 32),
                SelectableText(exportPath!),
                TextButton.icon(
                  onPressed: busy
                      ? null
                      : () => native.invokeMethod('share', {
                          'token': exportToken,
                        }),
                  icon: const Icon(Icons.share),
                  label: const Text('Export JSON + CSV'),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}
