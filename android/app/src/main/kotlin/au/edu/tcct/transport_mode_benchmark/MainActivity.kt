package au.edu.tcct.transport_mode_benchmark

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.app.ActivityManager
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.Debug
import android.os.PowerManager
import android.os.Process
import android.os.SystemClock
import android.view.WindowManager
import androidx.core.content.FileProvider
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedReader
import java.io.File
import java.nio.FloatBuffer
import java.security.MessageDigest
import java.util.concurrent.Executors
import kotlin.math.abs

class MainActivity : FlutterActivity() {
    private val worker = Executors.newSingleThreadExecutor()
    private lateinit var environment: OrtEnvironment
    private var session: OrtSession? = null
    private var options: OrtSession.SessionOptions? = null
    private var reader: BufferedReader? = null
    private var width = 0
    private var inputName = ""
    private var labelName = ""
    private var probabilityName = ""
    private var modelsLoaded = 0
    private val processStarted = SystemClock.elapsedRealtime()

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, "au.edu.tcct/benchmark")
            .setMethodCallHandler { call, reply ->
                worker.execute {
                    try {
                        val value = handle(call)
                        runOnUiThread { reply.success(value) }
                    } catch (error: Exception) {
                        runOnUiThread { reply.error("BENCHMARK", error.message, error.stackTraceToString()) }
                    }
                }
            }
    }

    private fun asset(path: String) = assets.open("flutter_assets/$path")
    private fun closeSession() {
        session?.close(); session = null
        options?.close(); options = null
    }
    private fun thermal(): Int = if (Build.VERSION.SDK_INT >= 29)
        (getSystemService(POWER_SERVICE) as PowerManager).currentThermalStatus else -1

    private fun handle(call: MethodCall): Any? = when (call.method) {
        "initialize" -> {
            environment = OrtEnvironment.getEnvironment()
            mapOf("runtime_version" to environment.version)
        }
        "launchConfig" -> mapOf(
            "model" to intent.getStringExtra("benchmark_model"),
            "mode" to intent.getStringExtra("benchmark_mode"),
            "token" to intent.getStringExtra("benchmark_token"))
        "verifyAssets" -> {
            val entries = call.argument<Map<String, String>>("assets")!!
            entries.forEach { (path, expected) ->
                val hash = MessageDigest.getInstance("SHA-256")
                asset(path).use { stream ->
                    val bytes = ByteArray(65536)
                    while (true) { val n = stream.read(bytes); if (n < 0) break; hash.update(bytes, 0, n) }
                }
                val actual = hash.digest().joinToString("") { "%02x".format(it.toInt() and 255) }
                require(actual == expected) { "Asset checksum mismatch: $path" }
            }; null
        }
        "load" -> {
            closeSession()
            width = call.argument<Int>("width")!!
            inputName = call.argument<String>("input")!!
            labelName = call.argument<String>("label")!!
            probabilityName = call.argument<String>("probabilities")!!
            options = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(1); setInterOpNumThreads(1)
                setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                addConfigEntry("session.intra_op.allow_spinning", "0")
                addConfigEntry("session.inter_op.allow_spinning", "0")
            }
            val start = System.nanoTime()
            val bytes = asset(call.argument<String>("asset")!!).use { it.readBytes() }
            session = environment.createSession(bytes, options)
            val duration = (System.nanoTime() - start) / 1000.0
            require(session!!.inputNames == setOf(inputName)) { "Input name mismatch" }
            require(session!!.outputNames.containsAll(listOf(labelName, probabilityName))) { "Output name mismatch" }
            modelsLoaded++
            mapOf("load_us" to duration)
        }
        "predict" -> predict(call.argument<FloatArray>("features")!!)
        "validate" -> {
            val rows = call.argument<List<Map<String, Any?>>>("rows")!!
            rows.forEachIndexed { index, row ->
                val values = (row["input"] as List<*>).map { (it as Number?)?.toFloat() ?: Float.NaN }.toFloatArray()
                val actual = predict(values)
                require(actual["label"] == (row["label"] as Number).toInt()) { "Label parity failed at chunk row $index" }
                val expected = row["probabilities"] as List<*>
                val probabilities = actual["probabilities"] as List<*>
                require(expected.size == probabilities.size) { "Probability shape mismatch" }
                expected.zip(probabilities).forEach { (e, a) ->
                    val target = (e as Number).toDouble()
                    val got = (a as Number).toDouble()
                    require(got.isFinite() && abs(got - target) <= 1e-6 + 1e-5 * abs(target)) {
                        "Probability parity failed at chunk row $index: $got vs $target"
                    }
                }
            }; mapOf("validated" to rows.size)
        }
        "close" -> { closeSession(); null }
        "thermal" -> thermal()
        "memory" -> memory()
        "metadata" -> metadata()
        "openReplay" -> { reader?.close(); reader = asset(call.argument<String>("asset")!!).bufferedReader(); null }
        "readReplay" -> {
            val lines = ArrayList<String>(256)
            for (i in 0 until 256) { val line = reader!!.readLine() ?: break; lines.add(line) }
            lines
        }
        "closeReplay" -> { reader?.close(); reader = null; null }
        "export" -> export(call.argument<String>("json")!!, call.argument<String>("token")!!)
        "share" -> {
            val token = safeToken(call.argument<String>("token")!!)
            val directory = File(getExternalFilesDir(null), "results")
            val uris = arrayListOf("json", "samples.csv", "memory.csv").mapTo(ArrayList()) { suffix ->
                FileProvider.getUriForFile(this, "$packageName.files", File(directory, "$token.$suffix"))
            }
            val share = Intent(Intent.ACTION_SEND_MULTIPLE).apply {
                type = "application/octet-stream"
                putParcelableArrayListExtra(Intent.EXTRA_STREAM, uris)
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            runOnUiThread { startActivity(Intent.createChooser(share, "Export benchmark results")) }; null
        }
        else -> throw IllegalArgumentException("Unknown method ${call.method}")
    }

    private fun predict(values: FloatArray): Map<String, Any> {
        require(values.size == width) { "Expected $width features, got ${values.size}" }
        require(values.none { it.isInfinite() }) { "Infinite features are unsupported" }
        val active = checkNotNull(session) { "Model is not loaded" }
        // Tensor is prepared before the native execution timer. App timer includes it.
        OnnxTensor.createTensor(environment, FloatBuffer.wrap(values), longArrayOf(1, width.toLong())).use { tensor ->
            val start = System.nanoTime()
            val result = active.run(mapOf(inputName to tensor))
            val elapsed = (System.nanoTime() - start) / 1000.0
            result.use {
                val labels = it.get(labelName).get().value as LongArray
                val probabilities = (it.get(probabilityName).get().value as Array<*>)[0] as FloatArray
                return mapOf("native_us" to elapsed, "label" to labels[0].toInt(),
                    "probabilities" to probabilities.map { p -> p.toDouble() })
            }
        }
    }

    private fun memory(): Map<String, Any?> {
        val info = Debug.MemoryInfo(); Debug.getMemoryInfo(info)
        val status = File("/proc/self/status").readLines().mapNotNull { line ->
            val key = line.substringBefore(':')
            if (key in listOf("VmRSS", "VmHWM")) key to line.substringAfter(':').trim().substringBefore(' ').toLong() else null
        }.toMap()
        return mapOf("pss_kib" to info.totalPss, "native_pss_kib" to info.nativePss,
            "dalvik_pss_kib" to info.dalvikPss, "rss_kib" to status["VmRSS"],
            "process_high_water_rss_kib" to status["VmHWM"], "thermal_status" to thermal(),
            "elapsed_realtime_ms" to SystemClock.elapsedRealtime())
    }

    private fun metadata(): Map<String, Any?> {
        val manager = getSystemService(ACTIVITY_SERVICE) as ActivityManager
        val ram = ActivityManager.MemoryInfo(); manager.getMemoryInfo(ram)
        val battery = registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val power = getSystemService(POWER_SERVICE) as PowerManager
        return mapOf("manufacturer" to Build.MANUFACTURER, "model" to Build.MODEL,
            "device_id" to android.provider.Settings.Secure.getString(contentResolver,
                android.provider.Settings.Secure.ANDROID_ID),
            "os_build_fingerprint" to Build.FINGERPRINT,
            "chipset" to if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else Build.HARDWARE,
            "ram_bytes" to ram.totalMem, "android" to Build.VERSION.RELEASE,
            "sdk" to Build.VERSION.SDK_INT, "abi" to Build.SUPPORTED_ABIS.toList(),
            "build_type" to BuildConfig.BUILD_TYPE, "app_version" to BuildConfig.VERSION_NAME,
            "runtime_version" to environment.version, "pid" to Process.myPid(),
            "models_loaded" to modelsLoaded, "process_started_elapsed_ms" to processStarted,
            "thermal_status" to thermal(), "power_save" to power.isPowerSaveMode,
            "battery_level" to battery?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1),
            "battery_scale" to battery?.getIntExtra(BatteryManager.EXTRA_SCALE, -1),
            "plugged" to battery?.getIntExtra(BatteryManager.EXTRA_PLUGGED, -1),
            "battery_temperature_tenths_c" to battery?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, -1),
            "screen_brightness" to android.provider.Settings.System.getInt(contentResolver,
                android.provider.Settings.System.SCREEN_BRIGHTNESS, -1))
    }

    private fun safeToken(token: String): String {
        require(token.matches(Regex("[A-Za-z0-9_-]{1,100}"))) { "Invalid export token" }; return token
    }

    private fun export(json: String, name: String): String {
        val token = safeToken(name)
        val directory = File(getExternalFilesDir(null), "results").apply { mkdirs() }
        val document = JSONObject(json)
        fun csv(array: JSONArray, target: File) {
            val keys = linkedSetOf<String>()
            for (i in 0 until array.length()) array.getJSONObject(i).keys().forEach { keys.add(it) }
            fun escape(value: String) = "\"" + value.replace("\"", "\"\"") + "\""
            target.bufferedWriter().use { writer ->
                writer.appendLine(keys.joinToString(",") { escape(it) })
                for (i in 0 until array.length()) {
                    val row = array.getJSONObject(i)
                    writer.appendLine(keys.joinToString(",") { escape(row.opt(it)?.toString() ?: "") })
                }
            }
        }
        csv(document.getJSONArray("samples"), File(directory, "$token.samples.csv"))
        csv(document.getJSONArray("memory"), File(directory, "$token.memory.csv"))
        val target = File(directory, "$token.json")
        // Publish JSON last, after CSVs: host polling uses this completion marker.
        val temporary = File(directory, "$token.json.tmp")
        temporary.writeText(json)
        check(temporary.renameTo(target)) { "Could not publish results" }
        return target.absolutePath
    }

    override fun onDestroy() {
        worker.execute { reader?.close(); closeSession() }
        worker.shutdown()
        super.onDestroy()
    }
}
