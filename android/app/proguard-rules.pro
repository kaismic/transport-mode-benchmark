# ORT's JNI resolves tensor/value classes and constructors by their Java names.
# Retain these entry points in release builds as required by the Android binding.
-keep class ai.onnxruntime.** { *; }
