# ONNX Runtime resolves Java classes and members from JNI.
# See https://onnxruntime.ai/docs/build/android.html#proguard-rules-for-r8-minimization-android-app-builds-to-work
-keep class ai.onnxruntime.** { *; }

# DJL discovers these providers through META-INF/services.
-keep class ai.djl.engine.rust.RsEngineProvider { public <init>(); }
-keep class ai.djl.huggingface.zoo.HfZooProvider { public <init>(); }
-keep class ai.djl.engine.rust.zoo.RsZooProvider { public <init>(); }

# DJL's tokenizer Java surface binds to libdjl_tokenizer.so through JNI.
-keep class ai.djl.huggingface.tokenizers.jni.** { *; }

# Keep the OpenMedKit inference seam available to reflective integrations.
-keep class com.openmed.openmedkit.** implements com.openmed.openmedkit.OnnxTokenClassifier { *; }
-keep class com.openmed.openmedkit.onnx.OnnxTokenClassifier { public <init>(...); }

# kotlinx.serialization ships generated-serializer rules. OpenMedKit parses the
# catalog through JsonElement, so only its public catalog boundary needs keeping.
-keep class com.openmed.openmedkit.catalog.ModelCatalog { public *; }
-keep class com.openmed.openmedkit.catalog.ModelCatalogEntry { public *; }
