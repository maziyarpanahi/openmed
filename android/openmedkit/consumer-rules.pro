# OpenMedKit inference entry points can be created by class name from host
# integrations. Preserve their names and public constructors while allowing R8
# to optimize their implementations.
-keep,allowoptimization class com.openmed.openmedkit.BackendOnnxTokenClassifier {
    public <init>(...);
}
-keep,allowoptimization class com.openmed.openmedkit.onnx.OnnxTokenClassifier {
    public <init>(...);
}

# ONNX Runtime's Java API is backed by libonnxruntime4j_jni. Preserve native
# method and declaring-class names, along with every type used in JNI method
# descriptors.
-keepclasseswithmembernames,includedescriptorclasses class ai.onnxruntime.** {
    native <methods>;
}

# Model catalog rows are parsed with kotlinx.serialization JSON. Preserve the
# concrete model and its schema-facing field names without disabling method
# optimization for the rest of the catalog implementation.
-keepnames class com.openmed.openmedkit.catalog.ModelCatalogEntry
-keepclassmembers,allowoptimization class com.openmed.openmedkit.catalog.ModelCatalogEntry {
    <fields>;
}
-keepattributes RuntimeVisibleAnnotations,AnnotationDefault
