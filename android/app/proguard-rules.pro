# Los metodos nativos y la clase puente no se pueden renombrar: JNI los resuelve por nombre.
-keepclasseswithmembernames class * { native <methods>; }
-keep class com.birdgraph.app.NativeAnalyzer { *; }
-keep interface com.birdgraph.app.NativeAnalyzer$Progress { *; }
