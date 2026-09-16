Pod::Spec.new do |s|
  s.name             = 'openmedkit_flutter'
  s.version          = '0.1.0'
  s.summary          = 'Local-first OpenMed token classification for Flutter.'
  s.description      = <<-DESC
Typed Dart FFI bindings and offline platform registration for OpenMed.
                       DESC
  s.homepage         = 'https://github.com/maziyarpanahi/openmed'
  s.license          = { :type => 'Apache-2.0', :file => '../../../LICENSE' }
  s.author           = { 'Maziyar Panahi' => 'maziyar.panahi@iscpif.fr' }
  s.source           = { :path => '.' }
  s.source_files     = 'Classes/**/*'
  s.dependency 'Flutter'
  s.dependency 'onnxruntime-c', '1.20.0'
  s.platform = :ios, '13.0'
  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'GCC_C_LANGUAGE_STANDARD' => 'c11',
    'GCC_PREPROCESSOR_DEFINITIONS' => '$(inherited) OPENMED_FFI_BUILDING_LIBRARY=1 OPENMED_FFI_ONNXRUNTIME=1'
  }
  s.swift_version = '5.0'
end
