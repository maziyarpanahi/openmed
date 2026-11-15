// Objective-C export file that registers the Swift `OpenMedKitRN` class with
// React Native. React Native discovers Swift modules only through these
// macros, so every selector here must match an `@objc(...)` name in
// OpenMedKitRN.swift and a method of `OpenMedKitNativeModule` in src/index.ts.

#import <React/RCTBridgeModule.h>

@interface RCT_EXTERN_MODULE(OpenMedKitRN, NSObject)

RCT_EXTERN_METHOD(loadModel:(NSDictionary *)options
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)

RCT_EXTERN_METHOD(analyzeText:(NSString *)text
                  options:(NSDictionary *)options
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)

RCT_EXTERN_METHOD(extractPii:(NSString *)text
                  options:(NSDictionary *)options
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)

RCT_EXTERN_METHOD(deidentify:(NSString *)text
                  options:(NSDictionary *)options
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)

@end
