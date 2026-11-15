// Registers the Swift OpenMedKitParityProbe class with React Native.
// The probe is example-only: it is linked into the parity test app, never
// into apps that depend on @openmed/openmedkit-react-native.

#import <React/RCTBridgeModule.h>

@interface RCT_EXTERN_MODULE (OpenMedKitParityProbe, NSObject)

RCT_EXTERN_METHOD(describeEnvironment
                  : (RCTPromiseResolveBlock)resolve rejecter
                  : (RCTPromiseRejectBlock)reject)

RCT_EXTERN_METHOD(prepareModelFixture
                  : (RCTPromiseResolveBlock)resolve rejecter
                  : (RCTPromiseRejectBlock)reject)

RCT_EXTERN_METHOD(createEphemeralHashSecret
                  : (RCTPromiseResolveBlock)resolve rejecter
                  : (RCTPromiseRejectBlock)reject)

RCT_EXTERN_METHOD(runDirect
                  : (NSString *)requestJSON resolver
                  : (RCTPromiseResolveBlock)resolve rejecter
                  : (RCTPromiseRejectBlock)reject)

RCT_EXTERN_METHOD(writeReport
                  : (NSString *)reportJSON resolver
                  : (RCTPromiseResolveBlock)resolve rejecter
                  : (RCTPromiseRejectBlock)reject)

@end
