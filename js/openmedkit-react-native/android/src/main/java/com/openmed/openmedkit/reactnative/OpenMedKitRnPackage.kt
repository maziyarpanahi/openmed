package com.openmed.openmedkit.reactnative

import com.facebook.react.ReactPackage
import com.facebook.react.bridge.NativeModule
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.uimanager.ViewManager

/**
 * Autolinking entry point that registers [OpenMedKitRnModule] as `OpenMedKitRN`.
 *
 * The module is a legacy bridge module; React Native 0.77 and later expose it
 * under the New Architecture through the automatic TurboModule interop layer.
 */
class OpenMedKitRnPackage : ReactPackage {
    override fun createNativeModules(
        reactContext: ReactApplicationContext,
    ): List<NativeModule> = listOf(OpenMedKitRnModule(reactContext))

    override fun createViewManagers(
        reactContext: ReactApplicationContext,
    ): List<ViewManager<*, *>> = emptyList()
}
