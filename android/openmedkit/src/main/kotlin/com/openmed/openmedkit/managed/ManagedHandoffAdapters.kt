package com.openmed.openmedkit.managed

import android.content.ContentResolver
import android.content.Context
import android.content.RestrictionsManager
import android.net.Uri
import java.io.FileNotFoundException
import java.io.InputStream

/**
 * Android managed configuration: the app restrictions an enterprise mobility
 * management (EMM) console or device policy controller sets, read through
 * [RestrictionsManager.getApplicationRestrictions].
 *
 * Declare the keys with `@xml/openmed_managed_bundle_restrictions` (shipped in
 * this library) as the `android.content.APP_RESTRICTIONS` meta-data of the app
 * manifest, or copy its entries into the app's own restrictions file. An empty
 * restrictions bundle reads as "no managed configuration".
 */
class RestrictionsManagerConfigurationSource(context: Context) : ManagedConfigurationSource {
    private val context: Context = context.applicationContext ?: context

    @Suppress("DEPRECATION")
    override fun managedConfiguration(): Map<String, Any?>? {
        val manager = context.getSystemService(Context.RESTRICTIONS_SERVICE) as? RestrictionsManager ?: return null
        val restrictions = manager.applicationRestrictions ?: return null
        if (restrictions.isEmpty) return null
        return restrictions.keySet().associateWith { restrictions.get(it) }
    }

    override fun toString(): String = "RestrictionsManagerConfigurationSource"
}

/**
 * Opens `content://` handoffs (for example a file shared by an EMM agent's
 * `FileProvider`) through [ContentResolver.openInputStream]. Any other scheme
 * is refused, so the importer never reaches a network URL.
 */
class ContentResolverStreamOpener(private val resolver: ContentResolver) : ManagedContentStreamOpener {
    override fun open(uri: String): InputStream {
        val parsed = Uri.parse(uri)
        if (parsed.scheme != ContentResolver.SCHEME_CONTENT) throw FileNotFoundException("scheme_unsupported")
        return resolver.openInputStream(parsed) ?: throw FileNotFoundException("content_unavailable")
    }

    override fun toString(): String = "ContentResolverStreamOpener"
}
