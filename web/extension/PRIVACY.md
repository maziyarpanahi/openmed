# OpenMed PHI Guard privacy notice

OpenMed PHI Guard examines page text only inside the browser. Detection runs in
the extension's private background context with the bundled OpenMed web
package. The extension does not send page text, detected spans, or settings to
OpenMed or to any other service.

The extension requests only browser storage, which holds the enabled state and
selected policy profile for each site. It does not store page text or detected
identifiers. Its manifest declares no host permissions for outbound access, and
its extension-page content security policy disables network connections.

The content script runs on HTTP and HTTPS pages so it can warn before text is
submitted. It does not inspect password fields, browser-internal pages, or
non-text content. Users can disable scanning for any site from the in-page
OpenMed panel.

This privacy guard is an assistive safeguard, not a guarantee that every
identifier will be found. Review masked text before sharing it. OpenMed is not a
medical device and must not be used to make autonomous clinical decisions.
