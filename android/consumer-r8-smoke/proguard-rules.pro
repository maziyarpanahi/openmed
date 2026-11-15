# Intentionally empty of rules.
#
# This app stands in for an integrator that adds OpenMedKit and turns on R8
# without writing any OpenMedKit-specific rules: everything it needs must come
# from the consumer rules packaged in the AAR (proguard.txt). Do not add -keep,
# -dontwarn or -ignorewarnings rules here; the R8 consumer gate rejects any
# -dontwarn filter that is not in r8-dontwarn-allowlist.txt and any
# -ignorewarnings.
