{{/*
Expand the chart name.
*/}}
{{- define "openmed-service.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Create a fully qualified app name.
*/}}
{{- define "openmed-service.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
Common labels.
*/}}
{{- define "openmed-service.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | quote }}
app.kubernetes.io/name: {{ include "openmed-service.name" . | quote }}
app.kubernetes.io/instance: {{ .Release.Name | quote }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service | quote }}
{{- end -}}

{{/*
Selector labels.
*/}}
{{- define "openmed-service.selectorLabels" -}}
app.kubernetes.io/name: {{ include "openmed-service.name" . | quote }}
app.kubernetes.io/instance: {{ .Release.Name | quote }}
{{- end -}}

{{/*
Name of the generated ConfigMap.
*/}}
{{- define "openmed-service.configMapName" -}}
{{- printf "%s-config" (include "openmed-service.fullname" .) -}}
{{- end -}}

{{/*
Name of the model-cache PVC.
*/}}
{{- define "openmed-service.modelCacheClaimName" -}}
{{- if .Values.persistence.existingClaim -}}
{{- .Values.persistence.existingClaim -}}
{{- else -}}
{{- printf "%s-model-cache" (include "openmed-service.fullname" .) -}}
{{- end -}}
{{- end -}}

{{/*
Trusted hosts needed for loopback probes and in-cluster Service DNS.
*/}}
{{- define "openmed-service.trustedHosts" -}}
{{- $fullname := include "openmed-service.fullname" . -}}
{{- $hosts := list "localhost" "127.0.0.1" "[::1]" $fullname (printf "%s.%s" $fullname .Release.Namespace) (printf "%s.%s.svc" $fullname .Release.Namespace) (printf "%s.%s.svc.cluster.local" $fullname .Release.Namespace) -}}
{{- range .Values.config.trustedHosts -}}
{{- $hosts = append $hosts . -}}
{{- end -}}
{{- join "," (uniq $hosts) -}}
{{- end -}}
