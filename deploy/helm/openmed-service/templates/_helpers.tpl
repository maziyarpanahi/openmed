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

{{- define "openmed-service.journeyArtifactClaimName" -}}
{{- if .Values.journey.artifacts.existingClaim -}}
{{- .Values.journey.artifacts.existingClaim -}}
{{- else -}}
{{- printf "%s-journey-artifacts" (include "openmed-service.fullname" .) -}}
{{- end -}}
{{- end -}}

{{- define "openmed-service.journeyWorkerName" -}}
{{- printf "%s-journey-worker" (include "openmed-service.fullname" .) -}}
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

{{/*
How the Journey DSN reaches containers: "file" (a read-only Secret volume plus
OPENMED_JOURNEY_POSTGRES_DSN_FILE) or "env" (a secretKeyRef environment value).
*/}}
{{- define "openmed-service.journeyDatabaseSource" -}}
{{- $source := default "file" .Values.journey.database.source -}}
{{- if not (has $source (list "file" "env")) -}}
{{- fail "journey.database.source must be \"file\" or \"env\"" -}}
{{- end -}}
{{- $source -}}
{{- end -}}

{{- define "openmed-service.journeyDatabaseEnv" -}}
{{- if eq (include "openmed-service.journeyDatabaseSource" .) "file" -}}
- name: OPENMED_JOURNEY_POSTGRES_DSN_FILE
  value: {{ printf "%s/dsn" (trimSuffix "/" .Values.journey.database.mountPath) | quote }}
{{- else -}}
- name: OPENMED_JOURNEY_POSTGRES_DSN
  valueFrom:
    secretKeyRef:
      name: {{ .Values.journey.database.secretName | quote }}
      key: {{ .Values.journey.database.secretKey | quote }}
{{- end -}}
{{- end -}}

{{- define "openmed-service.journeyDatabaseVolumeMount" -}}
{{- if eq (include "openmed-service.journeyDatabaseSource" .) "file" -}}
- name: journey-database
  mountPath: {{ .Values.journey.database.mountPath | quote }}
  readOnly: true
{{- end -}}
{{- end -}}

{{/*
Secret volumes are root-owned. With the pod fsGroup the kubelet makes the files
group-readable (0440); the service accepts group read for root-owned files only.
*/}}
{{- define "openmed-service.journeyDatabaseVolume" -}}
{{- if eq (include "openmed-service.journeyDatabaseSource" .) "file" -}}
- name: journey-database
  secret:
    secretName: {{ .Values.journey.database.secretName | quote }}
    defaultMode: 0440
    items:
      - key: {{ .Values.journey.database.secretKey | quote }}
        path: dsn
{{- end -}}
{{- end -}}

{{/*
Validate one secretFiles entry and return its mount directory.
*/}}
{{- define "openmed-service.secretFileDir" -}}
{{- $env := required "secretFiles[].env is required" .item.env -}}
{{- if not (regexMatch "^[A-Z][A-Z0-9_]*$" $env) -}}
{{- fail "secretFiles[].env must be an upper-case environment variable name" -}}
{{- end -}}
{{- if hasSuffix "_FILE" $env -}}
{{- fail "secretFiles[].env names the credential setting without the _FILE suffix" -}}
{{- end -}}
{{- $_ := required "secretFiles[].secretName is required" .item.secretName -}}
{{- $_ = required "secretFiles[].secretKey is required" .item.secretKey -}}
{{- printf "%s/%s" (trimSuffix "/" .root.Values.secretFilesMountPath) (lower $env | replace "_" "-") -}}
{{- end -}}

{{- define "openmed-service.secretFileEnv" -}}
{{- range $item := .Values.secretFiles }}
- name: {{ printf "%s_FILE" $item.env }}
  value: {{ printf "%s/value" (include "openmed-service.secretFileDir" (dict "root" $ "item" $item)) | quote }}
{{- end }}
{{- end -}}

{{- define "openmed-service.secretFileVolumeMounts" -}}
{{- range $index, $item := .Values.secretFiles }}
- name: {{ printf "secret-file-%d" $index }}
  mountPath: {{ include "openmed-service.secretFileDir" (dict "root" $ "item" $item) | quote }}
  readOnly: true
{{- end }}
{{- end -}}

{{- define "openmed-service.secretFileVolumes" -}}
{{- range $index, $item := .Values.secretFiles }}
- name: {{ printf "secret-file-%d" $index }}
  secret:
    secretName: {{ $item.secretName | quote }}
    defaultMode: 0440
    items:
      - key: {{ $item.secretKey | quote }}
        path: value
{{- end }}
{{- end -}}
