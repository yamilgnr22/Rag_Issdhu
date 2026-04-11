# Perfiles de ejecución

## dev-gpu

Perfil para desarrollo y benchmarking en estación local con GPU.

- admite modelos más pesados
- prioriza experimentación y calidad
- no representa la latencia final de producción

## prod-cpu

Perfil para despliegue on-prem en Windows Server sin GPU.

- embeddings y reranker locales optimizados para CPU
- generación evaluada contra presupuesto real de latencia
- preferencia por modo híbrido si la generación local no alcanza calidad o rendimiento

## Regla

La promoción a producción debe basarse en métricas y latencia del perfil `prod-cpu`, no en el rendimiento observado con `dev-gpu`.

## Fase 1

- desarrollo local puede usar SQLite y filesystem storage
- despliegues más cercanos a producción pueden activar PostgreSQL y MinIO mediante variables de entorno
