# Prometheus and Grafana

This directory mirrors the existing monitoring stack in
`/home/quyifei/multimodal/prometheus_grafana`.

It launches Prometheus and Grafana with Docker Compose. Prometheus scrapes the
vLLM metrics endpoint on the host:

```text
host.docker.internal:8333
```

## Launch

```bash
cd /home/zhongyu/project/vllm-omni-team/benchmarks/workload_generation/monitoring
docker compose up
```

Open:

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000
- vLLM metrics: http://localhost:8333/metrics

Grafana default login:

```text
admin / admin
```

## Grafana Dashboard

Add the Prometheus datasource manually:

1. Open http://localhost:3000/connections/datasources/new
2. Select Prometheus
3. Set the Prometheus server URL to:

```text
http://prometheus:9090
```

Then import `grafana.json` from this directory at:

```text
http://localhost:3000/dashboard/import
```

Select the Prometheus datasource during import.
