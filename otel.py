import os
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "llm-agent")
OTLP_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")    # 예: http://localhost:4318/v1/traces

def setup_tracing():
    if not OTLP_ENDPOINT:
        return  # 엔드포인트 없으면 비활성(로컬 기본)

    resource = Resource.create({"service.name": SERVICE_NAME})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=OTLP_ENDPOINT)
    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

def get_tracer():
    return trace.get_tracer(SERVICE_NAME)
