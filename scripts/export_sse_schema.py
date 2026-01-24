import sys
import os

# Add the project root to sys.path so we can import sse_events
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pydantic import TypeAdapter
from sse_events import SSEEvent
import json

adapter = TypeAdapter(SSEEvent)
schema = adapter.json_schema()

print(json.dumps(schema, ensure_ascii=False, indent=2))
