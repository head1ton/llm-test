.PHONY: test test-contract test-integration

test:
	uv run pytest -q

test-contract:
	uv run pytest -q -m "not integration"

test-integration:
	ENABLE_INTEGRATION_TESTS=1 uv run pytest -q -m integration
