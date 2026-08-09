# Makefile for openmed package management

.PHONY: help build publish release clean install lock-check lint type-check format format-check lint-swift format-swift quality test sbom grpc-proto grpc-proto-check brand-check docs-serve docs-build docs-stage docs-browser-test docs-deploy

UV ?= uv

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

build: ## Build the package
	@echo "🔨 Building package..."
	$(UV) run --frozen --extra dev --with build python -m build

publish: ## Publish to PyPI using Hatch
	@echo "📤 Publishing to PyPI..."
	hatch publish

release: clean build publish ## Full release cycle: clean, build, publish

clean: ## Clean build artifacts
	@echo "🧹 Cleaning build artifacts..."
	rm -rf dist/ build/ *.egg-info/

install: ## Install the package locally
	@echo "📦 Syncing the locked uv development environment..."
	$(UV) sync --frozen --extra dev

lock-check: ## Verify that uv.lock matches pyproject.toml
	@echo "🔎 Checking uv.lock against pyproject.toml..."
	$(UV) lock --check

lint: ## Run Ruff lint checks
	@echo "🔎 Running Ruff lint checks..."
	$(UV) run --frozen --extra dev ruff check .

type-check: ## Type-check the annotated public-module scope
	@echo "🔎 Running scoped mypy checks..."
	$(UV) run --frozen --extra dev mypy

format: ## Apply Ruff import fixes and formatting
	@echo "🎨 Formatting Python code with Ruff..."
	$(UV) run --frozen --extra dev ruff check --fix .
	$(UV) run --frozen --extra dev ruff format .

format-check: ## Check Ruff formatting without modifying files
	@echo "🔎 Checking Ruff formatting..."
	$(UV) run --frozen --extra dev ruff format --check .

lint-swift: ## Run Swift format lint checks for OpenMedKit
	@echo "🔎 Running Swift format lint checks..."
	scripts/lint_swift.sh

format-swift: ## Apply Swift formatting for OpenMedKit
	@echo "🎨 Formatting Swift code with swift-format..."
	scripts/format_swift.sh

quality: lint type-check format-check test ## Run the local quality gate

test: ## Run the test suite
	@echo "🧪 Running tests..."
	$(UV) run --frozen --extra dev pytest

sbom: ## Generate a CycloneDX 1.6 SBOM (sbom.cdx.json) for the runtime dependencies
	@echo "📦 Generating CycloneDX SBOM..."
	uv sync --frozen
	uv run --no-project --with 'cyclonedx-bom>=4.6,<7' python scripts/security/generate_sbom.py

grpc-proto: ## Regenerate committed gRPC protobuf stubs
	@echo "🔁 Generating gRPC protobuf stubs..."
	uv run --extra dev python scripts/generate_grpc_stubs.py

grpc-proto-check: ## Verify committed gRPC protobuf stubs are current
	@echo "🔎 Checking gRPC protobuf stubs..."
	uv run --extra dev python scripts/generate_grpc_stubs.py --check

docs-serve: ## Run the MkDocs dev server with live reload
	@echo "📚 Serving docs at http://127.0.0.1:8008 ..."
	uv run mkdocs serve -a 127.0.0.1:8008

brand-check: ## Validate brand sources, claims, consumers, and generated art
	@echo "🧭 Validating the repository-owned brand system..."
	uv run --extra dev --extra docs --frozen python scripts/brand/validate_system.py
	uv run --extra dev --extra docs --frozen python scripts/brand/update_claims.py --check
	uv run --extra dev --extra docs --frozen python scripts/brand/sync_consumers.py --check
	uv run --extra dev --extra docs --frozen python scripts/brand/render_social_assets.py --check

docs-build: docs-stage ## Build and verify the exact Pages artifact

docs-stage: brand-check ## Build docs, generated surfaces, and marketing into site/
	@echo "📦 Staging the verified GitHub Pages artifact..."
	uv run --extra dev --extra docs --frozen python scripts/docs/stage_pages.py

docs-browser-test: docs-stage ## Run the pinned Chromium, Firefox, and WebKit brand matrix
	@echo "🌐 Installing pinned browser-test dependencies..."
	npm ci --prefix tests/browser/brand
	npm exec --prefix tests/browser/brand -- playwright install chromium firefox webkit
	@echo "🧭 Running the cross-browser Pages matrix..."
	npm --prefix tests/browser/brand test

docs-deploy: docs-stage ## Publish marketing site + docs bundle to GitHub Pages (gh-pages branch)
	@echo "🚀 Deploying documentation to GitHub Pages..."
	ghp-import site -f -p

test-build: ## Test build without publishing
	@echo "🧪 Testing build..."
	$(UV) run --frozen --extra dev --with build python -m build
	@echo "✅ Build successful! Check dist/ directory"

bump-patch: ## Bump patch version (0.1.1 -> 0.1.2)
	@echo "📈 Bumping patch version..."
	$(UV) run --frozen --extra dev python scripts/release/release.py patch

bump-minor: ## Bump minor version (0.1.1 -> 0.2.0)
	@echo "📈 Bumping minor version..."
	$(UV) run --frozen --extra dev python scripts/release/release.py minor

bump-major: ## Bump major version (0.1.1 -> 1.0.0)
	@echo "📈 Bumping major version..."
	$(UV) run --frozen --extra dev python scripts/release/release.py major

# Quick commands for common workflows
patch: bump-patch release ## Bump patch version and release
minor: bump-minor release ## Bump minor version and release
major: bump-major release ## Bump major version and release
