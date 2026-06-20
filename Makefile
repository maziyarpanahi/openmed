# Makefile for openmed package management

.PHONY: help build publish release clean install lint format format-check lint-swift format-swift quality test docs-serve docs-build docs-stage docs-deploy

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

build: ## Build the package
	@echo "🔨 Building package..."
	python3 -m build

publish: ## Publish to PyPI using Hatch
	@echo "📤 Publishing to PyPI..."
	hatch publish

release: clean build publish ## Full release cycle: clean, build, publish

clean: ## Clean build artifacts
	@echo "🧹 Cleaning build artifacts..."
	rm -rf dist/ build/ *.egg-info/

install: ## Install the package locally
	@echo "📦 Installing package locally..."
	pip install -e .

lint: ## Run Ruff lint checks
	@echo "🔎 Running Ruff lint checks..."
	ruff check .

format: ## Apply Ruff import fixes and formatting
	@echo "🎨 Formatting Python code with Ruff..."
	ruff check --fix .
	ruff format .

format-check: ## Check Ruff formatting without modifying files
	@echo "🔎 Checking Ruff formatting..."
	ruff format --check .

lint-swift: ## Run Swift format lint checks for OpenMedKit
	@echo "🔎 Running Swift format lint checks..."
	scripts/lint_swift.sh

format-swift: ## Apply Swift formatting for OpenMedKit
	@echo "🎨 Formatting Swift code with swift-format..."
	scripts/format_swift.sh

quality: lint format-check test ## Run the local quality gate

test: ## Run the test suite
	@echo "🧪 Running tests..."
	pytest

docs-serve: ## Run the MkDocs dev server with live reload
	@echo "📚 Serving docs at http://127.0.0.1:8008 ..."
	uv run mkdocs serve -a 127.0.0.1:8008

docs-build: ## Build the MkDocs site (strict mode)
	@echo "🏗️ Building documentation..."
	uv run mkdocs build --strict

docs-stage: docs-build ## Build docs and bundle them with the marketing site into site/
	@echo "📦 Bundling marketing site with docs..."
	rsync -av docs/website/ site/

docs-deploy: docs-stage ## Publish marketing site + docs bundle to GitHub Pages (gh-pages branch)
	@echo "🚀 Deploying documentation to GitHub Pages..."
	ghp-import site -f -p

test-build: ## Test build without publishing
	@echo "🧪 Testing build..."
	python3 -m build
	@echo "✅ Build successful! Check dist/ directory"

bump-patch: ## Bump patch version (0.1.1 -> 0.1.2)
	@echo "📈 Bumping patch version..."
	python3 scripts/release/release.py patch

bump-minor: ## Bump minor version (0.1.1 -> 0.2.0)
	@echo "📈 Bumping minor version..."
	python3 scripts/release/release.py minor

bump-major: ## Bump major version (0.1.1 -> 1.0.0)
	@echo "📈 Bumping major version..."
	python3 scripts/release/release.py major

# Quick commands for common workflows
patch: bump-patch release ## Bump patch version and release
minor: bump-minor release ## Bump minor version and release
major: bump-major release ## Bump major version and release
