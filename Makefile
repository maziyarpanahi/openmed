# Makefile for openmed package management

.PHONY: help build publish release clean install test

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

build: ## Build the package
	@echo "🔨 Building package..."
	python -m build

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

test-build: ## Test build without publishing
	@echo "🧪 Testing build..."
	python -m build
	@echo "✅ Build successful! Check dist/ directory"

bump-patch: ## Bump patch version (0.1.1 -> 0.1.2)
	@echo "📈 Bumping patch version..."
	python scripts/release/release.py patch

bump-minor: ## Bump minor version (0.1.1 -> 0.2.0)
	@echo "📈 Bumping minor version..."
	python scripts/release/release.py minor

bump-major: ## Bump major version (0.1.1 -> 1.0.0)
	@echo "📈 Bumping major version..."
	python scripts/release/release.py major

# Quick commands for common workflows
patch: bump-patch release ## Bump patch version and release
minor: bump-minor release ## Bump minor version and release
major: bump-major release ## Bump major version and release
