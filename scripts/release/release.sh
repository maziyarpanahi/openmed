#!/bin/bash
# Simple release script for openmed
# Usage: ./release.sh [patch|minor|major]

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting release process...${NC}"

# Determine bump type
BUMP_TYPE=${1:-patch}
if [[ ! "$BUMP_TYPE" =~ ^(patch|minor|major)$ ]]; then
    echo -e "${RED}Error: Invalid bump type. Use patch, minor, or major${NC}"
    exit 1
fi

echo -e "${YELLOW}📦 Bump type: $BUMP_TYPE${NC}"

# Clean and build
echo -e "${YELLOW}🧹 Cleaning and building...${NC}"
rm -rf dist/
python -m build

# Publish using Hatch (it will handle credentials)
echo -e "${YELLOW}📤 Publishing to PyPI...${NC}"
hatch publish

echo -e "${GREEN}✅ Successfully published to PyPI!${NC}"

# Optional: Git operations
if git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${YELLOW}💾 Creating git commit and tag...${NC}"
    git add .
    git commit -m "Release: bump $BUMP_TYPE version" || echo "No changes to commit"
    git tag "v$(python -c "import toml; print(toml.load('pyproject.toml')['project']['version'])")" || echo "Tag may already exist"
    echo -e "${GREEN}✅ Git operations completed${NC}"
else
    echo -e "${YELLOW}⚠️  Git repository not initialized - skipping git operations${NC}"
fi
