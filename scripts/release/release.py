#!/usr/bin/env python3
"""
Automated PyPI release script for openmed
Handles version bumping, building, and publishing in one command.
"""

import subprocess
import sys
import re
from pathlib import Path


def run_command(cmd, cwd=None):
    """Run a shell command and return True if successful."""
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, check=True,
                              capture_output=True, text=True)
        return True, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr


def get_current_version():
    """Get current version from pyproject.toml."""
    with open('pyproject.toml', 'r') as f:
        content = f.read()

    match = re.search(r'version\s*=\s*"([^"]+)"', content)
    if match:
        return match.group(1)
    raise ValueError("Could not find version in pyproject.toml")


def bump_version(current_version, bump_type='patch'):
    """Bump version according to semantic versioning."""
    major, minor, patch = map(int, current_version.split('.'))

    if bump_type == 'major':
        major += 1
        minor = 0
        patch = 0
    elif bump_type == 'minor':
        minor += 1
        patch = 0
    else:  # patch
        patch += 1

    return f"{major}.{minor}.{patch}"


def update_version_files(new_version):
    """Update version in all relevant files."""
    # Update pyproject.toml
    with open('pyproject.toml', 'r') as f:
        content = f.read()

    content = re.sub(r'(version\s*=\s*)"[^"]*"', f'\\1"{new_version}"', content)

    with open('pyproject.toml', 'w') as f:
        f.write(content)

    # Update __init__.py
    init_file = Path('openmed/__init__.py')
    if init_file.exists():
        with open(init_file, 'r') as f:
            content = f.read()

        content = re.sub(r'(__version__\s*=\s*)"[^"]*"', f'\\1"{new_version}"', content)

        with open(init_file, 'w') as f:
            f.write(content)


def main():
    """Main release process."""
    if len(sys.argv) > 1:
        bump_type = sys.argv[1].lower()
        if bump_type not in ['major', 'minor', 'patch']:
            print("Usage: python release.py [major|minor|patch]")
            print("Default: patch")
            bump_type = 'patch'
    else:
        bump_type = 'patch'

    print(f"🚀 Starting {bump_type} release process...")

    # Get current version
    try:
        current_version = get_current_version()
        print(f"📦 Current version: {current_version}")
    except Exception as e:
        print(f"❌ Error reading current version: {e}")
        return 1

    # Calculate new version
    new_version = bump_version(current_version, bump_type)
    print(f"🎯 New version: {new_version}")

    # Update version files
    try:
        update_version_files(new_version)
        print("✅ Version files updated")
    except Exception as e:
        print(f"❌ Error updating version files: {e}")
        return 1

    # Clean previous builds
    print("🧹 Cleaning previous builds...")
    Path('dist').mkdir(exist_ok=True)
    for file in Path('dist').glob('*'):
        file.unlink()

    # Build package
    print("🔨 Building package...")
    success, stdout, stderr = run_command("python -m build")
    if not success:
        print(f"❌ Build failed:\n{stderr}")
        return 1
    print("✅ Package built successfully")

    # Publish to PyPI
    print("📤 Publishing to PyPI...")
    success, stdout, stderr = run_command("python -m twine upload dist/*")
    if not success:
        print(f"❌ Publish failed:\n{stderr}")
        return 1

    print("✅ Successfully published to PyPI!")
    print(f"🔗 View at: https://pypi.org/project/openmed/{new_version}/")

    # Git commit (optional)
    try:
        run_command(f"git add .")
        run_command(f"git commit -m 'Release version {new_version}'")
        run_command(f"git tag v{new_version}")
        print("✅ Git commit and tag created")
    except:
        print("⚠️  Git operations skipped (repository not set up)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
