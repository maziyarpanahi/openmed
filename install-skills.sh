#!/usr/bin/env bash
# Install OpenMed Agent Skills into Claude Code and/or OpenAI Codex.
#
# Skills are symlinked (not copied) so a later `git pull` updates every
# installed skill in place. Each skill is a self-contained folder following the
# open Agent Skills standard (https://agentskills.io), so the same folder works
# in both agents unchanged.
#
# Usage:
#   ./install-skills.sh            # install into both agents (default)
#   ./install-skills.sh claude     # Claude Code only  (~/.claude/skills)
#   ./install-skills.sh codex      # Codex only        (~/.codex/skills)
#   ./install-skills.sh all        # both
set -euo pipefail

SRC="$(cd "$(dirname "$0")/skills" && pwd)"
TARGET="${1:-all}"

link_into() {
  local dest="$1"
  mkdir -p "$dest"
  local n=0
  for skill in "$SRC"/*/; do
    name="$(basename "$skill")"
    [ -f "$skill/SKILL.md" ] || continue   # only real skills, skip helpers/_template
    ln -sfn "$skill" "$dest/$name"
    n=$((n + 1))
  done
  echo "  $n skills -> $dest"
}

case "$TARGET" in
  claude) echo "Installing OpenMed skills into Claude Code:"; link_into "$HOME/.claude/skills" ;;
  codex)  echo "Installing OpenMed skills into Codex:";       link_into "$HOME/.codex/skills" ;;
  all)
    echo "Installing OpenMed skills into Claude Code and Codex:"
    link_into "$HOME/.claude/skills"
    link_into "$HOME/.codex/skills"
    ;;
  *) echo "usage: $0 [claude|codex|all]" >&2; exit 1 ;;
esac

echo "Done. Claude Code detects new skills live; restart Codex to pick them up."
echo "On Windows, symlinks need Developer Mode or admin — otherwise copy:"
echo "    cp -r skills/*/ ~/.claude/skills/"
