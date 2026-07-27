#!/usr/bin/env bash
# Install OpenMed Agent Skills into your coding agent(s).
#
# The same SKILL.md folders follow the open Agent Skills standard
# (https://agentskills.io), so one install works across compatible agents.
# Skills are symlinked (not copied), so a later `git pull` updates every
# installed skill. Existing files, directories, and unrelated symlinks are
# preserved.
#
# Usage:
#   ./install-skills.sh              # install into every supported agent (default)
#   ./install-skills.sh claude       # Claude Code    ~/.claude/skills
#   ./install-skills.sh codex        # OpenAI Codex   ~/.codex/skills
#   ./install-skills.sh opencode     # OpenCode       ~/.config/opencode/skills
#   ./install-skills.sh agents       # cross-client convention  ~/.agents/skills
#   ./install-skills.sh all          # all of the above
set -euo pipefail

SRC="$(cd "$(dirname "$0")/skills" && pwd)"
TARGET="${1:-all}"

link_into() {
  local dest="$1"
  local existing
  local link_path
  local name
  local skill
  local skill_path
  mkdir -p "$dest"
  local n=0
  local skipped=0
  for skill in "$SRC"/*/; do
    skill_path="${skill%/}"
    name="$(basename "$skill_path")"
    [ -f "$skill/SKILL.md" ] || continue   # only real skills, skip helpers/_template
    link_path="$dest/$name"
    if [ -L "$link_path" ]; then
      existing="$(readlink "$link_path")"
      if [ "$existing" = "$skill" ] || [ "$existing" = "$skill_path" ]; then
        n=$((n + 1))
        continue
      fi
      echo "  skip $link_path (an unrelated symlink already exists)" >&2
      skipped=$((skipped + 1))
      continue
    fi
    if [ -e "$link_path" ]; then
      echo "  skip $link_path (a file or directory already exists)" >&2
      skipped=$((skipped + 1))
      continue
    fi
    ln -s "$skill_path" "$link_path"
    n=$((n + 1))
  done
  echo "  $n skills -> $dest"
  if [ "$skipped" -gt 0 ]; then
    echo "  $skipped existing entries preserved" >&2
  fi
}

install_target() {
  case "$1" in
    claude)   echo "Claude Code:";  link_into "$HOME/.claude/skills" ;;
    codex)    echo "OpenAI Codex:"; link_into "$HOME/.codex/skills" ;;
    opencode) echo "OpenCode:";     link_into "$HOME/.config/opencode/skills" ;;
    agents)   echo "Cross-client convention (~/.agents/skills):"; link_into "$HOME/.agents/skills" ;;
  esac
}

case "$TARGET" in
  claude|codex|opencode|agents)
    echo "Installing OpenMed skills:"
    install_target "$TARGET"
    ;;
  all)
    echo "Installing OpenMed skills into every supported agent:"
    for t in claude codex opencode agents; do install_target "$t"; done
    ;;
  *)
    echo "usage: $0 [claude|codex|opencode|agents|all]" >&2
    exit 1
    ;;
esac

echo "Done. Restart your agent if newly installed skills do not appear."
echo "On Windows, symlinks need Developer Mode or admin — otherwise copy:"
echo "    cp -r skills/*/ ~/.claude/skills/    # (swap the path for your agent)"
