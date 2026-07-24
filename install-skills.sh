#!/usr/bin/env bash
# Install OpenMed Agent Skills into your coding agent(s).
#
# The same SKILL.md folders follow the open Agent Skills standard
# (https://agentskills.io), so one install works across agents. Skills are
# symlinked (not copied), so a later `git pull` updates every installed skill.
#
# Usage:
#   ./install-skills.sh              # install into every supported agent (default)
#   ./install-skills.sh claude       # Claude Code    ~/.claude/skills
#   ./install-skills.sh codex        # OpenAI Codex   ~/.codex/skills
#   ./install-skills.sh opencode     # OpenCode       ~/.config/opencode/skills
#   ./install-skills.sh agents       # any other agent on the standard  ~/.agents/skills
#   ./install-skills.sh all          # all of the above
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

install_target() {
  case "$1" in
    claude)   echo "Claude Code:";  link_into "$HOME/.claude/skills" ;;
    codex)    echo "OpenAI Codex:"; link_into "$HOME/.codex/skills" ;;
    opencode) echo "OpenCode:";     link_into "$HOME/.config/opencode/skills" ;;
    agents)   echo "Generic (~/.agents/skills — any agent on the standard):"; link_into "$HOME/.agents/skills" ;;
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

echo "Done. Claude Code detects new skills live; restart Codex/OpenCode to pick them up."
echo "On Windows, symlinks need Developer Mode or admin — otherwise copy:"
echo "    cp -r skills/*/ ~/.claude/skills/    # (swap the path for your agent)"
