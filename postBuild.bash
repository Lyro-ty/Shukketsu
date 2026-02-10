#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

# User-owned global prefix (no sudo needed)
mkdir -p "$HOME/.npm-global"
npm config set prefix "$HOME/.npm-global"

# Ensure shells can find it later
for f in "$HOME/.bashrc" "$HOME/.profile"; do
  touch "$f"
  if ! grep -q 'export PATH="$HOME/.npm-global/bin:$PATH"' "$f"; then
    echo 'export PATH="$HOME/.npm-global/bin:$PATH"' >> "$f"
  fi
done

export PATH="$HOME/.npm-global/bin:$PATH"

npm install -g @anthropic-ai/claude-code

which claude
claude --version