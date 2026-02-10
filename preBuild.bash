#!/usr/bin/env bash
set -euo pipefail

# Pin Node LTS (20.x). You can change this later if needed.
NODE_VER="v20.19.6"

INSTALL_ROOT="${HOME}/.local/node"
BIN_DIR="${HOME}/.local/bin"

mkdir -p "${INSTALL_ROOT}" "${BIN_DIR}"

ARCH="$(uname -m)"
case "${ARCH}" in
  x86_64)  NODE_ARCH="linux-x64" ;;
  aarch64) NODE_ARCH="linux-arm64" ;;
  *) echo "Unsupported arch: ${ARCH}"; exit 1 ;;
esac

TARBALL="node-${NODE_VER}-${NODE_ARCH}.tar.gz"
URL="https://nodejs.org/dist/${NODE_VER}/${TARBALL}"
TMP="/tmp/${TARBALL}"

# Download with curl if available, else python3
if command -v curl >/dev/null 2>&1; then
  curl -fsSL "${URL}" -o "${TMP}"
elif command -v python3 >/dev/null 2>&1; then
  python3 - <<PY
import urllib.request
urllib.request.urlretrieve("${URL}", "${TMP}")
print("Downloaded ${TMP}")
PY
else
  echo "Neither curl nor python3 is available to download ${URL}"
  exit 1
fi

# Extract (.tar.gz avoids needing xz)
rm -rf "${INSTALL_ROOT}/current"
mkdir -p "${INSTALL_ROOT}"
tar -xzf "${TMP}" -C "${INSTALL_ROOT}"
mv "${INSTALL_ROOT}/node-${NODE_VER}-${NODE_ARCH}" "${INSTALL_ROOT}/current"

# Symlinks into ~/.local/bin
ln -sf "${INSTALL_ROOT}/current/bin/node" "${BIN_DIR}/node"
ln -sf "${INSTALL_ROOT}/current/bin/npm"  "${BIN_DIR}/npm"
ln -sf "${INSTALL_ROOT}/current/bin/npx"  "${BIN_DIR}/npx"

# Ensure future shells see it
for f in "${HOME}/.bashrc" "${HOME}/.profile"; do
  touch "$f"
  if ! grep -q 'export PATH="$HOME/.local/bin:$PATH"' "$f"; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$f"
  fi
done

# Sanity check inside the build step
export PATH="${BIN_DIR}:$PATH"
node -v
npm -v