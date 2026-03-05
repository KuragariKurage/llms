#!/bin/bash
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BOLD='\033[1m'
RESET='\033[0m'

info()  { echo -e "${GREEN}[ok]${RESET} $1"; }
warn()  { echo -e "${YELLOW}[warn]${RESET} $1"; }
error() { echo -e "${RED}[error]${RESET} $1"; exit 1; }

echo -e "${BOLD}llms installer${RESET}"
echo ""

# Check fzf
if command -v fzf &>/dev/null; then
    info "fzf found: $(fzf --version | head -1)"
else
    error "fzf is required. Install it: https://github.com/junegunn/fzf#installation"
fi

# Check uv
if command -v uv &>/dev/null; then
    info "uv found: $(uv --version)"
else
    error "uv is required. Install it: https://docs.astral.sh/uv/getting-started/installation/"
fi

# Check Python version
python_version=$(uv python find 2>/dev/null || echo "")
if [ -n "$python_version" ]; then
    info "Python found: $python_version"
else
    warn "Python 3.13+ not found. uv will install it automatically."
fi

echo ""

# Install
echo -e "${BOLD}Installing llms...${RESET}"
uv tool install -e "$(cd "$(dirname "$0")" && pwd)"

echo ""
info "Done! Run ${BOLD}llms${RESET} to start."
