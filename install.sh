#!/bin/bash
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
BOLD='\033[1m'
RESET='\033[0m'

info()  { echo -e "${GREEN}[ok]${RESET} $1"; }
error() { echo -e "${RED}[error]${RESET} $1"; exit 1; }

echo -e "${BOLD}llms installer${RESET}"
echo ""

# Check fzf (required for interactive mode)
if command -v fzf &>/dev/null; then
    info "fzf found: $(fzf --version | head -1)"
else
    error "fzf is required for interactive mode. Install it: https://github.com/junegunn/fzf#installation"
fi

echo ""

# Install with the best available tool
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if command -v uv &>/dev/null; then
    echo -e "${BOLD}Installing llms via uv...${RESET}"
    uv tool install -e "$SCRIPT_DIR"
elif command -v pipx &>/dev/null; then
    echo -e "${BOLD}Installing llms via pipx...${RESET}"
    pipx install -e "$SCRIPT_DIR"
elif command -v pip &>/dev/null; then
    echo -e "${BOLD}Installing llms via pip...${RESET}"
    pip install -e "$SCRIPT_DIR"
else
    error "No installer found. Install one of: uv, pipx, or pip"
fi

echo ""
info "Done! Run ${BOLD}llms${RESET} to start."
