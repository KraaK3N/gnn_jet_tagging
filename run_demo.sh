#!/usr/bin/env bash
# ==============================================================================
# Canned Demo Launcher for CS6012 Teaching Team
# Group 11: Pradipta Sadhukhan (25D1080) & Rahul Kumar (25M2015)
# Supervisor: Prof. Sharat Chandran | IIT Bombay
# ==============================================================================

set -e

# Change directory to the repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Execute the self-contained demo
python3 demo.py "$@"
