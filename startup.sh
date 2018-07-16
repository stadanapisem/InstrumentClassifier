#!/usr/bin/env bash
set -e

Xvfb $DISPLAY -screen 0 1920x1080x16 &>/dev/null &

fluxbox &>/dev/null &

x11vnc -forever -passwd asdasd -display $DISPLAY &>/dev/null &

exec "$@"

