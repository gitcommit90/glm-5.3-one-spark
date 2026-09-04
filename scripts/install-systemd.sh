#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
mkdir -p "$UNIT_DIR"
cat > "$UNIT_DIR/glm53-one-spark.service" <<UNIT
[Unit]
Description=GLM-5.3 One-Spark EXL3 2.05 + DFlash2 K5
After=network-online.target docker.service
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$ROOT
ExecStart=$ROOT/start.sh
ExecStop=/usr/bin/docker stop -t 90 glm53-one-spark
TimeoutStartSec=0
TimeoutStopSec=120

[Install]
WantedBy=default.target
UNIT
systemctl --user daemon-reload
echo "Installed $UNIT_DIR/glm53-one-spark.service"
