#!/bin/bash
#
# Astro-Otter Stacking Pipeline — interactive launcher
#
set -uo pipefail
# (no -e: whiptail returns a non-zero code on "Cancel", that's a normal
#  script flow, not an error that should abort everything abruptly)

# --- Colors -----------------------------------------------------------------
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

SESSIONS_DIR="./sessions"
LOG_DIR="./logs"
BACKTITLE="Astro-Otter — Stacking Pipeline"

trap 'clear; echo -e "${RED}Interrupted.${NC}"; exit 130' INT TERM

# --- Pre-flight checks -------------------------------------------------------
if ! command -v whiptail >/dev/null 2>&1; then
    echo -e "${RED}Error: whiptail is not installed (sudo apt install whiptail).${NC}" >&2
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo -e "${RED}Error: python3 not found in PATH.${NC}" >&2
    exit 1
fi

if [ ! -d "$SESSIONS_DIR" ]; then
    echo -e "${RED}Error: directory $SESSIONS_DIR does not exist.${NC}" >&2
    exit 1
fi

if [ -z "$(ls -A "$SESSIONS_DIR" 2>/dev/null)" ]; then
    whiptail --backtitle "$BACKTITLE" --title "Error" \
        --msgbox "No session found in $SESSIONS_DIR." 10 60
    exit 1
fi

mkdir -p "$LOG_DIR"

# --- Banner -------------------------------------------------------------------
clear
echo -e "${BLUE}${BOLD}"
cat <<'EOF'
  ╔══════════════════════════════════════════════╗
  ║        ASTRO-OTTER STACKING PIPELINE          ║
  ╚══════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# --- 1. Session selection -----------------------------------------------------
options=()
for dir in "$SESSIONS_DIR"/*/; do
    [ -d "$dir" ] || continue
    dir_name=$(basename "$dir")
    frame_count=$(find "$dir/lights" -maxdepth 1 -iname "*.fit*" 2>/dev/null | wc -l | tr -d ' ')
    options+=("$dir_name" "$frame_count frame(s)")
done

SELECTED_SESSION=$(whiptail --backtitle "$BACKTITLE" --title "Step 1/4: Session" \
    --menu "Select a session:" 20 70 8 "${options[@]}" \
    3>&1 1>&2 2>&3) || { echo -e "${YELLOW}Cancelled.${NC}"; exit 1; }

# --- 2. Format ------------------------------------------------------------
# WEBP is the default: it's the format used across every test session so far
# (the old default was PNG, never actually used in practice — that was
# probably "the bug"). Change the ON/OFF flags below if needed.
FORMAT=$(whiptail --backtitle "$BACKTITLE" --title "Step 2/4: Format" \
    --radiolist "Output format:" 15 60 4 \
    "webp" "WEBP (modern, lightweight)" ON \
    "png"  "PNG (lossless)"             OFF \
    "jpg"  "JPG (compressed)"           OFF \
    "tiff" "TIFF (raw)"                 OFF \
    3>&1 1>&2 2>&3) || { echo -e "${YELLOW}Cancelled.${NC}"; exit 1; }

# --- 3. DSO name (validated, won't accept an empty name) ---------
while true; do
    DSO=$(whiptail --backtitle "$BACKTITLE" --title "Step 3/4: Target" \
        --inputbox "DSO name (e.g. M31):" 10 60 \
        3>&1 1>&2 2>&3) || { echo -e "${YELLOW}Cancelled.${NC}"; exit 1; }
    DSO="$(echo "$DSO" | xargs)"   # trim whitespace
    [ -n "$DSO" ] && break
    whiptail --backtitle "$BACKTITLE" --title "Error" \
        --msgbox "The DSO name cannot be empty." 10 60
done

# --- 4. Verbose ---------------------------------------------------------------
VERBOSE=""
whiptail --backtitle "$BACKTITLE" --title "Step 4/4: Options" \
    --yesno "Enable verbose logging?" 10 60 && VERBOSE="--verbose"

# --- Summary & confirmation --------------------------------------------------
SUMMARY="Session: $SELECTED_SESSION\nFormat:  $FORMAT\nTarget:  $DSO\nVerbose: ${VERBOSE:-No}"
if ! whiptail --backtitle "$BACKTITLE" --title "Summary" \
        --yesno "$SUMMARY\n\nLaunch the stacking process?" 15 60; then
    clear
    echo -e "${RED}Process aborted by user.${NC}"
    exit 0
fi

# --- Run, with a live status line derived from stacking.py's JSON events ----
clear
LOG_FILE="$LOG_DIR/${DSO}_$(date +%Y%m%d-%H%M%S).log"
echo -e "${GREEN}${BOLD}Starting pipeline...${NC}"
echo -e "Session: ${BLUE}$SELECTED_SESSION${NC}  Format: ${BLUE}$FORMAT${NC}  Target: ${BLUE}$DSO${NC}"
echo -e "Full log: ${BLUE}$LOG_FILE${NC}"
echo ""

START_TIME=$(date +%s)

# stacking.py emits one JSON line per step on stderr (see emit() in the
# script). We copy it in full to the log file, and just display the
# "step"/"status" fields live on a single self-updating line.
python3 stacking.py "$SELECTED_SESSION" --format="$FORMAT" --dso="$DSO" $VERBOSE \
    2> >(tee "$LOG_FILE" | while IFS= read -r line; do
        step=$(printf '%s' "$line" | sed -n 's/.*"step": *"\([^"]*\)".*/\1/p')
        status=$(printf '%s' "$line" | sed -n 's/.*"status": *"\([^"]*\)".*/\1/p')
        [ -n "$step" ] && printf "\r\033[K${BLUE}[%s]${NC} %s" "${status:-...}" "$step"
    done)
EXIT_CODE=$?

echo ""
echo ""
ELAPSED=$(( $(date +%s) - START_TIME ))

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}${BOLD}✓ Completed successfully in ${ELAPSED}s${NC}"
    whiptail --backtitle "$BACKTITLE" --title "Done" \
        --msgbox "Processing succeeded in ${ELAPSED}s.\n\nLog: $LOG_FILE" 12 60
else
    echo -e "${RED}${BOLD}✗ Failed (code $EXIT_CODE)${NC} — see $LOG_FILE"
    whiptail --backtitle "$BACKTITLE" --title "Failed" \
        --msgbox "Processing failed (code $EXIT_CODE).\n\nCheck the log:\n$LOG_FILE" 12 60
fi

exit $EXIT_CODE
