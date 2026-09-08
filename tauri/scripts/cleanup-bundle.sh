#!/bin/bash
# Clean up tauri build artifacts that pollute Launchpad with a duplicate
# Voicebox.app icon. Voicebox.app is the development build output and should
# only be copied to /Applications/Voicebox.app, not left in the bundle/
# directory where macOS Launchpad (via Spotlight) picks it up and shows it
# alongside the real install.

set -e

BUNDLE_DIR="$(cd "$(dirname "$0")/.." && pwd)/src-tauri/target/release/bundle/macos"

if [ -d "$BUNDLE_DIR/Voicebox.app" ]; then
  echo "Removing $BUNDLE_DIR/Voicebox.app (was: $(du -sh "$BUNDLE_DIR/Voicebox.app" 2>/dev/null | awk '{print $1}'))"
  rm -rf "$BUNDLE_DIR/Voicebox.app"
fi

# Also remove tar.gz and DMG that the bundle_dmg.sh step tries to build
# (they're created even when the dmg step fails, taking up extra disk).
find "$BUNDLE_DIR" -maxdepth 1 \( -name "Voicebox.app.tar.gz" -o -name "rw.*.dmg" \) -delete 2>/dev/null

echo "Bundle cleanup done. Real install: /Applications/Voicebox.app"
