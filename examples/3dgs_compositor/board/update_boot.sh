#!/usr/bin/env bash
# Explicit, hash-guarded BOOT replacement; kernel/DTB/uEnv stay identical.
set -euo pipefail
[[ $# == 3 ]] || { echo 'usage: update_boot.sh apply|rollback staging_dir new_sha256'; exit 2; }
mode=$1
root=$2
new=$3
old=ff350477e624c50d2f8180fb4b9130ec7688fbc7ca553412ed7c3dd2a68b31ef
[[ "$root" == /root/fpga43dgs_compositor/boot_* && "$new" =~ ^[0-9a-f]{64}$ ]] || exit 2
fat="$root/fat"
mounted=0
cleanup(){
  local rc=$?
  trap - EXIT
  if ((mounted)); then sync; umount "$fat" || rc=1; fi
  exit "$rc"
}
trap cleanup EXIT
hash(){ sha256sum "$1" | awk '{print $1}'; }
[[ $EUID == 0 && $(uname -m) == aarch64 ]] || exit 1
[[ $(findmnt -n -o SOURCE /) == /dev/mmcblk0p2 ]] || { echo 'Unexpected root device'; exit 1; }
case "$mode" in
  apply) source="$root/BOOT.new.bin"; before=$old; after=$new ;;
  rollback) source="$root/BOOT.previous.bin"; before=$new; after=$old ;;
  *) exit 2 ;;
esac
[[ $(hash "$source") == "$after" ]] || { echo 'Source hash mismatch'; exit 1; }
if findmnt -rn -S /dev/mmcblk0p1 >/dev/null; then echo 'Boot partition already mounted'; exit 1; fi
[[ $(blkid -s TYPE -o value /dev/mmcblk0p1) == vfat ]] || exit 1
mkdir -p "$fat"
mount -t vfat -o rw /dev/mmcblk0p1 "$fat"
mounted=1
[[ $(hash "$fat/BOOT.bin") == "$before" ]] || { echo 'Unexpected current BOOT'; exit 1; }
if [[ $mode == apply ]]; then
  [[ ! -e "$root/BOOT.previous.bin" && ! -e "$fat/BOOT.pre_gsc1.bin" ]] || { echo 'Backup exists; inspect state'; exit 1; }
  cp -- "$fat/BOOT.bin" "$root/BOOT.previous.bin"
  cp -- "$fat/BOOT.bin" "$fat/BOOT.pre_gsc1.bin"
  [[ $(hash "$root/BOOT.previous.bin") == "$old" && $(hash "$fat/BOOT.pre_gsc1.bin") == "$old" ]] || exit 1
fi
(cd "$fat"; sha256sum Image fmqlmp-verify.dtb uEnv.txt) > "$root/unrelated_before_$mode.sha256"
[[ ! -e "$fat/BOOT.next_gsc1.bin" ]] || { echo 'Staged BOOT already exists'; exit 1; }
cp -- "$source" "$fat/BOOT.next_gsc1.bin"
[[ $(hash "$fat/BOOT.next_gsc1.bin") == "$after" ]] || exit 1
sync
mv -f -- "$fat/BOOT.next_gsc1.bin" "$fat/BOOT.bin"
sync
umount "$fat"
mounted=0
mount -t vfat -o ro /dev/mmcblk0p1 "$fat"
mounted=1
[[ $(hash "$fat/BOOT.bin") == "$after" ]] || exit 1
(cd "$fat"; sha256sum -c "$root/unrelated_before_$mode.sha256")
sha256sum "$fat/BOOT.bin" "$fat/BOOT.pre_gsc1.bin"
echo "PASS: BOOT $mode verified after remount. Reboot separately."
