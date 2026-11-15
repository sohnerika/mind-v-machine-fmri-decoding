#!/bin/bash
# Download BOLD5000 Release 2.0 TYPED betas for CSI1, plus the brain mask and the
# image-name list, and verify each file against Figshare's MD5 checksum.
#
# Usage (from the folder that holds ds001499 and BOLD5000_Stimuli):
#     cd ~/Documents/bold5000
#     caffeinate -i bash download_csi1_typed.sh
#
# Safe to rerun: files that are already complete and verified are skipped, and
# interrupted downloads resume where they stopped.

set -u
OUT="betas/CSI1"
mkdir -p "$OUT"
UA="Mozilla/5.0"

md5of() {
  if command -v md5 >/dev/null 2>&1; then md5 -q "$1"; else md5sum "$1" | cut -d' ' -f1; fi
}

# figshare_id  md5  filename
FILES="
27662166 045d7dfb30ae15bf0905f04c0092a2ff CSI1_GLMbetas-TYPED-FITHRF-GLMDENOISE-RR_ses-01.nii.gz
27662172 af4029a8db3155b374a66467de3a92b0 CSI1_GLMbetas-TYPED-FITHRF-GLMDENOISE-RR_ses-02.nii.gz
27662175 45e96b37fa75e063e96dc741700cca93 CSI1_GLMbetas-TYPED-FITHRF-GLMDENOISE-RR_ses-03.nii.gz
27662181 2e4c5565f58965a04f9e9d85698a69b8 CSI1_GLMbetas-TYPED-FITHRF-GLMDENOISE-RR_ses-04.nii.gz
27662184 f0efa84aaef81841eacc2856983e6860 CSI1_GLMbetas-TYPED-FITHRF-GLMDENOISE-RR_ses-05.nii.gz
27662187 dd96ebb4d5f87ee79f4e1f8c36b75a5c CSI1_GLMbetas-TYPED-FITHRF-GLMDENOISE-RR_ses-06.nii.gz
27662193 4812559ce01a2a311b819eadab0225f3 CSI1_GLMbetas-TYPED-FITHRF-GLMDENOISE-RR_ses-07.nii.gz
27662196 18b2469a55e167b2243e2ffc5edc4123 CSI1_GLMbetas-TYPED-FITHRF-GLMDENOISE-RR_ses-08.nii.gz
27662199 c44fb407e26a7b23d2c11e0e228e758d CSI1_GLMbetas-TYPED-FITHRF-GLMDENOISE-RR_ses-09.nii.gz
27662202 a26041b75a288e20f6ba62e755fd0c1d CSI1_GLMbetas-TYPED-FITHRF-GLMDENOISE-RR_ses-10.nii.gz
27662208 0c3baf40887c69e1307f7db0bf25667b CSI1_GLMbetas-TYPED-FITHRF-GLMDENOISE-RR_ses-11.nii.gz
27662211 22d89f08e01ff0a88d3908b51c7ee8ea CSI1_GLMbetas-TYPED-FITHRF-GLMDENOISE-RR_ses-12.nii.gz
27662214 7851082e254ef664e60fd0f8389da570 CSI1_GLMbetas-TYPED-FITHRF-GLMDENOISE-RR_ses-13.nii.gz
27662217 9f986bdfa5977cfbef23e38c8e987bbb CSI1_GLMbetas-TYPED-FITHRF-GLMDENOISE-RR_ses-14.nii.gz
27662223 a18befa073d5683477d4b68e9509697e CSI1_GLMbetas-TYPED-FITHRF-GLMDENOISE-RR_ses-15.nii.gz
27688734 bc9d7e17977579e738fdcfa5fb908f1f CSI1_brainmask.nii.gz
27688737 8d669115f9cc0e5c716a9f723855dfe0 CSI1_imgnames.txt
"

ok=0; bad=0
while read -r id md5 name; do
  [ -z "${id:-}" ] && continue
  dest="$OUT/$name"
  if [ -f "$dest" ] && [ "$(md5of "$dest")" = "$md5" ]; then
    echo "already have  $name"; ok=$((ok+1)); continue
  fi
  echo "downloading   $name"
  curl -L -A "$UA" -C - --retry 5 --retry-delay 10 -o "$dest" \
       "https://ndownloader.figshare.com/files/$id" < /dev/null
  if [ -f "$dest" ] && [ "$(md5of "$dest")" = "$md5" ]; then
    echo "  checksum OK"; ok=$((ok+1))
  else
    echo "  CHECKSUM MISMATCH: delete $dest and rerun this script"; bad=$((bad+1))
  fi
done <<< "$FILES"

echo
echo "Done: $ok verified, $bad failed (expected 17 verified)"
