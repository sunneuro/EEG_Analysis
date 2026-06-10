# scripts/03_import_filter_eeg.py
# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 OF EEG PREPROCESSING: Import raw EEG and filter
#
# What this script does:
#   1. Loads the raw BrainVision EEG file (.vhdr / .vmrk / .eeg)
#   2. Sets the standard 10-20 electrode montage
#   3. Adds Cz back as a flat channel (online reference — not saved in raw)
#   4. Re-sets the montage so Cz has a physical position
#   5. Applies bandpass filter: 0.1–40 Hz (FIR, Hamming window)
#   6. Applies notch filter: 50 Hz + 100 Hz harmonic (UK mains)
#   7. Prints event summary (trigger codes from E-Prime)
#   8. Saves filtered data as .fif for script 04
#
# Filter rationale (consistent with Luck & Kappenman 2012):
#   0.1 Hz highpass : removes slow DC drift and sweat artefacts
#                     (does not distort N2 or P3b components)
#   40 Hz lowpass   : removes high-frequency muscle noise
#                     (ERP components of interest are all < 30 Hz)
#   50 Hz notch     : removes UK electrical mains interference
#   100 Hz notch    : removes first harmonic of mains
#
# Recording parameters (actiCHamp, BrainVision):
#   Channels    : 31 recorded + Cz (online reference, re-added here) = 32
#   Sample rate : 500 Hz
#   Reference   : Cz (online) → average reference applied in script 04
#   Hardware filter: DC–140 Hz, no notch (applied here in software)
#
# Usage:
#   python3 scripts/03_import_filter_eeg.py <participant_id> <task> <vhdr_file>
#
# Example:
#   python3 scripts/03_import_filter_eeg.py P01 stroop JD_Stroop_22_05_2026.vhdr
#   python3 scripts/03_import_filter_eeg.py P02 nback  AB_nback_25_05_2026.vhdr
#
# Input  : data/raw/<vhdr_file>
# Output : data/processed/<participant_id>_<task>_filtered_raw.fif
# ─────────────────────────────────────────────────────────────────────────────

import sys
import mne
from pathlib import Path
from datetime import datetime

# ── Command-line arguments ────────────────────────────────────────────────────
if len(sys.argv) != 4:
    print("Usage: python3 scripts/03_import_filter_eeg.py "
          "<participant_id> <task> <vhdr_filename>")
    print("Example: python3 scripts/03_import_filter_eeg.py "
          "P01 stroop JD_Stroop_22_05_2026.vhdr")
    sys.exit(1)

participant_id = sys.argv[1]                          # e.g. P01
task           = sys.argv[2]                          # e.g. stroop or nback
vhdr_filename  = sys.argv[3]                          # e.g. JD_Stroop_22_05_2026.vhdr

print("=" * 60)
print(f"EEG IMPORT AND FILTERING")
print(f"Participant : {participant_id}")
print(f"Task        : {task}")
print(f"Input file  : {vhdr_filename}")
print(f"Started     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# ── Paths ─────────────────────────────────────────────────────────────────────
raw_file    = Path('data/raw') / vhdr_filename
output_file = Path(f'data/processed/{participant_id}_{task}_filtered_raw.fif')

if not raw_file.exists():
    print(f"ERROR: {raw_file} not found.")
    print(f"Make sure {vhdr_filename} (and matching .vmrk and .eeg) "
          f"are in data/raw/")
    sys.exit(1)

# Filter parameters
HIGHPASS_HZ = 0.1     # Hz — highpass cutoff
LOWPASS_HZ  = 40.0    # Hz — lowpass cutoff
NOTCH_HZ    = [50, 100]  # Hz — mains frequency + first harmonic

# ── Step 1: Load raw data ─────────────────────────────────────────────────────
print("\nSTEP 1: Loading raw data")

raw = mne.io.read_raw_brainvision(
    str(raw_file),
    preload=True,
    verbose=False
)

print(f"File       : {raw_file.name}")
print(f"Duration   : {raw.times[-1]:.1f}s ({raw.times[-1]/60:.1f} min)")
print(f"Channels   : {len(raw.ch_names)}")
print(f"Sample rate: {raw.info['sfreq']:.0f} Hz")
print(f"Channel names: {raw.ch_names}")

# ── Step 2: Set electrode montage ─────────────────────────────────────────────
# Assigns physical 3D positions to each electrode name
# Required for topographic maps and ICLabel in script 04
print("\nSTEP 2: Setting standard 10-20 montage")

montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage, match_case=False, on_missing='warn')
print("Montage set.")

# ── Step 3: Add Cz back as flat channel ───────────────────────────────────────
# Cz was the online reference during recording — not saved in the raw file
# because it had zero signal (it IS the reference point)
# Must be added back before average referencing so the average is computed
# correctly across all 32 electrodes, not just 31
print("\nSTEP 3: Adding Cz (online reference) as flat channel")

raw = mne.add_reference_channels(raw, ref_channels=['Cz'])
print(f"Cz added. Now have {len(raw.ch_names)} channels.")

# ── Step 4: Re-set montage after adding Cz ────────────────────────────────────
# IMPORTANT: must re-set montage AFTER add_reference_channels
# so that Cz gets a physical position in the montage
# Without this, Cz has no position and ICLabel will fail
print("\nSTEP 4: Re-setting montage so Cz has a position")

raw.set_montage(montage, match_case=False, on_missing='warn')
print("Montage re-set. All 32 channels now have positions.")

# ── Step 5: Bandpass filter ───────────────────────────────────────────────────
# Filter the continuous data BEFORE epoching
# Filtering after epoching introduces edge artefacts at epoch boundaries
# FIR filter with Hamming window — standard for ERP analysis
print(f"\nSTEP 5: Bandpass filter ({HIGHPASS_HZ}–{LOWPASS_HZ} Hz)")
print("This may take 30–60 seconds...")

raw.filter(
    l_freq=HIGHPASS_HZ,
    h_freq=LOWPASS_HZ,
    method='fir',
    fir_window='hamming',
    verbose=False
)
print("Bandpass filter applied.")

# ── Step 6: Notch filter ──────────────────────────────────────────────────────
# Removes 50 Hz UK mains interference and 100 Hz first harmonic
# Hardware notch was disabled during recording — applied here in software
print(f"\nSTEP 6: Notch filter ({NOTCH_HZ} Hz)")

raw.notch_filter(
    freqs=NOTCH_HZ,
    verbose=False
)
print("Notch filter applied.")

# ── Step 7: Event summary ─────────────────────────────────────────────────────
# Trigger codes sent by E-Prime are stored in the .vmrk file
# and automatically read alongside the .vhdr
print("\nSTEP 7: Event summary (trigger codes from E-Prime)")

events, event_id = mne.events_from_annotations(raw, verbose=False)
print(f"Total events: {len(events)}")
print(f"\nEvent types:")
for name, code in sorted(event_id.items(), key=lambda x: x[1]):
    count = (events[:, 2] == code).sum()
    print(f"  code {code:6d}  ({name:35s}): {count} events")

# ── Step 8: Save filtered data ────────────────────────────────────────────────
print("\nSTEP 8: Saving filtered data")

output_file.parent.mkdir(parents=True, exist_ok=True)
raw.save(str(output_file), overwrite=True, verbose=False)
print(f"Saved: {output_file}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"COMPLETE — {participant_id} {task}")
print(f"Finished : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)
print(f"\nFiltering applied:")
print(f"  Bandpass : {HIGHPASS_HZ}–{LOWPASS_HZ} Hz (FIR, Hamming)")
print(f"  Notch    : {NOTCH_HZ} Hz")
print(f"  Channels : {len(raw.ch_names)} (31 recorded + Cz re-added)")
print(f"\nNext: python3 scripts/04_ica.py {participant_id} {task}")
