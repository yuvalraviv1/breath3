# Belly Motion Audio Plan

- Goal: single-page HTML/JS that runs on iPhone Safari, reads belly motion via `DeviceMotionEvent`, and produces a continuous soft sine tone with pitch/volume driven by movement (live feedback, no external assets).

## Architecture

### Git/Deployment
- Initialize Git repository locally
- Add remote: https://github.com/yuvalraviv1/breath3
- Create `index.html` in root for GitHub Pages accessibility
- Push to GitHub and enable GitHub Pages to serve from main branch
- Access via: https://yuvalraviv1.github.io/breath3

### Permissions
- On load, request `DeviceMotionEvent.requestPermission()`
- Must be called in user gesture handler (tap) to resume AudioContext per iOS requirements
- Start streaming motion at ~60 Hz after permission granted

### Sensor Input & Axis Selection
- Use `accelerationIncludingGravity` from DeviceMotionEvent
- **Primary axis**: Z-axis (perpendicular to screen - typical belly breathing orientation when phone laid on torso)
- **Fallback**: Auto-detect strongest signal across X/Y/Z during first 2 seconds
- Manual axis selector (X/Y/Z toggle) + invert toggle for reversed direction
- Display current axis name and raw value in UI

### Calibration
- **Auto-calibration on start**: capture baseline over first 3 seconds (60 samples)
  - Calculate mean of selected axis during calibration window
  - Store as zero-point reference
- **Deviation calculation**: all subsequent readings are relative to baseline (current - baseline)
- **Recalibrate button**: allows user to reset baseline if they shift position
- Display calibration status ("Calibrating... 2s remaining" / "Ready")

### Signal Processing
- **Magnitude calculation**: Use deviation from baseline (current_value - baseline_value)
  - Positive deviation = belly rising (inhale), negative = belly falling (exhale)
  - Absolute value determines volume intensity
- **Filtering**: Exponential moving average (α = 0.15) to smooth jitter for both pitch and volume signals
  - `smoothed = α × new_value + (1 - α) × smoothed`

### Mapping
- **Frequency (pitch)**: Map smoothed deviation to 180–520 Hz range
  - Positive deviation (inhale) → higher pitch
  - Negative deviation (exhale) → lower pitch
  - Center (baseline) → ~350 Hz (middle of range)
- **Gain (volume)**: Map absolute magnitude of deviation to 0.15–0.6 gain range
  - Larger movement → louder
  - Still/minimal movement → softer but audible (0.15 floor prevents silence)
  - Clamp to avoid harshness above 0.6
- Make ranges adjustable via UI sliders if initial sensitivity feels off

### Audio Engine
- Web Audio API with one sine `OscillatorNode` feeding a `GainNode` to destination
- Start oscillator once (before user interaction) and update frequency/gain params per frame
- AudioContext must be resumed in user gesture handler (same tap as permission request)

### UI/UX
- Minimal full-screen dark page (easy on eyes for meditation/breathing practice)
- **Main controls**:
  - Large "Start" button (requests permissions + starts audio)
  - Recalibrate button
  - Axis selector (X/Y/Z) + invert checkbox
- **Live readouts**:
  - Calibration status
  - Current axis and raw value
  - Smoothed deviation from baseline
  - Current frequency (Hz) and gain
- **Optional adjustments** (collapsed by default):
  - Frequency range sliders (min/max Hz)
  - Volume range sliders (min/max gain)
  - Smoothing factor (α)

### Error Handling
- Detect missing DeviceMotionEvent support → "Motion sensors not available"
- Permission denied → "Motion access denied. Please enable in Settings."
- No AudioContext support → "Web Audio not supported"

## Implementation Steps

1. **Git setup**
   - Initialize repo: `git init`
   - Add remote: `git remote add origin https://github.com/yuvalraviv1/breath3.git`
   - Create .gitignore for .DS_Store, etc.

2. **Build HTML shell** (`index.html`)
   - Inline CSS/JS for offline capability
   - Dark theme, mobile-optimized viewport
   - All UI controls and readout placeholders

3. **Initialize Web Audio**
   - Create AudioContext, OscillatorNode, GainNode
   - Connect nodes: oscillator → gain → destination
   - Start oscillator (muted initially with gain = 0)

4. **Wire Start button**
   - Request `DeviceMotionEvent.requestPermission()`
   - Resume AudioContext
   - Start calibration process (3-second window)
   - Begin devicemotion event listening

5. **Implement calibration**
   - Collect axis samples for 3 seconds
   - Calculate baseline mean for selected axis
   - Update UI status
   - After calibration, enable main processing loop

6. **Implement devicemotion handler**
   - Read selected axis from accelerationIncludingGravity
   - Calculate deviation from baseline
   - Apply exponential smoothing
   - Map deviation → frequency (180-520 Hz)
   - Map absolute deviation → gain (0.15-0.6)
   - Update oscillator.frequency.value and gainNode.gain.value
   - Update UI readouts

7. **Add controls**
   - Axis selector (X/Y/Z radio buttons)
   - Invert toggle checkbox
   - Recalibrate button → reset baseline
   - Range sliders (collapsed/expandable)

8. **Test on iPhone**
   - Push to GitHub: `git push -u origin main`
   - Enable GitHub Pages (Settings → Pages → Source: main branch)
   - Access https://yuvalraviv1.github.io/breath3 on iPhone
   - Verify axis selection matches belly movement
   - Adjust default ranges if sensitivity is off
   - Test recalibration after position change

## Critical Files
- `index.html` - Single-file application (HTML + inline CSS/JS)
- `.gitignore` - Exclude OS artifacts
