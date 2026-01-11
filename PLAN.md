# Breath3 - PCA-Based Motion Audio

- Goal: single-page HTML/JS that runs on iPhone Safari, reads 3-axis motion via `DeviceMotionEvent`, applies Principal Component Analysis (PCA) to find dominant movement direction, and produces a continuous soft sine tone with pitch/volume driven by the PCA signal (live feedback, no external assets).

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
- **Wake Lock API**: Request screen wake lock to prevent phone from sleeping during use
  - Note: App will not work in lock mode (iOS restrictions on DeviceMotion and Web Audio when locked)
  - Wake Lock keeps screen on during breathing sessions

### Sensor Input & PCA Analysis
- Use `accelerationIncludingGravity` from DeviceMotionEvent (3-axis: X, Y, Z)
- **PCA Calibration**: During 10-second calibration, collect 3D samples and compute Principal Component Analysis
- **Dominant Direction**: PCA automatically finds the direction of maximum variance (dominant movement pattern)
- **Projection**: Real-time motion is projected onto the principal component for a single signal
- **Adaptive Range**: Sliding window (3-30s) tracks signal statistics for dynamic range adjustment
- Display raw X/Y/Z values, PCA components, and axis contribution weights in UI

### Calibration
- **PCA Calibration**: 10-second calibration with 3D sample collection (~600 samples)
  - First 3 seconds: stabilization (no collection)
  - Next 7 seconds: collect 3-axis acceleration data
  - **Tick sound feedback**: plays 800Hz tick sound every second during calibration
- **PCA Computation**: Calculate covariance matrix and dominant eigenvector (principal component)
- **Baseline Signal**: Project calibration samples onto PCA and compute mean as baseline
- **Recalibrate button**: Allows user to reset PCA and baseline if they shift position
- Display calibration status ("Calibrating... Xs remaining" / "Ready")

### Signal Processing
- **PCA Projection**: Real-time 3D acceleration projected onto principal component
- **Adaptive Range**: Sliding window (configurable 3-30s) tracks min/max/median for dynamic range
- **Deviation calculation**: Signal deviation from window median (current - median)
- **Filtering**: Exponential moving average (α = 0.1, configurable) to smooth jitter
  - `smoothed = α × new_value + (1 - α) × smoothed`
- **Sensitivity**: Multiplier (0.1-10x) for fine-tuning response to subtle movements

### Mapping
- **Frequency (pitch)**: Map smoothed deviation to configurable frequency range (50-700 Hz)
  - Default: 50-420 Hz (center ~235 Hz)
  - Positive deviation → higher pitch (configurable inversion)
  - Negative deviation → lower pitch
  - **Adaptive**: Uses window median as center, window range for normalization
- **Gain (volume)**: Map absolute deviation to configurable gain range (0.15-1.0)
  - Default: 0.15-0.6 (prevents silence, avoids harshness)
  - Larger movement → louder
  - Still/minimal movement → softer but audible
  - **Adaptive**: Normalized to window range for consistent response
- **Stereo Panning**: Optional stereo based on breathing phase (configurable width 0-1)
- All ranges adjustable via UI sliders in Advanced Settings modal

### Audio Engine
- Web Audio API with one sine `OscillatorNode` feeding a `GainNode` to destination
- Start oscillator once (before user interaction) and update frequency/gain params per frame
- AudioContext must be resumed in user gesture handler (same tap as permission request)

### UI/UX
- Minimal full-screen dark page (easy on eyes for meditation/breathing practice)
- **Main controls**:
  - Large "Start" button (requests permissions + starts audio)
  - Recalibrate button
  - Stereo toggle (On/Off)
  - Activity indicator bar
- **PCA Information**:
  - Principal component vector display
  - Axis contribution weights with visual bars
  - Raw X/Y/Z accelerometer values
- **Live readouts**:
  - Calibration status
  - PCA signal value
  - Smoothed deviation from median
  - Current frequency (Hz) and gain
  - Stereo pan position (C/L/R/LL/RR)
  - Window statistics (min/max/median)
- **Advanced Settings** (modal):
  - Frequency range sliders (min/max Hz)
  - Volume range sliders (min/max gain)
  - Smoothing factor (α)
  - Sensitivity multiplier
  - Window size (3-30s)
  - Frequency direction inversion
  - Stereo width control

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
   - Request Wake Lock to keep screen on
   - Start calibration process (10-second window)
   - Begin devicemotion event listening

5. **Implement PCA calibration**
    - Collect 3D samples for 10 seconds (~600 samples)
    - Play tick sound (800Hz) every second for feedback
    - Compute PCA: mean vector, covariance matrix, dominant eigenvector
    - Calculate baseline signal from PCA projection
    - Update UI with PCA components and axis weights
    - After calibration, enable main processing loop

6. **Implement devicemotion handler**
    - Read X/Y/Z from accelerationIncludingGravity
    - Project onto principal component for single signal
    - Update sliding window and calculate statistics
    - Calculate deviation from window median
    - Apply exponential smoothing
    - Map deviation → frequency (adaptive range based on window)
    - Map absolute deviation → gain (adaptive range)
    - Update oscillator.frequency.value and gainNode.gain.value
    - Handle stereo panning if enabled
    - Update all UI readouts

7. **Add controls**
    - Recalibrate button → reset PCA and baseline
    - Stereo toggle → enable/disable stereo panning
    - Advanced Settings modal with all parameter sliders

8. **Test on iPhone**
   - Push to GitHub: `git push -u origin main`
   - Enable GitHub Pages (Settings → Pages → Source: main branch)
   - Access https://yuvalraviv1.github.io/breath3 on iPhone
   - Verify axis selection matches belly movement
   - Adjust default ranges if sensitivity is off
   - Test recalibration after position change

## Technical Implementation

### PCA Algorithm
- **Power Iteration**: 50 iterations to find dominant eigenvector of 3x3 covariance matrix
- **Real-time Projection**: `signal = pc[0]*x + pc[1]*y + pc[2]*z`
- **Adaptive Normalization**: Dynamic range based on sliding window statistics

### Audio Engine
- **Web Audio API**: Single sine oscillator → gain node → stereo panner → destination
- **Startup Sound**: Brief 440Hz tone to establish iOS audio routing
- **Wake Lock**: Screen wake lock to prevent sleep during sessions

### Performance Optimizations
- **Efficient PCA**: Power iteration vs full eigenvalue decomposition
- **Sliding Window**: O(1) updates with circular buffer behavior
- **Exponential Smoothing**: Minimal computational overhead

## Critical Files
- `index.html` - Single-file application (HTML + inline CSS/JS, 1186 lines)
- `.gitignore` - Exclude OS artifacts
- `PLAN.md` - This documentation file
