# Breath3 - PCA-Based Motion Audio

A single-page web application that transforms body movement into sound using Principal Component Analysis (PCA) and Web Audio API. Optimized for iPhone Safari and breathing practice.

## Features

- **PCA Motion Analysis**: Automatically finds dominant movement direction from 3-axis accelerometer data
- **Adaptive Audio**: Dynamic frequency and volume mapping based on real-time movement patterns  
- **Stereo Panning**: Optional spatial audio based on breathing phase
- **Calibration System**: 10-second PCA calibration with audio feedback
- **Advanced Settings**: Comprehensive parameter control via modal interface
- **Mobile Optimized**: Full-screen dark interface designed for meditation/breathing practice

## How It Works

1. **Calibration**: Collect 3D motion data for 10 seconds while breathing naturally
2. **PCA Analysis**: Compute principal component to find dominant movement direction
3. **Real-time Processing**: Project motion onto PCA and map to audio parameters
4. **Adaptive Range**: Sliding window tracks signal statistics for dynamic response

## Usage

1. Open [https://yuvalraviv1.github.io/breath3](https://yuvalraviv1.github.io/breath3) in iPhone Safari
2. Tap "Start" and grant motion sensor permissions
3. Follow calibration ticks (10 seconds) - breathe naturally
4. Listen to audio feedback as you breathe
5. Use "Stereo" toggle for spatial audio
6. Adjust settings via ⚙️ Advanced Settings

## Technical Details

- **Motion Input**: DeviceMotionEvent.accelerationIncludingGravity (X, Y, Z axes)
- **PCA Algorithm**: Power iteration on 3x3 covariance matrix
- **Audio Engine**: Web Audio API with sine oscillator
- **Adaptive Window**: Configurable 3-30 second sliding window
- **Frequency Range**: 50-700 Hz (default 50-420 Hz)
- **Volume Range**: 0.15-1.0 gain (default 0.15-0.6)

## Browser Support

- **iOS Safari 13+**: Full support with motion permissions
- **Android Chrome**: Motion sensors available without permissions
- **Desktop**: Limited motion sensor support

## Development

Single-file application (`index.html`) with:
- Inline HTML structure and semantic markup
- Embedded CSS with mobile-first responsive design
- Vanilla JavaScript with no external dependencies
- PCA implementation using power iteration algorithm

## Deployment

Deployed via GitHub Pages from the `main` branch.
Access: https://yuvalraviv1.github.io/breath3

## License

Private repository - all rights reserved.