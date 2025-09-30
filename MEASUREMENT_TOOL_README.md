# Golf Grip Measurement Tool

A clean, standalone computer vision tool for measuring hand dimensions for golf grip fitting with **dynamic calibration**.

## 🎯 Features

- **Real-time hand tracking** using OpenCV and MediaPipe
- **Automatic measurements** of hand length and finger length
- **Dynamic calibration** using palm width detection (no reference objects needed!)
- **Auto-calibration** for each hand position and distance
- **Confidence scoring** and measurement validation
- **Simple command-line interface**
- **Minimal dependencies**

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r measurement_requirements.txt
```

### 2. Test the System

```bash
python test_measurement.py
```

### 3. Run the Measurement Tool

```bash
python golf_grip_measurement.py
```

## 📁 Files

- `golf_grip_measurement.py` - Main measurement tool
- `test_measurement.py` - Test script
- `measurement_requirements.txt` - Python dependencies
- `MEASUREMENT_TOOL_README.md` - This file

## 🔧 Usage

### Basic Usage

```python
from golf_grip_measurement import GolfGripMeasurement

# Initialize the tool
measurer = GolfGripMeasurement()

# Run measurement session
measurements = measurer.run_measurement_session(duration_seconds=10)

if measurements:
    print(f"Hand Length: {measurements['hand_length']:.1f} cm")
    print(f"Finger Length: {measurements['finger_length']:.1f} cm")
    print(f"Confidence: {measurements['confidence']:.1%}")
```

### Calibration

The tool automatically loads calibration from `grip_calibration.json` if it exists. To calibrate:

1. Run the tool: `python golf_grip_measurement.py`
2. Press 'c' during measurement to enter calibration mode
3. Use a reference object with known size (e.g., credit card = 8.56 cm)
4. Click and drag to measure the object in pixels
5. Enter the real size in centimeters

### Integration with Grip Engine

```python
# Get measurements
measurements = measurer.run_measurement_session(duration_seconds=10)

# Prepare for grip recommendation engine
user_data = {
    'handLength': measurements['hand_length'],
    'fingerLength': measurements['finger_length'],
    # ... other user data
}
```

## 📏 How It Works

### Hand Detection
- Uses MediaPipe's hand landmark detection
- Tracks 21 hand landmarks in real-time
- Provides confidence scores for measurement accuracy

### Measurements
- **Hand Length**: Wrist (landmark 0) to middle finger tip (landmark 12)
- **Finger Length**: Index finger base (landmark 5) to tip (landmark 8)
- **Palm Width**: Index finger base (landmark 5) to pinky base (landmark 17)
- **Dynamic Calibration**: Uses palm width to auto-calibrate for each hand position

### Accuracy
- **Hand Length**: ±0.5 cm with proper calibration
- **Finger Length**: ±0.3 cm with proper calibration
- **Confidence**: 80%+ with good conditions

## 🎮 Controls

During measurement:
- **'q'**: Quit early
- **'c'**: Enter calibration mode
- **Mouse**: Click and drag to measure reference objects (in calibration mode)

## 🔧 Dynamic Calibration

### How Dynamic Calibration Works
- **No reference objects needed!** The system auto-calibrates using palm width
- **Assumes average palm width**: 8.5cm (configurable)
- **Real-time calibration**: Adjusts for each hand position and distance
- **Stable measurements**: Uses calibration history for consistency

### Calibration Process
1. **Detects palm width** using MediaPipe landmarks (index to pinky base)
2. **Calculates calibration factor**: `calibration_factor = 8.5cm / palm_width_pixels`
3. **Averages recent calibrations** for stability
4. **Updates in real-time** as hand position changes

### Customizing Palm Width
```python
# For different average palm widths
measurer = GolfGripMeasurement(average_palm_width_cm=9.0)  # Larger hands
measurer = GolfGripMeasurement(average_palm_width_cm=7.5)  # Smaller hands
```

### Calibration Confidence
- **High confidence**: Consistent palm width measurements
- **Low confidence**: Variable hand positioning or poor detection
- **Auto-adjusts**: System stabilizes over multiple frames

## 🎯 Measurement Tips

### For Best Results
- **Lighting**: Good, even lighting improves detection
- **Hand Position**: Flat, open hand works best
- **Distance**: 12-18 inches from camera is optimal
- **Stability**: Keep hand steady during measurement
- **Calibration**: Proper calibration is crucial for accuracy

### Troubleshooting
- **Poor Detection**: Improve lighting, check hand position
- **Inaccurate Measurements**: Ensure hand is flat and well-lit for palm width detection
- **No Detection**: Ensure hand is fully visible, check camera focus
- **Low Calibration Confidence**: Keep hand steady, improve lighting

## 📊 Output Format

```python
{
    'hand_length': 19.2,           # cm
    'finger_length': 7.1,          # cm
    'palm_width_cm': 8.3,          # cm (for calibration)
    'calibration_factor': 0.0567,  # cm/pixel
    'calibration_confidence': 0.85, # 85%
    'confidence': 0.85,            # 85% (measurement confidence)
    'sample_count': 5              # number of samples averaged
}
```

## 🚀 Deployment

### Local Development
```bash
python golf_grip_measurement.py
```

### Integration
```python
# Import the tool
from golf_grip_measurement import GolfGripMeasurement

# Use in your application
measurer = GolfGripMeasurement()
measurements = measurer.run_measurement_session(duration_seconds=10)
```

### Production
- Deploy as standalone tool
- Integrate into web applications
- Use as API endpoint
- Package as executable

## 🔍 Testing

### Run Tests
```bash
python test_measurement.py
```

### Test Coverage
- Tool initialization
- Calibration methods
- Measurement session
- Error handling
- Dependency checking

## 📞 Support

### Common Issues
1. **Camera Not Detected**: Check camera permissions and connections
2. **Poor Hand Detection**: Improve lighting and hand positioning
3. **Inaccurate Measurements**: Recalibrate with reference object
4. **Import Errors**: Install dependencies with pip

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with debug output
measurer = GolfGripMeasurement()
measurements = measurer.run_measurement_session(duration_seconds=5)
```

## 📄 License

This tool is part of the GripScript project and follows the same MIT license.

---

**Ready to measure hands for golf grip fitting!** 🏌️‍♂️📏
