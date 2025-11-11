# Plant Health Detector

An AI-powered web application for detecting plant diseases and providing treatment recommendations.

## Features

- ��️ **Image Upload**: Drag & drop or click to upload plant images
- 🤖 **AI Analysis**: Machine learning model analyzes plant health
- 🩺 **Disease Detection**: Identifies common plant diseases
- 💊 **Treatment Plans**: Provides detailed cure and prevention steps
- 📱 **Responsive Design**: Works on desktop and mobile devices
- ⚡ **Real-time Results**: Fast analysis with confidence scores

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Modern web browser
- Your trained ML model file

### Installation

1. **Clone or download this repository**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your ML model**:
   - Place your trained model file in the project directory
   - Update the model path in `app.py` (line 18)
   - Ensure your model expects 224x224 RGB images

4. **Run the application**:
   ```bash
   python app.py
   ```

5. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```

## Usage

1. **Upload Image**: Click the upload area or drag & drop a plant image
2. **Analyze**: Click "Analyze Plant Health" to process the image
3. **View Results**: See disease detection results, confidence scores, and treatment recommendations

## Customization

### Adding New Diseases

1. Update `DISEASE_CLASSES` in `app.py`
2. Add disease information to `DISEASE_INFO` dictionary
3. Retrain your model with the new classes

### Modifying the UI

- Edit `styles.css` for visual changes
- Modify `script.js` for functionality changes
- Update `index.html` for structural changes

### Model Integration

The current setup includes a simulation mode. To use your actual model:

1. Uncomment the model loading line in `app.py`
2. Update the preprocessing function to match your model's input requirements
3. Ensure your model outputs match the expected format

## File Structure

```
plant-health-detector/
├── index.html          # Main HTML file
├── styles.css          # CSS styling
├── script.js           # Frontend JavaScript
├── app.py              # Flask backend
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Technologies Used

- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Backend**: Python Flask
- **ML Framework**: TensorFlow (your model)
- **Styling**: Custom CSS with modern design principles

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve this application.

## License

This project is open source and available under the MIT License.

## Support

If you encounter any issues or have questions, please check the troubleshooting section or create an issue in the repository.
