# Facial Emotion Recognition

AI-powered facial expression analysis using a MobileNetV2-based deep learning model, wrapped in a premium Streamlit interface.

## Features

- Upload a face image (JPG / JPEG / PNG) and get an instant emotion prediction
- 7-class classification: Happy, Sad, Angry, Fear, Neutral, Disgust, Surprise
- Confidence score with a per-class probability breakdown
- Image metadata (dimensions, file size, format)
- Model info panel and graceful error handling — the app never crashes on a bad file or a missing model

## Project Structure

```
.
├── app.py                       # Streamlit frontend
├── emotion_recognition.keras    # Trained MobileNetV2 model
├── requirements.txt             # Python dependencies
└── README.md
```

`app.py` and `emotion_recognition.keras` must live in the **same folder** — the app loads the model from a relative path.

## Setup

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

4. Open the URL Streamlit prints (usually `http://localhost:8501`).

## Model Details

| Property     | Value        |
|--------------|--------------|
| Architecture | MobileNetV2  |
| Input size   | 160 × 160 × 3|
| Classes      | 7            |
| Framework    | TensorFlow / Keras |

## Notes

- The model has preprocessing (rescaling) built directly into its graph, so the app feeds it raw resized pixel values — no manual normalization is needed.
- Class order assumes the standard alphabetical mapping (`Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise`), matching Keras' default `class_indices` from `ImageDataGenerator` / `image_dataset_from_directory`. If your training data used a different folder order, update the `EMOTION_CLASSES` list near the top of `app.py`.

## Tech Stack

TensorFlow · Streamlit · Python · Pillow · NumPy
