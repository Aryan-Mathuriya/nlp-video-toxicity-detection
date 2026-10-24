# Video Toxicity Detector

**An end-to-end NLP application for detecting toxic speech in video content using Speech Recognition, TF-IDF, LightGBM, and Gradio.**

## Overview

Video Toxicity Detector is a Python-based machine learning application that analyzes spoken content in uploaded videos to identify potentially toxic language.

The application extracts audio from a video, converts speech into text using Google's Speech Recognition service, preprocesses the transcript using Natural Language Processing (NLP) techniques, and predicts toxicity across six categories using a pre-trained multi-label classification model.

An interactive Gradio interface allows users to upload videos, view the generated transcript, and examine toxicity predictions.

## Key Features

- **Video Upload:** Accepts video files through an interactive Gradio interface.
- **Audio Extraction:** Extracts audio from uploaded videos using MoviePy.
- **Speech-to-Text:** Converts spoken audio into text using Google's Speech Recognition service.
- **Text Preprocessing:** Applies lowercasing, contraction expansion, tokenization, stopword removal, and stemming.
- **TF-IDF Vectorization:** Converts processed transcripts into numerical features using a pre-trained TF-IDF vectorizer.
- **Multi-label Toxicity Classification:** Uses six LightGBM classifiers within a One-vs-Rest classification framework.
- **Toxicity Probability Estimation:** Estimates probabilities for six toxicity categories.
- **Prediction Visualization:** Displays the transcript and detected toxicity categories through Gradio.

## Toxicity Categories

The application evaluates six toxicity labels:

| Category | Description |
|---|---|
| Toxic | Generally offensive, abusive, or harmful language |
| Severe Toxic | Particularly aggressive or severely abusive language |
| Obscene | Vulgar or sexually explicit language |
| Threat | Language expressing threats of harm |
| Insult | Language containing personal insults |
| Identity Hate | Hateful language targeting identity-related characteristics |

The model uses a probability threshold of **0.50** to determine which categories are reported.

A transcript can be assigned multiple toxicity labels simultaneously.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Machine Learning | LightGBM, Scikit-learn |
| NLP Preprocessing | NLTK |
| Feature Extraction | TF-IDF |
| Speech Recognition | SpeechRecognition (Google) |
| Video Processing | MoviePy |
| User Interface | Gradio |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Model Serialization | Pickle |

## System Architecture

```text
              Video Upload
                   |
                   v
        Audio Extraction
              MoviePy
                   |
                   v
       Speech-to-Text
     Google Speech Recognition
                   |
                   v
         Text Preprocessing
               NLTK
                   |
                   v
        TF-IDF Vectorization
                   |
                   v
       One-vs-Rest Classifier
         6 LightGBM Models
                   |
                   v
      Toxicity Probabilities
                   |
                   v
         Threshold Filtering
              > 0.50
                   |
                   v
        Gradio Output
    Transcript + Toxicity Labels
```

## Machine Learning Approach

### 1. Text Preprocessing

The transcribed text undergoes several NLP preprocessing steps:

1. Convert text to lowercase.
2. Expand common English contractions.
3. Tokenize text into individual words.
4. Remove special characters and punctuation.
5. Remove English stopwords.
6. Apply Porter stemming.
7. Reconstruct the processed text.

### 2. Feature Extraction

The cleaned transcript is transformed using a saved TF-IDF vectorizer.

The provided vectorizer produces **65,671-dimensional feature vectors**, representing the vocabulary learned during training.

The same saved vectorizer is reused during inference to maintain compatibility with the trained model.

### 3. Toxicity Classification

The application uses a saved `OneVsRestClassifier` containing six LightGBM estimators.

Each estimator predicts the probability of a particular toxicity label.

Unlike traditional single-label classification, this approach supports multiple simultaneous predictions.

### 4. Prediction Threshold

For each toxicity category:

- Probability greater than 0.50: Category reported as detected.
- Probability less than or equal to 0.50: Category not reported.

If no category exceeds the threshold, the application returns a non-toxic prediction.

The predicted probabilities represent model estimates, not guaranteed measures of toxicity or correctness.

## Project Structure

The core files used in this project are:

```text
video-toxicity-detector/
│
├── app.py
├── Toxic_lightgbm_model.pkl
├── Toxic_tfidf_vectorizer.pkl
├── requirements.txt
├── README.md
│
└── screenshots/
```

**Note:** Rename the uploaded application script to `app.py` and the serialized model files to the names shown above. The current notebook/script uses Colab-specific paths and must be updated to use these local filenames.

## Installation and Setup

### Prerequisites

- Python
- pip
- FFmpeg, if required by the local audio/video setup
- Internet connectivity for Google's speech recognition service

### 1. Clone the repository

```bash
git clone https://github.com/Aryan-Mathuriya/nlp-video-toxicity-detection.git
cd nlp-video-toxicity-detection
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate the environment.

**Windows:**

```bash
venv\Scripts\activate
```

**Linux/macOS:**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Ensure that the required packages include:

```text
numpy
pandas
scikit-learn
lightgbm
nltk
SpeechRecognition
moviepy
gradio
matplotlib
seaborn
```

The serialized model was created using scikit-learn 1.6.1. For reliable deployment, use compatible library versions and test the model-loading process before publishing the application.

### 4. Download NLTK resources

The application requires NLTK tokenization and stopword resources.

```python
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
```

### 5. Configure model paths

Update the application to load the saved model and vectorizer from the project directory instead of Google Colab's `/content/` directory.

The current code also uses a fixed temporary WAV file path. This should be replaced with a temporary-file approach for reliable local execution.

### 6. Run the application

```bash
python app.py
```

Open the local Gradio URL displayed in the terminal.

**Important:** The uploaded application script was originally exported from Google Colab and contains notebook-specific installation commands. These must be removed or adapted before running it as a standalone Python script.

## Usage

1. Launch the Gradio application.
2. Upload a video containing spoken English.
3. Wait for audio extraction and transcription.
4. Review the generated transcript.
5. Examine the detected toxicity categories and their reported probabilities.

The processing time depends on video duration, audio quality, speech recognition performance, and system resources.

## Model Evaluation

The supplied application and saved model demonstrate a functioning inference pipeline.

However, the original training notebook, dataset split, and evaluation results have not been provided.

Therefore, the following metrics are not currently verified:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Per-label classification performance

These metrics should be reported only after evaluating the model on a suitable held-out dataset.

A single successful prediction does not establish the model's overall accuracy.

## Current Limitations

- **Speech Recognition Dependency:** Transcription relies on Google's speech recognition service and requires internet connectivity.
- **Language Support:** The current preprocessing pipeline is designed primarily for English.
- **Audio Quality:** Background noise, overlapping speakers, and unclear speech may affect transcription accuracy.
- **Context Sensitivity:** TF-IDF-based classification may struggle with sarcasm, contextual meaning, and nuanced expressions.
- **Fixed Threshold:** A uniform 0.50 threshold may not be optimal for every toxicity category.
- **Video Scope:** The model analyzes transcribed speech, not visual content, gestures, or on-screen text.
- **Deployment Compatibility:** The original code requires path handling and dependency adjustments for reliable standalone deployment.

## Future Improvements

- Evaluate the model on a held-out test dataset.
- Add per-label precision, recall, F1-score, and ROC-AUC.
- Introduce category-specific probability thresholds.
- Support multilingual speech recognition and toxicity classification.
- Add timestamp-based toxicity detection.
- Implement speaker diarization.
- Explore transformer-based models such as BERT for contextual text classification.
- Improve exception handling for unsupported files and transcription failures.
- Add secure temporary-file handling and automatic cleanup.
- Deploy the application using Docker or Hugging Face Spaces.

## Responsible Use

This application is intended as a demonstration of machine learning techniques for speech-based content moderation.

Predictions may contain false positives or false negatives. They should not be treated as definitive judgments about a speaker's intentions or used as the sole basis for consequential moderation decisions.

Human review is recommended for sensitive or high-impact applications.

## Acknowledgments

The application notebook was originally authored by **Anmol Talwar**, Founder and Trainer at Talent Catalyst AI.

The project uses that instructional material as its foundation. Any independently developed modifications, enhancements, or deployment work should be documented separately.

## Project Maintainer

**Aryan Mathuriya**

B.Tech — Artificial Intelligence and Data Science

GitHub: https://github.com/Aryan-Mathuriya

Areas of Interest: Machine Learning, Natural Language Processing, Data Science, and AI Applications.

---

**If you find this project useful, consider starring the repository.**
