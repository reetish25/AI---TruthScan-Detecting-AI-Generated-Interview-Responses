# AI-TruthScan: Detecting AI-Generated Interview Responses

## Project Description
AI-TruthScan is an innovative AI-driven tool designed to detect AI-generated interview responses using advanced speech and text analysis techniques. With the rise of AI chatbots like ChatGPT, Gemini, and Bard, candidates can generate polished answers in real-time, which can undermine the fairness of recruitment processes.

This tool leverages Natural Language Processing (NLP), Large Language Models (LLMs), and behavioral cues such as eye gaze tracking to classify responses as either **"Real (Human-Created)"** or **"Fake (AI-Generated)."**

### Key Objectives:
- Ensure fair recruitment practices.
- Enhance security and integrity in hiring.
- Provide recruiters with a reliable AI-assisted tool for assessing candidate authenticity.

## Features
- **AI-Generated Response Detection:** Identifies linguistic patterns, fluency, and coherence to detect AI-generated content.
- **Response Variability Analysis:** Detects human-like inconsistencies such as filler words and self-corrections.
- **Eye Gaze Tracking:** Analyzes unnatural reading behaviors.
- **Real-Time Feedback:** Provides instant classification and confidence scores.
- **User-Friendly Interface:** Built using Streamlit for seamless interaction.
- **Customizable Thresholds:** Allows users to adjust detection sensitivity.

## System Architecture
The AI-TruthScan system follows a modular architecture:
1. **Input Sources:** Audio (live recording or file upload), eye-tracking data.
2. **Preprocessing:** Speech-to-text conversion, filler word detection, data cleaning.
3. **AI Detection & Analysis:** NLP and LLM-based classification using Gemini 1.5 Pro.
4. **Decision & Output:** Confidence score calculation and result presentation.
5. **Deployment:** Web-based interface via Streamlit.

## Technical Stack
### Programming Languages:
- Python

### Frameworks & Libraries:
- **Streamlit** (Web Interface)
- **google-generativeai** (Gemini API integration)
- **speech_recognition** (Audio transcription)
- **re** (Regular expression parsing)

### APIs:
- **Google Speech Recognition API**
- **Gemini API**

### Tools:
- **GitHub** (Version Control)
- **Local development environment** (Cloud deployment optional)

## Prerequisites
Before setting up the project, ensure you have:
- Python 3.8+ installed.
- Git installed for cloning the repository.
- A stable internet connection (for API calls).
- A **Gemini API key** (sign up via Google Cloud).
- Microphone access for live recording (optional for file upload).

## Installation and Setup
### Step 1: Clone the Repository
```sh
git clone https://github.com/[your-username]/AI-TruthScan.git
cd AI-TruthScan
```

### Step 2: Install Dependencies
```sh
pip install -r requirements.txt
```

### Step 3: Set Up API Key
- Open `main.py` in a text editor.
- Replace `YOUR_API_KEY_HERE` with your actual Gemini API key:

```python
import google.generativeai as genai

genai.configure(api_key="your_actual_api_key")
```

## How to Use
### Run the Application:
```sh
streamlit run main.py
```

### Interface Overview:
- **Recording Section:** Click "Start Recording" or upload an audio file.
- **Analysis Section:** View transcription, probability score, and classification.
- **Settings:** Adjust threshold sensitivity using sliders.

### Example Workflow:
1. Record a response (e.g., *"Uh, I think my skills are, uh, pretty good for this role"*).
2. Wait for transcription and analysis.
3. Review output (e.g., *"Real (Human-Created), 85% confidence"*).

## Execution Guide
1. Clone the repository.
2. Install dependencies.
3. Set up API keys.
4. Run the application using:
   ```sh
   streamlit run main.py
   ```

## Results & Performance
### Performance Metrics:
- **Human Response Detection:** 85% accuracy.
- **AI-Generated Response Detection:** 90% accuracy.

### Sample Output:
| Response Type | Confidence Score |
|--------------|----------------|
| Real (Human-Created) | 85% |
| Fake (AI-Generated) | 90% |

## Screenshots
- **Home Screen:** Displays input options.
- **Voice Input Screen:** Shows recording or upload interface.
- **Processing:** Displays analysis in progress.
- **Results Screen:** Shows AI detection output.

## Demo
Watch the demo video for a walkthrough:
[Demo Video](https://drive.google.com/file/d/1eUJZOWN5XeO3HeA1ej7Afe2amaSt7ijZ/view?usp=drive_link)
