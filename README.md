# DermaDetect India 🇮🇳
> *AI-Powered Dermatology Assistant for Indian Skin Tones*

**DermaDetect India** is a full-stack AI healthcare application designed to provide preliminary skin condition analysis and localized medical advice. Unlike generic classifiers, it uses an **Agentic Workflow** (`LangGraph` + `Gemini`) to act as a personalized dermatologist assistant, offering advice tailored to Indian contexts (e.g., specific home remedies, available generic medicines).

## 🌟 Key Features

*   **🔍 AI Skin Diagnosis**: Utilizes a custom fine-tuned **EfficientNet-B0** (PyTorch) model to detect 12+ common skin conditions (Acne, Eczema, Melasma, etc.) with high accuracy.
*   **🤖 Dr. Derma Agent**: A stateful conversational agent built with **LangGraph** that explains the diagnosis, suggests Indian home remedies, and answers follow-up questions.
*   **🇮🇳 India-Centric**: Trained and prompted specifically for Indian skin tones and medical availability.
*   **🛡️ Safety First**: Automatically flags low-confidence predictions (<50%) as "Unclear/Normal" and encourages professional consultation.
*   **⚡ Modern Tech Stack**: Built with React (Vite) for a premium frontend experience and FastAPI for high-performance inference.

## 🛠️ Tech Stack

### Backend
*   **Framework**: FastAPI (Python)
*   **ML Core**: PyTorch (EfficientNet-B0)
*   **LLM Orchestration**: LangGraph, LangChain
*   **LLM Provider**: Google Gemini (`gemma-3-27b-it`)
*   **Deployment**: Docker (optimized for Hugging Face Spaces CPU tier)

### Frontend
*   **Framework**: React 19 (Vite)
*   **Styling**: Tailwind CSS v4, Framer Motion (Animations)
*   **Icons**: Lucide React
*   **HTTP Client**: Axios

## 🚀 Getting Started

### Prerequisites
*   Python 3.10+
*   Node.js 18+
*   Google Gemini API Key

### 1. Backend Setup
```bash
cd App/backend
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
echo "GEMINI_API_KEY=your_api_key_here" > .env

# Run Server
python main.py
```
*Server runs at `http://localhost:8000`*

### 2. Frontend Setup
```bash
cd App/frontend
npm install

# Run Development Server
npm run dev
```
*App runs at `http://localhost:5173`*

## 🧠 Model Details
The model was fine-tuned on a dataset of 12,000+ dermatological images.
- **Base Model**: EfficientNet-B0 (Pre-trained on ImageNet)
- **Classes**: Acne, Eczema, Atopic Dermatitis, Tinea (Ringworm), Vitiligo, etc.
- **Accuracy**: ~88% on validation set (Top-1).

## ⚠️ Disclaimer
**This tool is for educational and informational purposes only.** It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a certified dermatologist for any skin concerns.

## 📄 License
MIT License - feel free to use this for your own learning!

