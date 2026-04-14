<div align="center">

# 🔬 DermaGlass India

### *AI-Powered Dermatology Assistant for Indian Skin Tones*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic-6B48FF?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

**Upload a skin image → Get an instant AI diagnosis → Chat with Dr. Derma for personalized advice**

[Features](#-features) • [Architecture](#️-architecture) • [Getting Started](#-getting-started) • [Model Details](#-model-details) • [Deployment](#-deployment) • [Disclaimer](#️-disclaimer)

</div>

---

## 📖 Overview

**DermaGlass India** is a full-stack AI healthcare application that provides preliminary skin condition analysis and localized medical advice tailored for Indian patients.

Unlike generic skin classifiers, DermaGlass uses an **Agentic Workflow** powered by `LangGraph` + `OpenAI GPT` to function as a personalized dermatologist assistant — offering advice specific to Indian contexts such as local home remedies and available generic medicines.

> **Why India-centric?** Most dermatology AI tools are trained on Western datasets and skin tones. DermaGlass is specifically tuned for South Asian skin phenotypes and the Indian healthcare ecosystem.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **AI Skin Diagnosis** | Custom fine-tuned **EfficientNet-B0** (PyTorch) detects 12+ skin conditions |
| 🤖 **Dr. Derma Agent** | Stateful conversational AI built with **LangGraph** for in-depth follow-up Q&A |
| 🇮🇳 **India-Centric Advice** | Recommendations include local home remedies and generic Indian medicines |
| 🛡️ **Safety-First Design** | Low-confidence predictions (<50%) are flagged as "Unclear/Normal" |
| 💬 **Markdown-Rich Responses** | Reports structured with clear sections: Diagnosis, Care, Remedies, Precautions |
| ⚡ **Modern Full-Stack** | React 19 (Vite) frontend + FastAPI async backend |
| 🐳 **Dockerized** | Production-ready Docker deployment optimized for Hugging Face Spaces |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User (Browser)                              │
│                    React 19 + Vite Frontend                         │
│           Tailwind CSS v4 · Framer Motion · Lucide React            │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ REST API (Axios)
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                                 │
│                                                                      │
│  ┌──────────────────┐      ┌──────────────────────────────────── ──┐ │
│  │  EfficientNet-B0  │     │         LangGraph Agentic Workflow    │ │
│  │   (PyTorch)       │──>  │                                       │ │
│  │  12+ Classes      │     │  ┌──────────────────────────────────┐ │ │
│  │  ~88% Accuracy    │     │  │  Router Node                     │ │ │
│  └──────────────────┘      │  │  ┌─────────────┐  ┌───────────┐  │ │ │
│                            │  │  │  Analysis   │  │ Assistant │  │ │ │
│  POST /predict             │  │  │  Reporter   │  │  (Chat)   │  │ │ │
│  POST /chat                │  │  └─────────────┘  └───────────┘  │ │ │
│                            │  └──────────────────────────────────┘ │ │
│                            └──────────────┬────────────────────────┘ │
│                                           │ OpenAI GPT API           │
└───────────────────────────────────────────|──────────────────────────┘
                                            ▼
                               ┌───────────────────┐
                               │   OpenAI / GPT    │
                               │   (LLM Provider)  │
                               └───────────────────┘
```

**Agentic Flow:**
1. User uploads image → `/predict` endpoint
2. `EfficientNet-B0` runs inference → returns disease class + confidence
3. `LangGraph` routes to **Analysis Reporter** node → generates structured Markdown report via GPT
4. User asks follow-up → `/chat` endpoint → **Assistant** node with persistent memory handles the conversation

---

## 🛠️ Tech Stack

### Backend
| Tool | Purpose |
|---|---|
| **FastAPI** | Async REST API framework |
| **PyTorch** | ML model inference (EfficientNet-B0) |
| **LangGraph** | Stateful agentic workflow orchestration |
| **LangChain** | LLM integration layer |
| **OpenAI GPT** | LLM provider for diagnosis advice |
| **Docker** | Containerized deployment |

### Frontend
| Tool | Purpose |
|---|---|
| **React 19 + Vite** | UI framework & build tool |
| **Tailwind CSS v4** | Utility-first styling |
| **Framer Motion** | Smooth UI animations |
| **Lucide React** | Icon library |
| **Axios** | HTTP client for API calls |

### ML Training
| Tool | Purpose |
|---|---|
| **PyTorch + torchvision** | Model training & transfer learning |
| **EfficientNet-B0** | Base model (pre-trained on ImageNet) |
| **WeightedRandomSampler** | Handles class imbalance in dataset |
| **Mixed Precision (AMP)** | Faster training on NVIDIA GPUs |

---

## 📁 Project Structure

```
DermaGlass/
├── App/
│   ├── backend/
│   │   ├── main.py              # FastAPI app + LangGraph agentic workflow
│   │   ├── requirements.txt     # Python dependencies
│   │   ├── Dockerfile           # Production container config
│   │   ├── .env                 # API keys (not committed)
│   │   └── models/
│   │       ├── best_model.pth   # Trained EfficientNet-B0 weights
│   │       └── class_names.json # Skin condition class labels
│   └── frontend/
│       ├── src/                 # React components & pages
│       ├── public/              # Static assets
│       ├── index.html
│       ├── vite.config.js
│       └── package.json
├── Training/
│   ├── train_final.py           # 2-stage fine-tuning training script
│   ├── split_data.py            # Train/val dataset splitter
│   └── check_models.py          # Model validation utility
├── Dataset/                     # (Not committed) training images
├── DEPLOY.md                    # Deployment guide
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python **3.10+**
- Node.js **18+**
- An **OpenAI API Key**
- *(Optional, for training)* NVIDIA GPU with CUDA

---

### 1. Clone the Repository

```bash
git clone https://github.com/Ganapathi-G2005/DermaGlass.git
cd DermaGlass
```

---

### 2. Backend Setup

```bash
cd App/backend

# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env and add your OpenAI API key:
# OPENAI_API_KEY=sk-your-key-here

# Start the development server
python main.py
```

> **Backend runs at:** `http://localhost:8000`  
> **API Docs (Swagger):** `http://localhost:8000/docs`

---

### 3. Frontend Setup

```bash
cd App/frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

> **Frontend runs at:** `http://localhost:5173`

---

### 4. (Optional) Train Your Own Model

```bash
# Ensure your dataset is placed in the Dataset/ folder:
# Dataset/
#   train/  ← subfolders per class
#   val/    ← subfolders per class

cd Training
python train_final.py
```

The script uses a **2-stage fine-tuning** strategy:
- **Stage 1** (5 epochs): Trains only the classifier head (backbone frozen)
- **Stage 2** (20 epochs): Fine-tunes the full model with differential learning rates

The best model is saved to `best_model.pth` automatically.

---

## 🧠 Model Details

| Property | Value |
|---|---|
| **Base Architecture** | EfficientNet-B0 (ImageNet pre-trained) |
| **Training Strategy** | 2-Stage Transfer Learning (Warmup → Fine-tune) |
| **Dataset Size** | 12,000+ dermatological images |
| **Validation Accuracy** | ~88% (Top-1) |
| **Low Confidence Threshold** | < 50% → flagged as "Unclear / Normal" |
| **Input Size** | 224×224 RGB |
| **Augmentations** | RandomCrop, Flip, Rotation ±30°, ColorJitter |
| **Imbalance Handling** | WeightedRandomSampler + Label Smoothing (0.1) |

### Detectable Conditions (12+ classes)
Acne · Eczema · Atopic Dermatitis · Tinea (Ringworm) · Vitiligo · Melasma · Psoriasis · Seborrheic Dermatitis · Rosacea · Warts · Nail Fungus · and more.

---

## 🌐 API Reference

### `POST /predict`
Upload a skin image to get a diagnosis and initial advice report.

**Request:** `multipart/form-data`
| Field | Type | Description |
|---|---|---|
| `file` | `File` | Skin image (JPG, PNG, WEBP) |

**Response:**
```json
{
  "disease": "Acne",
  "confidence": 87.42,
  "advice": "## What is it?\n...",
  "thread_id": "global_user_session"
}
```

---

### `POST /chat`
Ask follow-up questions about the diagnosis to the Dr. Derma agent.

**Request Body:**
```json
{
  "disease": "Acne",
  "confidence": 87.42,
  "question": "Can I use Clotrimazole cream for this?"
}
```

**Response:**
```json
{
  "reply": "Clotrimazole is an antifungal and is not recommended for Acne..."
}
```

---

## 🚢 Deployment

The recommended free-tier production stack is:

| Service | Purpose |
|---|---|
| **Hugging Face Spaces** (Docker) | Backend API hosting (16GB RAM free tier) |
| **Vercel** | Frontend hosting (auto-deploy from GitHub) |

See [`DEPLOY.md`](DEPLOY.md) for the full step-by-step deployment guide.

**Environment Variables (Production):**

| Variable | Where | Description |
|---|---|---|
| `OPENAI_API_KEY` | HF Space Secrets | Your OpenAI API key |
| `VITE_API_URL` | Vercel Env Vars | Full URL of your HF Space backend |

---

## ⚠️ Disclaimer

> **This tool is for educational and informational purposes only.**
>
> DermaGlass is **not** a substitute for professional medical advice, diagnosis, or treatment. AI predictions can be incorrect. Always consult a certified dermatologist for any skin concerns, especially before starting any medication.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details. Feel free to use, modify, and distribute for your own learning or projects.

---

## 👤 Author

**Ganapathi G**

---

<div align="center">
  <sub>Built with ❤️ for accessible AI-powered healthcare in India</sub>
</div>
