
# Deployment Guide (Feb 2026)

This guide will help you deploy your full-stack Dermatology AI application for **free** using Industry Best Practices.

## 1. Backend Deployment (Hugging Face Spaces)
We use Hugging Face Spaces because it offers generous free tiers for ML apps (16GB RAM) which is required for PyTorch.

1.  **Create a Space**:
    *   Go to [huggingface.co/spaces](https://huggingface.co/spaces) and create a new Space.
    *   **Space Name**: `derma-api` (or similar)
    *   **SDK**: Select **Docker** (Blank).
    *   **Hardware**: **CPU Basic (Free)** is sufficient now that we successfully optimized the model.

2.  **Upload Code**:
    *   You can upload via Git or the Web UI.
    *   **Method A (Web UI - Easiest)**:
        *   In your Space, go to "Files".
        *   Click "Add file" -> "Upload files".
        *   Drag and drop the **contents** of your `App/backend` folder:
            *   `Dockerfile`
            *   `main.py`
            *   `requirements.txt`
            *   `models/` (The whole folder with `best_model.pth` and `class_names.json`)
        *   Commit changes.

3.  **Set Secrets (Environment Variables)**:
    *   Go to "Settings" tab in your Space.
    *   Scroll to "Variables and secrets".
    *   Add a **Secret**:
        *   Name: `GEMINI_API_KEY`
        *   Value: (Your actual API Key from `.env`)

4.  **Wait for Build**:
    *   The "Building" status will appear. Once "Running", click the "Embed this space" or look at the direct URL (e.g., `https://huggingface.co/spaces/YOUR_USERNAME/derma-api`).
    *   **Copy the Direct URL**. It will look like: `https://yourusername-derma-api.hf.space`.

---

## 2. Frontend Deployment (Vercel)
Vercel is the standard for hosting React/Vite apps.

1.  **Push to GitHub**:
    *   Push your entire project to a GitHub repository.

2.  **Import to Vercel**:
    *   Go to [vercel.com](https://vercel.com) and "Add New Project".
    *   Import your GitHub repo.

3.  **Configure Build**:
    *   **Framework Preset**: Vite (Auto-detected).
    *   **Root Directory**: Click "Edit" and select `App/frontend`.

4.  **Environment Variables**:
    *   Add a new variable:
        *   Name: `VITE_API_URL`
        *   Value: `https://yourusername-derma-api.hf.space` (The URL you copied from Hugging Face).
        *   **Important**: Remove any trailing slash `/` from the URL.

5.  **Deploy**:
    *   Click "Deploy". Vercel will build your frontend.

---

## 3. Verification
*   Open your new Vercel URL (e.g., `https://derma-app.vercel.app`).
*   Upload an image.
*   It should talk to your Hugging Face backend and return results!
