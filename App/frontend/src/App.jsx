import { useEffect, useState } from 'react';
import axios from 'axios';
import { AnimatePresence, motion } from 'framer-motion';
import { Home, Camera, History as HistoryIcon } from 'lucide-react';
import './index.css';

// Components
import HomeView from './components/HomeView';
import CameraCapture from './components/CameraCapture';
import ResultCard from './components/ResultCard';
import HistoryView from './components/HistoryView';

function App() {
  const [view, setView] = useState('HOME'); // 'HOME', 'CAMERA', 'LOADING', 'RESULT', 'HISTORY'
  const [activeTab, setActiveTab] = useState('HOME'); // Bottom nav tab
  const [file, setFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);

  // Loading / scanning experience
  const loadingSteps = [
    'Analyzing texture...',
    'Checking pigmentation...',
    'Consulting AI dermatologist...'
  ];
  const [loadingStepIndex, setLoadingStepIndex] = useState(0);

  useEffect(() => {
    if (view !== 'LOADING') return;

    setLoadingStepIndex(0);
    const interval = setInterval(() => {
      setLoadingStepIndex(prev => (prev + 1) % loadingSteps.length);
    }, 700);

    return () => clearInterval(interval);
  }, [view]);

  // Load stored history (basic user memory) on first mount
  useEffect(() => {
    try {
      const stored = window.localStorage.getItem('derma_history_v1');
      if (stored) {
        const parsed = JSON.parse(stored);
        if (Array.isArray(parsed)) {
          setHistory(parsed);
        }
      }
    } catch {
      // Ignore storage errors / corrupt data
    }
  }, []);

  // Persist history whenever it changes
  useEffect(() => {
    try {
      window.localStorage.setItem('derma_history_v1', JSON.stringify(history));
    } catch {
      // Ignore quota / private mode errors
    }
  }, [history]);

  // -- Handlers --

  const handleStartCamera = () => {
    setView('CAMERA');
  };

  const handleFileSelected = (selectedFile) => {
    if (selectedFile) {
      setFile(selectedFile);
      setImagePreview(URL.createObjectURL(selectedFile));
      processUpload(selectedFile);
    }
  };

  const handleCameraCapture = (capturedFile) => {
    setFile(capturedFile);
    setImagePreview(URL.createObjectURL(capturedFile));
    setView('LOADING'); // Show loading state immediately while processing
    processUpload(capturedFile);
  };

  const processUpload = async (fileToUpload) => {
    setView('LOADING');
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', fileToUpload);

    try {
      // Ensure a minimum "trust-building" scanning time
      const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const request = axios.post(`${API_URL}/predict`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      const delay = new Promise(resolve => setTimeout(resolve, 2000));
      const response = await Promise.all([request, delay]).then(([res]) => res);

      setResult(response.data);
      setHistory(prev => [
        {
          id: Date.now(),
          disease: response.data.disease,
          confidence: response.data.confidence,
          advice: response.data.advice,
          createdAt: new Date().toISOString(),
          preview: URL.createObjectURL(fileToUpload),
        },
        ...prev,
      ]);
      setView('RESULT');
      setActiveTab('HOME');
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || err.message || 'Failed to analyze image');
      setView('HOME');
    }
  };

  const handleReset = () => {
    setFile(null);
    setImagePreview(null);
    setResult(null);
    setError(null);
    setView('HOME');
    setActiveTab('HOME');
  };

  const handleTabChange = (tab) => {
    setActiveTab(tab);
    if (tab === 'HOME' || tab === 'HISTORY') {
      setView(tab);
    }
  };

  const handleScanPress = () => {
    handleStartCamera();
  };

  return (
    <div className={`min-h-screen font-sans pb-24 ${view === 'HOME' || view === 'HISTORY' ? 'bg-[#0d0d0f] text-white' : 'bg-gradient-to-br from-blue-50 via-white to-blue-50 text-brand-dark'}`}>
      {/* Global Error Banner */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="bg-red-500 text-white p-3 text-center text-sm font-bold flex justify-between px-6 z-50"
          >
            <span>Error: {error}</span>
            <button onClick={() => setError(null)}>✕</button>
          </motion.div>
        )}
      </AnimatePresence>

      <main className="mx-auto px-4 pt-6 pb-4 max-w-md min-h-screen flex flex-col">
        <AnimatePresence mode="wait">
          {/* VIEW: HOME */}
          {view === 'HOME' && (
            <HomeView
              key="home"
              onStartCamera={handleStartCamera}
              onFileSelected={handleFileSelected}
              history={history}
              onSelectItem={(item) => {
                if (!item) return;
                setResult({
                  disease: item.disease,
                  confidence: item.confidence,
                  advice: item.advice || 'Previous AI advice not stored. Run a new scan for fresh guidance.',
                });
                setImagePreview(item.preview || imagePreview);
                setView('RESULT');
              }}
            />
          )}

          {/* VIEW: HISTORY */}
          {view === 'HISTORY' && (
            <HistoryView
              key="history"
              history={history}
              onSelectItem={(item) => {
                if (!item) return;
                setResult({
                  disease: item.disease,
                  confidence: item.confidence,
                  advice: item.advice || 'Previous AI advice not stored. Run a new scan for fresh guidance.',
                });
                setImagePreview(item.preview || imagePreview);
                setView('RESULT');
              }}
            />
          )}

        </AnimatePresence>
      </main>

      {/* Full-screen Result overlay (ChatGPT-style dark report) */}
      <AnimatePresence>
        {view === 'RESULT' && (
          <ResultCard
            key="result"
            result={result}
            imagePreview={imagePreview}
            onReset={handleReset}
          />
        )}
      </AnimatePresence>

      {/* CAMERA MODAL */}
      <AnimatePresence>
        {view === 'CAMERA' && (
          <motion.div
            key="camera"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-40 bg-black"
          >
            <CameraCapture
              onCapture={handleCameraCapture}
              onCancel={() => setView('HOME')}
            />
          </motion.div>
        )}
      </AnimatePresence>

      {/* LOADING / SCANNING EXPERIENCE */}
      <AnimatePresence>
        {view === 'LOADING' && (
          <motion.div
            key="loading"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-40 flex flex-col items-center justify-center bg-slate-900/90 text-center px-6"
          >
            <div className="relative w-40 h-40 mb-10">
              <div className="absolute inset-0 rounded-3xl bg-gradient-to-br from-sky-300/20 via-sky-500/10 to-sky-200/5 blur-3xl" />
              <div className="absolute inset-4 rounded-3xl border border-cyan-200/40 bg-slate-900/40 backdrop-blur-2xl" />

              {/* Radar pulse rings */}
              <div className="absolute inset-6 rounded-3xl border border-cyan-300/30" />
              <div className="absolute inset-10 rounded-3xl border border-cyan-400/40" />
              <div className="absolute inset-14 rounded-3xl bg-cyan-300/20 animate-pulse" />

              {/* Center dot */}
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="w-6 h-6 rounded-full bg-cyan-300 shadow-[0_0_30px_rgba(34,211,238,0.7)]" />
              </div>

              {/* Expanding ripple */}
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="w-24 h-24 rounded-3xl border border-cyan-200/40 animate-[ping_1.8s_ease-out_infinite]" />
              </div>
            </div>

            <h2 className="text-2xl font-black text-white mb-2 tracking-tight">
              Scanning your skin
            </h2>
            <p className="text-sm text-slate-300 mb-6">
              {loadingSteps[loadingStepIndex]}
            </p>
            <p className="text-[11px] text-slate-500 max-w-xs">
              This takes a few seconds while our dermatology models analyze texture, color and patterns.
            </p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Persistent Disclaimer (hidden on Camera, Loading, and Result view) */}
      {view !== 'CAMERA' && view !== 'LOADING' && view !== 'RESULT' && (
        <motion.footer
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="fixed bottom-2 left-0 right-0 text-center px-4 pointer-events-none"
        >
          <p className="max-w-md mx-auto px-4 py-2 rounded-full inline-block text-[10px] text-slate-500 bg-white/5 border border-white/10">
            DermaDetect India • AI Health Assistant • Not a Doctor
          </p>
        </motion.footer>
      )}
    </div>
  );
}

export default App;
