import React, { useRef, useState, useEffect } from 'react';
import { Camera, RefreshCw, X } from 'lucide-react';

const CameraCapture = ({ onCapture, onCancel }) => {
    const videoRef = useRef(null);
    const [stream, setStream] = useState(null);
    const [facingMode, setFacingMode] = useState('environment'); // 'user' (front) or 'environment' (back)
    const [error, setError] = useState(null);

    useEffect(() => {
        startCamera();
        return () => stopCamera();
    }, [facingMode]);

    const startCamera = async () => {
        stopCamera();
        setError(null);
        try {
            const newStream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: facingMode }
            });
            setStream(newStream);
            if (videoRef.current) {
                videoRef.current.srcObject = newStream;
            }
        } catch (err) {
            setError("Unable to access camera. Please allow permissions.");
            console.error(err);
        }
    };

    const stopCamera = () => {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            setStream(null);
        }
    };

    const handleCapture = () => {
        if (videoRef.current) {
            const canvas = document.createElement('canvas');
            canvas.width = videoRef.current.videoWidth;
            canvas.height = videoRef.current.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(videoRef.current, 0, 0);

            canvas.toBlob((blob) => {
                if (blob) {
                    const file = new File([blob], "camera-capture.jpg", { type: "image/jpeg" });
                    onCapture(file);
                    stopCamera();
                }
            }, 'image/jpeg', 0.95);
        }
    };

    const fileInputRef = useRef(null); // Fallback for no camera support

    return (
        <div className="fixed inset-0 bg-black z-50">

            {/* Video Feed - Full Screen */}
            <div className="absolute inset-0">
                {error ? (
                    <div className="w-full h-full flex flex-col items-center justify-center text-white p-6 bg-slate-900">
                        <p className="mb-4 text-red-400 text-lg">{error}</p>
                        <div className="flex flex-col gap-4">
                            <button
                                onClick={() => fileInputRef.current.click()}
                                className="px-6 py-3 bg-brand-primary rounded-xl font-bold text-white shadow-lg active:scale-95 transition-transform"
                            >
                                Upload File Instead
                            </button>
                            <button
                                onClick={onCancel}
                                className="px-6 py-3 bg-white/10 rounded-xl font-bold text-white border border-white/20 active:scale-95 transition-transform"
                            >
                                Cancel
                            </button>
                        </div>
                        <input
                            type="file"
                            ref={fileInputRef}
                            accept="image/*"
                            className="hidden"
                            onChange={(e) => {
                                if (e.target.files[0]) onCapture(e.target.files[0]);
                            }}
                        />
                    </div>
                ) : (
                    <video
                        ref={videoRef}
                        autoPlay
                        playsInline
                        className="w-full h-full object-cover"
                    />
                )}
            </div>

            {/* Overlay UI */}
            {/* Top Bar */}
            <div className="absolute top-0 left-0 right-0 z-20 p-6 flex justify-between items-start bg-gradient-to-b from-black/70 to-transparent pt-8">
                <button onClick={onCancel} className="p-3 bg-black/40 backdrop-blur-md rounded-full text-white hover:bg-black/60 transition-colors">
                    <X size={24} />
                </button>
                <button
                    onClick={() => setFacingMode(prev => prev === 'user' ? 'environment' : 'user')}
                    className="p-3 bg-black/40 backdrop-blur-md rounded-full text-white hover:bg-black/60 transition-colors"
                >
                    <RefreshCw size={24} />
                </button>
            </div>

            {/* Guide Frame (Center) */}
            {!error && (
                <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-10">
                    <div className="w-64 h-64 border-2 border-white/50 rounded-2xl relative">
                        <div className="absolute top-0 left-0 w-6 h-6 border-t-4 border-l-4 border-white -mt-1 -ml-1 rounded-tl-lg"></div>
                        <div className="absolute top-0 right-0 w-6 h-6 border-t-4 border-r-4 border-white -mt-1 -mr-1 rounded-tr-lg"></div>
                        <div className="absolute bottom-0 left-0 w-6 h-6 border-b-4 border-l-4 border-white -mb-1 -ml-1 rounded-bl-lg"></div>
                        <div className="absolute bottom-0 right-0 w-6 h-6 border-b-4 border-r-4 border-white -mb-1 -mr-1 rounded-br-lg"></div>
                    </div>
                </div>
            )}

            {/* Bottom Controls */}
            <div className="absolute bottom-0 left-0 right-0 z-20 h-40 bg-gradient-to-t from-black/80 via-black/40 to-transparent flex flex-col items-center justify-end pb-10">
                <button
                    onClick={handleCapture}
                    className="group relative flex flex-col items-center justify-center"
                    disabled={!!error}
                >
                    <div className={`w-20 h-20 rounded-full border-4 border-white flex items-center justify-center transition-all duration-200 ${!error && 'group-active:scale-95 group-hover:bg-white/10 cursor-pointer'}`}>
                        <div className="w-16 h-16 bg-white rounded-full shadow-lg"></div>
                    </div>
                    <span className="text-white text-sm font-medium mt-3 opacity-90 shadow-sm">Tap to Capture</span>
                </button>
            </div>
        </div>
    );
};

export default CameraCapture;
