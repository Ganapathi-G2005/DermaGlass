import React from 'react';
import { Camera, Image as ImageIcon, Sparkles } from 'lucide-react';
import { useDropzone } from 'react-dropzone';
import { motion } from 'framer-motion';

const HomeView = ({ onStartCamera, onFileSelected, history, onSelectItem }) => {
    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        accept: { 'image/*': [] },
        onDrop: acceptedFiles => {
            if (acceptedFiles?.length > 0) {
                onFileSelected(acceptedFiles[0]);
            }
        },
        noClick: true
    });

    const handleFileChange = (e) => {
        if (e.target.files?.[0]) {
            onFileSelected(e.target.files[0]);
        }
    };

    const recent = (history || []).slice(0, 8);

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="flex flex-col gap-5 pb-10"
            {...getRootProps()}
        >
            {/* Header */}
            <header className="flex items-center justify-between gap-3 pt-2">
                <div>
                    <p className="text-xs font-semibold uppercase tracking-[0.25em] text-sky-400 mb-1">
                        DermaGlass
                    </p>
                    <h1 className="text-xl font-extrabold text-white tracking-tight">
                        Hello Friend
                    </h1>
                    <p className="text-sm text-slate-400 mt-1">
                        Let&apos;s do a quick skin check-in.
                    </p>
                </div>
                <div className="relative">
                    <div className="w-10 h-10 rounded-full bg-gradient-to-br from-sky-400 to-indigo-500 flex items-center justify-center text-white text-xs font-semibold shadow-[0_8px_25px_rgba(59,130,246,0.55)]">
                        <span>DG</span>
                    </div>
                    <span className="absolute -bottom-1 -right-0.5 h-3 w-3 rounded-full border-2 border-[#0d0d0f] bg-emerald-400 shadow-sm" />
                </div>
            </header>

            {/* Hero / primary CTA card */}
            <div className="mt-1 rounded-2xl bg-white/5 border border-white/10 px-4 py-4 flex gap-4 items-center">
                <div className="relative">
                    <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-sky-500 to-blue-600 flex items-center justify-center text-white shadow-[0_10px_30px_rgba(56,189,248,0.4)]">
                        <Sparkles className="w-6 h-6" />
                    </div>
                </div>
                <div className="flex-1">
                    <p className="text-xs font-semibold text-sky-400 tracking-wide uppercase mb-0.5">
                        Check your skin health
                    </p>
                    <p className="text-sm font-semibold text-white leading-snug">
                        Capture a close-up photo to get an instant AI perspective.
                    </p>
                    <p className="text-xs text-slate-400 mt-1.5">
                        Works best with clear lighting and one focused area.
                    </p>
                </div>
            </div>

            {/* Actions */}
            <div className="space-y-3 mt-2">
                <motion.button
                    whileHover={{ scale: 1.01 }}
                    whileTap={{ scale: 0.97 }}
                    onClick={onStartCamera}
                    className="w-full py-4 px-5 bg-gradient-to-r from-sky-500 via-sky-600 to-blue-700 text-white rounded-2xl shadow-[0_18px_45px_rgba(56,189,248,0.4)] flex items-center justify-center gap-3 text-sm font-semibold relative overflow-hidden group"
                >
                    <div className="absolute inset-0 bg-white/15 translate-y-full group-hover:translate-y-0 transition-transform duration-300" />
                    <Camera size={22} />
                    <span>Start a live scan</span>
                </motion.button>

                <div className="relative">
                    <input
                        {...getInputProps()}
                        id="file-upload"
                        className="hidden"
                    />
                    <input
                        type="file"
                        id="hidden-file-input"
                        className="hidden"
                        accept="image/*"
                        onChange={handleFileChange}
                    />

                    <motion.button
                        whileHover={{ scale: 1.01 }}
                        whileTap={{ scale: 0.97 }}
                        onClick={() => document.getElementById('hidden-file-input').click()}
                        className={`w-full py-4 px-5 rounded-2xl flex items-center justify-center gap-3 text-sm font-semibold border transition-all ${isDragActive
                                ? 'bg-white/15 border-sky-500/50 text-white'
                                : 'bg-white/5 border-white/10 text-white hover:bg-white/10 hover:border-white/20'
                            }`}
                    >
                        <ImageIcon size={20} className="text-sky-400" />
                        <span>Upload a photo from gallery</span>
                    </motion.button>
                </div>
            </div>

            {/* Recent scans carousel */}
            <section className="mt-3">
                <div className="flex items-center justify-between mb-2">
                    <p className="text-xs font-semibold text-slate-400">Recent scans</p>
                    <p className="text-xs text-slate-500">
                        {recent.length > 0 ? `${recent.length} saved` : 'No scans yet'}
                    </p>
                </div>

                <div className="flex gap-3 overflow-x-auto pb-1 hide-scrollbar">
                    {recent.length === 0 && (
                        <div className="px-4 py-3 rounded-2xl border border-dashed border-white/20 text-xs text-slate-500 bg-white/5">
                            Your next few scans will show up here for quick reference.
                        </div>
                    )}

                    {recent.map((item) => (
                        <button
                            key={item.id}
                            type="button"
                            onClick={(e) => {
                                e.stopPropagation();
                                onSelectItem?.(item);
                            }}
                            className="min-w-[96px] max-w-[96px] text-left rounded-2xl overflow-hidden bg-white/5 border border-white/10 hover:bg-white/10 hover:border-white/20 active:scale-[0.98] transition-all cursor-pointer"
                        >
                            <div className="relative">
                                {item.preview && (
                                    <img
                                        src={item.preview}
                                        alt={item.disease}
                                        className="w-full h-[88px] object-cover"
                                    />
                                )}
                                <div className="absolute inset-x-0 bottom-0 h-10 bg-gradient-to-t from-black/80 to-transparent" />
                                <div className="absolute bottom-1.5 left-2 right-2">
                                    <p className="text-[10px] text-white font-semibold truncate">
                                        {item.disease}
                                    </p>
                                    <p className="text-[9px] text-slate-300">
                                        {item.confidence}% confidence
                                    </p>
                                </div>
                            </div>
                        </button>
                    ))}
                </div>
            </section>

            <p className="mt-2 text-xs text-slate-500 max-w-sm">
                This is an AI health assistant. It&apos;s smart, but it&apos;s not a doctor.
            </p>
        </motion.div>
    );
};

export default HomeView;
