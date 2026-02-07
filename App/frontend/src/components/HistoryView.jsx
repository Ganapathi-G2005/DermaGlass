import React from 'react';
import { Activity, Clock3 } from 'lucide-react';
import { motion } from 'framer-motion';

const formatTime = (iso) => {
    if (!iso) return '';
    const d = new Date(iso);
    return d.toLocaleString(undefined, {
        hour: '2-digit',
        minute: '2-digit',
        day: '2-digit',
        month: 'short',
    });
};

const HistoryView = ({ history, onSelectItem }) => {
    const hasItems = history && history.length > 0;

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="flex flex-col gap-4 pb-10"
        >
            <header className="mb-4">
                <p className="text-xs font-semibold uppercase tracking-[0.2em] text-sky-400 mb-1 flex items-center gap-1.5">
                    <Clock3 size={14} />
                    Scan history
                </p>
                <h1 className="text-2xl font-extrabold text-white tracking-tight">
                    Recent assessments
                </h1>
                <p className="text-sm text-slate-400 mt-1">
                    Tap any past scan to revisit its diagnosis.
                </p>
            </header>

            {!hasItems && (
                <div className="rounded-2xl bg-white/5 border border-white/10 px-5 py-6 flex flex-col items-start gap-3">
                    <div className="w-10 h-10 rounded-2xl bg-white/10 flex items-center justify-center text-slate-400 mb-1">
                        <Clock3 size={22} />
                    </div>
                    <p className="font-semibold text-white">No history yet</p>
                    <p className="text-sm text-slate-400">
                        Your future scans will appear here so you can track changes over time.
                    </p>
                </div>
            )}

            {hasItems && (
                <div className="space-y-3">
                    {history.map((item) => (
                        <button
                            key={item.id}
                            onClick={() => onSelectItem?.(item)}
                            className="w-full rounded-2xl bg-white/5 border border-white/10 px-4 py-3 flex items-center gap-3 text-left hover:bg-white/10 transition-colors"
                        >
                            <div className="w-14 h-14 rounded-2xl overflow-hidden bg-white/10 flex-shrink-0">
                                {item.preview ? (
                                    <img
                                        src={item.preview}
                                        alt={item.disease}
                                        className="w-full h-full object-cover"
                                    />
                                ) : (
                                    <div className="w-full h-full flex items-center justify-center text-slate-500 text-xs">
                                        No image
                                    </div>
                                )}
                            </div>
                            <div className="flex-1 min-w-0">
                                <p className="text-xs text-slate-400 mb-0.5">
                                    {formatTime(item.createdAt)}
                                </p>
                                <p className="text-sm font-semibold text-white truncate">
                                    {item.disease}
                                </p>
                                <div className="flex items-center gap-2 mt-1">
                                    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-white/10 text-[11px] text-slate-300">
                                        <Activity size={12} />
                                        {item.confidence}% confidence
                                    </span>
                                </div>
                            </div>
                        </button>
                    ))}
                </div>
            )}
        </motion.div>
    );
};

export default HistoryView;
