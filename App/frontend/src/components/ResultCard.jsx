import React, { useState } from 'react';
import { motion } from 'framer-motion';
import Markdown from 'react-markdown';
import { Activity, AlertTriangle, Send, MessageCircle, ChevronLeft } from 'lucide-react';
import axios from 'axios';

const ResultCard = ({ result, imagePreview, onReset }) => {
    // ... (state matches existing)
    const [chatMessages, setChatMessages] = useState([]);
    const [chatInput, setChatInput] = useState('');
    const [isSending, setIsSending] = useState(false);

    if (!result) return null;

    const { disease, confidence, advice } = result;
    const isHighConfidence = confidence > 70;

    // ... (handleSend matches existing)
    const handleSend = async (e) => {
        e?.preventDefault();
        const trimmed = chatInput.trim();
        if (!trimmed || isSending) return;

        const userMessage = { role: 'user', content: trimmed, ts: Date.now() };
        setChatMessages((prev) => [...prev, userMessage]);
        setChatInput('');
        setIsSending(true);

        try {
            const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
            const response = await axios.post(`${API_URL}/chat`, {
                disease,
                confidence,
                question: trimmed,
            });
            const replyText = response.data?.reply || 'Sorry, I could not generate a response right now.';
            setChatMessages((prev) => [
                ...prev,
                { role: 'assistant', content: replyText, ts: Date.now() + 1 },
            ]);
        } catch (err) {
            console.error(err);
            setChatMessages((prev) => [
                ...prev,
                {
                    role: 'assistant',
                    content:
                        `Connection Error: ${err.message}. ${err.response?.data?.detail || ''}. Please ensure the backend is running.`,
                    ts: Date.now() + 1,
                },
            ]);
        } finally {
            setIsSending(false);
        }
    };

    // ... (MarkdownComponents matches existing)
    const MarkdownComponents = {
        h2: ({ node, ...props }) => (
            <h2 className="text-[13px] font-bold uppercase tracking-wider text-sky-400 border-b border-sky-500/30 pb-2 mt-6 mb-3 flex items-center gap-2" {...props} />
        ),
        p: ({ node, ...props }) => (
            <p className="text-sm leading-7 text-slate-200 mb-3" {...props} />
        ),
        strong: ({ node, ...props }) => (
            <strong className="font-bold text-amber-300" {...props} />
        ),
        ul: ({ node, ...props }) => (
            <ul className="list-none space-y-2 my-3 pl-1" {...props} />
        ),
        li: ({ node, ...props }) => (
            <li className="relative pl-5 text-sm text-slate-200 before:content-['•'] before:absolute before:left-0 before:text-sky-400 before:font-bold" {...props} />
        ),
        blockquote: ({ node, ...props }) => (
            <blockquote className="border-l-4 border-amber-500/50 bg-amber-500/10 pl-4 py-3 pr-2 my-4 rounded-r-lg italic text-amber-100/90 text-sm" {...props} />
        ),
    };

    return (
        <motion.div
            key="result"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-30 flex flex-col bg-[#0d0d0f] overflow-hidden"
        >
            {/* Fixed header */}
            <header className="flex-shrink-0 px-4 pt-6 pb-4 border-b border-white/10 bg-[#0d0d0f]/95 backdrop-blur-md">
                <div className="max-w-2xl mx-auto flex items-center gap-4">
                    <button
                        onClick={onReset}
                        className="p-2 -ml-2 rounded-full text-slate-400 hover:text-white hover:bg-white/10 transition-all"
                    >
                        <ChevronLeft size={24} />
                    </button>

                    <div className="relative w-12 h-12 rounded-xl overflow-hidden bg-white/5 flex-shrink-0 ring-1 ring-white/10">
                        {imagePreview && (
                            <img
                                src={imagePreview}
                                alt="Skin condition"
                                className="w-full h-full object-cover"
                            />
                        )}
                        <div
                            className={`absolute -top-0.5 -right-0.5 rounded-bl-lg px-1.5 py-0.5 text-[9px] font-bold shadow-sm ${isHighConfidence
                                ? 'bg-emerald-500 text-white'
                                : 'bg-amber-500 text-white'
                                }`}
                        >
                            {confidence}%
                        </div>
                    </div>
                    <div className="min-w-0 flex-1">
                        <p className="text-[10px] uppercase tracking-widest text-slate-400 mb-0.5 font-medium">
                            AI Assessment
                        </p>
                        <h1 className="text-lg font-bold text-white truncate leading-tight">{disease}</h1>
                    </div>
                </div>
            </header>

            {/* Scrollable content area – ONLY this scrolls */}
            <div className="flex-1 min-h-0 overflow-y-auto overscroll-contain scroll-smooth">
                <div className="max-w-2xl mx-auto px-5 py-6 space-y-8">
                    {/* AI Advice */}
                    <section className="animate-in fade-in slide-in-from-bottom-4 duration-700">
                        <div className="flex items-center gap-2.5 mb-2">
                            <div className="h-8 w-8 rounded-xl bg-emerald-500/10 flex items-center justify-center ring-1 ring-emerald-500/20">
                                <Activity size={16} className="text-emerald-400" />
                            </div>
                            <h2 className="text-sm font-bold text-white tracking-wide">Analysis Report</h2>
                        </div>

                        {/* Custom Markdown Renderer */}
                        <div className="bg-white/5 rounded-2xl p-5 border border-white/5 shadow-inner">
                            <Markdown components={MarkdownComponents}>{advice}</Markdown>
                        </div>
                    </section>

                    {/* Disclaimer */}
                    <div className="rounded-xl bg-amber-500/10 border border-amber-500/20 p-4 flex gap-4 items-start">
                        <AlertTriangle size={20} className="text-amber-400 flex-shrink-0 mt-0.5" />
                        <div className="text-xs text-amber-100/80 leading-relaxed">
                            <p className="font-bold text-amber-200 mb-1 uppercase tracking-wider text-[10px]">Medical Disclaimer</p>
                            <p>
                                This app provides AI-generated insights and is{' '}
                                <strong className="text-amber-300">not a substitute for a doctor.</strong> If
                                symptoms worsen, spread, or cause pain, seek professional care immediately.
                            </p>
                        </div>
                    </div>

                    {/* Chat */}
                    <section>
                        <div className="flex items-center gap-2.5 mb-4">
                            <div className="h-8 w-8 rounded-xl bg-sky-500/10 flex items-center justify-center ring-1 ring-sky-500/20">
                                <MessageCircle size={16} className="text-sky-400" />
                            </div>
                            <h2 className="text-sm font-bold text-white tracking-wide">Ask a Follow-up</h2>
                        </div>

                        {chatMessages.length === 0 && (
                            <div className="text-center py-6 px-4 rounded-2xl bg-white/5 border border-dashed border-white/10 mb-4">
                                <p className="text-xs text-slate-400">
                                    Common questions: <br />
                                    <span className="text-sky-400/80">"Is this contagious?"</span> • <span className="text-sky-400/80">"Home remedies?"</span>
                                </p>
                            </div>
                        )}

                        <div className="space-y-4">
                            {chatMessages.map((msg) => (
                                <div
                                    key={msg.ts}
                                    className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                                >
                                    <div
                                        className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-relaxed shadow-sm ${msg.role === 'user'
                                            ? 'bg-gradient-to-br from-sky-500 to-blue-600 text-white rounded-br-none'
                                            : 'bg-[#1a1b1e] border border-white/10 text-slate-200 rounded-bl-none'
                                            }`}
                                    >
                                        {msg.role === 'assistant' ? (
                                            <div className="markdown-chat">
                                                <Markdown components={{
                                                    p: ({ node, ...props }) => <p className="mb-2 last:mb-0" {...props} />,
                                                    strong: ({ node, ...props }) => <strong className="text-sky-300 font-bold" {...props} />,
                                                    ul: ({ node, ...props }) => <ul className="list-disc pl-4 space-y-1 mb-2" {...props} />,
                                                    li: ({ node, ...props }) => <li className="pl-1" {...props} />
                                                }}>{msg.content}</Markdown>
                                            </div>
                                        ) : (
                                            <span>{msg.content}</span>
                                        )}
                                    </div>
                                </div>
                            ))}

                            {/* Thinking State */}
                            {isSending && (
                                <div className="flex justify-start animate-pulse">
                                    <div className="bg-[#1a1b1e] border border-white/10 rounded-2xl rounded-bl-none px-4 py-3 flex items-center gap-1.5">
                                        <div className="w-1.5 h-1.5 rounded-full bg-sky-400/60 animate-bounce [animation-delay:-0.3s]"></div>
                                        <div className="w-1.5 h-1.5 rounded-full bg-sky-400/60 animate-bounce [animation-delay:-0.15s]"></div>
                                        <div className="w-1.5 h-1.5 rounded-full bg-sky-400/60 animate-bounce"></div>
                                        <span className="text-xs text-slate-400 ml-1.5 font-medium">Thinking...</span>
                                    </div>
                                </div>
                            )}
                        </div>
                    </section>

                    {/* Spacer for fixed bottom */}
                    <div className="h-28" />
                </div>
            </div>

            {/* Fixed bottom bar */}
            <div className="flex-shrink-0 px-4 py-4 border-t border-white/10 bg-[#0d0d0f]/95 backdrop-blur-md z-40">
                <div className="max-w-2xl mx-auto space-y-3">
                    <form onSubmit={handleSend} className="flex gap-2">
                        <input
                            type="text"
                            value={chatInput}
                            onChange={(e) => setChatInput(e.target.value)}
                            placeholder="Type a question..."
                            disabled={isSending}
                            className="flex-1 rounded-xl border border-white/20 bg-white/5 px-4 py-3 text-sm text-white placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-sky-500/50 focus:border-sky-500/50 transition-all disabled:opacity-50"
                        />
                        <button
                            type="submit"
                            disabled={isSending || !chatInput.trim()}
                            className="h-11 w-11 rounded-xl bg-sky-500 hover:bg-sky-400 text-white flex items-center justify-center disabled:opacity-40 disabled:cursor-not-allowed active:scale-95 transition-all shadow-lg shadow-sky-500/20"
                        >
                            <Send size={18} />
                        </button>
                    </form>
                    <button
                        onClick={onReset}
                        className="w-full py-3 rounded-xl bg-white/5 border border-white/10 text-slate-300 text-sm font-medium hover:bg-white/10 hover:text-white active:scale-[0.99] transition-all"
                    >
                        Start New Assessment
                    </button>
                </div>
            </div>
        </motion.div>
    );
};

export default ResultCard;
