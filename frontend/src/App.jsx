import React, { useState, useRef, useCallback, useEffect } from 'react';
import Webcam from 'react-webcam';
import {
  Camera, Upload, Play, Pause, Languages, FileText,
  Zap, Loader2, CheckCircle, AlertTriangle, RotateCcw,
  ChevronDown, Mic, BookOpen, BarChart2, Send, Copy,
  X, Download, Info, Sparkles, History, TrendingUp,
  Clock, Hash, Globe, Volume2, VolumeX, Trash2,
  RefreshCw, Shield, Database, Award, Home,
  ArrowRight, Star, Eye
} from 'lucide-react';

const API_BASE = 'http://localhost:8000';

const LANGUAGES = {
  en: 'English', hi: 'Hindi', ta: 'Tamil',
  kn: 'Kannada', te: 'Telugu', ml: 'Malayalam',
  fr: 'French', de: 'German', es: 'Spanish',
  'zh-CN': 'Chinese', ar: 'Arabic', ja: 'Japanese',
};

const ENGINE_COLORS = {
  easyocr: 'linear-gradient(135deg, #6366f1, #818cf8)',
  'trocr-handwritten': 'linear-gradient(135deg, #10b981, #34d399)',
  correction_cache: 'linear-gradient(135deg, #a78bfa, #c4b5fd)',
  unknown: 'linear-gradient(135deg, #64748b, #94a3b8)',
};

// ─── Toast System ───────────────────────────────────────────────────────────
function ToastContainer({ toasts, removeToast }) {
  return (
    <div className="toast-container">
      {toasts.map((t) => (
        <div key={t.id} className={`toast toast-${t.type}`}>
          <span style={{ flexShrink: 0 }}>
            {t.type === 'success' && <CheckCircle size={15} />}
            {t.type === 'error' && <AlertTriangle size={15} />}
            {t.type === 'warn' && <AlertTriangle size={15} />}
            {t.type === 'info' && <Info size={15} />}
          </span>
          <span style={{ flex: 1 }}>{t.message}</span>
          <button className="toast-close" onClick={() => removeToast(t.id)} aria-label="Close">
            <X size={13} />
          </button>
        </div>
      ))}
    </div>
  );
}

function useToasts() {
  const [toasts, setToasts] = useState([]);
  const addToast = useCallback((message, type = 'success', duration = 3500) => {
    const id = Date.now() + Math.random();
    setToasts(prev => [...prev, { id, message, type }]);
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), duration);
  }, []);
  const removeToast = useCallback((id) => setToasts(prev => prev.filter(t => t.id !== id)), []);
  return { toasts, addToast, removeToast };
}

// ─── Step Indicator ─────────────────────────────────────────────────────────
function StepIndicator({ current }) {
  const steps = [
    { id: 1, label: 'Upload', icon: Upload },
    { id: 2, label: 'Extract', icon: Sparkles },
    { id: 3, label: 'Translate', icon: Languages },
    { id: 4, label: 'Listen', icon: Volume2 },
  ];
  return (
    <div className="step-indicator" aria-label="Progress steps">
      {steps.map((step, i) => {
        const Icon = step.icon;
        const state = current > step.id ? 'done' : current === step.id ? 'active' : 'idle';
        return (
          <React.Fragment key={step.id}>
            <div className={`step ${state}`} aria-label={`Step ${step.id}: ${step.label}`}>
              <div className="step-circle">
                {state === 'done' ? <CheckCircle size={14} /> : <Icon size={14} />}
              </div>
              <span className="step-label">{step.label}</span>
            </div>
            {i < steps.length - 1 && <div className={`step-line ${current > step.id ? 'done' : ''}`} />}
          </React.Fragment>
        );
      })}
    </div>
  );
}

// ─── Word Badge ──────────────────────────────────────────────────────────────
function WordBadge({ word, confidence, flagged, isHighlighted, corrected, onCorrect, onDelete }) {
  const [editing, setEditing] = useState(false);
  const [editVal, setEditVal] = useState(word);

  useEffect(() => { setEditVal(word); }, [word]);

  const handleCorrect = () => {
    if (editVal.trim()) onCorrect(editVal.trim());
    setEditing(false);
  };

  const confPct = Math.round((confidence || 0) * 100);

  return (
    <span
      className={`word-badge ${flagged ? 'flagged' : ''} ${isHighlighted ? 'highlighted' : ''} ${corrected ? 'corrected' : ''} word-badge-editable`}
      title={flagged ? `Low confidence (${confPct}%) — click to edit` : `Click to edit · Confidence: ${confPct}%`}
    >
      {editing ? (
        <span className="word-edit-wrap">
          <input
            className="word-edit-input"
            value={editVal}
            onChange={e => setEditVal(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter') handleCorrect();
              if (e.key === 'Escape') setEditing(false);
            }}
            autoFocus
          />
          <button className="word-edit-confirm" onClick={handleCorrect} aria-label="Confirm"><CheckCircle size={13} /></button>
          <button className="word-edit-cancel" onClick={() => setEditing(false)} aria-label="Cancel"><X size={13} /></button>
        </span>
      ) : (
        <span className="word-badge-inner">
          <span
            className="word-clickable"
            onClick={() => setEditing(true)}
          >
            {word}
            {flagged && !corrected && <span className="word-conf-dot" />}
          </span>
          {/* Delete button — visible on hover */}
          <button
            className="word-delete-btn"
            onClick={e => { e.stopPropagation(); onDelete(); }}
            aria-label={`Delete word "${word}"`}
            title="Delete this word"
          >
            <X size={10} />
          </button>
        </span>
      )}
    </span>
  );
}

// ─── Confidence Bar ──────────────────────────────────────────────────────────
function ConfidenceBar({ score }) {
  const pct = Math.round((score || 0) * 100);
  const color = pct >= 80 ? '#10b981' : pct >= 60 ? '#f59e0b' : '#ef4444';
  const label = pct >= 80 ? 'High' : pct >= 60 ? 'Medium' : 'Low';
  return (
    <div className="conf-bar-wrap" title={`Model confidence: ${pct}%`}>
      <span className="conf-bar-label">Confidence</span>
      <div className="conf-bar-track">
        <div className="conf-bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="conf-bar-pct" style={{ color }}>
        {pct}% <span className="conf-label">{label}</span>
      </span>
    </div>
  );
}

// ─── Copy Button ─────────────────────────────────────────────────────────────
function CopyButton({ text }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  return (
    <button className={`copy-btn ${copied ? 'copied' : ''}`} onClick={copy} title="Copy to clipboard">
      {copied ? <CheckCircle size={12} /> : <Copy size={12} />}
      {copied ? 'Copied!' : 'Copy'}
    </button>
  );
}

// ─── Backend Status ──────────────────────────────────────────────────────────
function BackendStatus({ status }) {
  return (
    <div className={`backend-status ${status}`}>
      <span className={`status-dot ${status}`} />
      {status === 'online' && 'Backend Online'}
      {status === 'offline' && 'Backend Offline'}
      {status === 'checking' && 'Connecting...'}
    </div>
  );
}

// ─── History Page ────────────────────────────────────────────────────────────
function HistoryPage({ onRestoreSession, addToast }) {
  const [history, setHistory] = useState(null);
  const [loading, setLoading] = useState(true);

  const fetchHistory = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/history?limit=30`);
      const data = await res.json();
      setHistory(data);
    } catch {
      addToast('Could not load history', 'error');
    } finally {
      setLoading(false);
    }
  }, [addToast]);

  useEffect(() => { fetchHistory(); }, [fetchHistory]);

  const deleteSession = async (imageId, e) => {
    e.stopPropagation();
    try {
      const res = await fetch(`${API_BASE}/api/history/${imageId}`, { method: 'DELETE' });
      if (res.ok) {
        addToast('Session removed from history', 'success');
        fetchHistory();
      }
    } catch {
      addToast('Delete failed', 'error');
    }
  };

  const formatTime = (ts) => {
    if (!ts) return '';
    const d = new Date(ts);
    return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
  };

  const getEngineColor = (engine) => {
    if (engine === 'easyocr') return '#6366f1';
    if (engine === 'trocr-handwritten') return '#10b981';
    if (engine === 'correction_cache') return '#a78bfa';
    return '#64748b';
  };

  return (
    <div className="history-page">
      <div className="page-header">
        <div>
          <h1 className="page-title">Session History</h1>
          <p className="page-subtitle">Your recent OCR sessions — click to restore</p>
        </div>
        <button className="btn btn-ghost btn-sm" onClick={fetchHistory} disabled={loading}>
          <RefreshCw size={14} className={loading ? 'spin' : ''} />
          Refresh
        </button>
      </div>

      {loading ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          {[...Array(4)].map((_, i) => (
            <div key={i} className="loading-shimmer" style={{ height: '72px', borderRadius: '18px' }} />
          ))}
        </div>
      ) : !history?.sessions?.length ? (
        <div className="history-empty">
          <div className="empty-icon-wrap"><History size={32} className="empty-icon" /></div>
          <p style={{ fontSize: '0.95rem', fontWeight: 600, color: 'var(--text-300)' }}>No sessions yet</p>
          <p>Upload and extract text from an image to start building history.</p>
        </div>
      ) : (
        <div className="history-grid">
          {history.sessions.map((session, i) => (
            <div
              key={session.image_id}
              className="history-card"
              onClick={() => onRestoreSession(session)}
              role="button"
              tabIndex={0}
              onKeyDown={e => e.key === 'Enter' && onRestoreSession(session)}
              style={{ animationDelay: `${i * 0.04}s` }}
            >
              <div className="history-card-icon">
                <FileText size={20} />
              </div>
              <div className="history-card-content">
                <div className="history-card-text">
                  {session.extracted_text || '(empty)'}
                </div>
                <div className="history-card-meta">
                  <span className="history-meta-item">
                    <Clock size={10} />
                    {formatTime(session.timestamp)}
                  </span>
                  <span className="history-meta-item" style={{ color: getEngineColor(session.engine) }}>
                    <Zap size={10} />
                    {session.engine}
                  </span>
                  <span className="history-meta-item">
                    <Hash size={10} />
                    {session.word_count || 0} words
                  </span>
                  {session.from_correction && (
                    <span className="meta-pill ok" style={{ fontSize: '0.65rem', padding: '0.1rem 0.45rem' }}>
                      ✓ Cached
                    </span>
                  )}
                </div>
              </div>
              <div className="history-card-actions">
                <button
                  className="btn btn-ghost btn-xs"
                  onClick={(e) => deleteSession(session.image_id, e)}
                  title="Remove from history"
                  aria-label="Delete session"
                >
                  <Trash2 size={12} />
                </button>
                <ArrowRight size={14} style={{ color: 'var(--text-600)', flexShrink: 0 }} />
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Dashboard Page ──────────────────────────────────────────────────────────
function DashboardPage({ addToast }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  const fetchDashboard = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/dashboard`);
      const d = await res.json();
      setData(d);
    } catch {
      addToast('Could not load dashboard', 'error');
    } finally {
      setLoading(false);
    }
  }, [addToast]);

  useEffect(() => { fetchDashboard(); }, [fetchDashboard]);

  const engineColors = {
    easyocr: '#6366f1',
    'trocr-handwritten': '#10b981',
    correction_cache: '#a78bfa',
  };

  const maxEngine = data
    ? Math.max(...Object.values(data.sessions?.engine_breakdown || {}).map(Number), 1)
    : 1;

  return (
    <div className="dashboard-page">
      <div className="page-header">
        <div>
          <h1 className="page-title">Dashboard</h1>
          <p className="page-subtitle">System analytics and model performance overview</p>
        </div>
        <button className="btn btn-ghost btn-sm" onClick={fetchDashboard} disabled={loading}>
          <RefreshCw size={14} className={loading ? 'spin' : ''} />
          Refresh
        </button>
      </div>

      {loading ? (
        <div className="stat-grid">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="loading-shimmer" style={{ height: '120px', borderRadius: '18px' }} />
          ))}
        </div>
      ) : data ? (
        <>
          {/* Stat Cards */}
          <div className="stat-grid">
            <div className="stat-card stat-primary">
              <div className="stat-icon"><FileText size={20} /></div>
              <div className="stat-value">{data.sessions?.total || 0}</div>
              <div className="stat-label">Total Sessions</div>
              <div className="stat-sub">OCR extractions run</div>
            </div>
            <div className="stat-card stat-success">
              <div className="stat-icon"><Database size={20} /></div>
              <div className="stat-value">{data.feedback?.total_corrections || 0}</div>
              <div className="stat-label">Corrections Saved</div>
              <div className="stat-sub">Training data collected</div>
            </div>
            <div className="stat-card stat-warn">
              <div className="stat-icon"><Hash size={20} /></div>
              <div className="stat-value">{data.sessions?.total_words_extracted || 0}</div>
              <div className="stat-label">Words Extracted</div>
              <div className="stat-sub">Across all sessions</div>
            </div>
            <div className="stat-card stat-accent">
              <div className="stat-icon"><Award size={20} /></div>
              <div className="stat-value">
                {Math.round((data.sessions?.avg_confidence || 0) * 100)}%
              </div>
              <div className="stat-label">Avg Confidence</div>
              <div className="stat-sub">Model accuracy score</div>
            </div>
          </div>

          {/* Engine Breakdown */}
          {data.sessions?.engine_breakdown && Object.keys(data.sessions.engine_breakdown).length > 0 && (
            <div className="dashboard-section">
              <h2 className="section-title"><Zap size={16} /> Engine Breakdown</h2>
              <div className="panel" style={{ gap: '0.85rem' }}>
                <div className="engine-chart">
                  {Object.entries(data.sessions.engine_breakdown).map(([engine, count]) => (
                    <div key={engine} className="engine-row">
                      <div className="engine-row-header">
                        <span className="engine-name">{engine}</span>
                        <span className="engine-count">{count} session{count !== 1 ? 's' : ''}</span>
                      </div>
                      <div className="engine-bar-track">
                        <div
                          className="engine-bar-fill"
                          style={{
                            width: `${(count / maxEngine) * 100}%`,
                            background: engineColors[engine] || '#64748b',
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Feedback Details */}
          <div className="dashboard-section">
            <h2 className="section-title"><Shield size={16} /> Training Data Status</h2>
            <div className="panel">
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem' }}>
                <div style={{ flex: 1, minWidth: '160px' }}>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-500)', marginBottom: '0.3rem', textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 }}>Words Corrected</div>
                  <div style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--success)', fontFamily: 'JetBrains Mono, monospace' }}>
                    {data.feedback?.total_words_corrected || 0}
                  </div>
                </div>
                <div style={{ flex: 1, minWidth: '160px' }}>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-500)', marginBottom: '0.3rem', textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 }}>Cache Hits</div>
                  <div style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--accent)', fontFamily: 'JetBrains Mono, monospace' }}>
                    {data.sessions?.from_cache || 0}
                  </div>
                </div>
                <div style={{ flex: 1, minWidth: '160px' }}>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-500)', marginBottom: '0.3rem', textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 }}>Last Updated</div>
                  <div style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-300)' }}>
                    {data.feedback?.last_updated
                      ? new Date(data.feedback.last_updated).toLocaleDateString()
                      : 'Never'}
                  </div>
                </div>
              </div>

              {data.feedback?.total_corrections > 0 && (
                <div style={{ marginTop: '0.5rem' }}>
                  <a
                    href={`${API_BASE}/api/feedback/export`}
                    target="_blank"
                    rel="noreferrer"
                    className="btn btn-success btn-sm"
                    style={{ display: 'inline-flex', textDecoration: 'none' }}
                  >
                    <Download size={14} />
                    Export Training Dataset ({data.feedback.total_corrections} records)
                  </a>
                </div>
              )}
            </div>
          </div>
        </>
      ) : (
        <div className="history-empty">
          <div className="empty-icon-wrap"><TrendingUp size={32} className="empty-icon" /></div>
          <p style={{ fontSize: '0.9rem', color: 'var(--text-400)' }}>No data yet. Start by extracting some text!</p>
        </div>
      )}
    </div>
  );
}

// ─── Main App ────────────────────────────────────────────────────────────────
export default function App() {
  // Navigation
  const [activeTab, setActiveTab] = useState('workspace');

  // Backend health
  const [backendStatus, setBackendStatus] = useState('checking');

  // Image state
  const [imageSrc, setImageSrc] = useState(null);
  const [useWebcam, setUseWebcam] = useState(false);
  const [isDragging, setIsDragging] = useState(false);

  // OCR state
  const [imageId, setImageId] = useState(null);
  const [imageHash, setImageHash] = useState('');
  const [ocrResult, setOcrResult] = useState(null);
  const [editedText, setEditedText] = useState('');
  const [selectedModel, setSelectedModel] = useState('auto');
  const [isOcring, setIsOcring] = useState(false);

  // Translation + TTS
  const [targetLang, setTargetLang] = useState('hi');
  const [sourceLang, setSourceLang] = useState('en');  // OCR always extracts English
  const [transResult, setTransResult] = useState(null);
  const [isTranslating, setIsTranslating] = useState(false);

  // Audio
  const audioRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackPct, setPlaybackPct] = useState(0);
  const [highlightedWordIdx, setHighlightedWordIdx] = useState(-1);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);

  // Corrections
  const [wordCorrections, setWordCorrections] = useState({});
  const [feedbackSent, setFeedbackSent] = useState(false);
  const [feedbackStats, setFeedbackStats] = useState(null);

  // Toasts
  const { toasts, addToast, removeToast } = useToasts();

  const webcamRef = useRef(null);
  const fileInputRef = useRef(null);

  const currentStep = !imageSrc ? 1 : !ocrResult ? 2 : !transResult ? 3 : 4;

  // ── Backend Health Check ───────────────────────────────────────────────────
  useEffect(() => {
    const check = async () => {
      try {
        const res = await fetch(`${API_BASE}/`, { signal: AbortSignal.timeout(4000) });
        setBackendStatus(res.ok ? 'online' : 'offline');
      } catch {
        setBackendStatus('offline');
      }
    };
    check();
    const iv = setInterval(check, 30000);
    return () => clearInterval(iv);
  }, []);

  // ── Feedback Stats ─────────────────────────────────────────────────────────
  useEffect(() => {
    fetch(`${API_BASE}/api/feedback/stats`)
      .then(r => r.json())
      .then(setFeedbackStats)
      .catch(() => {});
  }, [feedbackSent]);

  // ── Drag & Drop ────────────────────────────────────────────────────────────
  const handleDragOver = (e) => { e.preventDefault(); setIsDragging(true); };
  const handleDragLeave = () => setIsDragging(false);
  const handleDrop = (e) => {
    e.preventDefault(); setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) loadFile(file);
  };

  const loadFile = (file) => {
    const reader = new FileReader();
    reader.onloadend = () => { setImageSrc(reader.result); resetResults(); };
    reader.readAsDataURL(file);
  };

  const capture = useCallback(() => {
    const image = webcamRef.current.getScreenshot();
    setImageSrc(image); setUseWebcam(false); resetResults();
  }, [webcamRef]);

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    loadFile(file);
    e.target.value = '';
  };

  const resetResults = () => {
    setOcrResult(null); setEditedText(''); setTransResult(null);
    setHighlightedWordIdx(-1); setWordCorrections({});
    setFeedbackSent(false); setImageId(null); setImageHash('');
    setIsPlaying(false); setPlaybackPct(0); setPlaybackSpeed(1);
  };

  const dataURLtoBlob = (dataurl) => {
    const arr = dataurl.split(',');
    const mime = arr[0].match(/:(.*?);/)[1];
    const bstr = atob(arr[1]);
    const u8arr = new Uint8Array(bstr.length);
    for (let i = 0; i < bstr.length; i++) u8arr[i] = bstr.charCodeAt(i);
    return new Blob([u8arr], { type: mime });
  };

  // ── OCR ────────────────────────────────────────────────────────────────────
  const extractText = async () => {
    if (!imageSrc) return;
    setIsOcring(true); setOcrResult(null); setTransResult(null);
    setEditedText(''); setWordCorrections({}); setFeedbackSent(false);

    const blob = dataURLtoBlob(imageSrc);
    const formData = new FormData();
    formData.append('file', blob, 'image.jpg');
    formData.append('model_choice', selectedModel);
    formData.append('preprocess', 'true');

    try {
      const res = await fetch(`${API_BASE}/api/extract-text`, { method: 'POST', body: formData });
      const data = await res.json();
      if (res.ok) {
        setOcrResult(data);
        setEditedText(data.extracted_text || '');
        setImageId(data.image_id);
        setImageHash(data.image_hash || '');
        const flagged = data.words?.filter(w => w.flagged).length || 0;
        if (data.from_correction) {
          addToast('✓ Loaded your human-corrected text for this image', 'success');
        } else {
          addToast(
            flagged > 0
              ? `Extracted! ${flagged} uncertain word${flagged !== 1 ? 's' : ''} flagged — click to correct.`
              : '✓ Text extracted successfully!',
            flagged > 0 ? 'warn' : 'success'
          );
        }
      } else {
        setOcrResult({ error: data.detail });
        addToast(data.detail || 'OCR failed', 'error');
      }
    } catch {
      setOcrResult({ error: 'Cannot connect to backend. Is it running on port 8000?' });
      addToast('Backend not reachable', 'error');
    } finally {
      setIsOcring(false);
    }
  };

  // ── Word Correction & Deletion ─────────────────────────────────────────────
  // wordCorrections[idx] = string  → replacement word
  // wordCorrections[idx] = null    → word deleted
  //
  // We pass `words` as a param to avoid stale closure inside setState updater
  const rebuildText = (corrections, words) => {
    if (!words) return '';
    return words
      .map((w, i) => (corrections[i] === null ? null : corrections[i] ?? w.word))
      .filter(w => w !== null)
      .join(' ');
  };

  const handleWordCorrect = (idx, correctedWord) => {
    setWordCorrections(prev => {
      const next = { ...prev, [idx]: correctedWord };
      setEditedText(rebuildText(next, ocrResult?.words));
      return next;
    });
    addToast(`Word updated: "${correctedWord}"`, 'success', 1800);
  };

  const handleWordDelete = (idx) => {
    setWordCorrections(prev => {
      const next = { ...prev, [idx]: null };
      setEditedText(rebuildText(next, ocrResult?.words));
      return next;
    });
    addToast('Word removed', 'warn', 1600);
  };

  const sendFeedback = async () => {
    if (!imageId || !ocrResult) return;
    // Include both corrections (string) and deletions (null → empty string for training)
    const correctionsList = Object.entries(wordCorrections)
      .filter(([, v]) => v !== undefined)
      .map(([idx, corrected]) => ({
        index: parseInt(idx),
        original: ocrResult.words[parseInt(idx)]?.word || '',
        corrected: corrected === null ? '' : corrected,
      }));
    try {
      const res = await fetch(`${API_BASE}/api/feedback/correct`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_id: imageId,
          image_hash: imageHash,
          original_ocr: ocrResult.extracted_text ?? ocrResult.full_text ?? '',
          corrected_text: editedText,
          word_corrections: correctionsList,
          model_used: ocrResult.engine,
          sequence_confidence: ocrResult.sequence_confidence,
        }),
      });
      if (res.ok) {
        setFeedbackSent(true);
        addToast(`✓ ${correctionsList.length} correction${correctionsList.length !== 1 ? 's' : ''} saved to training dataset!`, 'success');
      }
    } catch {
      addToast('Could not save corrections', 'error');
    }
  };

  // ── Translation + TTS ──────────────────────────────────────────────────────
  const translateAndSpeak = async () => {
    if (!editedText) return;
    setIsTranslating(true); setTransResult(null);
    setHighlightedWordIdx(-1);

    const formData = new FormData();
    formData.append('text', editedText);
    formData.append('target_language', targetLang);
    formData.append('source_language', sourceLang);

    try {
      const res = await fetch(`${API_BASE}/api/translate-and-tts`, { method: 'POST', body: formData });
      const data = await res.json();
      if (res.ok) {
        setTransResult(data);
        addToast(`✓ Translated to ${LANGUAGES[targetLang] || targetLang}`, 'success');
      } else {
        addToast(data.detail || 'Translation failed', 'error');
      }
    } catch {
      addToast('Translation error — check backend', 'error');
    } finally {
      setIsTranslating(false);
    }
  };

  // ── Audio ──────────────────────────────────────────────────────────────────
  const handlePlayPause = () => {
    if (!audioRef.current || !transResult) return;
    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current.play();
      setIsPlaying(true);
    }
  };

  const handleAudioEnded = () => {
    setIsPlaying(false); setHighlightedWordIdx(-1);
    setPlaybackPct(100);
  };

  const handleAudioTimeUpdate = () => {
    if (!audioRef.current) return;
    const { currentTime, duration } = audioRef.current;
    if (duration) setPlaybackPct((currentTime / duration) * 100);

    // Sync highlight precisely based on time (handles play/pause & speed changes perfectly)
    if (transResult?.word_timestamps) {
      const ms = currentTime * 1000;
      const idx = transResult.word_timestamps.findIndex((wt, i, arr) => {
        const next = arr[i + 1];
        return ms >= wt.start_ms && (!next || ms < next.start_ms);
      });
      setHighlightedWordIdx(idx);
    }
  };

  const handleSpeedChange = () => {
    const nextSpeed = playbackSpeed === 1 ? 1.25 : playbackSpeed === 1.25 ? 1.5 : playbackSpeed === 1.5 ? 2 : 1;
    setPlaybackSpeed(nextSpeed);
    if (audioRef.current) {
      audioRef.current.playbackRate = nextSpeed;
    }
  };

  const handleDownloadAudio = () => {
    if (!transResult?.audio_base64) return;
    const url = `data:audio/mp3;base64,${transResult.audio_base64}`;
    const a = document.createElement('a');
    a.href = url;
    a.download = `translated_audio_${targetLang}.mp3`;
    a.click();
  };

  // ── Restore session from history ────────────────────────────────────────────
  const handleRestoreSession = (session) => {
    setOcrResult({
      extracted_text: session.extracted_text,
      words: session.extracted_text?.split(' ').map(w => ({ word: w, confidence: session.sequence_confidence, flagged: false })) || [],
      engine: session.engine,
      sequence_confidence: session.sequence_confidence,
      from_correction: session.from_correction,
      preprocessing: session.preprocessing || {},
      image_id: session.image_id,
      image_hash: session.image_hash,
    });
    setEditedText(session.extracted_text || '');
    setImageId(session.image_id);
    setImageHash(session.image_hash || '');
    setWordCorrections({});
    setFeedbackSent(false);
    setTransResult(null);
    setPlaybackSpeed(1);
    setActiveTab('workspace');
    addToast('Session restored from history', 'info');
  };

  // ── Derived values ─────────────────────────────────────────────────────────
  // flaggedCount: only non-deleted uncertain words that haven't been corrected
  const flaggedCount = ocrResult?.words?.filter(
    (w, i) => w.flagged && wordCorrections[i] === undefined
  ).length || 0;
  // correctionCount: number of edits (corrections OR deletions) made
  const correctionCount = Object.values(wordCorrections).filter(v => v !== undefined).length;

  const tabs = [
    { id: 'workspace', label: 'Workspace', icon: Home },
    { id: 'history', label: 'History', icon: History },
    { id: 'dashboard', label: 'Analytics', icon: TrendingUp },
  ];

  return (
    <div className="app">
      <ToastContainer toasts={toasts} removeToast={removeToast} />

      {/* ── Header ── */}
      <header className="header">
        <div className="header-inner">
          <a className="logo" href="#" onClick={e => { e.preventDefault(); setActiveTab('workspace'); }}>
            <div className="logo-icon-wrap">
              <BookOpen size={20} color="#fff" />
            </div>
            <div className="logo-text">
              <h1>ScriptBridge</h1>
              <p>AI Handwriting Digitizer</p>
            </div>
          </a>

          {/* Desktop Nav */}
          <nav className="header-nav" aria-label="Main navigation">
            {tabs.map(tab => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  className={`nav-tab ${activeTab === tab.id ? 'active' : ''}`}
                  onClick={() => setActiveTab(tab.id)}
                  aria-current={activeTab === tab.id ? 'page' : undefined}
                >
                  <Icon size={14} />
                  {tab.label}
                </button>
              );
            })}
          </nav>

          <div className="header-right">
            {feedbackStats && (
              <div className="stats-pill" title="Corrections saved for model retraining">
                <Database size={13} />
                <span>{feedbackStats.total_corrections || 0} corrections</span>
              </div>
            )}
            <BackendStatus status={backendStatus} />
          </div>
        </div>
      </header>

      {/* ── Workspace Page ── */}
      <div className={`page ${activeTab === 'workspace' ? 'active' : ''}`}>
        {/* Step bar */}
        <div className="step-bar">
          <StepIndicator current={currentStep} />
        </div>

        <div className="workspace">
          {/* ══ LEFT PANEL ════════════════════════════════════════════════════ */}
          <section className="panel" aria-label="Image Input">
            <div className="panel-title">
              <Camera size={16} />
              <span>Image Input</span>
              {imageSrc && (
                <button
                  className="btn btn-ghost btn-xs panel-title-action"
                  onClick={() => { setImageSrc(null); resetResults(); }}
                  title="Clear and start over"
                  id="clear-image-btn"
                >
                  <X size={12} /> Clear
                </button>
              )}
            </div>

            {/* Model Select */}
            <div className="field-group">
              <label className="field-label" htmlFor="model-select">
                <Zap size={12} /> OCR Engine
              </label>
              <div className="select-wrap">
                <select
                  id="model-select"
                  className="select"
                  value={selectedModel}
                  onChange={e => setSelectedModel(e.target.value)}
                >
                  <option value="auto">🤖 Auto — Automatically choose best model</option>
                  <option value="easyocr">⚡ EasyOCR — Best for printed / mixed text</option>
                  <option value="trocr-handwritten">✍ TrOCR Handwritten — Best for cursive</option>
                </select>
                <ChevronDown size={14} className="select-icon" />
              </div>
              <p className="field-hint">
                <Info size={11} />
                {selectedModel === 'auto' && 'Automatically determines handwritten vs printed and uses the best model.'}
                {selectedModel === 'easyocr' && 'Best overall speed+accuracy for mixed/printed text.'}
                {selectedModel === 'trocr-handwritten' && 'Microsoft TrOCR fine-tuned on handwriting datasets.'}
              </p>
            </div>

            {/* Image Area */}
            <div
              className={`image-area ${isDragging ? 'drag-active' : ''}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={!imageSrc && !useWebcam ? () => fileInputRef.current?.click() : undefined}
              role={!imageSrc && !useWebcam ? 'button' : undefined}
              aria-label={!imageSrc ? 'Click or drag an image here' : 'Image preview'}
              tabIndex={!imageSrc && !useWebcam ? 0 : undefined}
              onKeyDown={e => {
                if (!imageSrc && !useWebcam && (e.key === 'Enter' || e.key === ' '))
                  fileInputRef.current?.click();
              }}
            >
              {useWebcam ? (
                <Webcam
                  audio={false}
                  ref={webcamRef}
                  screenshotFormat="image/jpeg"
                  videoConstraints={{ facingMode: 'environment' }}
                  className="webcam-feed"
                />
              ) : imageSrc ? (
                <img src={imageSrc} alt="Uploaded note" className="preview-img" />
              ) : (
                <div className="image-placeholder">
                  <div className="placeholder-icon-wrap">
                    <Upload size={32} className="placeholder-icon" />
                  </div>
                  <p>Drop image here or click to upload</p>
                  <p className="placeholder-sub">JPG · PNG · WebP · Handwriting · Printed · Mixed</p>
                </div>
              )}
            </div>

            {/* Controls */}
            <div className="btn-row">
              {useWebcam ? (
                <>
                  <button id="capture-btn" className="btn btn-primary" onClick={capture}>
                    <Camera size={15} /> Capture
                  </button>
                  <button className="btn btn-ghost" onClick={() => setUseWebcam(false)}>Cancel</button>
                </>
              ) : (
                <>
                  <button id="camera-btn" className="btn btn-ghost" onClick={() => setUseWebcam(true)}>
                    <Camera size={15} /> Camera
                  </button>
                  <button id="upload-btn" className="btn btn-ghost" onClick={() => fileInputRef.current?.click()}>
                    <Upload size={15} /> Upload
                  </button>
                  <input
                    type="file" ref={fileInputRef} accept="image/*"
                    className="file-input" onChange={handleFileUpload} id="file-input"
                  />
                </>
              )}
            </div>

            <button
              id="extract-btn"
              className="btn btn-primary btn-full"
              onClick={extractText}
              disabled={!imageSrc || isOcring}
            >
              {isOcring
                ? <><Loader2 size={17} className="spin" /> Extracting text...</>
                : <><Sparkles size={17} /> Extract Text</>
              }
            </button>

            {/* Preprocessing metadata */}
            {ocrResult?.preprocessing && Object.keys(ocrResult.preprocessing).length > 0 && (
              <div className="meta-pills">
                <span className={`meta-pill ${ocrResult.preprocessing.is_blurry ? 'warn' : 'ok'}`}>
                  {ocrResult.preprocessing.is_blurry ? '⚠ Blurry' : '✓ Sharp'}
                </span>
                {ocrResult.preprocessing.skew_corrected && (
                  <span className="meta-pill ok"><RotateCcw size={10} /> Deskewed</span>
                )}
                <span className="meta-pill neutral">
                  Quality: {Math.round(ocrResult.preprocessing.quality_score || 0)}
                </span>
              </div>
            )}

            {/* Restore tip */}
            {!imageSrc && !ocrResult && (
              <div style={{
                fontSize: '0.75rem', color: 'var(--text-600)', textAlign: 'center',
                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.4rem'
              }}>
                <History size={12} />
                Or restore a previous session from the{' '}
                <button
                  style={{ background: 'none', border: 'none', color: 'var(--primary-light)', cursor: 'pointer', fontSize: 'inherit', fontWeight: 600, padding: 0 }}
                  onClick={() => setActiveTab('history')}
                >
                  History tab
                </button>
              </div>
            )}
          </section>

          {/* ══ RIGHT PANEL ═══════════════════════════════════════════════════ */}
          <section className="panel panel-right" aria-label="Results">

            {/* ── OCR Result ── */}
            {ocrResult && !ocrResult.error && (
              <div className="result-block">
                <div className="result-header">
                  <span className="result-title"><FileText size={15} />Extracted Text</span>
                  <div className="result-header-actions">
                    {ocrResult.from_correction && (
                      <span className="engine-badge engine-badge-corrected" title="From your human corrections">
                        ✓ Human-Corrected
                      </span>
                    )}
                    <span
                      className="engine-badge"
                      style={{ background: `${ENGINE_COLORS[ocrResult.engine] || ENGINE_COLORS.unknown}20` }}
                    >
                      {ocrResult.engine}
                    </span>
                    <CopyButton text={editedText} />
                  </div>
                </div>

                <ConfidenceBar score={ocrResult.sequence_confidence} />

                {flaggedCount > 0 && (
                  <div className="flagged-notice">
                    <AlertTriangle size={14} />
                    <span>
                      {flaggedCount} uncertain {flaggedCount === 1 ? 'word' : 'words'} — click{' '}
                      <span className="flagged-notice-highlight">red words</span> to correct
                    </span>
                  </div>
                )}

                {/* Word badges — all editable, all deletable */}
                <div className="words-display" aria-label="Word-level extraction">
                  {ocrResult.words?.map((w, idx) => {
                    // Skip words marked as deleted (null)
                    if (wordCorrections[idx] === null) return null;
                    return (
                      <WordBadge
                        key={idx}
                        word={wordCorrections[idx] ?? w.word}
                        confidence={w.confidence}
                        flagged={w.flagged && !wordCorrections[idx]}
                        corrected={!!wordCorrections[idx]}
                        isHighlighted={false}
                        onCorrect={(corrected) => handleWordCorrect(idx, corrected)}
                        onDelete={() => handleWordDelete(idx)}
                      />
                    );
                  })}
                </div>

                {/* Editable text */}
                <div className="textarea-wrap">
                  <textarea
                    id="extracted-text"
                    className="textarea"
                    value={editedText}
                    onChange={e => setEditedText(e.target.value)}
                    placeholder="Edit the extracted text here..."
                    rows={4}
                    aria-label="Extracted text — editable"
                  />
                  <span className="char-count">{editedText.length} chars</span>
                </div>

                {/* Feedback */}
                {correctionCount > 0 && !feedbackSent && (
                  <button id="send-feedback-btn" className="btn btn-success btn-sm" onClick={sendFeedback}>
                    <Send size={13} />
                    Save {correctionCount} correction{correctionCount !== 1 ? 's' : ''} to training data
                  </button>
                )}
                {feedbackSent && (
                  <div className="feedback-ok">
                    <CheckCircle size={14} />
                    Corrections saved! They'll improve model accuracy in the next fine-tuning run.
                  </div>
                )}
              </div>
            )}

            {ocrResult?.error && (
              <div className="error-block">
                <AlertTriangle size={16} style={{ flexShrink: 0 }} /> {ocrResult.error}
              </div>
            )}

            {/* ── Translate & TTS ── */}
            {ocrResult && !ocrResult.error && (
              <div className="result-block" style={{ marginTop: '0.75rem' }}>
                <div className="result-header">
                  <span className="result-title"><Languages size={15} />Translate & Speak</span>
                </div>

                <div className="translate-controls">
                  {/* From is always English — OCR extracts English text */}
                  <div className="field-group" style={{ flex: '0 0 auto' }}>
                    <label className="field-label" htmlFor="source-lang-display">From</label>
                    <div
                      id="source-lang-display"
                      className="lang-fixed-badge"
                      title="OCR always extracts English text"
                    >
                      🇬🇧 English
                    </div>
                  </div>

                  <div className="field-group flex-1">
                    <label className="field-label" htmlFor="target-lang-select">To</label>
                    <div className="select-wrap">
                      <select id="target-lang-select" className="select select-sm" value={targetLang} onChange={e => setTargetLang(e.target.value)}>
                        {Object.entries(LANGUAGES).map(([code, name]) => (
                          <option key={code} value={code}>{name}</option>
                        ))}
                      </select>
                      <ChevronDown size={14} className="select-icon" />
                    </div>
                  </div>

                  <button
                    id="translate-btn"
                    className="btn btn-primary translate-btn"
                    onClick={translateAndSpeak}
                    disabled={!editedText || isTranslating}
                  >
                    {isTranslating
                      ? <><Loader2 size={15} className="spin" />Translating...</>
                      : <><Globe size={15} />Translate & Listen</>
                    }
                  </button>
                </div>

                {/* Translation result */}
                {transResult && (
                  <div className="trans-result">
                    <div className="trans-result-header">
                      <div className="trans-lang-badge">
                        <Globe size={11} />
                        {LANGUAGES[transResult.target_lang] || transResult.target_lang}
                      </div>
                      <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                        {transResult.word_count && (
                          <span className="meta-pill neutral">{transResult.word_count} words</span>
                        )}
                        <CopyButton text={transResult.translated_text} />
                      </div>
                    </div>

                    {/* Word highlighting */}
                    <div className="words-display" aria-label="Translated words with audio sync">
                      {transResult.word_timestamps?.map((wt, idx) => (
                        <span
                          key={idx}
                          className={`word-badge ${idx === highlightedWordIdx ? 'highlighted' : ''}`}
                        >
                          {wt.word}
                        </span>
                      ))}
                    </div>

                    <div className="textarea-wrap">
                      <textarea
                        id="translated-text"
                        className="textarea textarea-sm"
                        value={transResult.translated_text}
                        readOnly rows={3}
                        aria-label="Translated text"
                      />
                    </div>

                    {/* Audio player */}
                    {transResult.audio_base64 && (
                      <div>
                        <div className="audio-row" style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                          <button
                            id="play-pause-btn"
                            className={`btn btn-play ${isPlaying ? 'playing' : ''}`}
                            onClick={handlePlayPause}
                            aria-label={isPlaying ? 'Pause' : 'Play'}
                          >
                            {isPlaying ? <Pause size={16} /> : <Play size={16} />}
                            {isPlaying ? 'Pause' : 'Play'}
                          </button>

                          <button 
                            className="btn btn-ghost btn-sm" 
                            onClick={handleSpeedChange}
                            title="Playback Speed"
                          >
                            {playbackSpeed}x
                          </button>

                          <button 
                            className="btn btn-ghost btn-sm" 
                            onClick={handleDownloadAudio}
                            title="Download Audio"
                          >
                            <Download size={14} />
                          </button>

                          <audio
                            ref={audioRef}
                            src={`data:audio/mp3;base64,${transResult.audio_base64}`}
                            onEnded={handleAudioEnded}
                            onTimeUpdate={handleAudioTimeUpdate}
                          />
                          <span className="audio-hint" style={{ marginLeft: 'auto' }}>Words light up as they're spoken</span>
                        </div>
                        {isPlaying && (
                          <div className="playback-bar" style={{ marginTop: '0.5rem' }}>
                            <div className="playback-bar-fill" style={{ width: `${playbackPct}%` }} />
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}

            {/* Empty state */}
            {!ocrResult && (
              <div className="empty-state">
                <div className="empty-icon-wrap">
                  <Sparkles size={36} className="empty-icon" />
                </div>
                <h2>Ready to digitize</h2>
                <p>
                  Upload or capture an image, then click{' '}
                  <strong>Extract Text</strong> to begin.
                </p>
                <div className="empty-features">
                  <span className="feature-tag"><Zap size={11} /> 2 OCR Engines</span>
                  <span className="feature-tag"><Globe size={11} /> 12 Languages</span>
                  <span className="feature-tag"><Volume2 size={11} /> Text to Speech</span>
                  <span className="feature-tag"><Database size={11} /> AI Fine-Tuning</span>
                  <span className="feature-tag"><History size={11} /> Session History</span>
                  <span className="feature-tag"><Shield size={11} /> Correction Cache</span>
                </div>
              </div>
            )}
          </section>
        </div>
      </div>

      {/* ── History Page ── */}
      <div className={`page ${activeTab === 'history' ? 'active' : ''}`}>
        <HistoryPage onRestoreSession={handleRestoreSession} addToast={addToast} />
      </div>

      {/* ── Dashboard Page ── */}
      <div className={`page ${activeTab === 'dashboard' ? 'active' : ''}`}>
        <DashboardPage addToast={addToast} />
      </div>

      {/* ── Mobile Bottom Nav ── */}
      <nav className="mobile-nav" aria-label="Mobile navigation">
        {tabs.map(tab => {
          const Icon = tab.icon;
          return (
            <button
              key={tab.id}
              className={`mobile-nav-item ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
              aria-current={activeTab === tab.id ? 'page' : undefined}
            >
              <Icon size={20} />
              {tab.label}
            </button>
          );
        })}
      </nav>
    </div>
  );
}
