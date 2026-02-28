import { useState, useRef, useEffect } from "react";

const API_URL = "http://localhost:5000/api/chat";

// Simple markdown-like rendering: bold, line breaks, bullet points
function formatMessage(text) {
  if (!text) return null;
  const lines = text.split("\n");
  return lines.map((line, i) => {
    // Bold markers
    let formatted = line.replace(
      /\*\*(.+?)\*\*/g,
      '<strong class="font-semibold text-amber-300">$1</strong>'
    );
    // Bullet points
    const isBullet = /^\s*[-•]\s/.test(line);
    const isNumbered = /^\s*\d+[.)]\s/.test(line);
    if (isBullet || isNumbered) {
      formatted = formatted.replace(/^\s*[-•]\s/, "").replace(/^\s*\d+[.)]\s/, (m) => m);
      return (
        <div
          key={i}
          className={`${isBullet ? "pl-4 before:content-['•'] before:absolute before:-left-0 before:text-amber-400 relative ml-4" : "pl-1"} py-0.5`}
          dangerouslySetInnerHTML={{ __html: formatted }}
        />
      );
    }
    if (line.trim() === "") return <div key={i} className="h-2" />;
    return (
      <div key={i} className="py-0.5" dangerouslySetInnerHTML={{ __html: formatted }} />
    );
  });
}

function TypingIndicator() {
  return (
    <div className="flex items-center gap-1.5 py-3 px-1">
      {[0, 1, 2].map((i) => (
        <div
          key={i}
          className="w-2 h-2 rounded-full bg-amber-400"
          style={{
            animation: "bounce-dot 1.2s ease-in-out infinite",
            animationDelay: `${i * 0.15}s`,
          }}
        />
      ))}
    </div>
  );
}

const SUGGESTIONS = [
  "Should I do compound exercises first?",
  "Give me a chest workout, 4 exercises, beginner",
  "I want a hamstring workout with barbells, advanced",
  "What is the correct exercise order in a program?",
];

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const scrollRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, loading]);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const sendMessage = async (text) => {
    const userMsg = text || input.trim();
    if (!userMsg || loading) return;

    setInput("");
    setError(null);
    setMessages((prev) => [...prev, { role: "user", content: userMsg }]);
    setLoading(true);

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMsg }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.error || `Server error ${res.status}`);
      }

      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.response },
      ]);
    } catch (err) {
      setError(err.message || "Failed to connect to the server");
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "Sorry, something went wrong. Please make sure the Flask server is running on port 5000.",
        },
      ]);
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const isEmpty = messages.length === 0;

  return (
    <div
      className="min-h-screen flex flex-col"
      style={{
        background: "linear-gradient(145deg, #0c0c0c 0%, #1a1510 50%, #0c0c0c 100%)",
        fontFamily: "'DM Sans', sans-serif",
      }}
    >
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700&family=Space+Mono:wght@400;700&display=swap');

        @keyframes bounce-dot {
          0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
          30% { transform: translateY(-6px); opacity: 1; }
        }

        @keyframes fade-up {
          from { opacity: 0; transform: translateY(12px); }
          to { opacity: 1; transform: translateY(0); }
        }

        .msg-enter {
          animation: fade-up 0.3s ease-out both;
        }

        .suggestion-chip:hover {
          background: rgba(251, 191, 36, 0.15);
          border-color: rgba(251, 191, 36, 0.5);
          transform: translateY(-1px);
        }

        textarea::placeholder { color: #6b6558; }
        textarea:focus { outline: none; }
        
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #2a2520; border-radius: 4px; }
      `}</style>

      {/* Header */}
      <header className="flex-shrink-0 border-b" style={{ borderColor: "#1f1b15" }}>
        <div className="max-w-3xl mx-auto px-6 py-4 flex items-center gap-3">
          <div
            className="w-9 h-9 rounded-lg flex items-center justify-center text-sm"
            style={{
              background: "linear-gradient(135deg, #f59e0b, #d97706)",
              fontFamily: "'Space Mono', monospace",
              fontWeight: 700,
              color: "#0c0c0c",
            }}
          >
            PT
          </div>
          <div>
            <h1
              className="text-base font-semibold tracking-tight"
              style={{ color: "#e8e0d4" }}
            >
              AI Personal Trainer
            </h1>
            <p className="text-xs" style={{ color: "#6b6558" }}>
              RAG-powered • Exercise ordering • Training rules
            </p>
          </div>
          {error && (
            <div className="ml-auto text-xs px-2 py-1 rounded" style={{ background: "#2d1515", color: "#ef4444" }}>
              Connection error
            </div>
          )}
        </div>
      </header>

      {/* Messages area */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto">
        <div className="max-w-3xl mx-auto px-6 py-6">
          {isEmpty && !loading && (
            <div className="flex flex-col items-center justify-center" style={{ minHeight: "55vh" }}>
              <div
                className="w-16 h-16 rounded-2xl flex items-center justify-center text-2xl mb-6"
                style={{
                  background: "linear-gradient(135deg, rgba(251,191,36,0.15), rgba(217,119,6,0.08))",
                  border: "1px solid rgba(251,191,36,0.12)",
                }}
              >
                🏋️
              </div>
              <h2
                className="text-xl font-semibold mb-2"
                style={{ color: "#e8e0d4", fontFamily: "'Space Mono', monospace" }}
              >
                Ready to train
              </h2>
              <p className="text-sm mb-8 text-center max-w-md" style={{ color: "#6b6558" }}>
                Ask about training rules, request a workout program, or get exercise recommendations.
              </p>
              <div className="flex flex-wrap gap-2 justify-center max-w-lg">
                {SUGGESTIONS.map((s, i) => (
                  <button
                    key={i}
                    onClick={() => sendMessage(s)}
                    className="suggestion-chip text-xs px-3 py-2 rounded-lg transition-all duration-200 cursor-pointer"
                    style={{
                      border: "1px solid #2a2520",
                      background: "rgba(26,21,16,0.6)",
                      color: "#a89a85",
                    }}
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg, i) => (
            <div
              key={i}
              className={`msg-enter mb-4 flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
              style={{ animationDelay: `${0.05}s` }}
            >
              <div
                className="max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-relaxed"
                style={
                  msg.role === "user"
                    ? {
                        background: "linear-gradient(135deg, #b45309, #92400e)",
                        color: "#fef3c7",
                        borderBottomRightRadius: "6px",
                      }
                    : {
                        background: "#161310",
                        border: "1px solid #1f1b15",
                        color: "#c4b8a4",
                        borderBottomLeftRadius: "6px",
                      }
                }
              >
                {msg.role === "assistant" ? formatMessage(msg.content) : msg.content}
              </div>
            </div>
          ))}

          {loading && (
            <div className="msg-enter flex justify-start mb-4">
              <div
                className="rounded-2xl px-4"
                style={{ background: "#161310", border: "1px solid #1f1b15" }}
              >
                <TypingIndicator />
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Input area */}
      <div className="flex-shrink-0 border-t" style={{ borderColor: "#1f1b15" }}>
        <div className="max-w-3xl mx-auto px-6 py-4">
          <div
            className="flex items-end gap-3 rounded-xl px-4 py-3"
            style={{
              background: "#131110",
              border: "1px solid #1f1b15",
            }}
          >
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask about training rules or request a workout..."
              rows={1}
              className="flex-1 bg-transparent text-sm resize-none"
              style={{
                color: "#e8e0d4",
                maxHeight: "120px",
                lineHeight: "1.5",
              }}
              onInput={(e) => {
                e.target.style.height = "auto";
                e.target.style.height = Math.min(e.target.scrollHeight, 120) + "px";
              }}
            />
            <button
              onClick={() => sendMessage()}
              disabled={!input.trim() || loading}
              className="flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center transition-all duration-200"
              style={{
                background:
                  input.trim() && !loading
                    ? "linear-gradient(135deg, #f59e0b, #d97706)"
                    : "#1f1b15",
                color: input.trim() && !loading ? "#0c0c0c" : "#4a4235",
                cursor: input.trim() && !loading ? "pointer" : "not-allowed",
              }}
            >
              <svg
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <line x1="22" y1="2" x2="11" y2="13" />
                <polygon points="22 2 15 22 11 13 2 9 22 2" />
              </svg>
            </button>
          </div>
          <p className="text-center mt-2 text-xs" style={{ color: "#3a3428" }}>
            Powered by RAG • Exercise data sourced from local database
          </p>
        </div>
      </div>
    </div>
  );
}
