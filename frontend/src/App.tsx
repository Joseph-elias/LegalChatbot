import { FormEvent, useEffect, useMemo, useState } from 'react';
import { LogOut, MessageSquarePlus, SendHorizontal, Settings, Trash2, X } from 'lucide-react';
import ChatMessage from './components/ChatMessage';
import Footer from './components/Footer';
import './App.css';

const API_BASE = (import.meta.env.VITE_API_BASE || '/api').replace(/\/+$/, '');
const POLICY_VERSION = '2026-04-16';
const COOKIE_CONSENT_KEY = 'cookie_consent_v1';

interface User {
  id: number;
  email: string;
  created_at: string;
  policy_version?: string | null;
  accepted_disclaimer?: boolean;
  accepted_privacy?: boolean;
  accepted_cookies?: boolean;
  accepted_disclaimer_at?: string | null;
  accepted_privacy_at?: string | null;
  accepted_cookie_at?: string | null;
  last_login_at?: string | null;
}

interface Source {
  title: string;
  excerpt: string;
  url?: string;
  citation_tag?: string;
  publisher?: string;
}

interface Message {
  id?: number;
  role: 'user' | 'assistant';
  content: string;
  created_at: string;
  sources?: Source[];
}

interface Conversation {
  id: number;
  title: string;
  created_at: string;
  updated_at: string;
}

interface AuthResponse {
  token: string;
  user: User;
}

interface AccountSettingsResponse {
  user: User;
  active_sessions: number;
  current_policy_version: string;
}

function App() {
  const [token, setToken] = useState<string | null>(() => localStorage.getItem('auth_token'));
  const [user, setUser] = useState<User | null>(null);
  const [authMode, setAuthMode] = useState<'login' | 'register'>('login');
  const [authEmail, setAuthEmail] = useState('');
  const [authPassword, setAuthPassword] = useState('');
  const [authError, setAuthError] = useState('');
  const [authBusy, setAuthBusy] = useState(false);

  const [acceptDisclaimer, setAcceptDisclaimer] = useState(false);
  const [acceptPrivacy, setAcceptPrivacy] = useState(false);
  const [acceptCookies, setAcceptCookies] = useState(
    () => typeof window !== 'undefined' && localStorage.getItem(COOKIE_CONSENT_KEY) === 'accepted'
  );
  const [showCookieBanner, setShowCookieBanner] = useState(
    () => !(typeof window !== 'undefined' && localStorage.getItem(COOKIE_CONSENT_KEY) === 'accepted')
  );

  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<number | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [loadingConversations, setLoadingConversations] = useState(false);
  const [deletingConversationId, setDeletingConversationId] = useState<number | null>(null);

  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settingsData, setSettingsData] = useState<AccountSettingsResponse | null>(null);
  const [settingsError, setSettingsError] = useState('');
  const [settingsLoading, setSettingsLoading] = useState(false);

  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [passwordBusy, setPasswordBusy] = useState(false);
  const [passwordMessage, setPasswordMessage] = useState('');
  const [passwordError, setPasswordError] = useState('');

  const [deletePassword, setDeletePassword] = useState('');
  const [deleteConfirmText, setDeleteConfirmText] = useState('');
  const [deleteBusy, setDeleteBusy] = useState(false);
  const [deleteError, setDeleteError] = useState('');

  const authHeaders = useMemo(() => {
    if (!token) return {};
    return { Authorization: `Bearer ${token}` };
  }, [token]);

  const allPoliciesAccepted = acceptDisclaimer && acceptPrivacy && acceptCookies;

  useEffect(() => {
    if (!token) return;
    bootstrapSession().catch((err) => {
      console.error(err);
      clearSession();
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  const clearSession = () => {
    setToken(null);
    setUser(null);
    setConversations([]);
    setActiveConversationId(null);
    setMessages([]);
    setSettingsOpen(false);
    localStorage.removeItem('auth_token');
  };

  const bootstrapSession = async () => {
    const me = await fetchJson<{ user: User }>(`${API_BASE}/auth/me`, {
      headers: { ...authHeaders },
    });
    setUser(me.user);
    await refreshConversations(true);
  };

  const refreshConversations = async (pickFirstIfNone = false) => {
    if (!token) return;
    setLoadingConversations(true);
    try {
      const data = await fetchJson<{ conversations: Conversation[] }>(`${API_BASE}/conversations`, {
        headers: { ...authHeaders },
      });
      setConversations(data.conversations);
      if (data.conversations.length === 0) {
        setActiveConversationId(null);
        setMessages([]);
        return;
      }

      const exists = data.conversations.some((c) => c.id === activeConversationId);
      if (!exists || (pickFirstIfNone && activeConversationId === null)) {
        await openConversation(data.conversations[0].id);
      }
    } finally {
      setLoadingConversations(false);
    }
  };

  const openConversation = async (conversationId: number) => {
    if (!token) return;
    const data = await fetchJson<{ messages: Message[] }>(`${API_BASE}/conversations/${conversationId}/messages`, {
      headers: { ...authHeaders },
    });
    setActiveConversationId(conversationId);
    setMessages(data.messages);
  };

  const handleAcceptCookies = () => {
    localStorage.setItem(COOKIE_CONSENT_KEY, 'accepted');
    setAcceptCookies(true);
    setShowCookieBanner(false);
  };

  const handleAuthSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setAuthError('');
    if (!allPoliciesAccepted) {
      setAuthError('Please accept disclaimer, privacy terms, and essential cookies first.');
      return;
    }

    setAuthBusy(true);
    try {
      const endpoint = authMode === 'login' ? '/auth/login' : '/auth/register';
      const data = await fetchJson<AuthResponse>(`${API_BASE}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email: authEmail,
          password: authPassword,
          accept_disclaimer: acceptDisclaimer,
          accept_privacy: acceptPrivacy,
          accept_cookies: acceptCookies,
          policy_version: POLICY_VERSION,
        }),
      });
      localStorage.setItem('auth_token', data.token);
      localStorage.setItem(COOKIE_CONSENT_KEY, 'accepted');
      setToken(data.token);
      setUser(data.user);
      setAuthPassword('');
      setShowCookieBanner(false);
    } catch (err: any) {
      setAuthError(err.message || 'Authentication failed');
    } finally {
      setAuthBusy(false);
    }
  };

  const handleLogout = async () => {
    try {
      if (token) {
        await fetchJson<{ ok: boolean }>(`${API_BASE}/auth/logout`, {
          method: 'POST',
          headers: { ...authHeaders },
        });
      }
    } catch (err) {
      console.warn('logout warning', err);
    } finally {
      clearSession();
    }
  };

  const handleCreateConversation = async () => {
    if (!token) return;
    const created = await fetchJson<{ conversation: Conversation }>(`${API_BASE}/conversations`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...authHeaders },
      body: JSON.stringify({ title: 'New conversation' }),
    });
    const convo = created.conversation;
    setConversations((prev) => [convo, ...prev]);
    setActiveConversationId(convo.id);
    setMessages([]);
  };

  const handleDeleteConversation = async (conversationId: number) => {
    if (!token) return;
    const convo = conversations.find((c) => c.id === conversationId);
    const label = (convo?.title || 'this conversation').trim();
    const confirmed = window.confirm(`Delete "${label}"? This cannot be undone.`);
    if (!confirmed) return;

    setDeletingConversationId(conversationId);
    try {
      await fetchJson<{ ok: boolean }>(`${API_BASE}/conversations/${conversationId}`, {
        method: 'DELETE',
        headers: { ...authHeaders },
      });

      const remaining = conversations.filter((c) => c.id !== conversationId);
      setConversations(remaining);

      if (activeConversationId === conversationId) {
        if (remaining.length > 0) {
          await openConversation(remaining[0].id);
        } else {
          setActiveConversationId(null);
          setMessages([]);
        }
      }
    } finally {
      setDeletingConversationId(null);
    }
  };

  const openSettings = async () => {
    if (!token) return;
    setSettingsOpen(true);
    setSettingsLoading(true);
    setSettingsError('');
    try {
      const data = await fetchJson<AccountSettingsResponse>(`${API_BASE}/account/settings`, {
        headers: { ...authHeaders },
      });
      setSettingsData(data);
    } catch (err: any) {
      setSettingsError(err.message || 'Failed to load account settings');
    } finally {
      setSettingsLoading(false);
    }
  };

  const handleChangePassword = async (e: FormEvent) => {
    e.preventDefault();
    if (!token) return;
    setPasswordBusy(true);
    setPasswordMessage('');
    setPasswordError('');
    try {
      const res = await fetchJson<{ ok: boolean; message: string }>(`${API_BASE}/account/password`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json', ...authHeaders },
        body: JSON.stringify({
          current_password: currentPassword,
          new_password: newPassword,
        }),
      });
      setPasswordMessage(res.message || 'Password updated');
      setCurrentPassword('');
      setNewPassword('');
      clearSession();
    } catch (err: any) {
      setPasswordError(err.message || 'Unable to update password');
    } finally {
      setPasswordBusy(false);
    }
  };

  const handleDeleteAccount = async (e: FormEvent) => {
    e.preventDefault();
    if (!token) return;
    setDeleteBusy(true);
    setDeleteError('');
    try {
      await fetchJson<{ ok: boolean }>(`${API_BASE}/account`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json', ...authHeaders },
        body: JSON.stringify({
          password: deletePassword,
          confirm_text: deleteConfirmText,
        }),
      });
      clearSession();
    } catch (err: any) {
      setDeleteError(err.message || 'Unable to delete account');
    } finally {
      setDeleteBusy(false);
    }
  };

  const handleSend = async (e: FormEvent) => {
    e.preventDefault();
    if (!token || !activeConversationId || !input.trim() || isSending) return;

    const content = input.trim();
    const localUserMessage: Message = {
      role: 'user',
      content,
      created_at: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, localUserMessage]);
    setInput('');
    setIsSending(true);

    try {
      const data = await fetchJson<{
        conversation: Conversation;
        assistant_message: Message;
      }>(`${API_BASE}/conversations/${activeConversationId}/messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...authHeaders },
        body: JSON.stringify({ content, top_k: 40 }),
      });

      setMessages((prev) => [...prev, data.assistant_message]);
      setConversations((prev) => {
        const updated = prev.map((c) => (c.id === data.conversation.id ? data.conversation : c));
        updated.sort((a, b) => (a.updated_at < b.updated_at ? 1 : -1));
        return updated;
      });
    } catch (err: any) {
      const localErrorMessage: Message = {
        role: 'assistant',
        content: `Sorry, I could not process your request.\n${err.message || String(err)}`,
        created_at: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, localErrorMessage]);
    } finally {
      setIsSending(false);
    }
  };

  if (!token || !user) {
    return (
      <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col relative overflow-hidden">
        <div className="absolute inset-0 pointer-events-none">
          <div
            className="absolute inset-0 bg-center bg-cover"
            style={{
              backgroundImage: "url('/landing-bg.png')",
              opacity: 0.2,
              filter: 'blur(1.5px)',
              transform: 'scale(1.03)',
              backgroundPosition: 'center 62%',
            }}
          />
          <div className="absolute inset-0 bg-gradient-to-b from-slate-950/85 via-slate-950/75 to-slate-950/92" />
        </div>

        <main className="relative z-10 flex-1 flex items-center justify-center p-6">
          <div className="w-full max-w-[980px] grid md:grid-cols-2 gap-6 items-stretch">
            <div className="rounded-2xl border border-slate-800 bg-gradient-to-br from-slate-900 via-slate-900 to-blue-950 p-6">
              <div className="flex items-center gap-3 mb-5">
                <img src="/logo.png" alt="Lebanese Legal RAG Assistant" className="h-10 w-10 rounded object-cover" />
                <h1 className="text-2xl leading-tight font-semibold">Lebanese Legal Assistant</h1>
              </div>
              <p className="text-slate-300 mb-5 text-[15px]">
                Production-style legal RAG assistant with authenticated sessions, persistent chat history, and multi-conversation workspace.
              </p>
              <ul className="space-y-2 text-[14px] text-slate-300">
                <li>- Login-based private conversations</li>
                <li>- Conversation history saved in database</li>
                <li>- Multiple chat threads per user</li>
                <li>- Source-grounded legal answers</li>
              </ul>
            </div>

            <div className="rounded-2xl border border-slate-800 bg-slate-900 p-6">
              <h2 className="text-lg font-semibold mb-2">{authMode === 'login' ? 'Welcome back' : 'Create account'}</h2>
              <p className="text-slate-400 text-sm mb-5">
                {authMode === 'login' ? 'Login to continue your conversations.' : 'Register to start saving your legal chats.'}
              </p>

              <form onSubmit={handleAuthSubmit} className="space-y-4">
                <input
                  type="email"
                  placeholder="Email"
                  value={authEmail}
                  onChange={(e) => setAuthEmail(e.target.value)}
                  required
                  className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 outline-none focus:border-blue-500"
                />
                <input
                  type="password"
                  placeholder="Password"
                  value={authPassword}
                  onChange={(e) => setAuthPassword(e.target.value)}
                  required
                  minLength={10}
                  className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 outline-none focus:border-blue-500"
                />

                <div className="rounded-lg border border-slate-700 bg-slate-950/60 p-3 text-xs text-slate-300 space-y-2">
                  <label className="flex gap-2 items-start">
                    <input type="checkbox" checked={acceptDisclaimer} onChange={(e) => setAcceptDisclaimer(e.target.checked)} className="mt-1" />
                    <span>I accept the legal disclaimer: this tool is educational and not a substitute for legal counsel.</span>
                  </label>
                  <label className="flex gap-2 items-start">
                    <input type="checkbox" checked={acceptPrivacy} onChange={(e) => setAcceptPrivacy(e.target.checked)} className="mt-1" />
                    <span>I consent to processing my account and conversation data to provide the service.</span>
                  </label>
                  <label className="flex gap-2 items-start">
                    <input type="checkbox" checked={acceptCookies} onChange={(e) => setAcceptCookies(e.target.checked)} className="mt-1" />
                    <span>I accept essential cookies/local storage for authentication sessions and security state.</span>
                  </label>
                </div>

                {authError && <p className="text-sm text-red-400">{authError}</p>}
                <button
                  type="submit"
                  disabled={authBusy || !allPoliciesAccepted}
                  className="w-full rounded-lg bg-blue-600 hover:bg-blue-500 disabled:bg-blue-900 px-3 py-2 font-medium"
                >
                  {authBusy ? 'Please wait...' : authMode === 'login' ? 'Login' : 'Register'}
                </button>
              </form>

              <button
                onClick={() => {
                  setAuthMode((m) => (m === 'login' ? 'register' : 'login'));
                  setAuthError('');
                }}
                className="mt-4 text-sm text-blue-300 hover:text-blue-200"
              >
                {authMode === 'login' ? 'Need an account? Register' : 'Already have an account? Login'}
              </button>
            </div>
          </div>
        </main>

        {showCookieBanner && (
          <div className="relative z-20 mx-4 mb-4 rounded-xl border border-slate-700 bg-slate-900/95 p-4 text-sm text-slate-200">
            <p className="mb-3">
              We use essential cookies/local storage for secure login, session persistence, and account protection.
            </p>
            <button onClick={handleAcceptCookies} className="rounded-lg bg-blue-600 hover:bg-blue-500 px-4 py-2 text-white">
              Accept essential cookies
            </button>
          </div>
        )}

        <div className="relative z-10">
          <Footer compact />
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen overflow-hidden bg-slate-100 text-slate-900">
      <div className="h-full min-h-0 grid grid-cols-12">
        <aside className="col-span-12 md:col-span-3 lg:col-span-2 border-r border-slate-300 bg-slate-900 text-slate-100 p-3 flex flex-col gap-3">
          <button
            onClick={handleCreateConversation}
            className="w-full flex items-center justify-center gap-2 rounded-lg bg-blue-600 hover:bg-blue-500 px-3 py-2 text-sm font-medium"
          >
            <MessageSquarePlus className="h-4 w-4" />
            New chat
          </button>

          <div className="text-xs text-slate-400 px-1">Conversations</div>
          <div className="flex-1 overflow-auto space-y-1">
            {loadingConversations && <p className="text-xs text-slate-400 px-1">Loading...</p>}
            {conversations.map((conv) => (
              <div
                key={conv.id}
                className={`w-full text-left rounded-md px-3 py-2 text-sm ${
                  conv.id === activeConversationId ? 'bg-slate-700' : 'hover:bg-slate-800'
                }`}
              >
                <div className="flex items-center gap-2">
                  <button onClick={() => openConversation(conv.id)} className="flex-1 text-left truncate">
                    {conv.title}
                  </button>
                  <button
                    type="button"
                    onClick={() => handleDeleteConversation(conv.id)}
                    disabled={deletingConversationId === conv.id}
                    className="rounded p-1 text-slate-400 hover:text-red-300 hover:bg-slate-600 disabled:opacity-50"
                    aria-label={`Delete conversation ${conv.title}`}
                    title="Delete conversation"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </div>
              </div>
            ))}
          </div>

          <div className="border-t border-slate-700 pt-3 space-y-2">
            <div className="text-xs text-slate-400 truncate">{user.email}</div>
            <button
              onClick={openSettings}
              className="w-full flex items-center justify-center gap-2 rounded-lg border border-slate-600 hover:bg-slate-800 px-3 py-2 text-sm"
            >
              <Settings className="h-4 w-4" />
              Settings
            </button>
            <button
              onClick={handleLogout}
              className="w-full flex items-center justify-center gap-2 rounded-lg border border-slate-600 hover:bg-slate-800 px-3 py-2 text-sm"
            >
              <LogOut className="h-4 w-4" />
              Logout
            </button>
          </div>
        </aside>

        <main className="col-span-12 md:col-span-9 lg:col-span-10 flex flex-col min-h-0 overflow-hidden">
          <div className="border-b border-slate-300 px-4 py-3 bg-white flex items-center gap-2">
            <img src="/logo.png" alt="Lebanese Legal RAG Assistant" className="h-7 w-7 rounded object-cover" />
            <h2 className="font-semibold">
              {conversations.find((c) => c.id === activeConversationId)?.title || 'No conversation selected'}
            </h2>
          </div>

          <div className="flex-1 min-h-0 overflow-y-auto p-4 space-y-4 bg-slate-50 law-pattern">
            {messages.length === 0 ? (
              <div className="h-full flex items-center justify-center text-slate-500">Start your legal conversation.</div>
            ) : (
              messages.map((m, idx) => (
                <ChatMessage
                  key={`${m.created_at}-${idx}`}
                  message={m.content}
                  isUser={m.role === 'user'}
                  timestamp={new Date(m.created_at)}
                  sources={m.sources}
                />
              ))
            )}
            {isSending && <div className="text-sm text-slate-500 px-2">Assistant is thinking...</div>}
          </div>

          <form onSubmit={handleSend} className="border-t border-slate-300 bg-white p-3 flex items-center gap-2">
            <input
              type="text"
              placeholder="Ask a legal question..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={!activeConversationId}
              className="flex-1 rounded-lg border border-slate-300 px-3 py-2 outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-slate-100"
            />
            <button
              type="submit"
              disabled={!activeConversationId || !input.trim() || isSending}
              className="rounded-lg bg-blue-600 hover:bg-blue-500 disabled:bg-blue-300 text-white px-3 py-2"
            >
              <SendHorizontal className="h-4 w-4" />
            </button>
          </form>
        </main>
      </div>

      {settingsOpen && (
        <div className="fixed inset-0 z-50 bg-black/60 flex items-center justify-center p-4">
          <div className="w-full max-w-2xl max-h-[90vh] overflow-y-auto rounded-2xl bg-white border border-slate-200 shadow-xl">
            <div className="px-5 py-4 border-b border-slate-200 flex items-center justify-between">
              <h3 className="font-semibold text-lg">Account settings</h3>
              <button onClick={() => setSettingsOpen(false)} className="rounded p-1 hover:bg-slate-100" aria-label="Close settings">
                <X className="h-5 w-5" />
              </button>
            </div>

            <div className="p-5 space-y-6">
              {settingsLoading && <p className="text-sm text-slate-500">Loading settings...</p>}
              {settingsError && <p className="text-sm text-red-600">{settingsError}</p>}

              {settingsData && (
                <>
                  <section className="rounded-lg border border-slate-200 p-4">
                    <h4 className="font-semibold mb-2">Profile & compliance</h4>
                    <div className="text-sm text-slate-700 space-y-1">
                      <p>Email: {settingsData.user.email}</p>
                      <p>Account created: {formatDate(settingsData.user.created_at)}</p>
                      <p>Last login: {formatDate(settingsData.user.last_login_at)}</p>
                      <p>Active sessions: {settingsData.active_sessions}</p>
                      <p>Policy version: {settingsData.user.policy_version || settingsData.current_policy_version}</p>
                      <p>Disclaimer accepted: {settingsData.user.accepted_disclaimer ? 'Yes' : 'No'}</p>
                      <p>Privacy accepted: {settingsData.user.accepted_privacy ? 'Yes' : 'No'}</p>
                      <p>Cookies accepted: {settingsData.user.accepted_cookies ? 'Yes' : 'No'}</p>
                    </div>
                  </section>

                  <section className="rounded-lg border border-slate-200 p-4">
                    <h4 className="font-semibold mb-2">Security</h4>
                    <form onSubmit={handleChangePassword} className="space-y-3">
                      <input
                        type="password"
                        placeholder="Current password"
                        value={currentPassword}
                        onChange={(e) => setCurrentPassword(e.target.value)}
                        required
                        className="w-full rounded-lg border border-slate-300 px-3 py-2"
                      />
                      <input
                        type="password"
                        placeholder="New password"
                        value={newPassword}
                        onChange={(e) => setNewPassword(e.target.value)}
                        required
                        className="w-full rounded-lg border border-slate-300 px-3 py-2"
                      />
                      <p className="text-xs text-slate-500">
                        Use at least 10 characters with uppercase, lowercase, number, and special character.
                      </p>
                      {passwordError && <p className="text-sm text-red-600">{passwordError}</p>}
                      {passwordMessage && <p className="text-sm text-green-600">{passwordMessage}</p>}
                      <button
                        type="submit"
                        disabled={passwordBusy}
                        className="rounded-lg bg-blue-600 hover:bg-blue-500 text-white px-4 py-2 disabled:bg-blue-300"
                      >
                        {passwordBusy ? 'Updating...' : 'Update password'}
                      </button>
                    </form>
                  </section>

                  <section className="rounded-lg border border-red-200 bg-red-50 p-4">
                    <h4 className="font-semibold text-red-800 mb-2">Danger zone</h4>
                    <form onSubmit={handleDeleteAccount} className="space-y-3">
                      <p className="text-sm text-red-700">
                        Deleting your account permanently removes profile, sessions, and all conversations.
                      </p>
                      <input
                        type="password"
                        placeholder="Current password"
                        value={deletePassword}
                        onChange={(e) => setDeletePassword(e.target.value)}
                        required
                        className="w-full rounded-lg border border-red-300 px-3 py-2"
                      />
                      <input
                        type="text"
                        placeholder='Type DELETE to confirm'
                        value={deleteConfirmText}
                        onChange={(e) => setDeleteConfirmText(e.target.value)}
                        required
                        className="w-full rounded-lg border border-red-300 px-3 py-2"
                      />
                      {deleteError && <p className="text-sm text-red-700">{deleteError}</p>}
                      <button
                        type="submit"
                        disabled={deleteBusy}
                        className="rounded-lg bg-red-600 hover:bg-red-500 text-white px-4 py-2 disabled:bg-red-300"
                      >
                        {deleteBusy ? 'Deleting...' : 'Delete account'}
                      </button>
                    </form>
                  </section>
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function formatDate(value?: string | null): string {
  if (!value) return 'N/A';
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString();
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const text = await res.text();
    let detail = text;
    try {
      const parsed = JSON.parse(text);
      detail = parsed.detail || parsed.message || text;
    } catch {
      // no-op
    }
    throw new Error(detail || `HTTP ${res.status}`);
  }
  return (await res.json()) as T;
}

export default App;
