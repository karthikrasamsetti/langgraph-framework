import streamlit as st
import requests
import json
import uuid
import time
from datetime import datetime

st.set_page_config(
    page_title="LangGraph Agent",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.tool-badge {
    display: inline-block;
    background: #EAF3DE; color: #27500A;
    border: 1px solid #97C459;
    border-radius: 999px; padding: 2px 10px;
    font-size: 11px; margin: 2px;
}
.stat-card {
    background: #f7f7f5; border-radius: 8px;
    padding: 12px 16px; text-align: center;
}
.stat-card .val { font-size: 22px; font-weight: 600; }
.stat-card .lbl { font-size: 11px; color: #6b6b64; }
.node-pill {
    display: inline-block; padding: 2px 8px;
    border-radius: 999px; font-size: 11px;
    border: 1px solid #e5e4dc; background: #f7f7f5;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────
if "session_id"   not in st.session_state: st.session_state.session_id   = str(uuid.uuid4())[:8].upper()
if "messages"     not in st.session_state: st.session_state.messages     = []
if "total_iters"  not in st.session_state: st.session_state.total_iters  = 0
if "total_tools"  not in st.session_state: st.session_state.total_tools  = 0
if "all_sessions" not in st.session_state: st.session_state.all_sessions = {}
if "streaming"    not in st.session_state: st.session_state.streaming    = True


# ── Streaming helper ───────────────────────────────────────────────────────────
def stream_response(api_base: str, message: str, session_id: str):
    """
    Calls /chat/stream and yields text chunks one by one.
    st.write_stream() consumes this generator and renders
    tokens in real time as they arrive.
    """
    url = f"{api_base.rstrip('/')}/chat/stream"

    try:
        with requests.post(
            url,
            json={"message": message, "session_id": session_id},
            stream=True,       # ← tells requests not to buffer the response
            timeout=120,
        ) as resp:
            resp.raise_for_status()

            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue

                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line

                # SSE format is:  data: {...json...}
                if not line.startswith("data:"):
                    continue

                payload = line[5:].strip()   # strip "data: " prefix

                if payload == "[DONE]":
                    break

                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                if "error" in data:
                    yield f"\n\n⚠️ Error: {data['error']}"
                    break

                if "response" in data:
                    # Yield word by word so Streamlit renders
                    # progressively instead of all at once
                    for word in data["response"].split(" "):
                        yield word + " "
                        time.sleep(0.01)   # tiny delay for visual smoothness

    except requests.exceptions.ConnectionError:
        yield "\n\n⚠️ Cannot reach backend. Run: `uvicorn api.server:app --reload`"
    except requests.exceptions.Timeout:
        yield "\n\n⚠️ Request timed out after 120 seconds."
    except Exception as e:
        yield f"\n\n⚠️ Unexpected error: {e}"


# ── Non-streaming fallback ─────────────────────────────────────────────────────
def fetch_response(api_base: str, message: str, session_id: str) -> dict:
    """Falls back to /chat when streaming is toggled off."""
    resp = requests.post(
        f"{api_base.rstrip('/')}/chat",
        json={"message": message, "session_id": session_id},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("LangGraph Agent")
    st.caption("Production Framework UI")

    st.divider()

    api_base = st.text_input("Backend URL", value="http://localhost:8000")

    provider = st.selectbox("LLM Provider", [
        "groq", "openai", "anthropic",
        "bedrock", "ollama", "huggingface", "azure_openai"
    ])

    # ── Streaming toggle ───────────────────────────────────────────────────────
    st.session_state.streaming = st.toggle(
        "Stream responses",
        value=st.session_state.streaming,
        help="Uses /chat/stream for real-time token output. Turn off to use /chat instead."
    )

    st.caption(
        "Endpoint: `/chat/stream`" if st.session_state.streaming
        else "Endpoint: `/chat`"
    )

    # ── Health check ───────────────────────────────────────────────────────────
    st.divider()
    try:
        r = requests.get(f"{api_base}/health", timeout=2)
        if r.ok:
            d = r.json()
            st.success(f"Connected · {d.get('provider','')}")
            if d.get("langsmith_enabled"):
                st.info(f"LangSmith: {d.get('langsmith_project','')}")
        else:
            st.error("Backend error")
    except:
        st.warning("Backend offline — start uvicorn")

    st.divider()

    # ── Stats ──────────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""<div class="stat-card">
            <div class="val">{st.session_state.total_iters}</div>
            <div class="lbl">total steps</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="stat-card">
            <div class="val">{st.session_state.total_tools}</div>
            <div class="lbl">tool calls</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Session management ─────────────────────────────────────────────────────
    st.markdown("**Sessions**")
    if st.button("+ New Session", use_container_width=True):
        st.session_state.all_sessions[st.session_state.session_id] = {
            "messages":  st.session_state.messages.copy(),
            "iters":     st.session_state.total_iters,
            "tools":     st.session_state.total_tools,
        }
        st.session_state.session_id  = str(uuid.uuid4())[:8].upper()
        st.session_state.messages    = []
        st.session_state.total_iters = 0
        st.session_state.total_tools = 0
        st.rerun()

    for sid, data in st.session_state.all_sessions.items():
        last = data["messages"][-1]["content"][:30]+"…" if data["messages"] else "empty"
        if st.button(f"{sid}  ·  {last}", key=f"sess_{sid}", use_container_width=True):
            st.session_state.all_sessions[st.session_state.session_id] = {
                "messages":  st.session_state.messages.copy(),
                "iters":     st.session_state.total_iters,
                "tools":     st.session_state.total_tools,
            }
            st.session_state.session_id  = sid
            st.session_state.messages    = data["messages"]
            st.session_state.total_iters = data["iters"]
            st.session_state.total_tools = data["tools"]
            st.rerun()


# ── Main area ──────────────────────────────────────────────────────────────────
mode_label = "streaming" if st.session_state.streaming else "standard"
st.markdown(
    f"**Session** `{st.session_state.session_id}` · "
    f"provider: `{provider}` · mode: `{mode_label}`"
)

# Message history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("tools"):
            badges = " ".join(
                f'<span class="tool-badge">{t}</span>'
                for t in msg["tools"]
            )
            st.markdown(badges, unsafe_allow_html=True)
        if msg.get("meta"):
            st.caption(msg["meta"])


# ── Input + response ───────────────────────────────────────────────────────────
if prompt := st.chat_input("Message the agent…"):

    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Agent response
    with st.chat_message("assistant"):

        if st.session_state.streaming:
            # ── Streaming path ─────────────────────────────────────────────────
            # st.write_stream consumes the generator and renders
            # each yielded chunk in real time
            full_response = st.write_stream(
                stream_response(api_base, prompt, st.session_state.session_id)
            )
            tool_names = []   # streaming doesn't return tool metadata
            iters      = 1

        else:
            # ── Non-streaming path ─────────────────────────────────────────────
            with st.spinner("Agent thinking…"):
                try:
                    data          = fetch_response(api_base, prompt, st.session_state.session_id)
                    full_response = data.get("response") or data.get("final_response", "")
                    tool_names    = [t.get("tool", "tool") for t in data.get("tool_results", [])]
                    iters         = data.get("iteration_count", 1)

                    st.write(full_response)

                    if tool_names:
                        badges = " ".join(
                            f'<span class="tool-badge">{t}</span>'
                            for t in tool_names
                        )
                        st.markdown(badges, unsafe_allow_html=True)

                except requests.exceptions.ConnectionError:
                    full_response = "Cannot reach backend."
                    tool_names    = []
                    iters         = 0
                    st.error(full_response)

                except Exception as e:
                    full_response = f"Error: {e}"
                    tool_names    = []
                    iters         = 0
                    st.error(full_response)

        # ── Save to history ────────────────────────────────────────────────────
        meta = (
            f"{datetime.now().strftime('%H:%M')} · "
            f"{iters} steps · "
            f"{provider} · "
            f"{'streamed' if st.session_state.streaming else 'standard'}"
        )
        if tool_names:
            st.markdown(
                " ".join(f'<span class="tool-badge">{t}</span>' for t in tool_names),
                unsafe_allow_html=True
            )
        st.caption(meta)

        st.session_state.total_iters += iters
        st.session_state.total_tools += len(tool_names)
        st.session_state.messages.append({
            "role":    "assistant",
            "content": full_response,
            "tools":   tool_names,
            "meta":    meta,
        })

    st.rerun()