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
.agent-badge {
    display: inline-block;
    padding: 2px 10px; border-radius: 999px;
    font-size: 11px; margin: 2px; font-weight: 500;
}
.agent-research { background:#E1F5EE; color:#085041; border:1px solid #9FE1CB; }
.agent-code     { background:#EEEDFE; color:#26215C; border:1px solid #AFA9EC; }
.agent-general  { background:#FAEEDA; color:#412402; border:1px solid #EF9F27; }
.stat-card {
    background: #f7f7f5; border-radius: 8px;
    padding: 12px 16px; text-align: center;
}
.stat-card .val { font-size: 22px; font-weight: 600; }
.stat-card .lbl { font-size: 11px; color: #6b6b64; }
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────
if "session_id"   not in st.session_state: st.session_state.session_id   = str(uuid.uuid4())[:8].upper()
if "messages"     not in st.session_state: st.session_state.messages     = []
if "total_iters"  not in st.session_state: st.session_state.total_iters  = 0
if "total_tools"  not in st.session_state: st.session_state.total_tools  = 0
if "all_sessions" not in st.session_state: st.session_state.all_sessions = {}
if "streaming"    not in st.session_state: st.session_state.streaming    = False
if "agent_mode"   not in st.session_state: st.session_state.agent_mode   = "multi"


# ── Streaming helper ───────────────────────────────────────────────────────────
def stream_response(api_base: str, message: str, session_id: str):
    url = f"{api_base.rstrip('/')}/chat/stream"
    try:
        with requests.post(
            url,
            json={"message": message, "session_id": session_id},
            stream=True,
            timeout=120,
        ) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
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
                    for word in data["response"].split(" "):
                        yield word + " "
                        time.sleep(0.01)
    except requests.exceptions.ConnectionError:
        yield "\n\n⚠️ Cannot reach backend."
    except Exception as e:
        yield f"\n\n⚠️ Error: {e}"


# ── API calls ──────────────────────────────────────────────────────────────────
def fetch_single_agent(api_base: str, message: str, session_id: str) -> dict:
    resp = requests.post(
        f"{api_base.rstrip('/')}/chat",
        json={"message": message, "session_id": session_id},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_multi_agent(api_base: str, message: str, session_id: str) -> dict:
    resp = requests.post(
        f"{api_base.rstrip('/')}/multi-agent/chat",
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
        "bedrock", "groq", "openai", "anthropic",
        "ollama", "huggingface", "azure_openai"
    ])

    st.divider()

    # ── Agent mode ─────────────────────────────────────────────────────────────
    st.markdown("**Agent mode**")
    agent_mode = st.radio(
        label="agent_mode_radio",
        options=["multi", "single"],
        format_func=lambda x: "Multi-agent (supervisor)" if x == "multi" else "Single agent",
        index=0,
        label_visibility="collapsed",
    )
    st.session_state.agent_mode = agent_mode

    if agent_mode == "multi":
        st.caption("Supervisor routes to: research · code · general")
    else:
        st.session_state.streaming = st.toggle(
            "Stream responses",
            value=st.session_state.streaming,
        )
        st.caption(
            "Endpoint: `/chat/stream`" if st.session_state.streaming
            else "Endpoint: `/chat`"
        )

    st.divider()

    # ── Health check ───────────────────────────────────────────────────────────
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
mode_label = (
    "multi-agent" if st.session_state.agent_mode == "multi"
    else ("streaming" if st.session_state.streaming else "standard")
)
st.markdown(
    f"**Session** `{st.session_state.session_id}` · "
    f"provider: `{provider}` · mode: `{mode_label}`"
)

# ── Message history ────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        # Show which agent handled it (multi-agent mode only)
        if msg.get("agent_used"):
            agent = msg["agent_used"]
            css   = f"agent-{agent}"
            st.markdown(
                f'<span class="agent-badge {css}">{agent} agent</span>',
                unsafe_allow_html=True
            )

        if msg.get("tools"):
            badges = " ".join(
                f'<span class="tool-badge">{t}</span>'
                for t in msg["tools"]
            )
            st.markdown(badges, unsafe_allow_html=True)

        if msg.get("meta"):
            st.caption(msg["meta"])


# ── Input ──────────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Message the agent…"):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):

        # ── Multi-agent path ───────────────────────────────────────────────────
        if st.session_state.agent_mode == "multi":
            with st.spinner("Routing to specialist…"):
                try:
                    data       = fetch_multi_agent(api_base, prompt, st.session_state.session_id)
                    full_resp  = data.get("response", "")
                    agent_used = data.get("agent_used", "")
                    tool_names = [t.get("tool","tool") for t in data.get("tool_results",[])]
                    iters      = data.get("iteration_count", 1)

                    st.write(full_resp)

                    # Show agent badge
                    if agent_used:
                        css = f"agent-{agent_used}"
                        st.markdown(
                            f'<span class="agent-badge {css}">{agent_used} agent</span>',
                            unsafe_allow_html=True
                        )

                except requests.exceptions.ConnectionError:
                    full_resp  = "Cannot reach backend."
                    agent_used = ""
                    tool_names = []
                    iters      = 0
                    st.error(full_resp)
                except Exception as e:
                    full_resp  = f"Error: {e}"
                    agent_used = ""
                    tool_names = []
                    iters      = 0
                    st.error(full_resp)

        # ── Single agent streaming path ────────────────────────────────────────
        elif st.session_state.streaming:
            full_resp  = st.write_stream(
                stream_response(api_base, prompt, st.session_state.session_id)
            )
            agent_used = ""
            tool_names = []
            iters      = 1

        # ── Single agent standard path ─────────────────────────────────────────
        else:
            with st.spinner("Agent thinking…"):
                try:
                    data       = fetch_single_agent(api_base, prompt, st.session_state.session_id)
                    full_resp  = data.get("response") or data.get("final_response","")
                    tool_names = [t.get("tool","tool") for t in data.get("tool_results",[])]
                    iters      = data.get("iteration_count", 1)
                    agent_used = ""

                    st.write(full_resp)

                    if tool_names:
                        badges = " ".join(
                            f'<span class="tool-badge">{t}</span>'
                            for t in tool_names
                        )
                        st.markdown(badges, unsafe_allow_html=True)

                except requests.exceptions.ConnectionError:
                    full_resp  = "Cannot reach backend."
                    tool_names = []
                    iters      = 0
                    agent_used = ""
                    st.error(full_resp)
                except Exception as e:
                    full_resp  = f"Error: {e}"
                    tool_names = []
                    iters      = 0
                    agent_used = ""
                    st.error(full_resp)

        # ── Save to history ────────────────────────────────────────────────────
        meta = (
            f"{datetime.now().strftime('%H:%M')} · "
            f"{iters} steps · "
            f"{provider} · "
            f"{mode_label}"
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
            "role":       "assistant",
            "content":    full_resp,
            "tools":      tool_names,
            "agent_used": agent_used,
            "meta":       meta,
        })

    st.rerun()