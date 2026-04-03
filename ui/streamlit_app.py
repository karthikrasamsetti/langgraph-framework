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
.agent-badge { display:inline-block; padding:2px 10px; border-radius:999px;
    font-size:11px; margin:2px; font-weight:500; }
.agent-research { background:#E1F5EE; color:#085041; border:1px solid #9FE1CB; }
.agent-code     { background:#EEEDFE; color:#26215C; border:1px solid #AFA9EC; }
.agent-general  { background:#FAEEDA; color:#412402; border:1px solid #EF9F27; }
.stat-card { background:#f7f7f5; border-radius:8px; padding:12px 16px; text-align:center; }
.stat-card .val { font-size:22px; font-weight:600; }
.stat-card .lbl { font-size:11px; color:#6b6b64; }
.pending-box { background:#FFF8E1; border:1px solid #EF9F27;
    border-radius:8px; padding:16px; margin:8px 0; }
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
if "pending"      not in st.session_state: st.session_state.pending      = None


# ── API helpers ────────────────────────────────────────────────────────────────
def stream_response(api_base: str, message: str, session_id: str):
    url = f"{api_base.rstrip('/')}/chat/stream"
    try:
        with requests.post(
                url,
                json={"message": message, "session_id": session_id},
                headers=get_headers(),
                stream=True, timeout=120,
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


def fetch_single(api_base: str, message: str, session_id: str) -> dict:
    resp = requests.post(
        f"{api_base.rstrip('/')}/chat",
        json={"message": message, "session_id": session_id},
        headers=get_headers(),
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_multi(api_base: str, message: str, session_id: str) -> dict:
    resp = requests.post(
        f"{api_base.rstrip('/')}/multi-agent/chat",
        json={"message": message, "session_id": session_id},
        headers=get_headers(),
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def check_pending(api_base: str, session_id: str, mode: str) -> dict:
    try:
        resp = requests.get(
            f"{api_base.rstrip('/')}/chat/pending/{session_id}",
            params={"mode": mode},
            headers=get_headers(),
            timeout=5,
        )
        if resp.ok:
            return resp.json()
    except Exception:
        pass
    return {}


def resume_session(api_base: str, session_id: str,
                   action: str, mode: str,
                   next_agent: str = None) -> dict:
    payload = {"action": action, "mode": mode}
    if next_agent:
        payload["next_agent"] = next_agent
    resp = requests.post(
        f"{api_base.rstrip('/')}/chat/resume/{session_id}",
        json=payload,
        headers=get_headers(),
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def save_agent_message(full_resp: str, tool_names: list,
                       agent_used: str, iters: int,
                       provider: str, mode_label: str):
    meta = (
        f"{datetime.now().strftime('%H:%M')} · "
        f"{iters} steps · {provider} · {mode_label}"
    )
    st.session_state.total_iters += iters
    st.session_state.total_tools += len(tool_names)
    st.session_state.messages.append({
        "role":       "assistant",
        "content":    full_resp,
        "tools":      tool_names,
        "agent_used": agent_used,
        "meta":       meta,
    })
    st.session_state.pending = None


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("LangGraph Agent")
    st.caption("Production Framework UI")
    st.divider()

    api_base = st.text_input("Backend URL", value="http://localhost:8000")

    api_key = st.text_input(
        "API Key",
        value="",
        type="password",
        placeholder="sk-xxx (leave empty if auth disabled)",
    )

    def get_headers() -> dict:
        """Returns auth headers if key is set."""
        if api_key:
            return {"X-API-Key": api_key}
        return {}

    provider = st.selectbox("LLM Provider", [
        "bedrock", "groq", "openai", "anthropic",
        "ollama", "huggingface", "azure_openai"
    ])

    st.divider()

    st.markdown("**Agent mode**")
    agent_mode = st.radio(
        label="mode",
        options=["multi", "single"],
        format_func=lambda x: "Multi-agent (supervisor)" if x=="multi" else "Single agent",
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

    st.markdown("**Sessions**")
    if st.button("+ New Session", use_container_width=True):
        st.session_state.all_sessions[st.session_state.session_id] = {
            "messages": st.session_state.messages.copy(),
            "iters":    st.session_state.total_iters,
            "tools":    st.session_state.total_tools,
        }
        st.session_state.session_id  = str(uuid.uuid4())[:8].upper()
        st.session_state.messages    = []
        st.session_state.total_iters = 0
        st.session_state.total_tools = 0
        st.session_state.pending     = None
        st.rerun()

    for sid, data in st.session_state.all_sessions.items():
        last = data["messages"][-1]["content"][:30]+"…" if data["messages"] else "empty"
        if st.button(f"{sid}  ·  {last}", key=f"sess_{sid}", use_container_width=True):
            st.session_state.all_sessions[st.session_state.session_id] = {
                "messages": st.session_state.messages.copy(),
                "iters":    st.session_state.total_iters,
                "tools":    st.session_state.total_tools,
            }
            st.session_state.session_id  = sid
            st.session_state.messages    = data["messages"]
            st.session_state.total_iters = data["iters"]
            st.session_state.total_tools = data["tools"]
            st.session_state.pending     = None
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


# ── Pending approval UI ────────────────────────────────────────────────────────
# Check if current session has a pending interrupt
pending = check_pending(
    api_base,
    st.session_state.session_id,
    st.session_state.agent_mode
)
pending_node = pending.get("pending_node")

if pending_node:
    st.divider()

    if pending_node == "tool_executor":
        # ── Single agent tool approval ─────────────────────────────────────────
        st.markdown("### Approval needed — tool call")
        st.markdown(pending.get("message","Agent wants to run a tool"))

        tools = pending.get("pending_tools", [])
        for t in tools:
            with st.container():
                col_a, col_b = st.columns([1,3])
                with col_a:
                    st.markdown(f"**Tool**")
                    st.markdown(f"**Args**")
                with col_b:
                    st.code(t["name"])
                    st.code(json.dumps(t["args"], indent=2))

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✓  Approve", type="primary",
                         use_container_width=True, key="approve_tool"):
                with st.spinner("Resuming agent…"):
                    try:
                        data      = resume_session(
                            api_base, st.session_state.session_id,
                            "approve", "single"
                        )
                        full_resp = data.get("response","")
                        tool_names = [
                            t.get("tool","tool")
                            for t in data.get("tool_results",[])
                        ]
                        save_agent_message(
                            full_resp, tool_names, "",
                            data.get("iteration_count",1),
                            provider, "approved"
                        )
                    except Exception as e:
                        st.error(f"Resume failed: {e}")
                st.rerun()

        with col2:
            if st.button("✗  Reject", type="secondary",
                         use_container_width=True, key="reject_tool"):
                try:
                    data = resume_session(
                        api_base, st.session_state.session_id,
                        "reject", "single"
                    )
                    save_agent_message(
                        data.get("response","Action rejected."),
                        [], "", 0, provider, "rejected"
                    )
                except Exception as e:
                    st.error(f"Reject failed: {e}")
                st.rerun()

    elif pending_node == "supervisor_run_agent":
        # ── Multi-agent routing approval ───────────────────────────────────────
        next_agent = pending.get("next_agent","unknown")
        st.markdown("### Approval needed — agent routing")
        st.markdown(pending.get("message","Supervisor made a routing decision"))

        col_a, col_b = st.columns([1,2])
        with col_a:
            st.markdown("**Chosen agent**")
        with col_b:
            css = f"agent-{next_agent}"
            st.markdown(
                f'<span class="agent-badge {css}">{next_agent} agent</span>',
                unsafe_allow_html=True
            )

        override = st.selectbox(
            "Override routing",
            options=["research","code","general"],
            index=["research","code","general"].index(next_agent)
                  if next_agent in ["research","code","general"] else 0,
            key="override_select"
        )

        is_override = override != next_agent
        btn_label   = f"Override → {override}" if is_override else "✓  Approve"

        col1, col2 = st.columns(2)
        with col1:
            if st.button(btn_label, type="primary",
                         use_container_width=True, key="approve_route"):
                action = "override" if is_override else "approve"
                with st.spinner("Resuming supervisor…"):
                    try:
                        data = resume_session(
                            api_base, st.session_state.session_id,
                            action, "multi",
                            next_agent=override if is_override else None,
                        )
                        full_resp  = data.get("response","")
                        agent_used = data.get("agent_used", override)
                        save_agent_message(
                            full_resp, [], agent_used,
                            data.get("iteration_count",1),
                            provider, action
                        )
                    except Exception as e:
                        st.error(f"Resume failed: {e}")
                st.rerun()

        with col2:
            if st.button("✗  Reject", type="secondary",
                         use_container_width=True, key="reject_route"):
                try:
                    data = resume_session(
                        api_base, st.session_state.session_id,
                        "reject", "multi"
                    )
                    save_agent_message(
                        data.get("response","Action rejected."),
                        [], "", 0, provider, "rejected"
                    )
                except Exception as e:
                    st.error(f"Reject failed: {e}")
                st.rerun()

    st.divider()


# ── Chat input ─────────────────────────────────────────────────────────────────
# Disable input while waiting for approval
input_disabled = pending_node is not None
placeholder    = "Approve or reject the pending action above first…" \
                 if input_disabled else "Message the agent…"

if prompt := st.chat_input(placeholder, disabled=input_disabled):

    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):

        # ── Multi-agent ────────────────────────────────────────────────────────
        if st.session_state.agent_mode == "multi":
            with st.spinner("Routing to specialist…"):
                try:
                    data       = fetch_multi(api_base, prompt, st.session_state.session_id)
                    error      = data.get("error","")
                    full_resp  = data.get("response","")
                    agent_used = data.get("agent_used","")
                    iters      = data.get("iteration_count",1)
                    tool_names = []

                    if error and error.startswith("pending:"):
                        st.info("Waiting for your approval — see above")
                        full_resp = ""
                    else:
                        st.write(full_resp)
                        if agent_used:
                            css = f"agent-{agent_used}"
                            st.markdown(
                                f'<span class="agent-badge {css}">{agent_used} agent</span>',
                                unsafe_allow_html=True
                            )

                except requests.exceptions.ConnectionError:
                    full_resp = "Cannot reach backend."
                    agent_used = ""; iters = 0; tool_names = []
                    st.error(full_resp)
                except Exception as e:
                    full_resp = f"Error: {e}"
                    agent_used = ""; iters = 0; tool_names = []
                    st.error(full_resp)

        # ── Single agent streaming ─────────────────────────────────────────────
        elif st.session_state.streaming:
            full_resp  = st.write_stream(
                stream_response(api_base, prompt, st.session_state.session_id)
            )
            agent_used = ""; tool_names = []; iters = 1

        # ── Single agent standard ──────────────────────────────────────────────
        else:
            with st.spinner("Agent thinking…"):
                try:
                    data       = fetch_single(api_base, prompt, st.session_state.session_id)
                    error      = data.get("error","")
                    full_resp  = data.get("response") or data.get("final_response","")
                    tool_names = [t.get("tool","tool") for t in data.get("tool_results",[])]
                    iters      = data.get("iteration_count",1)
                    agent_used = ""

                    if error and error.startswith("pending:"):
                        st.info("Waiting for your approval — see above")
                        full_resp = ""
                    else:
                        st.write(full_resp)
                        if tool_names:
                            badges = " ".join(
                                f'<span class="tool-badge">{t}</span>'
                                for t in tool_names
                            )
                            st.markdown(badges, unsafe_allow_html=True)

                except requests.exceptions.ConnectionError:
                    full_resp = "Cannot reach backend."
                    tool_names = []; iters = 0; agent_used = ""
                    st.error(full_resp)
                except Exception as e:
                    full_resp = f"Error: {e}"
                    tool_names = []; iters = 0; agent_used = ""
                    st.error(full_resp)

        # ── Save message ───────────────────────────────────────────────────────
        meta = (
            f"{datetime.now().strftime('%H:%M')} · "
            f"{iters} steps · {provider} · {mode_label}"
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