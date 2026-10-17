import os

import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(page_title="Gemini Chat (Text + Image)", page_icon="✨")
st.title("✨ Gemini Chatbot")
st.caption("Send text, images, or both. The chat remembers earlier messages.")


# ---------- Sidebar ----------
def default_key() -> str:
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.getenv("GEMINI_API_KEY", "")


with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Gemini API key", type="password", value=default_key())
    model_name = st.text_input(
        "Model",
        value="gemini-3.5-flash",
        help="Any Gemini model that accepts images, e.g. gemini-3.5-flash or gemini-2.5-flash.",
    )
    system_prompt = st.text_area(
        "System instruction (optional)",
        value="You are a helpful assistant. When an image is provided, describe and reason about it accurately.",
    )
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if not api_key:
    st.info("AIzaSyAzI-ND91fmvHPkg8WFdggF9AcOiG4Xvx8")
    st.stop()

client = genai.Client(api_key=api_key)

# ---------- State ----------
# Each message: {"role": "user" | "model", "text": str, "images": [(bytes, mime_type), ...]}
if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------- Helpers ----------
def build_contents(messages):
    contents = []
    for m in messages:
        parts = [types.Part.from_bytes(data=b, mime_type=mt) for b, mt in m["images"]]
        if m["text"]:
            parts.append(types.Part.from_text(text=m["text"]))
        contents.append(types.Content(role=m["role"], parts=parts))
    return contents


def stream_reply(contents):
    config = types.GenerateContentConfig(system_instruction=system_prompt or None)
    for chunk in client.models.generate_content_stream(
        model=model_name, contents=contents, config=config
    ):
        if chunk.text:
            yield chunk.text


def render_message(m):
    with st.chat_message("assistant" if m["role"] == "model" else "user"):
        for img_bytes, _ in m["images"]:
            st.image(img_bytes, width=300)
        if m["text"]:
            st.markdown(m["text"])


# ---------- Show history ----------
for m in st.session_state.messages:
    render_message(m)

# ---------- Input ----------
prompt = st.chat_input(
    "Ask something, or attach an image...",
    accept_file="multiple",
    file_type=["png", "jpg", "jpeg", "webp"],
)

if prompt:
    text = (prompt.text or "").strip()
    images = [(f.getvalue(), f.type or "image/jpeg") for f in (prompt.files or [])]

    if not text and images:
        text = "Describe this image."

    user_msg = {"role": "user", "text": text, "images": images}
    st.session_state.messages.append(user_msg)
    render_message(user_msg)

    with st.chat_message("assistant"):
        try:
            reply = st.write_stream(stream_reply(build_contents(st.session_state.messages)))
            st.session_state.messages.append({"role": "model", "text": reply, "images": []})
        except Exception as e:
            st.session_state.messages.pop()  # drop the failed user turn
            st.error(f"Gemini API error: {e}")
