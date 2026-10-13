
import streamlit as st
import requests

st.set_page_config(
    page_title="Multi-Agent RAG Assistant",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 Multi-Agent RAG Assistant")
st.caption("LangGraph + FastAPI + FAISS")

api_url = st.sidebar.text_input(
    "FastAPI endpoint",
    value="http://127.0.0.1:8000/query",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("trace"):
            with st.expander("Agent execution trace"):
                st.write(message["trace"])

question = st.chat_input("Ask about orders, returns, refunds, or accounts...")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Agents are processing your question..."):
            try:
                response = requests.post(
                    api_url,
                    json={"query": question},
                    timeout=120,
                )
                response.raise_for_status()
                data = response.json()

                if isinstance(data, dict):
                    answer = (
                        data.get("final_answer")
                        or data.get("answer")
                        or data.get("response")
                        or data.get("output")
                    )

                    if not answer and isinstance(data.get("result"), dict):
                        answer = (
                            data["result"].get("final_answer")
                            or data["result"].get("answer")
                        )

                    trace = data.get("trace", [])

                elif isinstance(data, str):
                    answer = data
                    trace = []

                else:
                    answer = None
                    trace = []

                if not answer:
                    answer = (
                        "The API responded, but its answer field could "
                        "not be identified. Inspect the API response."
                    )
                    st.json(data)

                st.markdown(answer)

                if trace:
                    with st.expander("Agent execution trace"):
                        st.write(trace)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "trace": trace,
                })

            except requests.RequestException as error:
                st.error(
                    "Could not connect to FastAPI. Check that the API "
                    "server is running and the endpoint URL is correct."
                )
                st.code(str(error))

st.sidebar.markdown("---")
if st.sidebar.button("Clear conversation"):
    st.session_state.messages = []
    st.rerun()
