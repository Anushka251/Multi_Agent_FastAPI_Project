# Multi-Agent LangGraph + FastAPI Project

This is a **separate project** for the multi-agent capstone. It is not the fraud-detection project.

## Architecture

`POST /query` → Planner → Research Agent / Calculator Agent / Writer Agent → Evaluator → response

The starter runs without API keys so you can test FastAPI first. The agent graph uses LangGraph and includes:
- `GET /health`
- `POST /query`
- `GET /docs` (Swagger UI)

The default agents are lightweight starter implementations. After the API works, we can connect the local Qwen model from Colab/Transformers and optionally add web search.

## 1. Open this folder in VS Code

Extract the ZIP, then in VS Code choose **File → Open Folder** and select `Multi_Agent_FastAPI_Project`.

## 2. Create and activate a virtual environment

Windows PowerShell:
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use:
```powershell
.venv\Scripts\activate.bat
```

## 3. Install packages

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Run the API

From the project root:
```powershell
uvicorn app.main:app --reload
```

Open:
- Swagger UI: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

## 5. Test `/query` in Swagger

Select `POST /query` → **Try it out** and enter:
```json
{
  "query": "What is 125 * (8 + 4)?"
}
```

Then click **Execute**. The response includes the answer and the agent trace.

Try a non-math prompt too:
```json
{
  "query": "Explain what an agentic workflow is in simple terms."
}
```

## Project files

- `app/main.py` — FastAPI endpoints and request validation
- `app/agents.py` — LangGraph state, agents, routing, and graph
- `tests/test_api.py` — basic API tests
- `requirements.txt` — dependencies

## Important

This starter does not load the large Qwen model at startup. First verify that the API and graph work. Then we can wire in the local Qwen model and add the evidence/screenshots required by the capstone.
