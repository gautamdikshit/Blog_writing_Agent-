# 📝 AI Blog Writing Agent

An AI-powered technical blog generator built using **LangGraph**, **LLMs**, and **Streamlit**.  
The system can optionally perform research before generating structured, actionable technical blog posts.

Live on : https://blog-agent-latest-mkir.onrender.com
(DISCLAIMER: Here I am using Render Free Web Service, so this can take 20–60 seconds or maybe even more. The reason is, the application goes to sleep after ~15 minutes of inactivity, the Container shuts down completely and restarts when someone visits the URL. What happens is: Render wakes up the container --> Pulls the image --> Starts the server --> Loads dependencies. Using the free tier, so lets just accept it.)

🔄 Restarts when someone visits the URL
---

## 🚀 Features

- 🔀 Intelligent routing (`closed_book`, `hybrid`, `open_book`)
- 🔎 Optional research using Tavily
- 🧠 Structured blog planning (task-based outline)
- 🏗 Section-wise generation using LangGraph
- 🖥 Clean Streamlit UI
- 📚 Sidebar with recent blog history
- 🐳 Fully Dockerized

## 🧠 Architecture Overview

- Router → Determines blog generation mode

- Research Node → Collects evidence (if needed)

- Orchestrator → Creates structured blog plan

- Worker Nodes → Generate each section

- Reducer → Combines sections into final Markdown

## 🏗 Project Structure

```
blog-writing-agent/
│
├── agent.py              # LangGraph-based blog agent
├── streamlit_app.py      # Streamlit UI
├── main.py               # FastAPI API endpoint
├── pyproject.toml        # Project configuration (uv)
├── uv.lock               # Locked dependencies
├── Dockerfile
├── .dockerignore
├── .gitignore
│
└── blogs/                # Generated blog storage
```

---

## ⚙️ Local Development (Using uv)

### 1️⃣ Install uv (if not installed)

```bash
pip install uv
```

### 2️⃣Install dependencies
```bash
uv sync
```
### 3️⃣ Run the app
```bash
uv run streamlit run streamlit_app.py
```

🔐 Environment Variables

```
 OPENAI_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
```


