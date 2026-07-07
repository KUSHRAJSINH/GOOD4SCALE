# BrightBox Voice Support Agent

A real-time phone support agent for BrightBox (subscription box company). The system answers incoming phone calls, answers questions about plans, billing, and shipping using retrieval-augmented generation (RAG), and detects escalation/goodbye intents to cleanly manage call state.

This project was built for the GOOD4SCALE voice agent task.

---

## System Architecture

```
                 +---------------------------+
                 |       Twilio Voice        |
                 +-------------+-------------+
                               |
                   HTTP POST   | (opens media stream)
                               v
                 +-------------+-------------+
                 |    FastAPI Server (/ws)   |
                 +-------------+-------------+
                               |
                               | Bi-directional audio over WebSockets
                               v
  +----------------------------+----------------------------+
  |                      Pipecat Pipeline                    |
  |                                                         |
  |  1. FastAPIWebsocketTransport (in/out)                  |
  |     - Receives & sends raw telephony audio              |
  |                                                         |
  |  2. VADProcessor (Silero VAD)                           |
  |     - Identifies when the user starts/stops speaking    |
  |                                                         |
  |  3. WhisperSTTService (Local Whisper)                   |
  |     - Decodes audio segments to text (runs on CPU)      |
  |                                                         |
  |  4. BrightBoxRAGProcessor (Custom)                      |
  |     - Queries LocalVectorDB for relevant Q&A context    |
  |     - Generates Gemini response & triggers escalations  |
  |                                                         |
  |  5. EdgeTTSService (Microsoft Edge Read-Aloud API)      |
  |     - Dynamically synthesizes replies back into PCM audio|
  +---------------------------------------------------------+
```

---

## Technical Stack & Choices

* **Telephony**: Twilio Media Streams (WebSocket-based raw audio streaming).
* **Pipeline Orchestration**: Pipecat-AI (`1.5.0`).
* **LLM**: Gemini (`models/gemini-2.5-flash`) via the `google-generativeai` SDK.
* **Vector Store**: Pure-Python local vector database (`agent/vector_db.py`) storing embeddings in a simple JSON file (`vector_db.json`). This avoids compiled C-extension crashes (like ChromaDB/hnswlib) on Windows while retaining 100% local retrieval using numpy cosine similarity.
* **Embeddings**: Gemini Hosted Embeddings (`models/gemini-embedding-2`), replacing local PyTorch/sentence-transformers to run stably on standard CPU hardware without DLL conflicts.
* **TTS**: `edge-tts` (Microsoft Edge neural speech generation) converted to PCM using a dynamically configured PATH-patched `pydub`/`ffmpeg` pipeline.
* **STT**: Local Faster-Whisper (`base` model configured to run on CPU).

---

## Getting Started

### Prerequisites
* Python 3.10 or 3.11
* `ffmpeg` installed on your machine.
  * *Windows*: `winget install ffmpeg`
  * *macOS*: `brew install ffmpeg`
  * *Linux*: `sudo apt-get install ffmpeg`
* A free [ngrok](https://ngrok.com) account.
* A Twilio account with a phone number.

### 1. Installation
Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Configuration
Copy the template and fill in your API keys:
```bash
cp .env.example .env
```
Update `.env` with your `GOOGLE_API_KEY`, Twilio credentials, and ngrok public domain.

### 3. Ingestion
Chunk the documentation and build your local knowledge base database:
```bash
python scripts/ingest_kb.py
```
This produces `vector_db.json` containing embeddings for the company overview, shipping FAQ, and escalation guidelines.

### 4. Running the Agent
Start the FastAPI server:
```bash
python server.py
```

Configure your Twilio Phone Number webhook to point to your public ngrok endpoint:
* **A Call Comes In Webhook**: `https://<your-subdomain>.ngrok-free.dev/voice` (HTTP POST)

Call your configured Twilio number to begin the conversation.

---

## Project Structure

```
good4scale/
├── server.py               # FastAPI server & WebSocket endpoint
├── requirements.txt        # Verified package dependencies
├── .gitignore              # Configured patterns (ignoring env/local DBs)
│
├── agent/
│   ├── __init__.py
│   ├── llm.py              # Gemini conversation loop
│   ├── pipeline.py         # Pipecat pipeline orchestration
│   ├── rag.py              # Retrieval & escalation keywords check
│   ├── tts_service.py      # Custom Edge TTS stream receiver
│   └── vector_db.py        # Local JSON-backed vector store
│
├── scripts/
│   └── ingest_kb.py        # File chunking & embedding generation
│
└── knowledge_base/         # Reference text documents
```

---

## Fallback & Escalation Flow

* **Out-of-Scope Qs**: The agent will politely state it does not have the information and ask how else it can help with BrightBox subscriptions.
* **Account/Billing Disputes**: When the user requests exceptions or mentions specific order numbers/billing queries, the agent triggers an escalation flow offering to transfer them to a human team member.
* **Clean Hang-up**: If the customer says goodbye, the agent sends a warm closing line and automatically triggers a Twilio hang-up (`EndFrame`).
