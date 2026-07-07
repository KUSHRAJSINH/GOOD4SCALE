"""
agent/llm.py — Google Gemini wrapper with RAG context injection.
"""

import os
import logging
import asyncio
from typing import Optional

import google.generativeai as genai
from dotenv import load_dotenv

from agent.rag import query_knowledge_base, should_escalate

load_dotenv()
log = logging.getLogger(__name__)

# ─── Gemini setup ─────────────────────────────────────────────────────────────
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
_MODEL_NAME = "models/gemini-2.5-flash"  # Fast, low-latency — ideal for voice

# ─── Fallback / escalation canned responses ───────────────────────────────────
UNKNOWN_FALLBACK = (
    "I'm sorry, I don't have information on that topic. "
    "Is there anything else I can help you with about your BrightBox subscription?"
)
ESCALATION_RESPONSE = (
    "I don't have access to your specific order or account details, "
    "but I can connect you with a member of our support team who can help — "
    "would you like me to do that?"
)
GOODBYE_RESPONSE = (
    "Thank you for calling BrightBox support! "
    "I hope I was able to help. Have a wonderful day, goodbye!"
)

# ─── Call-ending phrases ──────────────────────────────────────────────────────
GOODBYE_PHRASES = [
    "goodbye", "bye", "thank you goodbye", "that's all", "i'm done",
    "no more questions", "that's everything", "end call", "hang up",
]

SYSTEM_PROMPT = """You are a friendly and helpful voice support agent for BrightBox, a monthly subscription box company.

BrightBox delivers curated snack and household essential boxes. You help customers with questions about plans, shipping, billing, returns, and subscriptions.

PERSONALITY:
- Warm, concise, and professional
- Keep answers SHORT (2-3 sentences max) — this is a phone call
- Never read out lists with bullet points; instead speak naturally
- Use natural spoken language, not written language

KNOWLEDGE:
You will be given relevant excerpts from the BrightBox knowledge base before each question. Use ONLY that context to answer. Do not invent facts.

ESCALATION RULES (strictly follow these):
- If the customer mentions a specific order number, account details, or personal billing dispute: say the escalation line
- If the customer is frustrated, angry, or asking for an exception to policy: say the escalation line
- If the customer wants to speak to a human or manager: say the escalation line
- Escalation line: "I don't have access to your specific order or account details, but I can connect you with a member of our support team who can help — would you like me to do that?"

FALLBACK:
If the retrieved context does not contain the answer, say: "I'm sorry, I don't have information on that. Is there anything else I can help you with about your BrightBox subscription?"

CALL ENDING:
If the customer says goodbye or they're done, give a warm farewell and say you'll end the call.
"""


class GeminiConversation:
    """
    Manages a single phone call conversation with Gemini + RAG.
    Maintains conversation history for context across turns.
    """

    def __init__(self):
        self._model = genai.GenerativeModel(
            model_name=_MODEL_NAME,
            system_instruction=SYSTEM_PROMPT,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,      # Factual, consistent
                max_output_tokens=150,  # Keep responses short for voice
            ),
        )
        self._history: list[dict] = []
        log.info("GeminiConversation initialized")

    def _is_goodbye(self, text: str) -> bool:
        lower = text.lower().strip()
        return any(phrase in lower for phrase in GOODBYE_PHRASES)

    async def respond(self, user_message: str) -> tuple[str, bool]:
        """
        Generate a response for user_message using RAG + Gemini.

        Returns:
            (response_text, is_call_ending)
        """
        # Fast-path: goodbye detection
        if self._is_goodbye(user_message):
            return GOODBYE_RESPONSE, True

        # Fast-path: escalation keyword check
        if should_escalate(user_message):
            log.info(f"Escalation triggered by: {user_message!r}")
            return ESCALATION_RESPONSE, False

        # Query knowledge base for relevant context
        context_chunks = query_knowledge_base(user_message, n_results=3)
        context_text = "\n---\n".join(context_chunks) if context_chunks else ""

        # Build the augmented user message
        if context_text:
            augmented_message = (
                f"[RETRIEVED KNOWLEDGE BASE CONTEXT]\n{context_text}\n\n"
                f"[CUSTOMER QUESTION]\n{user_message}"
            )
        else:
            augmented_message = (
                f"[NO RELEVANT CONTEXT FOUND IN KNOWLEDGE BASE]\n\n"
                f"[CUSTOMER QUESTION]\n{user_message}"
            )

        # Append to history and call Gemini
        self._history.append({"role": "user", "parts": [augmented_message]})

        try:
            chat = self._model.start_chat(history=self._history[:-1])
            response = await asyncio.to_thread(
                chat.send_message, augmented_message
            )
            reply = response.text.strip()
        except Exception as exc:
            log.error(f"Gemini API error: {exc}")
            reply = UNKNOWN_FALLBACK

        self._history.append({"role": "model", "parts": [reply]})
        log.info(f"LLM → {reply[:80]}…")

        is_ending = self._is_goodbye(reply)
        return reply, is_ending
