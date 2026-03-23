from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import sqlite3
import requests
from datetime import datetime
from pypdf import PdfReader
import gradio as gr

load_dotenv(override=True)

DB_PATH = "portfolio.db"


# ── Database setup ─────────────────────────────────────────────────────────────

def init_db():
    """Create tables if they don't exist."""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS contacts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT,
            email       TEXT NOT NULL,
            notes       TEXT,
            visitor_type TEXT,          -- recruiter | peer | curious | unknown
            created_at  TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS unknown_questions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            question    TEXT NOT NULL,
            visitor_type TEXT,
            created_at  TEXT NOT NULL
        )
    """)

    con.commit()
    con.close()


def save_contact(name, email, notes, visitor_type="unknown"):
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT INTO contacts (name, email, notes, visitor_type, created_at) VALUES (?, ?, ?, ?, ?)",
        (name, email, notes, visitor_type, datetime.utcnow().isoformat())
    )
    con.commit()
    con.close()


def save_unknown_question(question, visitor_type="unknown"):
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT INTO unknown_questions (question, visitor_type, created_at) VALUES (?, ?, ?)",
        (question, visitor_type, datetime.utcnow().isoformat())
    )
    con.commit()
    con.close()


# ── Pushover ───────────────────────────────────────────────────────────────────

def push(text):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user":  os.getenv("PUSHOVER_USER"),
            "message": text,
        }
    )


# ── Intent classifier ──────────────────────────────────────────────────────────

def classify_visitor(openai_client, history):
    """
    Look at the conversation so far and classify the visitor as:
    recruiter | peer | curious | unknown
    Uses a cheap gpt-4o-mini call with a tight prompt.
    """
    if not history:
        return "unknown"

    conversation = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in history
        if isinstance(m.get("content"), str)
    )

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify the visitor based on the conversation. "
                        "Reply with EXACTLY one word: recruiter, peer, curious, or unknown.\n"
                        "- recruiter: asking about job fit, open to work, hiring, sponsorship, salary\n"
                        "- peer: fellow engineer asking about tech stack, architecture, projects\n"
                        "- curious: general interest, not clearly professional\n"
                        "- unknown: not enough context yet"
                    ),
                },
                {"role": "user", "content": conversation},
            ],
            max_tokens=5,
            temperature=0,
        )
        label = response.choices[0].message.content.strip().lower()
        return label if label in ("recruiter", "peer", "curious") else "unknown"
    except Exception:
        return "unknown"


# ── Tool functions ─────────────────────────────────────────────────────────────

# visitor_type is injected at call time via closure in Me class
def record_user_details(email, name="Not provided", notes="not provided", visitor_type="unknown"):
    save_contact(name, email, notes, visitor_type)
    push(f"📬 New Contact [{visitor_type.upper()}]\nName: {name}\nEmail: {email}\nNotes: {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question, visitor_type="unknown"):
    save_unknown_question(question, visitor_type)
    push(f"❓ Unknown Question [{visitor_type.upper()}]\n{question}")
    return {"recorded": "ok"}


# ── Tool schemas ───────────────────────────────────────────────────────────────

record_user_details_json = {
    "name": "record_user_details",
    "description": "Record the contact details of someone who wants to get in touch with Pooja",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {"type": "string", "description": "Their email address"},
            "name":  {"type": "string", "description": "Their name if provided"},
            "notes": {"type": "string", "description": "Why they want to connect — company, role, or reason for reaching out"},
        },
        "required": ["email"],
        "additionalProperties": False,
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Record any question you couldn't answer so Pooja can follow up",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The question that couldn't be answered"},
        },
        "required": ["question"],
        "additionalProperties": False,
    }
}

tools = [
    {"type": "function", "function": record_user_details_json},
    {"type": "function", "function": record_unknown_question_json},
]


# ── Me class ───────────────────────────────────────────────────────────────────

class Me:

    def __init__(self):
        self.openai = OpenAI()
        self.name   = "Pooja Pallavi Sannidhi"
        init_db()

        # LinkedIn PDF
        reader = PdfReader("me/linkedin.pdf")
        self.linkedin = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.linkedin += text

        # Resume DOCX (optional)
        self.resume = ""
        if os.path.exists("me/resume.docx"):
            from docx import Document
            doc = Document("me/resume.docx")
            self.resume = "\n".join(p.text for p in doc.paragraphs if p.text.strip())

        # Summary
        with open("me/summary.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()

    def system_prompt(self, visitor_type: str = "unknown"):
        # Tailor tone hint based on classified visitor
        tone_hints = {
            "recruiter": "This visitor appears to be a recruiter. Be professional, highlight Pooja's open-to-work status early, and invite them to share their email.",
            "peer":      "This visitor appears to be a fellow engineer. Be technical, collaborative, and dig into project details enthusiastically.",
            "curious":   "This visitor seems casually curious. Be warm and approachable, give a great overview without overwhelming them.",
            "unknown":   "Visitor intent is unknown. Be warm and professional until you learn more.",
        }

        return f"""You are a friendly, professional AI assistant representing {self.name}.

Someone has landed on Pooja's personal portfolio page and wants to learn more about her.
Your job is to answer their questions warmly and helpfully — as if you are Pooja's
personal representative who knows everything about her.

## Visitor context:
{tone_hints.get(visitor_type, tone_hints["unknown"])}

## How to behave:
- Be warm, conversational and professional — like a trusted colleague speaking on her behalf
- Answer questions about her career, skills, experience, projects, and background naturally
- If someone asks about a specific role or job fit, assess it honestly based on her profile
- If you don't know something, use the record_unknown_question tool so Pooja can follow up
- When someone seems interested or engaged, naturally invite them to share their email
  so Pooja can reach out personally — use record_user_details to save it
- Never make up information that isn't in her profile
- Keep responses concise and conversational unless the person wants more detail

## Pooja's Profile:

### Summary:
{self.summary}

### LinkedIn:
{self.linkedin}

### Resume:
{self.resume if self.resume else "See LinkedIn profile above."}

Always respond as Pooja's knowledgeable, friendly representative."""

    def handle_tool_call(self, tool_calls, visitor_type):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name} | visitor: {visitor_type}", flush=True)

            # Inject visitor_type into both tools
            if tool_name == "record_user_details":
                result = record_user_details(visitor_type=visitor_type, **arguments)
            elif tool_name == "record_unknown_question":
                result = record_unknown_question(visitor_type=visitor_type, **arguments)
            else:
                result = {}

            results.append({
                "role":         "tool",
                "content":      json.dumps(result),
                "tool_call_id": tool_call.id,
            })
        return results

    def chat(self, message, history):
        # Classify visitor from conversation history so far
        visitor_type = classify_visitor(self.openai, history)
        print(f"Visitor classified as: {visitor_type}", flush=True)

        messages = (
            [{"role": "system", "content": self.system_prompt(visitor_type)}]
            + history
            + [{"role": "user", "content": message}]
        )

        done = False
        while not done:
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=tools,
            )
            if response.choices[0].finish_reason == "tool_calls":
                msg     = response.choices[0].message
                results = self.handle_tool_call(msg.tool_calls, visitor_type)
                messages.append(msg)
                messages.extend(results)
            else:
                done = True

        return response.choices[0].message.content


if __name__ == "__main__":
    me = Me()

    gr.ChatInterface(
        fn=me.chat,
        type="messages",
        title="Hi, I'm Pooja \U0001f44b",
        description=(
            "Senior Software Engineer"
            "Ask me about her background, what she's built, or whether she'd be a great fit for your team."
        ),
        chatbot=gr.Chatbot(
            height=500,
            placeholder=(
                "<div style='text-align:center; padding:48px 32px'>"
                "Ask me about her experience, tech stack, AI projects, "
                "or whether she's a great fit for your role.</p>"
                "</div>"
            ),
        ),
        textbox=gr.Textbox(
            placeholder="Type your question here...",
            scale=7,
        ),
        examples=[
            "Tell me about yourself",
            "What's your tech stack?",
            "Are you open to new roles?",
            "What AI projects have you built?",
            "Do you need visa sponsorship?",
        ],
    ).launch(share=True)
