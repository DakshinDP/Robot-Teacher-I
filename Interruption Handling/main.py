from fastapi import FastAPI, WebSocket, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import asyncio
from enum import Enum
import random
from typing import Optional
import logging
from pydantic import BaseModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class InterruptionType(str, Enum):
    DOUBT = "doubt"
    NOISE = "noise"
    LATE = "late"
    INTRUDER = "intruder"

class InterruptionModel(BaseModel):
    type: str
    subtype: str
    text: Optional[str] = None
    intruder_type: Optional[str] = None

class LessonState:
    def __init__(self):
        self.lesson = [
            "Let's learn about estimating program execution time.",
            "A program step is a segment independent of instance characteristics.",
            "Instance characteristics are input-dependent variables.",
            "Step 1: Identify key operations and count their executions.",
            "Step 2: Determine total steps executed by the program.",
            "Time complexity is proportional to these operations.",
            "Now, let's discuss asymptotic notations.",
            "These describe time or space complexity behavior.",
            "We represent complexity using function f(n).",
            "There are four main notations:",
            "Big-O, Omega, Theta, and Little-O.",
            "Big-O provides an upper bound for f(n).",
            "f(n) = O(g(n)) if f(n) ≤ c*g(n) for n ≥ n₀.",
            "Example: 3n + 2 = O(n) with c=4, n₀=2.",
            "Another example: 10n² + 4n + 2 = O(n²) with c=11, n₀=6.",
            "Omega gives a lower bound for f(n).",
            "f(n) = Ω(g(n)) if f(n) ≥ c*g(n) for n ≥ n₀.",
            "Example: 3n + 2 = Ω(n) with c=3, n₀=0.",
            "Another example: 10n² + 4n + 2 = Ω(n²) with c=10, n₀=0.",
            "Theta bounds f(n) both above and below.",
            "f(n) = θ(g(n)) if c₁*g(n) ≤ f(n) ≤ c₂*g(n) for n ≥ n₀.",
            "Since 3n + 2 = O(n) and Ω(n), it's θ(n).",
            "This means it grows exactly like n.",
            "Remember: Big-O for worst case, Omega for best case, Theta for tight bounds."
        ]
        self.position = 0
        self.context = []
        self.late_count = 0
        self.paused = False
        self.current_interruption = None

    def reset(self):
        self.position = 0
        self.context = []
        self.late_count = 0
        self.paused = False
        self.current_interruption = None

lesson_state = LessonState()

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/teach")
async def teacher(websocket: WebSocket):
    await websocket.accept()
    logger.info("New WebSocket connection established")
    
    try:
        while lesson_state.position < len(lesson_state.lesson):
            if lesson_state.paused:
                try:
                    data = await asyncio.wait_for(websocket.receive_json(), timeout=None)
                    if data.get("type") == "resume":
                        lesson_state.paused = False
                        recap = "To recap: " + " ".join(lesson_state.context[-3:])
                        await websocket.send_json({
                            "type": "recap",
                            "text": recap
                        })
                        await asyncio.sleep(1)
                        await websocket.send_json({
                            "type": "resumed",
                            "text": "Let's continue our lesson...",
                            "position": lesson_state.position
                        })
                        continue
                except asyncio.TimeoutError:
                    continue
            
            current_sentence = lesson_state.lesson[lesson_state.position]
            lesson_state.context.append(current_sentence)
            if len(lesson_state.context) > 3:
                lesson_state.context.pop(0)
            
            await websocket.send_json({
                "type": "teach",
                "text": current_sentence,
                "position": lesson_state.position
            })
            lesson_state.position += 1
            
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=2.0)
                if data.get("type") == "interrupt":
                    lesson_state.paused = True
                    await handle_interruption(
                        websocket, 
                        InterruptionType(data["subtype"]),
                        data.get("text", ""),
                        data.get("intruder_type", "")
                    )
            except asyncio.TimeoutError:
                continue
        
        await websocket.send_json({"type": "complete"})
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        lesson_state.reset()

async def handle_interruption(websocket: WebSocket, subtype: InterruptionType, text: str, intruder_type: str = ""):
    lesson_state.current_interruption = {
        'type': subtype,
        'text': text,
        'intruder_type': intruder_type,
        'position': lesson_state.position
    }
    
    if subtype == InterruptionType.DOUBT:
        await handle_doubt(websocket, text)
    elif subtype == InterruptionType.NOISE:
        await handle_noise(websocket)
    elif subtype == InterruptionType.LATE:
        await handle_late(websocket)
    elif subtype == InterruptionType.INTRUDER:
        await handle_intruder(websocket, intruder_type)

async def handle_doubt(websocket: WebSocket, question: str):
    explanations = {
        "big-o": "Big-O describes the worst-case scenario growth rate. Example: O(n²) means runtime grows quadratically with input size.",
        "omega": "Omega describes the best-case scenario. Example: Ω(n) means runtime grows at least linearly.",
        "theta": "Theta gives tight bounds. θ(n) means runtime grows exactly linearly.",
        "time complexity": "Measures how runtime grows with input size. Common complexities: O(1), O(log n), O(n), O(n²).",
        "space complexity": "Measures memory usage growth. Important for memory-constrained systems.",
        "default": "This concept helps analyze algorithm efficiency as input size grows."
    }
    
    explanation = explanations.get(question.lower().split()[0], explanations["default"])
    await websocket.send_json({
        "type": "doubt_response",
        "text": f"Regarding '{question}': {explanation}",
        "show_resume": True
    })

async def handle_noise(websocket: WebSocket):
    await websocket.send_json({
        "type": "noise_response",
        "text": "Class, please maintain silence. I'll pause until it's quiet.",
        "show_resume": True
    })

async def handle_late(websocket: WebSocket):
    lesson_state.late_count += 1
    response = "Please be on time. " + (
        "This is your final warning." if lesson_state.late_count >= 2 
        else "Don't repeat this."
    )
    await websocket.send_json({
        "type": "late_response",
        "text": response,
        "show_resume": True,
        "late_count": lesson_state.late_count
    })

async def handle_intruder(websocket: WebSocket, intruder_type: str):
    if intruder_type == "announcement":
        await websocket.send_json({
            "type": "intruder_response",
            "text": "You may make your announcement now.",
            "action_required": True
        })
    else:
        await websocket.send_json({
            "type": "intruder_response",
            "text": "Please see me after class.",
            "show_resume": True
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
