from fastapi import FastAPI, WebSocket, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import asyncio

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/teach")
async def teacher(websocket: WebSocket):
    await websocket.accept()
    lesson = [
        # Estimating Execution Time
        "Let's learn about estimating program execution time.",
        "A program step is a segment independent of instance characteristics.",
        "Instance characteristics are input-dependent variables.",
        "Step 1: Identify key operations and count their executions.",
        "Step 2: Determine total steps executed by the program.",
        "Time complexity is proportional to these operations.",

        # Asymptotic Notations Introduction
        "Now, let's discuss asymptotic notations.",
        "These describe time or space complexity behavior.",
        "We represent complexity using function f(n).",
        "There are four main notations:",
        "Big-O, Omega, Theta, and Little-O.",

        # Big-O Notation
        "Big-O provides an upper bound for f(n).",
        "f(n) = O(g(n)) if f(n) ≤ c*g(n) for n ≥ n₀.",
        "Example: 3n + 2 = O(n) with c=4, n₀=2.",
        "Another example: 10n² + 4n + 2 = O(n²) with c=11, n₀=6.",

        # Omega Notation
        "Omega gives a lower bound for f(n).",
        "f(n) = Ω(g(n)) if f(n) ≥ c*g(n) for n ≥ n₀.",
        "Example: 3n + 2 = Ω(n) with c=3, n₀=0.",
        "Another example: 10n² + 4n + 2 = Ω(n²) with c=10, n₀=0.",

        # Theta Notation
        "Theta bounds f(n) both above and below.",
        "f(n) = θ(g(n)) if c₁*g(n) ≤ f(n) ≤ c₂*g(n) for n ≥ n₀.",
        "Since 3n + 2 = O(n) and Ω(n), it's θ(n).",
        "This means it grows exactly like n.",

        # Conclusion
        "Remember: Big-O for worst case, Omega for best case, Theta for tight bounds."
    ]
    
    for sentence in lesson:
        await websocket.send_json({
            "type": "teach", 
            "text": sentence,
            "completed": False
        })
        await asyncio.sleep(2)  # Simulate speaking pace
        
        try:
            # Check for interruption without blocking
            data = await asyncio.wait_for(websocket.receive_json(), timeout=0.1)
            if data.get("type") == "interrupt":
                await handle_interruption(websocket)
                break
        except asyncio.TimeoutError:
            continue
    
    await websocket.send_json({"type": "complete", "completed": True})

async def handle_interruption(websocket: WebSocket):
    await websocket.send_json({
        "type": "interrupted",
        "text": "Yes, what's your question?"
    })
    # Wait for question
    question = await websocket.receive_json()
    await websocket.send_json({
        "type": "answer",
        "text": f"Great question about '{question['text']}'. Let me explain..."
    })
    await asyncio.sleep(1)