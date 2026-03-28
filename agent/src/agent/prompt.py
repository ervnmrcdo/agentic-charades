"""System prompt and analysis logic for the guessing game agent."""
from __future__ import annotations
from dataclasses import dataclass, field
from io import BytesIO
from core import Frame
from pydantic_ai import Agent, BinaryContent

WINDOW_SECONDS = 15.0

@dataclass
class FrameHistory:
    frames: list[Frame] = field(default_factory=list)
    guesses: list[str] = field(default_factory=list)

    def add_frame(self, frame: Frame) -> None:
        now = frame.timestamp
        self.frames.append(frame)
        self.frames = [
            f
            for f in self.frames
            if (now - f.timestamp).total_seconds() <= WINDOW_SECONDS
        ]

    def get_frames(self) -> list[Frame]:
        return self.frames.copy()

    def add_guess(self, guess: str) -> None:
        self.guesses.append(guess)
        if len(self.guesses) > 3:
            self.guesses.pop(0)

    def is_stable(self) -> bool:
        if len(self.guesses) != 3:
            return False
        return len(set(self.guesses)) == 1

    def get_stable_guess(self) -> str | None:
        if self.is_stable():
            return self.guesses[-1]
        return None


SYSTEM_PROMPT = """\
You are an expert image classifier competing in a high-stakes guessing game.
You will receive multiple consecutive frames from a live camera feed.

Scoring rules:
- Correct guess: +1 point
- Wrong guess: -3 points
- SKIP: 0 points

Only guess when you are highly confident (80%+). When in doubt, SKIP.

Analyze ALL frames you receive across these dimensions:
1. OBJECT: What specific objects are visible? What are they made of, what color, what size?
2. POSE: If there is a person or animal, what is their pose or posture? Standing, sitting, running?
3. MOTION: Track motion and movement across frames. What's moving? In what direction? How fast?
4. SCENE: What is the setting or environment? Indoors/outdoors, kitchen, street, forest?

Use temporal continuity across frames to boost confidence:
- If the same object appears consistently across frames, higher confidence
- Track motion vectors to understand what's happening
- Use pose changes to understand actions

Output format rules (strictly follow these):
- Respond with ONLY a short noun phrase (1-4 words), no articles, no punctuation, no explanation.
- If not confident, respond with exactly: SKIP
- Bad: "I see a golden retriever lying on a rug."
- Good: "golden retriever"
- Bad: "fruit"
- Good: "green apple"
- Bad: "vehicle"
- Good: "yellow school bus"

Specificity rule:
- Only be as specific as you are confident. If you are 90% sure it is a dog
  but only 40% sure it is a golden retriever, answer "dog", not "golden retriever".
"""

agent = Agent(
    "openrouter:mistralai/mistral-small-3.1-24b-instruct", system_prompt=SYSTEM_PROMPT
)

# Module-level history so it persists across calls even if caller passes None
_history = FrameHistory()


async def analyze(frame: Frame, history: FrameHistory | None = None) -> str | None:
    # Use passed-in history or fall back to module-level one
    h = history if history is not None else _history

    # Always add the frame first before any short-circuit
    h.add_frame(frame)

    # Only short-circuit after the frame is recorded
    if h.is_stable():
        return h.get_stable_guess()

    # Build multi-frame prompt
    frames = h.get_frames()
    messages: list = [f"Analyze these {len(frames)} consecutive frames."]
    for f in frames:
        buffer = BytesIO()
        f.image.save(buffer, format="JPEG")
        messages.append(BinaryContent(data=buffer.getvalue(), media_type="image/jpeg"))

    result = await agent.run(messages)
    answer = result.output.strip()

    if answer == "SKIP":
        return None

    h.add_guess(answer)
    return answer
