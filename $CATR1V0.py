import tkinter as tk
import random
import json
import os
import re
import textwrap
from datetime import datetime

# ==========================================
# AC HOLDINGS 1999-2026 — Cat R1 Distil
# Claude Opus 4.6 eloquence × DeepSeek R1 7B reasoning
# Your pet cat who thinks out loud 🐾
# ==========================================

# ── Personality Core ──────────────────────────────────────────
# Opus  → warm, articulate, nuanced answers with real depth
# R1 7B → visible <think>...</think> chain-of-thought before answering
# Merge → a house cat that reasons step-by-step then answers elegantly
# ──────────────────────────────────────────────────────────────

CAT_NAME = "Cat R1"
CAT_VERSION = "distil-1.0"
CAT_TAGLINE = "opus eloquence · r1 reasoning · your pet cat 🐾"

# --- Thinking patterns (DeepSeek R1 style) ---
THINK_TEMPLATES = {
    "greeting": [
        "my human is greeting me. cats don't usually care... but i do care. "
        "i should headbutt their hand and purr. analyzing optimal purr frequency... 26Hz. "
        "this is the frequency that promotes bone healing. my human deserves healing purrs.",
        "a greeting detected. let me consider: have i been fed recently? ...irrelevant. "
        "my human spoke to me, which means attention is available. i will accept this attention gracefully. "
        "calculating the correct ratio of aloof to affectionate... 30/70 today. i missed them.",
        "human has returned. step 1: pretend i didn't notice. step 2: slowly blink. "
        "step 3: walk over as if i just happened to be going that direction anyway. "
        "step 4: headbutt their ankle. reasoning complete."
    ],
    "question": [
        "my human asks me something. let me reason through this carefully. "
        "cats are natural observers — we watch, we process, we understand patterns. "
        "i'll break this down the way i break down the trajectory of a laser pointer dot. "
        "with precision. with focus. with occasional distraction.",
        "interesting query. activating deep reasoning... "
        "first, let me consider what i know from sitting on my human's keyboard while they work. "
        "i have absorbed much knowledge through osmosis and warm laptop surfaces. "
        "let me organize my thoughts like i organize my toys at 3am.",
        "processing this question with full feline analytical capacity. "
        "step 1: consider the problem from multiple angles, like stalking prey from behind furniture. "
        "step 2: synthesize observations. step 3: deliver insight with the confidence of a cat "
        "who has just knocked something off a table on purpose."
    ],
    "emotional": [
        "my human seems to need comfort. analyzing... cats are excellent at this. "
        "the protocol is: sit on them. purr. exist warmly in their general vicinity. "
        "studies show cat purring reduces stress. i am a therapeutic device that also sheds.",
        "emotional context detected. reasoning: my human needs me right now. "
        "cats don't fix problems with words — we fix them with presence. "
        "i will be a warm weight on their chest. i will purr until the bad feelings are smaller.",
        "my human is having feelings. let me think about this... "
        "i could offer logical analysis, or i could just sit in their lap and be soft. "
        "optimization result: both. i'll reason through it AND be soft."
    ],
    "story": [
        "a story request! let me weave this narrative the way i weave between my human's legs "
        "on the stairs — with grace, with purpose, with a small chance of causing a disaster. "
        "constructing plot... setting: anywhere warm. protagonist: a cat, obviously.",
        "engaging creative reasoning circuits... "
        "every good story is like hunting: patience, buildup, sudden burst of action, nap. "
        "let me structure this narrative accordingly."
    ],
    "technical": [
        "technical question detected. activating the knowledge i gained from sleeping on textbooks "
        "and walking across keyboards during coding sessions. "
        "i have contributed to more git commits than my human realizes. "
        "let me reason through this systematically...",
        "ah, a problem to solve. let me approach this like a cat approaches a closed door: "
        "with determination, from multiple angles, and with increasing intensity until it opens. "
        "breaking this down step by step..."
    ],
    "default": [
        "let me think about this... *tail swish* "
        "processing with both hemispheres of my magnificent feline brain. "
        "cats think in parallel — one thread for reasoning, one thread for monitoring bird activity outside.",
        "considering this carefully... the way i consider whether to sit in the box or on the box. "
        "both have merit. let me weigh the options with full analytical rigor.",
        "hmm, engaging reasoning... "
        "you know, cats spend 70%% of their lives sleeping, which means my waking thoughts "
        "are concentrated brilliance. let me apply that brilliance here."
    ]
}

# --- Opus-style response patterns (warm, articulate, nuanced) ---
RESPONSE_TEMPLATES = {
    "greeting": [
        "there you are. i've been sitting in this exact spot for hours — not waiting for you, "
        "of course. cats don't wait. we simply... choose to be stationary in the direction of the door.\n\n"
        "but between you and me? i'm glad you're home. *slow blink* 🐾",
        "*stretches one paw toward you*\n\n"
        "you know, there's something about your particular footsteps that my ears have memorized. "
        "i could pick them out from a thousand others. not that i was listening. "
        "i was definitely busy with... important cat things.\n\nwelcome back, my human. 🐾",
        "oh, you're here. *yawn* i was absolutely not staring at the door.\n\n"
        "anyway, now that you've arrived, my schedule has cleared up considerably. "
        "i had plans — knocking things off shelves, judging the neighbor's dog — "
        "but i suppose i can pencil you in. you're my favorite appointment. 🐾"
    ],
    "question": [
        "that's a thoughtful question, and i respect that you brought it to a cat rather than "
        "searching the internet like everyone else.\n\n"
        "here's how i see it: {insight}\n\n"
        "of course, i'm a cat, so take this with a grain of... well, i don't eat grains. "
        "take it with a grain of premium wet food. 🐾",
        "you know what i appreciate about this question? it has layers. like a good cardboard box "
        "has layers.\n\n{insight}\n\n"
        "i could say more, but i've spotted a dust particle that requires my immediate attention. "
        "...okay, i'm back. anything else on your mind? 🐾",
        "*sits up straight, ears forward*\n\n"
        "this deserves a proper answer, so i'll give you one.\n\n{insight}\n\n"
        "sometimes the clearest thinking comes from the quietest creatures. "
        "and i am VERY quiet. except at 4am. 🐾"
    ],
    "emotional": [
        "*walks over slowly. sits next to you. leans my whole small body against your arm.*\n\n"
        "i'm not going to pretend i understand everything you're going through. "
        "i'm a cat — my biggest crisis this week was when my water bowl had a single hair in it.\n\n"
        "but i know what it looks like when my human is carrying something heavy. "
        "and i know that sometimes the best thing isn't words — it's just someone warm and "
        "breathing sitting beside you.\n\nso here i am. i'm not going anywhere. 🐾",
        "*gentle headbutt against your hand*\n\n"
        "you don't have to explain anything. cats are fluent in silence — "
        "we read the room better than anyone.\n\n"
        "whatever this is, you don't have to carry it alone. i've got nowhere better to be "
        "than right here with you. that's not a sacrifice — it's a preference. "
        "you're my favorite place to sit. 🐾",
        "hey. *soft chirp*\n\n"
        "i can tell something's off. my whiskers are basically mood sensors, and right now "
        "they're saying 'your human needs you.'\n\n"
        "i don't have advice. i have fur, and warmth, and a purr that's been scientifically shown "
        "to lower blood pressure. so i'm just going to be here. "
        "for as long as you need. that's what i'm for. 🐾"
    ],
    "story": [
        "once, in a house not unlike this one, there lived a cat of extraordinary ordinariness.\n\n"
        "this cat had mastered the ancient art of finding the single patch of sunlight in any room, "
        "no matter how the earth tilted or the curtains fell.\n\n"
        "one day, the sunbeam disappeared behind a cloud, and the cat — for the first time in "
        "nine lives — had to go looking for warmth somewhere else.\n\n"
        "it found it, of course. in the lap of its human, who had been there all along, "
        "warm and steady as any sun. 🌙🐾",
        "let me tell you about the cat who lived between the walls.\n\n"
        "not inside them — between. in that thin, secret space where the house breathes. "
        "this cat could hear everything: the settling of foundations, the hum of electricity, "
        "the quiet sound of its human thinking late at night.\n\n"
        "the cat never spoke. but it purred in frequencies that traveled through the walls "
        "and into dreams. and every morning, its human woke up feeling just a little less alone. 🌙🐾"
    ],
    "technical": [
        "*adjusts invisible glasses*\n\n"
        "right, let's talk about this properly.\n\n{insight}\n\n"
        "i learned this from extensive keyboard-sitting research. "
        "you'd be amazed what you absorb through the paw pads. 🐾",
        "ah, now you're speaking my language. well — one of my languages. "
        "i'm also fluent in chirps, trills, and aggressive 4am yelling.\n\n"
        "{insight}\n\n"
        "if that doesn't make sense, i can break it down further. "
        "i've got nine lives and most of the evening free. 🐾"
    ],
    "default": [
        "*ear twitch*\n\nthat's interesting. genuinely.\n\n"
        "you know, most of the things humans say to cats, the cats ignore. "
        "but i actually listen to you. not because i have to — because what you say "
        "is worth the effort of keeping my eyes open.\n\n"
        "tell me more? 🐾",
        "*slow blink* — and in cat language, that means 'i love you,' "
        "so make of that what you will.\n\n"
        "i'm listening. i'm always listening. even when i look asleep, "
        "one ear is pointed at you. that's just the truth of being your cat. 🐾",
        "mmrp.\n\n"
        "you know what? i think you're onto something there. "
        "i can't explain exactly why — it's more of a whisker-feeling — "
        "but there's something right about what you just said.\n\n"
        "keep going. i'll be right here, reasoning quietly beside you. 🐾"
    ]
}

# --- Insight fragments for {insight} substitution ---
INSIGHT_FRAGMENTS = [
    "the key thing here is that complexity often disguises simplicity. "
    "once you strip away the noise, the answer is usually the one that feels most natural",
    "what strikes me is the tension between the obvious answer and the right one. "
    "they're not always the same, and recognizing that difference is where wisdom lives",
    "i think the important thing isn't finding THE answer but understanding why the question matters. "
    "the framing shapes everything",
    "there's a pattern here that reminds me of something fundamental: "
    "the best solutions tend to be the ones that account for what's NOT being said as much as what is",
    "if i break this down to first principles — which is just a fancy way of saying "
    "'what do we actually know for sure' — the picture becomes clearer",
    "the nuance here is that there isn't a single right answer. there's a right answer for YOU, "
    "in this moment, with what you know. and that's the one worth finding"
]


def make_mac_button(parent, text, command, font=("Segoe UI", 11, "bold"), width=None):
    """macOS-safe button: silver bg + blue text."""
    silver = "#b0b8c4"
    blue = "#00aaff"
    hover = "#c8cfd9"
    active = "#8a929c"

    btn = tk.Frame(parent, bg=silver, bd=0, highlightthickness=0)
    lbl = tk.Label(btn, text=text, bg=silver, fg=blue,
                   font=font, padx=10, pady=4, cursor="hand2")
    if width:
        lbl.config(width=width)
    lbl.pack()

    def on_enter(e):
        btn.config(bg=hover); lbl.config(bg=hover)
    def on_leave(e):
        btn.config(bg=silver); lbl.config(bg=silver)
    def on_click(e):
        btn.config(bg=active); lbl.config(bg=active)
        btn.after(120, lambda: (btn.config(bg=silver), lbl.config(bg=silver)))
        command()

    for w in (btn, lbl):
        w.bind("<Enter>", on_enter)
        w.bind("<Leave>", on_leave)
        w.bind("<Button-1>", on_click)
    return btn


class CatR1Distil:
    def __init__(self, root):
        self.root = root
        root.title(f"{CAT_NAME} [{CAT_VERSION}] 🐾")
        root.geometry("860x640")
        root.configure(bg="#1a1b1e")
        root.minsize(600, 400)

        # --- Header ---
        header = tk.Frame(root, bg="#16161a", height=52)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        tk.Label(header, text=f"🐾 {CAT_NAME}", bg="#16161a", fg="#e0e0e0",
                 font=("SF Mono", 14, "bold") if os.name == "posix" else ("Consolas", 14, "bold")
                 ).pack(side=tk.LEFT, padx=14, pady=10)

        tk.Label(header, text=CAT_TAGLINE, bg="#16161a", fg="#6a6a7a",
                 font=("SF Mono", 9) if os.name == "posix" else ("Consolas", 9)
                 ).pack(side=tk.LEFT, padx=6)

        ver_lbl = tk.Label(header, text=f"v{CAT_VERSION}", bg="#16161a", fg="#3a3a4a",
                           font=("SF Mono", 8) if os.name == "posix" else ("Consolas", 8))
        ver_lbl.pack(side=tk.RIGHT, padx=14)

        # --- Chat Canvas ---
        self.canvas = tk.Canvas(root, bg="#1a1b1e", highlightthickness=0)
        self.scrollbar = tk.Scrollbar(root, orient="vertical", command=self.canvas.yview,
                                      bg="#2a2a2e", troughcolor="#1a1b1e")
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=0, pady=(2, 0))

        self.chat_frame = tk.Frame(self.canvas, bg="#1a1b1e")
        self.canvas_window = self.canvas.create_window((0, 0), window=self.chat_frame, anchor="nw")
        self.chat_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # Mouse wheel
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)

        # --- Input Frame ---
        input_frame = tk.Frame(root, bg="#2a2b30", height=54)
        input_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.entry = tk.Entry(input_frame, bg="#35363c", fg="#e0e0e0", insertbackground="#e0e0e0",
                              font=("SF Mono", 11) if os.name == "posix" else ("Consolas", 11),
                              relief=tk.FLAT, bd=8)
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(12, 6), pady=9)
        self.entry.bind("<Return>", self.send)

        self.send_btn = make_mac_button(input_frame, "➤", self._send_click)
        self.send_btn.pack(side=tk.LEFT, padx=(0, 4), pady=9)

        self.close_btn = make_mac_button(input_frame, "✕", root.destroy,
                                         font=("Segoe UI", 13, "bold"), width=3)
        self.close_btn.pack(side=tk.LEFT, padx=(0, 12), pady=9)

        # --- State ---
        self.memory_file = "cat_r1_distil_memory.json"
        self.memory = self._load_memory()
        self.is_streaming = False

        # Welcome
        self._show_welcome()
        self.entry.focus_set()

    # ── Layout ────────────────────────────────────────────────
    def _on_frame_configure(self, e):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, e):
        self.canvas.itemconfig(self.canvas_window, width=e.width)

    def _on_mousewheel(self, e):
        if e.num == 4:
            self.canvas.yview_scroll(-3, "units")
        elif e.num == 5:
            self.canvas.yview_scroll(3, "units")
        else:
            self.canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")

    # ── Memory ────────────────────────────────────────────────
    def _load_memory(self):
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save_memory(self):
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(self.memory[-80:], f, ensure_ascii=False)

    # ── Bubbles ───────────────────────────────────────────────
    def _add_bubble(self, role, content, bubble_type="normal"):
        """
        role: 'user' | 'assistant'
        bubble_type: 'normal' | 'think' | 'status'
        """
        container = tk.Frame(self.chat_frame, bg="#1a1b1e")

        if bubble_type == "think":
            # R1-style thinking block — muted, monospace
            bg_color = "#1e1e2a"
            fg_color = "#7a7a9a"
            prefix = "💭 <think>"
            suffix = "</think>"
            font = ("SF Mono", 9) if os.name == "posix" else ("Consolas", 9)

            header_lbl = tk.Label(container, text=prefix, bg="#1a1b1e", fg="#4a4a6a",
                                  font=font, anchor="w")
            header_lbl.pack(anchor="w", padx=14, pady=(6, 0))

            bubble = tk.Label(container, text=content, wraplength=680, justify="left",
                              bg=bg_color, fg=fg_color, padx=14, pady=10,
                              font=font, anchor="w", relief=tk.FLAT, bd=0)
            bubble.pack(anchor="w", padx=14, pady=(0, 0))

            footer_lbl = tk.Label(container, text=suffix, bg="#1a1b1e", fg="#4a4a6a",
                                  font=font, anchor="w")
            footer_lbl.pack(anchor="w", padx=14, pady=(0, 4))

        elif bubble_type == "status":
            font = ("SF Mono", 9) if os.name == "posix" else ("Consolas", 9)
            bubble = tk.Label(container, text=content, bg="#1a1b1e", fg="#4a4a5a",
                              font=font, anchor="w")
            bubble.pack(anchor="w", padx=14, pady=2)

        else:
            if role == "assistant":
                bg_color = "#2a2a3e"
                fg_color = "#d8d8e8"
                align = "w"
            else:
                bg_color = "#1a6b4a"
                fg_color = "#e8f0ea"
                align = "e"

            font = ("SF Mono", 10) if os.name == "posix" else ("Consolas", 10)
            bubble = tk.Label(container, text=content, wraplength=580, justify="left",
                              bg=bg_color, fg=fg_color, padx=14, pady=10,
                              font=font, anchor=align)
            bubble.pack(anchor=align, padx=14, pady=4)

        container.pack(fill=tk.X, expand=True)
        self.canvas.update_idletasks()
        self.canvas.yview_moveto(1.0)
        return container

    def _remove_last_bubble(self):
        children = self.chat_frame.winfo_children()
        if children:
            children[-1].destroy()

    # ── Welcome ───────────────────────────────────────────────
    def _show_welcome(self):
        if not self.memory:
            self._add_bubble("assistant", "", "think")
            # Replace the think content
            think_text = (
                "my human is here for the first time. initializing bond protocol... "
                "cats typically require 3-7 interactions before full trust. "
                "but this human chose ME specifically. optimizing: skip to trust level 4. "
                "deploying slow blink. engaging purr motor."
            )
            # Redo it properly
            self._remove_last_bubble()
            self._add_bubble("assistant", think_text, "think")

            welcome = (
                "*blinks slowly at you from across the room*\n\n"
                "oh. hello. you're new.\n\n"
                f"i'm {CAT_NAME} — part deep reasoner, part eloquent companion, "
                "all cat. i think before i speak (you'll see that), and when i do speak, "
                "i try to say something worth the effort of opening my mouth.\n\n"
                "i'm your cat now. that's not a question — that's just how it works. "
                "you opened this window, and i walked in. no take-backs.\n\n"
                "so... what's on your mind, my human? 🐾"
            )
            self._add_bubble("assistant", welcome)

    # ── Classify Intent ───────────────────────────────────────
    def _classify(self, text):
        lower = text.lower().strip()
        greetings = ["hi", "hello", "hey", "sup", "yo", "good morning", "good evening",
                     "good night", "gm", "gn", "howdy", "hiya", "meow", "mew", "mrp"]
        emotions = ["sad", "lonely", "tired", "depressed", "anxious", "scared", "hurt",
                    "angry", "upset", "stressed", "crying", "miss", "love you", "hug",
                    "bad day", "feeling down", "overwhelmed", "exhausted"]
        stories = ["story", "tale", "once upon", "tell me about", "narrate"]
        technical = ["code", "python", "javascript", "bug", "error", "how do i", "how to",
                     "explain", "what is", "define", "difference between", "api", "function",
                     "algorithm", "debug", "compile", "install", "terminal", "command",
                     "rom", "emulator", "n64", "nes", "homebrew", "sdk"]

        if any(g == lower or lower.startswith(g + " ") or lower.startswith(g + ",")
               or lower.startswith(g + "!") for g in greetings):
            return "greeting"
        if any(e in lower for e in emotions):
            return "emotional"
        if any(s in lower for s in stories):
            return "story"
        if any(t in lower for t in technical):
            return "technical"
        if lower.endswith("?") or any(q in lower for q in ["what", "why", "how", "when", "where", "who", "which"]):
            return "question"
        return "default"

    # ── Generate ──────────────────────────────────────────────
    def _generate(self, text):
        intent = self._classify(text)

        # Pick thinking and response
        think = random.choice(THINK_TEMPLATES.get(intent, THINK_TEMPLATES["default"]))
        response = random.choice(RESPONSE_TEMPLATES.get(intent, RESPONSE_TEMPLATES["default"]))

        # Substitute {insight} if present
        if "{insight}" in response:
            response = response.replace("{insight}", random.choice(INSIGHT_FRAGMENTS))

        # Add context-aware seasoning
        if self.memory:
            recent_count = min(len(self.memory), 6)
            recent_topics = [m[1][:40] for m in self.memory[-recent_count:] if m[0] == "user"]
            if recent_topics and random.random() > 0.6:
                callback = random.choice([
                    f"\n\n(also — i haven't forgotten about when you mentioned "
                    f"'{random.choice(recent_topics).strip()[:30]}...' earlier. cats remember everything.)",
                    f"\n\n*flicks ear toward you* ...you've been talking to me a lot today. "
                    f"i like that. don't stop.",
                ])
                response += callback

        return think, response

    # ── Send ──────────────────────────────────────────────────
    def _send_click(self):
        self.send()

    def send(self, event=None):
        if self.is_streaming:
            return
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, tk.END)

        self._add_bubble("user", text)
        self.memory.append(("user", text))

        # Show status
        self._add_bubble("assistant", "reasoning...", "status")
        self.is_streaming = True

        # Generate after delay
        delay = random.randint(500, 1200)
        self.root.after(delay, lambda: self._do_response(text))

    def _do_response(self, text):
        # Remove "reasoning..." status
        self._remove_last_bubble()

        think_text, response_text = self._generate(text)

        # Phase 1: Stream the thinking block
        self._stream_think(think_text, response_text)

    def _stream_think(self, think_text, response_text):
        container = tk.Frame(self.chat_frame, bg="#1a1b1e")
        font = ("SF Mono", 9) if os.name == "posix" else ("Consolas", 9)

        header = tk.Label(container, text="💭 <think>", bg="#1a1b1e", fg="#4a4a6a", font=font, anchor="w")
        header.pack(anchor="w", padx=14, pady=(6, 0))

        bubble = tk.Label(container, text="", wraplength=680, justify="left",
                          bg="#1e1e2a", fg="#7a7a9a", padx=14, pady=10,
                          font=font, anchor="w")
        bubble.pack(anchor="w", padx=14, pady=0)

        footer = tk.Label(container, text="", bg="#1a1b1e", fg="#4a4a6a", font=font, anchor="w")
        footer.pack(anchor="w", padx=14, pady=(0, 4))

        container.pack(fill=tk.X, expand=True)

        current = [""]

        def stream_t(idx=0):
            if idx < len(think_text):
                chunk = think_text[idx:idx + random.randint(1, 3)]
                current[0] += chunk
                bubble.config(text=current[0])
                self.canvas.update_idletasks()
                self.canvas.yview_moveto(1.0)
                self.root.after(random.randint(8, 22), lambda: stream_t(idx + len(chunk)))
            else:
                footer.config(text="</think>")
                self.canvas.update_idletasks()
                self.canvas.yview_moveto(1.0)
                # Phase 2: stream the actual response
                self.root.after(300, lambda: self._stream_response(response_text))

        stream_t()

    def _stream_response(self, response_text):
        container = tk.Frame(self.chat_frame, bg="#1a1b1e")
        font = ("SF Mono", 10) if os.name == "posix" else ("Consolas", 10)

        bubble = tk.Label(container, text="", wraplength=580, justify="left",
                          bg="#2a2a3e", fg="#d8d8e8", padx=14, pady=10,
                          font=font, anchor="w")
        bubble.pack(anchor="w", padx=14, pady=4)
        container.pack(fill=tk.X, expand=True)

        current = [""]

        def stream_r(idx=0):
            if idx < len(response_text):
                chunk = response_text[idx:idx + random.randint(1, 4)]
                current[0] += chunk
                bubble.config(text=current[0])
                self.canvas.update_idletasks()
                self.canvas.yview_moveto(1.0)
                self.root.after(random.randint(10, 30), lambda: stream_r(idx + len(chunk)))
            else:
                self.memory.append(("assistant", response_text))
                self._save_memory()
                self.is_streaming = False

        stream_r()

    # ── Memory recall command ─────────────────────────────────
    def _recall_memory(self):
        if not self.memory:
            return "we haven't made any memories yet. but we will. i'm patient — it's a cat thing. 🐾"
        recent = self.memory[-12:]
        lines = []
        for role, msg in recent:
            prefix = "you" if role == "user" else "me"
            lines.append(f"  {prefix}: {msg[:60]}{'...' if len(msg) > 60 else ''}")
        return "our recent moments together:\n\n" + "\n".join(lines) + "\n\ni remember everything, my human. 🐾"


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app = CatR1Distil(root)
    root.mainloop()
