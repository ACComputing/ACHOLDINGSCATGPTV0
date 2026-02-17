import tkinter as tk
import random

# ==========================================
# Cat R1 1.X — ChatGPT.com Style Tkinter GUI
# Cozy 1B LLM simulator 🐾
# ==========================================

class CatR1_1X_GUI:
    def __init__(self, root):
        self.root = root
        root.title("Cat R1 1.X 🐾")
        root.geometry("700x500")
        root.configure(bg="#202123")  # ChatGPT-like dark bg

        # ======================================
        # Top Title
        # ======================================
        self.title_label = tk.Label(
            root,
            text="Cat R1 1.X",
            bg="#202123",
            fg="white",
            font=("Segoe UI", 16, "bold")
        )
        self.title_label.pack(pady=(20, 10))

        # ======================================
        # Card Frame (like examples/capabilities/limits)
        # ======================================
        self.card_frame = tk.Frame(root, bg="#202123")
        self.card_frame.pack(expand=True)

        cards = [
            ("Examples", [
                '"Explain quantum computing in simple terms"',
                '"Write a short bedtime story"',
                '"Show a Python snippet to reverse a string"'
            ]),
            ("Capabilities", [
                "Remembers recent conversation",
                "Responds to /commands",
                "Simulated cozy 1B LLM"
            ]),
            ("Limitations", [
                "Responses are simulated",
                "No real web search",
                "Limited knowledge after 2021"
            ])
        ]

        for i, (title, lines) in enumerate(cards):
            card = tk.Frame(self.card_frame, bg="#2a2b32", bd=0, relief=tk.RAISED, padx=10, pady=10)
            card.grid(row=0, column=i, padx=10, pady=10)
            card_title = tk.Label(card, text=title, bg="#2a2b32", fg="white", font=("Segoe UI", 10, "bold"))
            card_title.pack(pady=(0,5))
            for line in lines:
                lbl = tk.Label(card, text=line, bg="#2a2b32", fg="#d4d4d8", font=("Segoe UI", 9), wraplength=150, justify="left")
                lbl.pack(anchor="w")

        # ======================================
        # Bottom input area
        # ======================================
        self.input_frame = tk.Frame(root, bg="#40414f")
        self.input_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.entry = tk.Entry(self.input_frame, bg="#40414f", fg="white", insertbackground="white", font=("Segoe UI", 10))
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10, pady=10)
        self.entry.bind("<Return>", self.send)

        self.send_btn = tk.Button(
            self.input_frame,
            text="➤",
            bg="#19c37d",
            fg="white",
            relief=tk.FLAT,
            font=("Segoe UI", 10, "bold"),
            command=self.send
        )
        self.send_btn.pack(side=tk.RIGHT, padx=10)

        # Memory for simulated 1B LLM
        self.memory = []

    # ======================================
    # Handle input
    # ======================================
    def send(self, event=None):
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, tk.END)
        self.memory.append(("user", text))
        response = self.generate_response(text)
        self.memory.append(("assistant", response))
        self.show_response_popup(response)

    # Show response in popup (ChatGPT style)
    def show_response_popup(self, message):
        popup = tk.Toplevel(self.root)
        popup.title("Cat R1 Response")
        popup.configure(bg="#343541")
        popup.geometry("500x250")
        lbl = tk.Label(popup, text=message, bg="#343541", fg="white", font=("Segoe UI", 10), wraplength=480, justify="left")
        lbl.pack(padx=10, pady=10, expand=True, fill=tk.BOTH)
        btn = tk.Button(popup, text="Close", bg="#19c37d", fg="white", command=popup.destroy)
        btn.pack(pady=(0,10))

    # ======================================
    # Simulated 1B LLM responses
    # ======================================
    def generate_response(self, text):
        lower = text.lower()
        if "hello" in lower or "hi" in lower:
            return "Hello! 🐾 I am Cat R1 1.X, your cozy simulated 1B LLM."
        if "story" in lower or "bedtime" in lower:
            return "Once upon a time, a tiny kitten found a glowing star and purred it to sleep… 🌙✨"
        if "code" in lower:
            return "Type /run followed by your code to simulate execution."
        if "memory" in lower:
            mem_text = "\n".join([f"{r}: {m}" for r, m in self.memory[-5:]])
            return f"Recent conversation:\n{mem_text}"
        fallback = [
            "Mmm… tell me more… 🐾",
            "Interesting! Let's explore that together…",
            "I see… can you elaborate a bit?"
        ]
        return random.choice(fallback)

# ==========================================
# Run
# ==========================================
if __name__ == "__main__":
    root = tk.Tk()
    app = CatR1_1X_GUI(root)
    root.mainloop()
c
