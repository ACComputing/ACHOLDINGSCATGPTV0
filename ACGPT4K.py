import tkinter as tk
import random

# ===============================
# Cat R1 1.X — Cozy O1-style Engine Tkinter
# ===============================

class CatR1_1X_O1Engine:
    def __init__(self, root):
        self.root = root
        root.title("Cat R1 1.X 🐾 O1 Engine")
        root.geometry("700x500")
        root.configure(bg="#202123")

        # Top title
        self.title_label = tk.Label(root, text="Cat R1 1.X 🐾 O1 Engine",
                                    bg="#202123", fg="white", font=("Segoe UI", 16, "bold"))
        self.title_label.pack(pady=(20, 10))

        # Card frame (examples/capabilities/limits)
        self.card_frame = tk.Frame(root, bg="#202123")
        self.card_frame.pack(expand=True)

        cards = [
            ("Examples", [
                '"Explain quantum computing simply"',
                '"Tell a bedtime story"',
                '"Simulate a Python snippet"'
            ]),
            ("Capabilities", [
                "Token-by-token streaming responses",
                "Memory-aware cozy guidance",
                "Simulated 1B LLM engine"
            ]),
            ("Limitations", [
                "Offline & simulated only",
                "No real web search",
                "Knowledge limited after 2021"
            ])
        ]

        for i, (title, lines) in enumerate(cards):
            card = tk.Frame(self.card_frame, bg="#2a2b32", padx=10, pady=10)
            card.grid(row=0, column=i, padx=10, pady=10)
            tk.Label(card, text=title, bg="#2a2b32", fg="white", font=("Segoe UI", 10, "bold")).pack(pady=(0,5))
            for line in lines:
                tk.Label(card, text=line, bg="#2a2b32", fg="#d4d4d8", font=("Segoe UI", 9),
                         wraplength=150, justify="left").pack(anchor="w")

        # Bottom input area
        self.input_frame = tk.Frame(root, bg="#40414f")
        self.input_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.entry = tk.Entry(self.input_frame, bg="#40414f", fg="white",
                              insertbackground="white", font=("Segoe UI", 10))
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10, pady=10)
        self.entry.bind("<Return>", self.send)

        self.send_btn = tk.Button(self.input_frame, text="➤", bg="#19c37d", fg="white",
                                  relief=tk.FLAT, font=("Segoe UI", 10, "bold"), command=self.send)
        self.send_btn.pack(side=tk.RIGHT, padx=10)

        # Memory
        self.memory = []

    # Handle input
    def send(self, event=None):
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, tk.END)
        self.memory.append(("user", text))
        response = self.generate_response(text)
        self.memory.append(("assistant", response))
        self.show_streaming_popup(response)

    # Streaming popup (O1 style)
    def show_streaming_popup(self, message):
        popup = tk.Toplevel(self.root)
        popup.title("Cat R1 Response 🐾")
        popup.configure(bg="#343541")
        popup.geometry("500x250")

        text_widget = tk.Text(popup, bg="#343541", fg="white", font=("Segoe UI", 10), wrap="word")
        text_widget.pack(padx=10, pady=10, expand=True, fill=tk.BOTH)
        text_widget.configure(state="disabled")

        def stream_text(idx=0):
            if idx < len(message):
                text_widget.configure(state="normal")
                text_widget.insert(tk.END, message[idx])
                text_widget.see(tk.END)
                text_widget.configure(state="disabled")
                popup.after(random.randint(20,40), lambda: stream_text(idx+1))
            else:
                tk.Button(popup, text="Close", bg="#19c37d", fg="white",
                          font=("Segoe UI", 10, "bold"), command=popup.destroy).pack(pady=(5,10))

        stream_text()

    # Cozy O1 engine response generation
    def generate_response(self, text):
        lower = text.lower()
        if "hello" in lower or "hi" in lower:
            return "Hello there! 🐾 I’m Cat R1 1.X, your cozy O1 engine companion."
        if "story" in lower or "bedtime" in lower:
            return "Once upon a time, a tiny kitten found a glowing star and purred it to sleep… 🌙✨"
        if "code" in lower or "python" in lower:
            return "Type /run followed by your code to simulate execution."
        if "memory" in lower:
            mem_text = "\n".join([f"{r}: {m}" for r, m in self.memory[-5:]])
            return f"Here’s what I remember from recent chat:\n{mem_text}"
        fallback = [
            "Mmm… tell me a little more… 🐾",
            "Interesting! Let's explore that together…",
            "I see… can you elaborate a bit?"
        ]
        return random.choice(fallback)


# ===============================
# Run
# ===============================
if __name__ == "__main__":
    root = tk.Tk()
    app = CatR1_1X_O1Engine(root)
    root.mainloop()
