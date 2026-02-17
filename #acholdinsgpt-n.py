import tkinter as tk

# ==========================================
# AC HOLDINGS 1999-2026 CAT R1 V0.X — Token-Streaming Edition
# ==========================================

class CatR1Home:
    def __init__(self, root):
        self.root = root
        self.root.title("AC HOLDINGS 1999-2026 CAT R1 V0.X")
        self.root.geometry("600x400")
        self.root.configure(bg="#202123")
        self.root.resizable(False, False)

        # ======================================
        # Top Title
        # ======================================
        self.title_label = tk.Label(
            root,
            text="AC HOLDINGS 1999-2026 CAT R1 V0.X",
            bg="#202123",
            fg="white",
            font=("Segoe UI", 16, "bold")
        )
        self.title_label.pack(pady=(20, 10))

        # ======================================
        # Card Frame (centered)
        # ======================================
        self.card_frame = tk.Frame(root, bg="#202123")
        self.card_frame.pack(expand=True)

        cards = [
            ("Examples", [
                '"Explain quantum computing in simple terms"',
                '"Got any creative ideas for a 10-year-old’s birthday?"',
                '"How do I make an HTTP request in Python?"'
            ]),
            ("Capabilities", [
                "Remembers what user said earlier in the conversation",
                "Allows user to provide follow-up corrections",
                "Trained to decline inappropriate requests"
            ]),
            ("Limitations", [
                "May occasionally generate incorrect information",
                "May occasionally produce harmful instructions or biased content",
                "Limited knowledge of world and events after 2021"
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
        # Bottom Input Area
        # ======================================
        self.input_frame = tk.Frame(root, bg="#40414f")
        self.input_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.entry = tk.Entry(self.input_frame, bg="#40414f", fg="white", insertbackground="white", font=("Segoe UI", 10))
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10, pady=10)
        self.entry.bind("<Return>", self.send)

        self.send_btn = tk.Button(
            self.input_frame,
            text="➤",
            bg="black",
            fg="#19c37d",
            relief=tk.FLAT,
            font=("Segoe UI", 10, "bold"),
            command=self.send
        )
        self.send_btn.pack(side=tk.RIGHT, padx=10)

        # ======================================
        # Memory
        # ======================================
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
        self.show_streaming_popup(response)

    # ======================================
    # Streaming popup
    # ======================================
    def show_streaming_popup(self, message):
        popup = tk.Toplevel(self.root)
        popup.title("Cat R1 Response")
        popup.configure(bg="#343541")
        popup.geometry("400x200")

        text_widget = tk.Text(popup, bg="#343541", fg="white", font=("Segoe UI", 10), wrap="word")
        text_widget.pack(padx=10, pady=10, expand=True, fill=tk.BOTH)
        text_widget.configure(state="disabled")

        def stream_text(index=0):
            if index < len(message):
                text_widget.configure(state="normal")
                text_widget.insert(tk.END, message[index])
                text_widget.see(tk.END)
                text_widget.configure(state="disabled")
                popup.after(30, lambda: stream_text(index+1))  # gentle token delay
            else:
                close_btn = tk.Button(popup, text="Close", bg="#19c37d", fg="white", command=popup.destroy)
                close_btn.pack(pady=(0,10))

        stream_text()

    # ======================================
    # Tiny 1-bit-style response engine
    # ======================================
    def generate_response(self, text):
        lower = text.lower()
        if "hello" in lower or "hi" in lower:
            return "Hello! Welcome to AC HOLDINGS 1999-2026 CAT R1 V0.X 🐾 How can I help you today?"
        if "code" in lower:
            return "Tell me the language and I can generate it for you."
        if "memory" in lower:
            return "Conversation memory is stored in RAM only."
        return "I understand. Can you give me more details?"

# ==========================================
# Run
# ==========================================
if __name__ == "__main__":
    root = tk.Tk()
    app = CatR1Home(root)
    root.mainloop()
