import tkinter as tk
import random

class CatR1_1X_1bit:
    def __init__(self, root):
        self.root = root
        self.root.title("Cat R1 1.X 1-bit 🐾")
        self.root.geometry("600x400")
        self.root.configure(bg="#101010")  # darker retro 1-bit feel
        self.root.resizable(False, False)

        # Top title
        self.title_label = tk.Label(
            root,
            text="Cat R1 1.X 1-bit 🐾",
            bg="#101010",
            fg="#fff",  # high contrast 1-bit
            font=("Courier New", 16, "bold")
        )
        self.title_label.pack(pady=(20, 10))

        # Cards frame
        self.card_frame = tk.Frame(root, bg="#101010")
        self.card_frame.pack(expand=True)

        cards = [
            ("Examples", ['"Hi"', '"Tell a story"', '"Python code?"']),
            ("Capabilities", ["Memory-aware", "Token-streaming", "1-bit LLM sim"]),
            ("Limitations", ["Very minimal", "Retro style", "Might guess"])
        ]

        for i, (title, lines) in enumerate(cards):
            card = tk.Frame(self.card_frame, bg="#202020", padx=8, pady=8)
            card.grid(row=0, column=i, padx=8, pady=8)
            tk.Label(card, text=title, bg="#202020", fg="#fff", font=("Courier New", 10, "bold")).pack(pady=(0,4))
            for line in lines:
                tk.Label(card, text=line, bg="#202020", fg="#ccc", font=("Courier New", 9),
                         wraplength=140, justify="left").pack(anchor="w")

        # Bottom input
        self.input_frame = tk.Frame(root, bg="#303030")
        self.input_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.entry = tk.Entry(self.input_frame, bg="#303030", fg="#fff", insertbackground="#fff", font=("Courier New", 10))
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8, pady=8)
        self.entry.bind("<Return>", self.send)

        self.send_btn = tk.Button(self.input_frame, text="→", bg="#101010", fg="#0f0", font=("Courier New", 10, "bold"),
                                  relief=tk.FLAT, command=self.send)
        self.send_btn.pack(side=tk.RIGHT, padx=8)

        # Memory
        self.memory = []

    def send(self, event=None):
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, tk.END)
        self.memory.append(("user", text))
        response = self.generate_response(text)
        self.memory.append(("assistant", response))
        self.show_streaming_popup(response)

    def show_streaming_popup(self, message):
        popup = tk.Toplevel(self.root)
        popup.title("Cat R1 1.X 1-bit Response 🐾")
        popup.configure(bg="#202020")
        popup.geometry("400x220")

        text_widget = tk.Text(popup, bg="#202020", fg="#0f0", font=("Courier New", 10), wrap="word")
        text_widget.pack(padx=8, pady=8, expand=True, fill=tk.BOTH)
        text_widget.configure(state="disabled")

        # Stream character-by-character like retro token stream
        def stream_text(index=0):
            if index < len(message):
                text_widget.configure(state="normal")
                text_widget.insert(tk.END, message[index])
                text_widget.see(tk.END)
                text_widget.configure(state="disabled")
                popup.after(40, lambda: stream_text(index+1))
            else:
                tk.Button(popup, text="CLOSE", bg="#0f0", fg="#101010", font=("Courier New", 10, "bold"),
                          command=popup.destroy).pack(pady=(4,8))

        stream_text()

    def generate_response(self, text):
        lower = text.lower()
        # Minimal 1-bit style responses
        if "hi" in lower or "hello" in lower:
            return "0101 Hello! I am Cat R1 1.X 1-bit 🐾"
        if "code" in lower or "python" in lower:
            return "0110 print('hi from 1-bit Cat R1')"
        if "story" in lower or "bedtime" in lower:
            return "1001 tiny kitten finds a glowing star. purr… 🌙"
        if "memory" in lower:
            mem_text = "\n".join([f"{r[0]}: {r[1]}" for r in self.memory[-5:]])
            return f"0011 Recent chat:\n{mem_text}"
        # Fallback 1-bit minimal phrases
        return random.choice(["0001 yes… tell more", "0010 hmm… interesting…", "0100 keep going… 🐾"])

# Run
if __name__ == "__main__":
    root = tk.Tk()
    app = CatR1_1X_1bit(root)
    root.mainloop()
