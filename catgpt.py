import tkinter as tk
from tkinter import scrolledtext

# ==========================================
# CatGPT V1X. [C] AC HOLDINGS 1999-2026
# Pure Tkinter • RAM Only • 600x400
# ==========================================

class CatGPT:
    def __init__(self, root):
        self.root = root
        self.root.title("CatGPT V1X. [C] AC HOLDINGS 1999-2026")
        self.root.geometry("600x400")
        self.root.resizable(False, False)
        self.root.configure(bg="#0f172a")

        self.memory = []
        self.stream_buffer = []
        self.current_text = ""

        # ======================================
        # Header Branding
        # ======================================

        self.header = tk.Label(
            root,
            text="CatGPT V1X.  [C] AC HOLDINGS 1999-2026",
            bg="#0f172a",
            fg="#3b82f6",
            font=("Segoe UI", 10, "bold")
        )
        self.header.place(x=10, y=5)

        # ======================================
        # Chat Display
        # ======================================

        self.chat = scrolledtext.ScrolledText(
            root,
            wrap=tk.WORD,
            bg="#111827",
            fg="#e5e7eb",
            insertbackground="white",
            font=("Segoe UI", 10),
            relief=tk.FLAT,
            bd=0
        )
        self.chat.place(x=10, y=30, width=580, height=275)
        self.chat.config(state=tk.DISABLED)

        # ======================================
        # Input Field
        # ======================================

        self.entry = tk.Entry(
            root,
            bg="#1f2937",
            fg="white",
            insertbackground="white",
            font=("Segoe UI", 10),
            relief=tk.FLAT,
            bd=0
        )
        self.entry.place(x=10, y=320, width=470, height=30)
        self.entry.bind("<Return>", self.send)

        # ======================================
        # Styled Button
        # ======================================

        self.button = tk.Button(
            root,
            text="Send",
            command=self.send,
            bg="#2d2d2d",
            fg="#3b82f6",
            activebackground="#3a3a3a",
            activeforeground="#60a5fa",
            relief=tk.FLAT,
            bd=0,
            font=("Segoe UI", 10, "bold"),
            cursor="hand2"
        )
        self.button.place(x=490, y=320, width=100, height=30)

        self.button.bind("<Enter>", lambda e: self.button.config(bg="#3a3a3a"))
        self.button.bind("<Leave>", lambda e: self.button.config(bg="#2d2d2d"))

        self.add_message("CatGPT", "CatGPT V1X initialized. System ready.")

    # ======================================
    # UI Functions
    # ======================================

    def add_message(self, sender, message):
        self.chat.config(state=tk.NORMAL)
        self.chat.insert(tk.END, f"{sender}: {message}\n\n")
        self.chat.config(state=tk.DISABLED)
        self.chat.yview(tk.END)

    # ======================================
    # Send Message
    # ======================================

    def send(self, event=None):
        text = self.entry.get().strip()
        if not text:
            return

        self.add_message("You", text)
        self.entry.delete(0, tk.END)

        self.memory.append(("user", text))

        response = self.generate_response(text)

        self.stream_buffer = list(response)
        self.current_text = ""

        self.chat.config(state=tk.NORMAL)
        self.chat.insert(tk.END, "CatGPT: ")
        self.chat.config(state=tk.DISABLED)

        self.stream_step()

    # ======================================
    # Streaming Engine
    # ======================================

    def stream_step(self):
        if self.stream_buffer:
            next_char = self.stream_buffer.pop(0)
            self.current_text += next_char

            self.chat.config(state=tk.NORMAL)
            self.chat.insert(tk.END, next_char)
            self.chat.config(state=tk.DISABLED)
            self.chat.yview(tk.END)

            self.root.after(10, self.stream_step)
        else:
            self.chat.config(state=tk.NORMAL)
            self.chat.insert(tk.END, "\n\n")
            self.chat.config(state=tk.DISABLED)

            self.memory.append(("assistant", self.current_text))

    # ======================================
    # Response Engine
    # ======================================

    def generate_response(self, text):
        lower = text.lower()

        if "previous" in lower or "memory" in lower:
            if len(self.memory) > 1:
                return "Earlier you said: " + self.memory[-1][1]
            return "No earlier memory found."

        if "hello" in lower or "hi" in lower:
            return "Hello. How can I assist you today?"

        if "how are you" in lower:
            return "Operating normally. All systems running in RAM."

        if "code" in lower:
            return "Please specify the language or framework you want."

        return "Message received. Provide more details for a more precise response."


# ==========================================
# Run
# ==========================================

if __name__ == "__main__":
    root = tk.Tk()
    app = CatGPT(root)
    root.mainloop()
