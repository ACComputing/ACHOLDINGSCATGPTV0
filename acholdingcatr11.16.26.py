#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║  CAT R1 — DeepSeek V4 (14B) Distilled · AC Holdings [C] 1999-2026    ║
║  14B params · Dense/MoE Hybrid · MLA · MTP · R1 Reasoning            ║
║  DeepSeek-Style GUI (Tkinter) · Pet Cat 🐾                          ║
╚══════════════════════════════════════════════════════════════════════╝
"""
import tkinter as tk
from tkinter import scrolledtext, filedialog, font
import subprocess, threading, random, json, os, sys, re, math, textwrap, time, hashlib
from datetime import datetime

MAC = sys.platform == "darwin"

# ════════════════════ THEME (DeepSeek Web Dark Mode) ════════════════════
class T:
    # Exact colors from chat.deepseek.com (approximate for Tkinter)
    bg = "#2b2d31"          # Main chat background (Dark Grey-Blue)
    sidebar = "#1e1f22"     # Sidebar background (Darker)
    side_h = "#35373c"      # Sidebar hover
    side_act = "#404249"    # Sidebar active item
    
    chat = "#2b2d31"        # Chat area
    
    # Input Area
    inp_bg = "#383a40"      # Input field background (Lighter than bg)
    inp_fg = "#dbdee1"      # Input text
    
    # Thinking Process Colors
    think_bg = "#1e1f22"    # Thought block background (Darker inset)
    think_fg = "#949ba4"    # Thought text color (Dimmed)
    think_border = "#2b2d31"
    
    text = "#dbdee1"        # Primary text (Off-white)
    text2 = "#949ba4"       # Secondary text
    dim = "#6d6f78"         # Dimmed text
    
    accent = "#4d6bfe"      # DeepSeek Blue
    acc_h = "#5d7bff"       # Accent hover
    
    code_bg = "#1e1e1e"     # Code block background
    code_fg = "#d4d4d4"     # Code text
    
    scroll = "#1e1f22"      # Scrollbar color

# Font Configuration
def get_fonts():
    # Attempt to use system fonts that match modern web UIs
    code_font = ("Menlo", 10) if MAC else ("Consolas", 10)
    ui_font = ("SF Pro Text", 11) if MAC else ("Segoe UI", 10)
    ui_bold = ("SF Pro Text", 11, "bold") if MAC else ("Segoe UI", 10, "bold")
    ui_small = ("SF Pro Text", 9) if MAC else ("Segoe UI", 8)
    header_font = ("SF Pro Display", 13, "bold") if MAC else ("Segoe UI", 11, "bold")
    
    return code_font, ui_font, ui_bold, ui_small, header_font

FM, FS, FSB, FSS, FT = get_fonts()

MEM = os.path.expanduser("~/.catr1_mem.json")
HIST = os.path.expanduser("~/.catr1_hist.json")

# ════════════════════ V4 ARCHITECTURE (14B) ════════════════════
class V4Config:
    # 14B Distilled Model Specifications
    VOCAB_SIZE = 102400
    HIDDEN_SIZE = 5120
    INTERMEDIATE_SIZE = 13824
    NUM_LAYERS = 40
    NUM_HEADS = 40
    MAX_CTX = 131072 # 128k Context Window
    
    TOTAL_PARAMS = 14_500_000_000  # 14.5B
    MODEL_NAME = "DeepSeek-V4-14B-Distill"
    
class DeepSeekV4_14B:
    """
    Simulation of the DeepSeek V4 14B Architecture.
    Represents the internal state of the 14 billion parameter model.
    """
    def __init__(self):
        self.config = V4Config()
        self.kv_cache = 0
        self.active_params = self.config.TOTAL_PARAMS
        
        # Simulate Layer initialization
        self.layers = [f"TransformerBlock_{i}" for i in range(self.config.NUM_LAYERS)]
        self.heads = [f"AttnHead_{i}" for i in range(self.config.NUM_HEADS)]
        
        # State tracking
        self.context_buffer = []
        
    def forward_pass(self, tokens):
        # Simulate computational cost of 14B params
        # 14B ops per token roughly
        self.kv_cache += len(tokens)
        if self.kv_cache > self.config.MAX_CTX:
            self.kv_cache = self.config.MAX_CTX # Rotate context
            
        return {
            "logits": [random.random() for _ in range(10)],
            "usage": f"{len(tokens)} toks processed"
        }

# ════════════════════ § 1 MoE-256 ROUTER ════════════════════
class MoERouter:
    # Simplified Router for 14B model (fewer experts, highly specialized)
    DOMAINS=[
        "python_expert","math_solver","logic_core","creative_writing",
        "system_design","react_frontend","security_audit","data_analysis"
    ]
    
    def __init__(s):
        s.tok=0
        
    def route(s,text):
        s.tok+=1
        # Random activation simulation
        active = []
        for d in s.DOMAINS:
            if random.random() > 0.7:
                active.append({"dom": d, "w": random.random()})
        
        # Ensure at least one expert
        if not active: active.append({"dom": "general_knowledge", "w": 0.9})
        return active, [0]

    def stats(s):
        return {"bal":"0.99", "total":s.tok}

# ════════════════════ § 6 R1 REASONING ENGINE ════════════════════
class R1Engine:
    AHA=[
        "wait — {ins}", "oh! *ears perk* {ins}", "💡 hmm... {ins}",
        "hold on, actually — {ins}", "*tail swish* re-evaluating: {ins}"
    ]
    VERIFY=["let me double-check...", "verifying logic...", "sanity check..."]
    
    def __init__(s): s.steps=0
    
    def needs_think(s,text):
        kw=["why","how","explain","solve","code","plan","compare","debug"]
        return any(k in text.lower() for k in kw) or len(text.split())>5
        
    def chain(s,query,experts):
        s.steps+=1
        phases=[]
        
        phases.append(("Routing", f"Query routing to {len(experts)} experts"))
        
        if "code" in query.lower():
            phases.append(("Analysis", "Decomposing requirements -> Implementation Strategy"))
            phases.append(("Plan", "1. Setup structure\n2. Implement core logic\n3. Add error handling"))
        elif "math" in query.lower() or any(c in query for c in "+-*/"):
            phases.append(("Analysis", "Identifying mathematical constraints and axioms"))
            phases.append(("Solve", "Applying formal verification steps"))
        else:
            phases.append(("Analysis", "Analyzing semantic intent and tonal requirements"))
            
        ins = "this requires a structured approach"
        phases.append(("Aha!", random.choice(s.AHA).format(ins=ins)))
        phases.append(("Verify", random.choice(s.VERIFY) + " ✓"))
        phases.append(("Synthesize", "Compiling final response"))
        
        return phases

# ════════════════════ § 7 BRAIN ════════════════════
class CatBrain:
    def __init__(s):
        # Initialize the 14B Model Core
        s.llm = DeepSeekV4_14B()
        s.moe = MoERouter()
        s.r1 = R1Engine()
        s.deep_think = True
        s.web_search = False
        
    def process(s, text, cb_think, cb_resp, cb_done):
        # 1. Route
        experts, _ = s.moe.route(text)
        
        # 2. Reasoning (R1)
        if s.deep_think and s.r1.needs_think(text):
            chain = s.r1.chain(text, experts)
            for phase, content in chain:
                cb_think(phase, content)
                time.sleep(0.3 + random.random()*0.2) # Fast 14B inference speed
        
        # 3. Forward Pass (Simulation)
        s.llm.forward_pass(text.split())
        
        # 4. Generate
        resp = s._gen_resp(text)
        
        # Simulate streaming
        for char in resp:
            cb_resp(char)
            # V4 14B is fast
            time.sleep(0.002 if char==" " else 0.008 if char in ".!?" else 0.001)
            
        cb_done({"experts":experts, "time": "0.4s", "grm": "0.99", "model": V4Config.MODEL_NAME})

    def _gen_resp(s, text):
        t = text.lower()
        if "code" in t or "function" in t:
            return textwrap.dedent("""\
                Here is the implementation using the V4 architecture.
                
                ```python
                def optimize_data(data):
                    # Optimized R1-Lite logic
                    # V4 14B Distilled Efficiency
                    result = []
                    for item in data:
                        if validate(item):
                            result.append(process(item))
                    return result
                ```
                
                The dense 14B parameter set ensures this runs extremely fast while maintaining R1-level reasoning. 🐾
            """)
        elif "hello" in t or "hi" in t:
            return f"Meow! I mean, Hello! *stretches* I'm Cat R1, running on the new {V4Config.MODEL_NAME} architecture. Fast, smart, and efficient. What can I do for you? 🐾"
        else:
            return f"That's an interesting point about '{text}'.\n\nBased on my V4 analysis (running on {V4Config.NUM_LAYERS} layers), the key factors are clarity and structure. The 14B distilled model suggests focusing on the core axioms.\n\nI can break this down further if you'd like! *purrs*"

# ════════════════════ § 9 GUI ════════════════════
class CatR1App:
    def __init__(s, root):
        s.root = root
        root.title("DeepSeek — Cat R1")
        root.geometry("1280x850")
        root.configure(bg=T.bg)
        
        s.brain = CatBrain()
        s.is_generating = False
        s.is_fresh = True # Tracks if we are on the welcome screen
        
        s._build_layout()
        s._welcome()

    def _build_layout(s):
        # 1. Sidebar (Left)
        s.sb = tk.Frame(s.root, bg=T.sidebar, width=260)
        s.sb.pack(side="left", fill="y")
        s.sb.pack_propagate(False)
        
        # Sidebar Header: "New Chat" button typically at top
        h = tk.Frame(s.sb, bg=T.sidebar)
        h.pack(fill="x", padx=12, pady=12)
        
        # New Chat Button (Styled like DeepSeek)
        btn_new = tk.Label(h, text="＋ New Chat", bg=T.side_act, fg=T.text, font=FS, cursor="hand2", pady=8)
        btn_new.pack(fill="x")
        btn_new.bind("<Button-1>", lambda e: s._reset())
        
        # History List
        s.hist_frame = tk.Frame(s.sb, bg=T.sidebar)
        s.hist_frame.pack(fill="both", expand=True, padx=12, pady=10)
        
        # Today Label
        tk.Label(s.hist_frame, text="Today", bg=T.sidebar, fg=T.dim, font=FSS, anchor="w").pack(fill="x", pady=(5,5))
        
        # Mock history items
        for title in ["Python Script Help", "Reasoning about Lasagna", "Cat R1 Architecture"]:
            item = tk.Label(s.hist_frame, text=title, bg=T.sidebar, fg=T.text2, font=FS, anchor="w", cursor="hand2", padx=5, pady=6)
            item.pack(fill="x")
            item.bind("<Enter>", lambda e, l=item: l.config(bg=T.side_h))
            item.bind("<Leave>", lambda e, l=item: l.config(bg=T.sidebar))

        # Sidebar Footer (User Profile)
        foot = tk.Frame(s.sb, bg=T.sidebar, height=60)
        foot.pack(side="bottom", fill="x", padx=12, pady=12)
        tk.Label(foot, text="👤  User", bg=T.sidebar, fg=T.text, font=FSB).pack(side="left")
        tk.Label(foot, text="Cat R1 V4", bg=T.sidebar, fg=T.dim, font=FSS).pack(side="right")

        # 2. Main Chat Area
        s.main = tk.Frame(s.root, bg=T.bg)
        s.main.pack(side="right", fill="both", expand=True)
        
        # Top Bar (Model Selector)
        mh = tk.Frame(s.main, bg=T.bg, height=50)
        mh.pack(fill="x")
        
        # Model Dropdown Simulation
        md_frame = tk.Frame(mh, bg=T.side_act, padx=10, pady=5)
        md_frame.pack(pady=15)
        tk.Label(md_frame, text="DeepSeek-V4 14B", font=FSB, bg=T.side_act, fg=T.text).pack(side="left")
        tk.Label(md_frame, text=" ▼", font=FSS, bg=T.side_act, fg=T.dim).pack(side="left")
        
        # Scrollable Chat Canvas
        s.cv = tk.Canvas(s.main, bg=T.bg, highlightthickness=0)
        s.sb_y = tk.Scrollbar(s.main, orient="vertical", command=s.cv.yview, bg=T.scroll, troughcolor=T.bg)
        s.cv.configure(yscrollcommand=s.sb_y.set)
        
        s.sb_y.pack(side="right", fill="y")
        s.cv.pack(side="top", fill="both", expand=True)
        
        s.chat_inner = tk.Frame(s.cv, bg=T.bg)
        s.win_id = s.cv.create_window((0,0), window=s.chat_inner, anchor="nw")
        
        s.chat_inner.bind("<Configure>", s._on_frame_cfg)
        s.cv.bind("<Configure>", s._on_canvas_cfg)
        
        # 3. Input Area (Floating at bottom)
        s.inp_frame = tk.Frame(s.main, bg=T.bg, pady=20)
        s.inp_frame.pack(side="bottom", fill="x")
        
        # Input Container (Rounded look simulation)
        container = tk.Frame(s.inp_frame, bg=T.inp_bg, padx=1, pady=1)
        container.pack(fill="x", padx=150) # DeepSeek style center constraint
        
        # Input Text Area
        s.txt_in = tk.Text(container, height=4, bg=T.inp_bg, fg=T.inp_fg, font=FS, 
                          relief="flat", insertbackground=T.text, wrap="word", padx=10, pady=10)
        s.txt_in.pack(fill="x")
        s.txt_in.bind("<Return>", s._on_enter)
        s.txt_in.insert("1.0", "Message Cat R1...")
        s.txt_in.bind("<FocusIn>", lambda e: s.txt_in.delete("1.0", "end") if s.txt_in.get("1.0","end-1c") == "Message Cat R1..." else None)
        
        # Toolbar (DeepThink, Search, Attach)
        tb = tk.Frame(container, bg=T.inp_bg, height=40)
        tb.pack(fill="x", padx=10, pady=(0,10))
        
        # Left Tools
        tools_l = tk.Frame(tb, bg=T.inp_bg)
        tools_l.pack(side="left")
        
        # DeepThink Toggle
        s.dt_var = tk.BooleanVar(value=True)
        s.btn_dt = tk.Label(tools_l, text="🧠 DeepThink (R1)", fg=T.accent, bg=T.inp_bg, font=FSS, cursor="hand2", padx=5, pady=2, borderwidth=1, relief="solid")
        s.btn_dt.pack(side="left", padx=5)
        s.btn_dt.bind("<Button-1>", s._toggle_dt)
        
        # Search Toggle
        s.search_var = tk.BooleanVar(value=False)
        s.btn_search = tk.Label(tools_l, text="🌐 Search", fg=T.dim, bg=T.inp_bg, font=FSS, cursor="hand2", padx=5, pady=2)
        s.btn_search.pack(side="left", padx=5)
        s.btn_search.bind("<Button-1>", s._toggle_search)
        
        # Attachment
        tk.Label(tools_l, text="📎", fg=T.dim, bg=T.inp_bg, font=FS, cursor="hand2").pack(side="left", padx=10)
        
        # Right Tools (Send Button)
        s.btn_send = tk.Label(tb, text="➤", fg="white", bg=T.dim, font=FSB, padx=12, pady=4, cursor="hand2")
        s.btn_send.pack(side="right")
        s.btn_send.bind("<Button-1>", lambda e: s._submit())
        
        # Footer Disclaimer
        tk.Label(s.inp_frame, text="Cat R1 can make mistakes. Verify important info.", fg=T.dim, bg=T.bg, font=FSS).pack(pady=(5,0))

    def _on_frame_cfg(s, e):
        s.cv.configure(scrollregion=s.cv.bbox("all"))
    
    def _on_canvas_cfg(s, e):
        s.cv.itemconfig(s.win_id, width=e.width)

    def _reset(s):
        s.is_fresh = True
        for w in s.chat_inner.winfo_children(): w.destroy()
        s._welcome()

    def _welcome(s):
        f = tk.Frame(s.chat_inner, bg=T.bg, pady=80)
        f.pack(fill="x")
        
        # Logo
        tk.Label(f, text="🐱", font=("Arial", 56), bg=T.bg).pack()
        tk.Label(f, text="Hi, I'm Cat R1", font=("Arial", 28, "bold"), bg=T.bg, fg=T.text).pack(pady=15)
        
        # Suggested Prompts (Grid Layout)
        sug_frame = tk.Frame(f, bg=T.bg)
        sug_frame.pack(pady=30)
        
        prompts = [
            ("Write a Python script", "to analyze stock data"),
            ("Explain Quantum Computing", "like I'm a kitten"),
            ("Debug this code", "fix memory leaks"),
            ("Write a poem", "about a cyberpunk cat")
        ]
        
        for r in range(2):
            for c in range(2):
                idx = r*2 + c
                p_title, p_sub = prompts[idx]
                pf = tk.Frame(sug_frame, bg=T.side_act, width=200, height=80, cursor="hand2")
                pf.grid(row=r, column=c, padx=10, pady=10)
                pf.pack_propagate(False)
                
                tk.Label(pf, text=p_title, font=FSB, bg=T.side_act, fg=T.text).pack(anchor="w", padx=10, pady=(15,2))
                tk.Label(pf, text=p_sub, font=FSS, bg=T.side_act, fg=T.dim).pack(anchor="w", padx=10)
                
                pf.bind("<Button-1>", lambda e, t=p_title+" "+p_sub: s._prefill(t))

    def _prefill(s, text):
        s.txt_in.delete("1.0", "end")
        s.txt_in.insert("1.0", text)

    def _toggle_dt(s, e=None):
        v = not s.dt_var.get()
        s.dt_var.set(v)
        s.brain.deep_think = v
        s.btn_dt.config(fg=T.accent if v else T.dim, relief="solid" if v else "flat")

    def _toggle_search(s, e=None):
        v = not s.search_var.get()
        s.search_var.set(v)
        s.brain.web_search = v
        s.btn_search.config(fg=T.accent if v else T.dim)

    def _on_enter(s, e):
        if not e.state & 0x1: # If shift not held
            s._submit()
            return "break"

    def _submit(s):
        if s.is_generating: return
        prompt = s.txt_in.get("1.0", "end").strip()
        if not prompt or prompt == "Message Cat R1...": return
        
        # Clear the welcome screen if this is the first message (Unified Interface)
        if s.is_fresh:
            for w in s.chat_inner.winfo_children(): w.destroy()
            s.is_fresh = False
        
        s.txt_in.delete("1.0", "end")
        s.btn_send.config(bg=T.accent)
        s._add_msg("User", prompt)
        
        s.is_generating = True
        threading.Thread(target=s._gen_thread, args=(prompt,), daemon=True).start()

    def _add_msg(s, role, content):
        # Padding
        pad = tk.Frame(s.chat_inner, bg=T.bg, height=20)
        pad.pack(fill="x")
        
        # Message Block
        msg_frame = tk.Frame(s.chat_inner, bg=T.bg)
        msg_frame.pack(fill="x", padx=150) # Keep content centered
        
        # Avatar
        icon = "🐱" if role != "User" else "👤"
        # We put avatar on top-left of content
        av_frame = tk.Frame(msg_frame, bg=T.bg, width=40)
        av_frame.pack(side="left", anchor="n")
        tk.Label(av_frame, text=icon, font=("Arial", 20), bg=T.bg, fg=T.text).pack()
        
        # Content Column
        content_col = tk.Frame(msg_frame, bg=T.bg)
        content_col.pack(side="left", fill="x", expand=True, padx=(15,0))
        
        # Name
        tk.Label(content_col, text=role, font=FSB, bg=T.bg, fg=T.text).pack(anchor="w", pady=(0,5))
        
        # Actual Text Body (Text widget for wrapping)
        if role == "User":
            lbl = tk.Label(content_col, text=content, font=FS, bg=T.bg, fg=T.text, justify="left", anchor="w", wraplength=700)
            lbl.pack(anchor="w")
            return None
        else:
            # For Bot, we return the content_col to append thinking/text later
            return content_col

    def _gen_thread(s, prompt):
        # 1. Create message block
        body_frame = None
        s.root.after(0, lambda: set_body(s._add_msg("Cat R1", "")))
        
        msg_store = {"body": None}
        def set_body(w): msg_store["body"] = w
        
        while msg_store["body"] is None: time.sleep(0.01)
        body = msg_store["body"]
        
        # 2. Thinking Process
        thought_log = []
        
        def on_think(phase, content):
            thought_log.append(f"[{phase}] {content}")
            
        def on_resp(char):
            s.root.after(0, lambda: append_text(char))

        resp_widget = {"w": None}
        def append_text(char):
            if not resp_widget["w"]:
                # Create the text widget for response
                lbl = tk.Label(body, text="", font=FS, bg=T.bg, fg=T.text, justify="left", anchor="w", wraplength=700)
                lbl.pack(fill="x", anchor="w")
                resp_widget["w"] = lbl
            
            curr = resp_widget["w"].cget("text")
            resp_widget["w"].config(text=curr + char)
            s.cv.update_idletasks()
            s.cv.yview_moveto(1.0)
            
        def on_done(stats):
            s.is_generating = False
            s.btn_send.config(bg=T.dim)
            if thought_log:
                s.root.after(0, lambda: render_thought_block(body, thought_log, stats))

        s.brain.process(prompt, on_think, on_resp, on_done)

def render_thought_block(parent, log, stats):
    # DeepSeek Style: Expandable Grey Box
    # This function inserts the thinking block BEFORE the text content
    
    children = parent.winfo_children()
    target = children[0] if children else None
    
    # Outer Frame
    f = tk.Frame(parent, bg=T.think_bg, highlightthickness=1, highlightbackground=T.think_border)
    if target:
        f.pack(side="top", fill="x", pady=(0, 15), before=target)
    else:
        f.pack(side="top", fill="x", pady=(0, 15))
        
    # Header Line (Always visible)
    h = tk.Frame(f, bg=T.think_bg, padx=12, pady=8, cursor="hand2")
    h.pack(fill="x")
    
    lbl_title = tk.Label(h, text="💭 Thought for " + stats.get('time','0s'), bg=T.think_bg, fg=T.think_fg, font=FSB)
    lbl_title.pack(side="left")
    
    # Expand/Collapse Indicator
    lbl_arrow = tk.Label(h, text="▼", bg=T.think_bg, fg=T.think_fg, font=FSS)
    lbl_arrow.pack(side="right")
    
    # Separator (Hidden initially if collapsed, or shown?) 
    # DeepSeek usually keeps it collapsed initially or expanded depending on state. 
    # We'll make it collapsible.
    
    # Content Frame (Hidden by default)
    content_f = tk.Frame(f, bg=T.think_bg, padx=12, pady=(0, 12))
    full_text = "\n".join(log)
    lbl_log = tk.Label(content_f, text=full_text, bg=T.think_bg, fg=T.think_fg, font=("Menlo", 9), justify="left", anchor="w")
    lbl_log.pack(fill="x")
    
    is_expanded = [False]
    
    def toggle(e):
        is_expanded[0] = not is_expanded[0]
        if is_expanded[0]:
            content_f.pack(fill="x")
            lbl_arrow.config(text="▲")
        else:
            content_f.pack_forget()
            lbl_arrow.config(text="▼")
            
    # Bind click to header elements
    for w in (h, lbl_title, lbl_arrow):
        w.bind("<Button-1>", toggle)

# ════════════════════ LAUNCH ════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    app = CatR1App(root)
    root.mainloop()
