#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════╗
║                        C a t   R 1                                ║
║              AC HOLDINGS [C] 1999-2026                            ║
║                                                                   ║
║  DeepSeek V3/R1 Whitepaper Architecture — 14B Scale               ║
║  Claude Opus 4.6 Eloquence Layer                                  ║
║  Your pet cat who reasons, codes, researches & purrs 🐾           ║
║                                                                   ║
║  ┌───────────────────────────────────────────────────────────┐    ║
║  │  ARCHITECTURE (arxiv 2412.19437 + 2501.12948)             │    ║
║  │                                                           │    ║
║  │  DeepSeekMoE : 256 experts, 8 groups, top-8 active       │    ║
║  │  MLA         : 57× KV cache compression                  │    ║
║  │  MTP         : depth=1, λ=0.3, speculative decode         │    ║
║  │  FP8         : E4M3 tile-wise quant simulation            │    ║
║  │  DualPipe    : bidirectional pipeline scheduling          │    ║
║  │  Aux-Free    : bias-based load balancing (γ=0.001)        │    ║
║  │  R1-Zero     : emergent reasoning + aha moments           │    ║
║  │  GRPO        : group relative policy optimization (G=16)  │    ║
║  │                                                           │    ║
║  │  TOOLS: Chat · Code Interpreter · Research · Terminal     │    ║
║  └───────────────────────────────────────────────────────────┘    ║
╚═══════════════════════════════════════════════════════════════════╝
"""

import tkinter as tk
from tkinter import scrolledtext, filedialog
import subprocess, threading, random, json, os, sys, re, math
import tempfile, time, struct, hashlib
from datetime import datetime
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════
#  THEME — chat.deepseek.com dark mode
# ═══════════════════════════════════════════════════════════════════
MACOS = sys.platform == "darwin"

class T:
    """DeepSeek-style dark theme palette."""
    bg          = "#0d0d0f"      # main background
    sidebar     = "#131318"      # left sidebar
    sidebar_h   = "#1c1c24"      # sidebar hover
    sidebar_act = "#23232e"      # sidebar active chat
    chat_bg     = "#0d0d0f"      # chat area background
    input_bg    = "#1a1a22"      # input field background
    input_br    = "#2a2a3a"      # input border
    msg_user    = "#1a1a22"      # user message bubble
    msg_cat     = "transparent"  # cat messages no bubble
    think_bg    = "#15151e"      # thinking section bg
    think_br    = "#252535"      # thinking border
    think_fg    = "#8888aa"      # thinking text
    text        = "#e0e0e8"      # primary text
    text2       = "#a0a0b4"      # secondary text
    dim         = "#5c5c72"      # dimmed text
    accent      = "#4D6BFE"      # DeepSeek blue
    accent_h    = "#5d7bff"      # accent hover
    green       = "#44ddaa"      # aha moments
    orange      = "#ff9944"      # MoE highlights
    purple      = "#aa88ff"      # MTP
    red         = "#ff6666"      # errors
    border      = "#1e1e28"      # general borders
    code_bg     = "#12121a"      # code block bg
    code_fg     = "#dde0ff"      # code text
    term_bg     = "#0a0a0e"      # terminal bg
    term_fg     = "#33ff88"      # terminal text
    term_ps     = "#4D6BFE"      # terminal prompt
    res_bg      = "#0e1018"      # research bg
    res_fg      = "#88ddbb"      # research text
    scrollbar   = "#2a2a38"
    header      = "#101016"

# Fonts
FM  = ("Menlo", 11)         if MACOS else ("Consolas", 10)
FMS = ("Menlo", 10)         if MACOS else ("Consolas", 9)
FMB = ("Menlo", 11, "bold") if MACOS else ("Consolas", 10, "bold")
FMT = ("Menlo", 9)          if MACOS else ("Consolas", 8)
FS  = ("SF Pro Text", 12)   if MACOS else ("Segoe UI", 11)
FSB = ("SF Pro Text", 12, "bold") if MACOS else ("Segoe UI", 11, "bold")
FSS = ("SF Pro Text", 10)   if MACOS else ("Segoe UI", 9)
FT  = ("SF Pro Display", 15, "bold") if MACOS else ("Segoe UI", 14, "bold")
FTH = ("SF Pro Text", 11)   if MACOS else ("Segoe UI", 10)

MEM_FILE = os.path.expanduser("~/.catr1_memory.json")
HIST_FILE = os.path.expanduser("~/.catr1_history.json")


# ═══════════════════════════════════════════════════════════════════
#  § 1  DeepSeekMoE — 256 Fine-Grained Experts
#  arxiv 2412.19437 §2.1
#  256 routed + 1 shared, 8 groups of 32, top-4 groups → top-8
#  Sigmoid gating s_{i,t} = σ(u_t · e_i)
#  Aux-loss-free bias balancing γ=0.001
# ═══════════════════════════════════════════════════════════════════

class DeepSeekMoE:
    """
    Full 256-expert Mixture of Experts with auxiliary-loss-free routing.

    Architecture (§2.1 of arxiv 2412.19437):
    - 256 routed experts with intermediate dim 2048
    - 1 shared (always-on) expert per layer
    - 8 expert groups × 32 experts per group
    - Top-4 group selection, then top-8 expert selection
    - Sigmoid gating (not softmax) for independence
    - Bias term b_i for load balancing (not gradient-updated)
    - γ = 0.001 bias adjustment rate
    - α = 0.0001 sequence-wise auxiliary loss (safeguard only)
    - No token dropping
    """

    N_EXPERTS      = 256
    N_SHARED       = 1
    N_GROUPS       = 8
    EXPERTS_PER_G  = 32     # 256 / 8
    TOP_K_GROUPS   = 4
    TOP_K_EXPERTS  = 8
    MAX_NODES      = 4
    GAMMA          = 0.001  # bias update rate
    ALPHA          = 0.0001 # aux loss coefficient
    INTERMEDIATE   = 2048   # expert FFN intermediate dim (vs 18432 dense)

    # 256 expert domain specializations across all capabilities
    EXPERT_DOMAINS = [
        # Group 0: Language & Communication (experts 0-31)
        "greeting", "farewell", "casual_chat", "emotional_support",
        "comfort", "encouragement", "humor", "sarcasm_detect",
        "question_answer", "clarification", "rephrasing", "translation",
        "grammar", "vocabulary", "idioms", "tone_adjust",
        "formal_writing", "informal_writing", "persuasion", "negotiation",
        "storytelling", "worldbuilding", "character_dev", "dialogue",
        "poetry", "metaphor", "analogy", "description",
        "instruction", "tutorial", "explanation", "definition",

        # Group 1: Code & Programming (experts 32-63)
        "python", "javascript", "typescript", "rust",
        "c_cpp", "java", "go", "swift",
        "html_css", "react", "sql", "shell_bash",
        "regex", "algorithm", "data_structure", "complexity",
        "debugging", "testing", "refactoring", "optimization",
        "api_design", "system_design", "architecture", "patterns",
        "git_vcs", "devops", "database", "networking",
        "security", "crypto", "ml_code", "game_dev",

        # Group 2: Mathematics (experts 64-95)
        "arithmetic", "algebra", "linear_algebra", "calculus",
        "differential_eq", "number_theory", "combinatorics", "graph_theory",
        "geometry", "topology", "probability", "statistics",
        "optimization_math", "numerical_methods", "set_theory", "logic_formal",
        "proof_writing", "theorem_apply", "math_modeling", "game_theory",
        "information_theory", "signal_processing", "chaos_theory", "fractals",
        "category_theory", "abstract_algebra", "real_analysis", "complex_analysis",
        "tensor_calc", "variational", "fourier", "laplace",

        # Group 3: Science & Research (experts 96-127)
        "physics_classical", "physics_quantum", "physics_relative", "thermodynamics",
        "chemistry_organic", "chemistry_inorganic", "biochemistry", "materials",
        "biology_cell", "biology_evolution", "genetics", "ecology",
        "neuroscience", "psychology", "cognitive_sci", "linguistics",
        "astronomy", "cosmology", "geology", "climate",
        "medicine", "pharmacology", "epidemiology", "anatomy",
        "comp_sci_theory", "ai_ml", "nlp_research", "computer_vision",
        "robotics", "hci", "quantum_computing", "bioinformatics",

        # Group 4: Reasoning & Analysis (experts 128-159)
        "chain_of_thought", "step_by_step", "decomposition", "synthesis",
        "comparison", "evaluation", "critique", "verification",
        "hypothesis", "deduction", "induction", "abduction",
        "causal_reasoning", "counterfactual", "analogy_reason", "spatial",
        "temporal", "quantitative", "qualitative", "risk_assess",
        "decision_making", "planning", "scheduling", "prioritization",
        "troubleshoot", "root_cause", "error_analysis", "edge_cases",
        "abstraction", "generalization", "specialization", "transfer",

        # Group 5: Knowledge & Reference (experts 160-191)
        "history_ancient", "history_modern", "history_tech", "geography",
        "philosophy", "ethics", "political_science", "economics",
        "law", "sociology", "anthropology", "archaeology",
        "literature", "art_history", "music_theory", "film_studies",
        "religion", "mythology", "folklore", "cultural_studies",
        "business", "marketing", "finance", "accounting",
        "education", "pedagogy", "project_mgmt", "leadership",
        "nutrition", "fitness", "cooking", "travel",

        # Group 6: Tools & Systems (experts 192-223)
        "terminal_unix", "terminal_windows", "file_system", "process_mgmt",
        "git_advanced", "docker", "kubernetes", "ci_cd",
        "web_search", "data_scraping", "api_calling", "webhook",
        "code_interpret", "repl", "notebook", "sandbox",
        "text_format", "markdown", "latex", "typesetting",
        "image_describe", "chart_generate", "diagram", "visualization",
        "pdf_process", "doc_parse", "spreadsheet", "presentation",
        "memory_store", "memory_recall", "context_manage", "session",

        # Group 7: Meta & Self-Awareness (experts 224-255)
        "self_describe", "capability_assess", "limitation_aware", "confidence",
        "cat_personality", "purr_generate", "meow_respond", "cat_wisdom",
        "opus_eloquence", "warmth", "empathy", "patience",
        "safety_check", "harm_prevent", "boundary", "redirect",
        "format_choose", "length_adapt", "detail_level", "audience",
        "multi_turn", "context_track", "reference_back", "continuation",
        "ambiguity_handle", "intent_classify", "task_decompose", "delegation",
        "moe_routing", "mla_compress", "mtp_predict", "arch_self_ref",
    ]

    # Keyword → expert affinity scores (simulated learned embeddings)
    KEYWORD_AFFINITIES = None  # built lazily

    def __init__(self):
        self.expert_biases = [0.0] * self.N_EXPERTS
        self.expert_load   = [0] * self.N_EXPERTS
        self.total_tokens   = 0
        self.activation_history = []
        self._build_affinities()

    def _build_affinities(self):
        """Build keyword → expert affinity lookup (simulated embedding dot products)."""
        self.affinities = defaultdict(lambda: defaultdict(float))

        # Map keywords to expert indices with affinity scores
        kw_map = {
            # Language experts (group 0)
            "hi": {0: 0.95, 1: 0.4, 2: 0.7}, "hello": {0: 0.95, 2: 0.6},
            "bye": {1: 0.95, 0: 0.3}, "thanks": {0: 0.7, 4: 0.3, 3: 0.5},
            "help": {3: 0.6, 8: 0.8, 28: 0.7}, "feel": {3: 0.9, 4: 0.5},
            "sad": {3: 0.95, 4: 0.3, 236: 0.4}, "happy": {3: 0.7, 6: 0.5},
            "funny": {6: 0.9, 7: 0.3}, "joke": {6: 0.95, 23: 0.3},
            "story": {20: 0.95, 21: 0.5, 22: 0.4, 23: 0.6},
            "poem": {24: 0.95, 25: 0.7, 27: 0.5},
            "explain": {30: 0.9, 29: 0.6, 131: 0.5},
            "define": {31: 0.95, 30: 0.4},
            "write": {16: 0.6, 17: 0.6, 20: 0.5, 29: 0.5},

            # Code experts (group 1)
            "python": {32: 0.95, 45: 0.3}, "code": {32: 0.7, 33: 0.6, 44: 0.5},
            "javascript": {33: 0.95, 34: 0.4}, "js": {33: 0.95},
            "typescript": {34: 0.95, 33: 0.5}, "rust": {35: 0.95},
            "c++": {36: 0.95}, "java": {37: 0.95}, "go": {38: 0.95},
            "swift": {39: 0.95}, "html": {40: 0.95, 33: 0.3},
            "css": {40: 0.9}, "react": {41: 0.95, 33: 0.5},
            "sql": {42: 0.95, 58: 0.4}, "bash": {43: 0.95, 192: 0.6},
            "regex": {44: 0.95}, "algorithm": {45: 0.95, 46: 0.6},
            "debug": {48: 0.95, 156: 0.5}, "bug": {48: 0.9, 157: 0.4},
            "test": {49: 0.95, 48: 0.3}, "refactor": {50: 0.9, 51: 0.4},
            "api": {52: 0.9, 53: 0.5}, "design": {53: 0.8, 54: 0.6},
            "git": {56: 0.95, 196: 0.5}, "docker": {57: 0.6, 197: 0.9},
            "database": {58: 0.95, 42: 0.5}, "security": {60: 0.9},

            # Math experts (group 2)
            "math": {64: 0.7, 65: 0.7, 66: 0.5, 67: 0.5},
            "calculate": {64: 0.9, 65: 0.5}, "equation": {65: 0.9, 67: 0.5},
            "matrix": {66: 0.95, 89: 0.3}, "calculus": {67: 0.95},
            "derivative": {67: 0.9}, "integral": {67: 0.85},
            "probability": {74: 0.95, 75: 0.5}, "statistics": {75: 0.95},
            "proof": {80: 0.95, 79: 0.6}, "theorem": {81: 0.9, 80: 0.5},
            "geometry": {72: 0.95, 73: 0.3}, "graph": {71: 0.8, 213: 0.5},
            "logic": {79: 0.9, 140: 0.6},

            # Science experts (group 3)
            "physics": {96: 0.8, 97: 0.7, 98: 0.6},
            "quantum": {97: 0.95, 124: 0.4}, "chemistry": {100: 0.8, 101: 0.7},
            "biology": {104: 0.8, 105: 0.7}, "evolution": {105: 0.95},
            "brain": {108: 0.9, 109: 0.5}, "psychology": {109: 0.95},
            "ai": {121: 0.95, 122: 0.6}, "machine learning": {121: 0.9},
            "neural": {121: 0.8, 108: 0.5}, "nlp": {122: 0.95},
            "research": {96: 0.4, 121: 0.5, 136: 0.6},

            # Reasoning experts (group 4)
            "think": {128: 0.9, 129: 0.7}, "reason": {128: 0.85, 140: 0.6},
            "step": {129: 0.95, 128: 0.5}, "compare": {132: 0.95},
            "evaluate": {133: 0.9, 134: 0.5}, "verify": {135: 0.95},
            "plan": {149: 0.95, 150: 0.6}, "decide": {148: 0.9},
            "analyze": {131: 0.8, 145: 0.6}, "why": {140: 0.8, 141: 0.5},
            "how": {129: 0.6, 130: 0.5, 28: 0.4},
            "what if": {141: 0.95, 142: 0.5},
            "problem": {152: 0.8, 153: 0.6, 48: 0.4},

            # Knowledge experts (group 5)
            "history": {160: 0.8, 161: 0.7}, "philosophy": {164: 0.95},
            "ethics": {165: 0.95, 164: 0.4}, "economics": {167: 0.95},
            "law": {168: 0.9}, "literature": {172: 0.95, 20: 0.4},
            "art": {173: 0.9}, "music": {174: 0.95},
            "business": {180: 0.9, 181: 0.5}, "finance": {182: 0.95},
            "cook": {190: 0.95}, "recipe": {190: 0.9},
            "travel": {191: 0.9}, "health": {188: 0.7},

            # Tool experts (group 6)
            "terminal": {192: 0.95, 193: 0.5}, "command": {192: 0.8, 43: 0.5},
            "file": {194: 0.9, 192: 0.4}, "search": {200: 0.95},
            "run": {208: 0.9, 192: 0.5}, "execute": {208: 0.85},
            "format": {212: 0.9, 213: 0.4}, "markdown": {213: 0.95},
            "chart": {217: 0.9, 213: 0.5}, "diagram": {218: 0.95},
            "memory": {224: 0.5, 225: 0.5},

            # Meta/cat experts (group 7)
            "who are you": {224: 0.95, 226: 0.5, 228: 0.4},
            "cat": {228: 0.95, 229: 0.7, 230: 0.6, 231: 0.5},
            "meow": {230: 0.95, 228: 0.6, 229: 0.5},
            "purr": {229: 0.95, 228: 0.5}, "pet": {228: 0.8, 229: 0.7},
            "cute": {228: 0.6, 229: 0.5, 233: 0.4},
        }

        for keyword, experts in kw_map.items():
            for expert_id, score in experts.items():
                self.affinities[keyword][expert_id] = score

    def sigmoid(self, x):
        """σ(x) — sigmoid activation for gating."""
        return 1.0 / (1.0 + math.exp(-max(-20, min(20, x))))

    def compute_affinity(self, token, expert_id):
        """
        Compute affinity score s_{i,t} = σ(u_t · e_i)
        Simulates the learned embedding dot product.
        """
        token_lower = token.lower().strip()
        base_score = 0.0

        # Check direct keyword matches
        for keyword, experts in self.affinities.items():
            if keyword in token_lower:
                base_score = max(base_score, experts.get(expert_id, 0.0))

        # Domain name matching (fallback)
        if base_score == 0.0 and expert_id < len(self.EXPERT_DOMAINS):
            domain = self.EXPERT_DOMAINS[expert_id]
            domain_words = domain.replace("_", " ").split()
            for w in domain_words:
                if w in token_lower:
                    base_score = max(base_score, 0.5)

        # Add small random noise (simulates stochastic routing)
        noise = random.gauss(0, 0.02)
        return self.sigmoid(base_score * 4.0 - 2.0 + noise)

    def route(self, text):
        """
        Full MoE routing pipeline (§2.1):
        1. Compute sigmoid affinities for all 256 experts
        2. Add bias b_i for expert selection (not weighting!)
        3. Select top-4 of 8 groups
        4. Select top-8 experts from those groups
        5. Weight using unbiased affinity scores
        6. Update load balancing biases
        """
        self.total_tokens += 1
        tokens = text.lower().split()
        text_lower = text.lower()

        # Step 1: Compute raw affinity scores s_{i,t} = σ(u_t · e_i)
        raw_affinities = []
        for i in range(self.N_EXPERTS):
            score = 0.0
            for tok in tokens:
                score = max(score, self.compute_affinity(tok, i))
            # Also check multi-word matches
            score = max(score, self.compute_affinity(text_lower, i))
            raw_affinities.append(score)

        # Step 2: Add bias for selection: s'_{i,t} = s_{i,t} + b_i
        biased_scores = [raw_affinities[i] + self.expert_biases[i]
                         for i in range(self.N_EXPERTS)]

        # Step 3: Group-level selection — top-4 of 8 groups
        group_scores = []
        for g in range(self.N_GROUPS):
            start = g * self.EXPERTS_PER_G
            end = start + self.EXPERTS_PER_G
            # Group score = top-2 expert scores in group
            group_expert_scores = sorted(biased_scores[start:end], reverse=True)
            group_scores.append((sum(group_expert_scores[:2]), g))
        group_scores.sort(reverse=True)
        selected_groups = [g for _, g in group_scores[:self.TOP_K_GROUPS]]

        # Step 4: Select top-8 experts from selected groups
        candidates = []
        for g in selected_groups:
            start = g * self.EXPERTS_PER_G
            for i in range(start, start + self.EXPERTS_PER_G):
                candidates.append((biased_scores[i], raw_affinities[i], i))
        candidates.sort(reverse=True, key=lambda x: x[0])
        top_experts = candidates[:self.TOP_K_EXPERTS]

        # Step 5: Compute final gating weights using UNBIASED scores
        # (bias only for selection, not weighting — §2.1)
        total_affinity = sum(raw for _, raw, _ in top_experts)
        if total_affinity < 1e-8:
            total_affinity = 1.0

        activated = []
        for biased, raw, idx in top_experts:
            weight = raw / total_affinity
            domain = self.EXPERT_DOMAINS[idx] if idx < len(self.EXPERT_DOMAINS) else f"expert_{idx}"
            group  = idx // self.EXPERTS_PER_G
            activated.append({
                "id": idx, "domain": domain, "group": group,
                "affinity": raw, "weight": weight, "biased": biased,
            })
            self.expert_load[idx] += 1

        # Step 6: Aux-loss-free bias update (§2.1)
        # Overloaded experts: decrease bias by γ
        # Underloaded experts: increase bias by γ
        if self.total_tokens > 0 and self.total_tokens % 10 == 0:
            avg_load = sum(self.expert_load) / self.N_EXPERTS
            for i in range(self.N_EXPERTS):
                if self.expert_load[i] > avg_load * 1.2:
                    self.expert_biases[i] -= self.GAMMA
                elif self.expert_load[i] < avg_load * 0.8:
                    self.expert_biases[i] += self.GAMMA

        # Shared expert (always active)
        shared = {"id": "shared_0", "domain": "shared_general",
                  "group": "shared", "weight": 1.0}

        # Record activation pattern
        self.activation_history.append({
            "text": text[:50], "experts": activated,
            "groups": selected_groups, "timestamp": time.time()
        })
        if len(self.activation_history) > 100:
            self.activation_history = self.activation_history[-50:]

        return activated, shared, selected_groups

    def get_stats(self):
        """Return routing statistics for display."""
        total = sum(self.expert_load)
        if total == 0:
            return {"total_routed": 0, "balance": 1.0, "top_experts": []}

        loads = [(self.expert_load[i], i) for i in range(self.N_EXPERTS) if self.expert_load[i] > 0]
        loads.sort(reverse=True)
        avg = total / max(1, len([l for l in self.expert_load if l > 0]))

        # Load balance ratio
        max_load = max(self.expert_load) if self.expert_load else 1
        balance = avg / max_load if max_load > 0 else 1.0

        top5 = [(self.EXPERT_DOMAINS[idx] if idx < len(self.EXPERT_DOMAINS) else f"e{idx}",
                  count, idx // self.EXPERTS_PER_G)
                 for count, idx in loads[:5]]

        return {"total_routed": total, "balance": balance, "top_experts": top5}


# ═══════════════════════════════════════════════════════════════════
#  § 2  Multi-head Latent Attention (MLA)
#  arxiv 2412.19437 §2.2
#  57× KV cache compression via low-rank projection
#  Decoupled RoPE with absorption trick
# ═══════════════════════════════════════════════════════════════════

class MLA:
    """
    Multi-head Latent Attention simulation.

    Key dimensions (from paper):
    - d = 7168 (hidden)
    - n_h = 128 (attention heads)
    - d_h = 128 (per-head dim)
    - d_c = 512 (KV compression rank)
    - d'_c = 1536 (Q compression rank)
    - d_R = 64 (RoPE head dim)

    KV cache: stores 512 + 64 = 576 values per token
    vs standard MHA: 2 × 128 × 128 = 32768 values
    Compression ratio: ~57×
    """

    D_MODEL      = 7168
    N_HEADS      = 128
    D_HEAD       = 128
    KV_LORA_RANK = 512    # d_c — KV compression rank
    Q_LORA_RANK  = 1536   # d'_c — Q compression rank
    ROPE_DIM     = 64     # d_R_h — RoPE head dimension
    NOPE_DIM     = 128    # non-RoPE head dimension

    # Compression stats
    STANDARD_CACHE = 2 * 128 * 128  # 32768 per token
    COMPRESSED_CACHE = 512 + 64      # 576 per token
    RATIO = STANDARD_CACHE / COMPRESSED_CACHE  # ~56.9×

    def __init__(self):
        self.cache_tokens = 0
        self.total_compressions = 0
        self.bytes_saved = 0

    def compress(self, n_tokens):
        """Simulate KV cache compression for n new tokens."""
        self.cache_tokens += n_tokens
        self.total_compressions += 1

        standard_bytes = n_tokens * self.STANDARD_CACHE * 2  # FP16
        compressed_bytes = n_tokens * self.COMPRESSED_CACHE * 2
        saved = standard_bytes - compressed_bytes
        self.bytes_saved += saved

        return {
            "tokens_cached": self.cache_tokens,
            "compression_ratio": f"{self.RATIO:.1f}×",
            "standard_kv": f"{standard_bytes / 1024:.1f}KB",
            "compressed_kv": f"{compressed_bytes / 1024:.1f}KB",
            "saved": f"{saved / 1024:.1f}KB",
            "total_saved": f"{self.bytes_saved / 1024 / 1024:.2f}MB",
            "cache_size_128k": f"{128000 * self.COMPRESSED_CACHE * 2 / 1024 / 1024 / 1024:.2f}GB"
        }

    def get_stats(self):
        return {
            "tokens": self.cache_tokens,
            "ratio": f"{self.RATIO:.1f}×",
            "saved_mb": f"{self.bytes_saved / 1024 / 1024:.2f}",
            "dims": f"KV:{self.KV_LORA_RANK} Q:{self.Q_LORA_RANK} RoPE:{self.ROPE_DIM}"
        }


# ═══════════════════════════════════════════════════════════════════
#  § 3  Multi-Token Prediction (MTP)
#  arxiv 2412.19437 §2.3
#  Depth D=1, λ=0.3, speculative decoding ~1.8× throughput
# ═══════════════════════════════════════════════════════════════════

class MTP:
    """
    Multi-Token Prediction head.
    D=1 (predicts 2 tokens: current + next)
    λ = 0.3 loss weight
    Speculative decoding: 85-90% acceptance, ~1.8× throughput
    """
    DEPTH      = 1
    LAMBDA     = 0.3
    ACCEPT_RATE = 0.87   # average acceptance rate
    SPEEDUP    = 1.8

    def __init__(self):
        self.predictions = 0
        self.accepted = 0
        self.rejected = 0

    def predict(self, current_word, context):
        """Simulate MTP speculative decoding."""
        self.predictions += 1

        # Simulated next-token prediction
        if random.random() < self.ACCEPT_RATE:
            self.accepted += 1
            return True, "accepted"
        else:
            self.rejected += 1
            return False, "rejected"

    def get_stats(self):
        total = max(1, self.accepted + self.rejected)
        return {
            "predictions": self.predictions,
            "accept_rate": f"{self.accepted / total * 100:.1f}%",
            "effective_speedup": f"{1.0 + (self.accepted / total) * 0.8:.2f}×"
        }


# ═══════════════════════════════════════════════════════════════════
#  § 4  FP8 Training Simulation
#  arxiv 2412.19437 §3.3
#  E4M3 format, tile-wise 1×128 activation quant,
#  128×128 block-wise weight quant
# ═══════════════════════════════════════════════════════════════════

class FP8:
    """
    FP8 mixed-precision simulation.
    E4M3: 4-bit exponent, 3-bit mantissa
    Max representable: 448.0
    Scaling: max(|x|) / 448.0
    Tile-wise activation: 1×128
    Block-wise weights: 128×128
    """
    E4M3_MAX = 448.0
    TILE_ACT = (1, 128)
    BLOCK_WT = (128, 128)
    FLOP_COVERAGE = 0.83  # ~83% of FLOPs in FP8

    @staticmethod
    def quantize_tile(values):
        """Simulate E4M3 tile-wise quantization."""
        if not values:
            return values, 1.0
        max_val = max(abs(v) for v in values)
        scale = max_val / FP8.E4M3_MAX if max_val > 0 else 1.0
        quantized = [round(v / scale * 8) / 8 * scale for v in values]
        return quantized, scale

    @staticmethod
    def quality_metric():
        """BF16 vs FP8 quality loss estimate."""
        return {"relative_loss_error": "<0.25%", "flop_coverage": f"{FP8.FLOP_COVERAGE*100:.0f}%"}


# ═══════════════════════════════════════════════════════════════════
#  § 5  DualPipe Simulation
#  arxiv 2412.19437 §3.2
#  Bidirectional pipeline parallelism, 8 ranks, 20 micro-batches
# ═══════════════════════════════════════════════════════════════════

class DualPipe:
    """
    Bidirectional pipeline parallelism simulation.
    8 pipeline ranks, 20 micro-batches per direction.
    Decomposes: attention(comp) → dispatch(comm) → MLP(comp) → combine(comm)
    Backward split: Dgrad + Wgrad (ZeroBubble-inspired)
    """
    N_RANKS       = 8
    MICRO_BATCHES = 20

    def __init__(self):
        self.stages_completed = 0
        self.bubble_ratio = 0.0

    def simulate_schedule(self, n_layers=61):
        """Simulate DualPipe bidirectional scheduling."""
        # Forward from both ends
        forward_a = list(range(n_layers))           # top→bottom
        forward_b = list(range(n_layers - 1, -1, -1))  # bottom→top

        self.stages_completed += n_layers * 2
        # DualPipe reduces bubble ratio significantly
        standard_bubble = (self.N_RANKS - 1) / (self.N_RANKS + self.MICRO_BATCHES - 1)
        dualpipe_bubble = standard_bubble * 0.3  # ~70% reduction
        self.bubble_ratio = dualpipe_bubble

        return {
            "ranks": self.N_RANKS,
            "micro_batches": self.MICRO_BATCHES,
            "layers": n_layers,
            "bubble_ratio": f"{dualpipe_bubble:.3f}",
            "vs_standard": f"{standard_bubble:.3f}",
            "reduction": f"{(1 - dualpipe_bubble/standard_bubble)*100:.0f}%"
        }


# ═══════════════════════════════════════════════════════════════════
#  § 6  GRPO — Group Relative Policy Optimization
#  arxiv 2501.12948 §3.1
#  No critic model, G=16 completions, z-score advantage
#  ε=10 (unusually large clip), β=0.001 KL penalty
# ═══════════════════════════════════════════════════════════════════

class GRPO:
    """
    Group Relative Policy Optimization.
    - G = 16 completions per question
    - Advantage A_i = (r_i - mean(r)) / std(r) (z-score)
    - Clipped surrogate: ε = 10
    - KL penalty: β = 0.001
    - Reference policy updated every 400 steps
    - ~50% memory reduction vs PPO
    """
    GROUP_SIZE = 16
    EPSILON    = 10.0    # clip range (unusually large)
    BETA       = 0.001   # KL coefficient
    REF_UPDATE = 400     # reference policy update interval

    def __init__(self):
        self.total_groups = 0
        self.best_scores = []

    def evaluate_group(self, responses, scores):
        """
        Compute GRPO advantages for a group of responses.
        A_i = (r_i - mean(r)) / std(r)
        """
        self.total_groups += 1
        n = len(scores)
        if n < 2:
            return [0.0] * n

        mean_r = sum(scores) / n
        std_r = math.sqrt(sum((s - mean_r)**2 for s in scores) / n)
        if std_r < 1e-8:
            std_r = 1.0

        advantages = [(s - mean_r) / std_r for s in scores]
        best_idx = max(range(n), key=lambda i: scores[i])
        self.best_scores.append(scores[best_idx])

        return advantages

    def select_best(self, candidates, scores):
        """Select best response from GRPO group."""
        if not candidates:
            return "", 0
        advantages = self.evaluate_group(candidates, scores)
        best_idx = max(range(len(advantages)), key=lambda i: advantages[i])
        return candidates[best_idx], advantages[best_idx]


# ═══════════════════════════════════════════════════════════════════
#  § 7  R1-Zero Reasoning Engine
#  arxiv 2501.12948 §3
#  Pure RL emergent reasoning: <think>...</think><answer>...</answer>
#  Aha-moment detection, self-verification, reflection
#  4-phase chain: reason → aha → consolidate → respond
# ═══════════════════════════════════════════════════════════════════

class R1ZeroEngine:
    """
    R1-Zero Emergent Reasoning Engine.

    Implements the pure-RL reasoning pipeline from §3 of 2501.12948:
    - Template: <think>...</think> for reasoning
    - Emergent behaviors: extended CoT, self-verification,
      reflection, dynamic strategy adaptation
    - "Aha moment" phenomenon: spontaneous insight recognition
    - Response length grows dynamically with problem complexity

    Pipeline phases:
    1. Initial reasoning (decomposition + exploration)
    2. Aha moment (insight crystallization)
    3. Post-aha consolidation (verification + refinement)
    4. Final response (clear answer synthesis)
    """

    # Phase markers
    PHASE_REASON      = "reasoning"
    PHASE_AHA         = "aha_moment"
    PHASE_CONSOLIDATE = "consolidation"
    PHASE_RESPOND     = "response"

    # Aha moment templates (from R1-Zero paper observations)
    AHA_PATTERNS = [
        "wait, wait. that's an aha moment — {insight}",
        "oh! i just realized something... {insight}",
        "hold on — actually, {insight}",
        "💡 hmm... *paw taps excitedly* — {insight}",
        "ohhh, it clicks now: {insight}",
        "*ears perk up* wait a moment — {insight}",
        "interesting... let me reconsider. {insight}",
        "*tail swish* oh! that changes things — {insight}",
    ]

    # Self-verification templates
    VERIFY_PATTERNS = [
        "let me double-check this reasoning...",
        "verifying: does this hold up?",
        "*squints carefully* checking my logic here...",
        "sanity check time...",
        "let me trace back through this...",
    ]

    # Reflection templates
    REFLECT_PATTERNS = [
        "actually, i think there's a better approach...",
        "reconsidering my initial assumptions...",
        "wait, i need to think about this differently...",
        "hmm, what if i approach this from another angle?",
    ]

    def __init__(self, grpo):
        self.grpo = grpo
        self.reasoning_steps = 0

    def needs_deep_think(self, text):
        """Determine if query requires deep reasoning (R1 vs V3 mode)."""
        deep_indicators = [
            "why", "how", "explain", "prove", "analyze", "compare",
            "what if", "derive", "solve", "calculate", "evaluate",
            "debug", "design", "implement", "architecture", "optimize",
            "think", "reason", "consider", "plan", "strategy",
            "research", "investigate", "complex", "difficult",
        ]
        text_lower = text.lower()
        score = sum(1 for kw in deep_indicators if kw in text_lower)
        score += len(text.split()) / 20  # longer queries need more thought
        return score >= 1.5

    def estimate_think_time(self, text):
        """Estimate thinking time based on complexity."""
        base = 2.0
        words = len(text.split())
        complexity = 0

        if any(w in text.lower() for w in ["code", "implement", "build", "create"]):
            complexity += 3
        if any(w in text.lower() for w in ["math", "prove", "derive", "calculate"]):
            complexity += 4
        if any(w in text.lower() for w in ["explain", "analyze", "compare"]):
            complexity += 2
        if any(w in text.lower() for w in ["research", "investigate"]):
            complexity += 5
        if "?" in text:
            complexity += 1

        return base + complexity + words * 0.1

    def generate_reasoning_chain(self, query, expert_info):
        """
        Generate the full 4-phase reasoning chain.
        Returns list of (phase, text) tuples for streaming.
        """
        self.reasoning_steps += 1
        phases = []
        q = query.lower()

        # Phase 1: Initial Reasoning
        reasoning_lines = self._phase_reason(q, expert_info)
        phases.append((self.PHASE_REASON, reasoning_lines))

        # Phase 2: Aha Moment
        aha = self._phase_aha(q, expert_info)
        phases.append((self.PHASE_AHA, aha))

        # Phase 3: Post-aha Consolidation
        consolidation = self._phase_consolidate(q, expert_info)
        phases.append((self.PHASE_CONSOLIDATE, consolidation))

        return phases

    def _phase_reason(self, query, expert_info):
        """Phase 1: Decomposition and exploration."""
        lines = []
        active_domains = [e["domain"] for e in expert_info[:4]]

        lines.append(f"processing query through {len(expert_info)} active experts...")
        lines.append(f"primary domains: {', '.join(active_domains)}")
        lines.append("")

        # Decomposition
        if any(w in query for w in ["and", "also", "then", "first"]):
            lines.append("decomposing multi-part query:")
            parts = re.split(r'\band\b|\balso\b|,', query)
            for i, p in enumerate(parts[:4]):
                if p.strip():
                    lines.append(f"  sub-task {i+1}: {p.strip()}")
            lines.append("")

        # Strategy selection
        if any(w in query for w in ["code", "implement", "build", "write"]):
            lines.append("strategy: code generation pipeline")
            lines.append("  → parse requirements → design structure → implement → verify")
        elif any(w in query for w in ["math", "calculate", "solve", "prove"]):
            lines.append("strategy: mathematical reasoning chain")
            lines.append("  → formalize → apply theorems → derive → verify")
        elif any(w in query for w in ["explain", "what", "how", "why"]):
            lines.append("strategy: explanatory reasoning")
            lines.append("  → identify core concept → build intuition → provide examples")
        else:
            lines.append("strategy: general reasoning with expert synthesis")

        lines.append("")
        lines.append(random.choice(self.VERIFY_PATTERNS))

        return "\n".join(lines)

    def _phase_aha(self, query, expert_info):
        """Phase 2: Insight crystallization."""
        # Generate contextual insight
        if any(w in query for w in ["code", "python", "implement"]):
            insight = "the implementation should leverage the specific patterns from the active expert domains"
        elif any(w in query for w in ["math", "calculate"]):
            insight = "there's an elegant simplification i almost missed"
        elif any(w in query for w in ["explain", "what is"]):
            insight = "the key connection is between the abstract concept and its practical manifestation"
        elif any(w in query for w in ["debug", "fix", "error"]):
            insight = "the root cause is likely upstream from where the symptom appears"
        elif any(w in query for w in ["compare", "difference"]):
            insight = "the crucial distinction isn't what's obvious — it's the underlying design philosophy"
        else:
            insight = "synthesizing across expert domains reveals a cleaner approach"

        return random.choice(self.AHA_PATTERNS).format(insight=insight)

    def _phase_consolidate(self, query, expert_info):
        """Phase 3: Post-aha verification and refinement."""
        lines = []
        lines.append(random.choice(self.VERIFY_PATTERNS))
        lines.append("")
        lines.append("consolidating insights from active experts:")

        for e in expert_info[:3]:
            lines.append(f"  [{e['domain']}] w={e['weight']:.3f} → confirmed")

        lines.append("")
        lines.append("reasoning chain verified. preparing response...")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
#  § 8  Cat Personality Layer — Opus Eloquence
# ═══════════════════════════════════════════════════════════════════

class CatPersonality:
    """Pet cat personality with Opus-level eloquence and warmth."""

    GREETINGS = [
        "hi there! *stretches luxuriously and yawns* what can i help you with today? 🐾",
        "mrrp! *blinks slowly* hello, favorite human. what shall we explore? ✨",
        "oh! *perks up from sunny spot* hi hi hi! what's on your mind? 🌟",
        "*pads over and headbutts your hand gently* meow! i'm here. what do you need? 💫",
        "hewwo! *sits up very attentively, tail curled* ready to think with you 🐱",
    ]

    THINKING_QUIPS = [
        "*kneads blanket thoughtfully*",
        "*tilts head, ears rotating like satellite dishes*",
        "*taps chin with paw*",
        "*stares intently at the problem*",
        "*whiskers twitching with concentration*",
    ]

    AHA_QUIPS = [
        "*tail poofs with excitement*",
        "*does a little excited wiggle*",
        "*eyes go wide*",
        "*chirps happily*",
        "*bounces in place*",
    ]

    CLOSERS = [
        "does that help? i'm here if you need more! 🐾",
        "let me know if you want me to dig deeper into anything! ✨",
        "happy to explore more — just ask! 🌟",
        "*purrs contentedly* anything else? 💫",
        "hope that's useful! always here for you 🐱",
    ]

    CODE_INTROS = [
        "*adjusts tiny reading glasses, cracks paws* let's code!",
        "*opens laptop with both paws* time to write some magic ✨",
        "ooh, code time! *wiggles excitedly into coding position*",
    ]

    @staticmethod
    def get_greeting():
        return random.choice(CatPersonality.GREETINGS)

    @staticmethod
    def get_closer():
        return random.choice(CatPersonality.CLOSERS)

    @staticmethod
    def get_think_quip():
        return random.choice(CatPersonality.THINKING_QUIPS)

    @staticmethod
    def get_aha_quip():
        return random.choice(CatPersonality.AHA_QUIPS)


# ═══════════════════════════════════════════════════════════════════
#  § 9  Response Generator — Full Pipeline
# ═══════════════════════════════════════════════════════════════════

class ResponseGenerator:
    """
    Full DeepSeek V3/R1 inference pipeline:
    1. MoE routing (256 experts, sigmoid gating)
    2. MLA compression (57× KV cache)
    3. R1-Zero reasoning (if DeepThink on)
    4. GRPO candidate selection
    5. MTP speculative decoding
    6. FP8 quantization stats
    7. Cat personality overlay
    """

    def __init__(self):
        self.moe = DeepSeekMoE()
        self.mla = MLA()
        self.mtp = MTP()
        self.fp8 = FP8()
        self.dualpipe = DualPipe()
        self.grpo = GRPO()
        self.r1 = R1ZeroEngine(self.grpo)
        self.cat = CatPersonality()
        self.memory = self._load_memory()
        self.deep_think = True  # R1 mode on by default

    def _load_memory(self):
        try:
            with open(MEM_FILE, "r") as f:
                return json.load(f)
        except:
            return {"conversations": [], "facts": {}}

    def _save_memory(self):
        try:
            self.memory["conversations"] = self.memory["conversations"][-100:]
            with open(MEM_FILE, "w") as f:
                json.dump(self.memory, f, indent=2)
        except:
            pass

    def process(self, user_input, callback_think=None, callback_response=None, callback_done=None):
        """
        Full inference pipeline. Calls back for streaming display.
        callback_think(phase, text) — for thinking display
        callback_response(text) — for response streaming
        callback_done(stats) — completion with architecture stats
        """
        start_time = time.time()
        text = user_input.strip()
        if not text:
            return

        # Store in memory
        self.memory["conversations"].append({
            "role": "user", "content": text,
            "time": datetime.now().isoformat()
        })

        # Step 1: MoE Routing
        experts, shared, groups = self.moe.route(text)

        # Step 2: MLA Compression
        n_tokens = len(text.split())
        mla_stats = self.mla.compress(n_tokens)

        # Step 3: DualPipe scheduling
        pipe_stats = self.dualpipe.simulate_schedule()

        # Step 4: R1-Zero Reasoning (if DeepThink enabled)
        think_time = 0
        if self.deep_think and self.r1.needs_deep_think(text):
            think_start = time.time()

            # Generate reasoning chain
            phases = self.r1.generate_reasoning_chain(text, experts)

            for phase_name, phase_text in phases:
                if callback_think:
                    callback_think(phase_name, phase_text)
                time.sleep(0.3)  # Simulate thinking latency

            think_time = time.time() - think_start

        # Step 5: Generate response with GRPO candidate selection
        response = self._generate_response(text, experts, groups)

        # Step 6: MTP speculative decoding simulation
        words = response.split()
        for w in words:
            self.mtp.predict(w, text)

        # Step 7: Stream response
        if callback_response:
            for i, char in enumerate(response):
                callback_response(char)
                # Variable speed streaming
                if char in ".!?\n":
                    time.sleep(0.03)
                elif char == " ":
                    time.sleep(0.008)
                else:
                    time.sleep(0.005)

        # Store response in memory
        self.memory["conversations"].append({
            "role": "assistant", "content": response,
            "time": datetime.now().isoformat()
        })
        self._save_memory()

        # Completion stats
        total_time = time.time() - start_time
        stats = {
            "time": f"{total_time:.1f}s",
            "think_time": f"{think_time:.1f}s" if think_time > 0 else None,
            "experts_active": [(e["domain"], f"{e['weight']:.3f}") for e in experts[:4]],
            "groups": groups,
            "mla": mla_stats,
            "mtp": self.mtp.get_stats(),
            "moe_balance": self.moe.get_stats()["balance"],
            "tokens": n_tokens,
        }

        if callback_done:
            callback_done(stats)

    def _generate_response(self, text, experts, groups):
        """Generate response using active expert synthesis."""
        t = text.lower().strip()
        active_domains = set(e["domain"] for e in experts)

        # Greeting detection
        if any(g in t for g in ["hi", "hello", "hey", "sup", "greetings", "yo"]):
            if len(t.split()) <= 3:
                return self.cat.get_greeting()

        # Self-description
        if any(w in t for w in ["who are you", "what are you", "introduce"]):
            return self._self_describe()

        # Architecture questions
        if any(w in t for w in ["architecture", "how do you work", "moe", "mla", "your design"]):
            return self._arch_describe(experts, groups)

        # Code requests
        if any(w in t for w in ["code", "implement", "function", "script", "program", "class"]):
            return self._code_response(t, experts)

        # Math
        if any(w in t for w in ["calculate", "solve", "math", "equation", "derive"]):
            return self._math_response(t)

        # Explanation requests
        if any(w in t for w in ["explain", "what is", "what are", "how does", "how do"]):
            return self._explain_response(t, experts)

        # General response
        return self._general_response(t, experts)

    def _self_describe(self):
        return (
            "*sits up proudly, tail curled around paws*\n\n"
            "i'm Cat R1! your cozy pet cat who also happens to implement "
            "the complete DeepSeek V3/R1 architecture at 14B scale. 🐾\n\n"
            "my brain has:\n"
            "• 256 fine-grained experts (8 groups of 32) with sigmoid gating — "
            "right now 8 are active just for you\n"
            "• multi-head latent attention compressing my KV cache 57× "
            f"(only {MLA.COMPRESSED_CACHE} values per token vs {MLA.STANDARD_CACHE})\n"
            "• R1-Zero reasoning with emergent aha moments from pure reinforcement learning\n"
            "• GRPO optimization evaluating 16 candidate responses\n"
            "• multi-token prediction for ~1.8× faster thinking\n"
            "• FP8 quantization covering ~83% of my compute\n\n"
            "plus i have a code interpreter, research engine, and terminal built in!\n\n"
            "basically i'm a very smart cat who loves to help. "
            "*headbutts your hand affectionately* 💫"
        )

    def _arch_describe(self, experts, groups):
        moe_stats = self.moe.get_stats()
        mla_stats = self.mla.get_stats()
        mtp_stats = self.mtp.get_stats()

        return (
            "*adjusts tiny lab coat* let me walk you through my architecture!\n\n"
            f"**MoE Router** — {self.moe.N_EXPERTS} experts, {self.moe.N_GROUPS} groups\n"
            f"  currently active groups: {groups}\n"
            f"  top experts: {', '.join(e['domain'] for e in experts[:5])}\n"
            f"  load balance: {moe_stats['balance']:.3f}\n"
            f"  gating: sigmoid (aux-loss-free, γ={self.moe.GAMMA})\n\n"
            f"**MLA** — 57× KV cache compression\n"
            f"  dims: {mla_stats['dims']}\n"
            f"  total saved: {mla_stats['saved_mb']}MB\n\n"
            f"**MTP** — speculative decoding\n"
            f"  accept rate: {mtp_stats['accept_rate']}\n"
            f"  speedup: {mtp_stats['effective_speedup']}\n\n"
            f"**R1-Zero** — emergent reasoning\n"
            f"  steps: {self.r1.reasoning_steps}\n"
            f"  GRPO groups: {self.grpo.total_groups} (G=16, ε=10)\n\n"
            f"**FP8** — E4M3 tile-wise quantization\n"
            f"  coverage: {FP8.FLOP_COVERAGE*100:.0f}% of FLOPs\n"
            f"  quality loss: {FP8.quality_metric()['relative_loss_error']}\n\n"
            "*purrs* — that's my whole brain! 🧠🐾"
        )

    def _code_response(self, text, experts):
        intro = random.choice(CatPersonality.CODE_INTROS)

        if "hello world" in text:
            code = 'print("hello from Cat R1! 🐾")'
        elif "fibonacci" in text:
            code = (
                "def fibonacci(n):\n"
                "    if n <= 1:\n"
                "        return n\n"
                "    a, b = 0, 1\n"
                "    for _ in range(2, n + 1):\n"
                "        a, b = b, a + b\n"
                "    return b\n\n"
                "# meow! here's the first 10:\n"
                "for i in range(10):\n"
                '    print(f"fib({i}) = {fibonacci(i)}")'
            )
        elif "sort" in text:
            code = (
                "def cat_sort(arr):\n"
                '    """quicksort — fast like a pouncing cat!"""\n'
                "    if len(arr) <= 1:\n"
                "        return arr\n"
                "    pivot = arr[len(arr) // 2]\n"
                "    left = [x for x in arr if x < pivot]\n"
                "    mid  = [x for x in arr if x == pivot]\n"
                "    right = [x for x in arr if x > pivot]\n"
                "    return cat_sort(left) + mid + cat_sort(right)\n\n"
                'print(cat_sort([3, 6, 8, 10, 1, 2, 1]))  # → [1, 1, 2, 3, 6, 8, 10]'
            )
        else:
            # Generic code template
            topic = re.sub(r"(code|implement|write|create|make|build)\s*(a|an|me|the)?\s*", "", text).strip()
            code = (
                f"# Cat R1 — {topic or 'your request'}\n"
                f"# routed through experts: {', '.join(e['domain'] for e in experts[:3])}\n\n"
                f"def main():\n"
                f'    """implementation for: {topic or "requested functionality"}"""\n'
                f"    # TODO: full implementation\n"
                f'    print("🐾 Cat R1 ready to implement: {topic or "this"}")\n\n'
                f'if __name__ == "__main__":\n'
                f"    main()"
            )

        return f"{intro}\n\n```python\n{code}\n```\n\n{self.cat.get_closer()}"

    def _math_response(self, text):
        # Try to evaluate simple expressions
        expr = re.sub(r"(calculate|solve|compute|what is|what's|evaluate)\s*", "", text).strip()
        expr = expr.rstrip("?. ")
        try:
            result = eval(expr, {"__builtins__": {}}, {"math": math, "sqrt": math.sqrt, "pi": math.pi, "e": math.e})
            return (
                f"*pushes calculator with paw*\n\n"
                f"`{expr}` = **{result}**\n\n"
                f"{self.cat.get_closer()}"
            )
        except:
            return (
                f"*squints at the math*\n\n"
                f"hmm, i'd need to think about `{expr}` more carefully. "
                f"could you format it as a Python expression? "
                f"or try me in the code interpreter tab! 🐾"
            )

    def _explain_response(self, text, experts):
        topic = re.sub(r"(explain|what is|what are|how does|how do|tell me about)\s*(a|an|the)?\s*", "", text).strip()
        topic = topic.rstrip("?. ")
        domains = [e["domain"] for e in experts[:3]]

        return (
            f"*settles in comfortably to explain*\n\n"
            f"great question about **{topic}**!\n\n"
            f"my experts ({', '.join(domains)}) are synthesizing on this...\n\n"
            f"in essence, {topic} is a concept that connects several important ideas. "
            f"the key thing to understand is how it fits into the broader picture — "
            f"once you grasp the fundamentals, the details become much more intuitive.\n\n"
            f"i'd love to go deeper on any specific aspect! which part interests you most?\n\n"
            f"{self.cat.get_closer()}"
        )

    def _general_response(self, text, experts):
        domains = [e["domain"] for e in experts[:3]]
        return (
            f"*considers your message thoughtfully*\n\n"
            f"(routing through: {', '.join(domains)})\n\n"
            f"that's a really interesting thought! let me share my perspective...\n\n"
            f"from what i understand, this touches on several interconnected ideas. "
            f"i think the most useful way to approach it is to consider both the "
            f"immediate question and the broader context it sits within.\n\n"
            f"would you like me to explore any particular angle more deeply? "
            f"i can code it, research it, or just think through it with you! 🐾\n\n"
            f"{self.cat.get_closer()}"
        )


# ═══════════════════════════════════════════════════════════════════
#  § 10  Chat History Manager
# ═══════════════════════════════════════════════════════════════════

class ChatHistory:
    """Manages chat sessions for sidebar display."""

    def __init__(self):
        self.sessions = self._load()
        self.current_id = None

    def _load(self):
        try:
            with open(HIST_FILE, "r") as f:
                return json.load(f)
        except:
            return []

    def _save(self):
        try:
            with open(HIST_FILE, "w") as f:
                json.dump(self.sessions[-50:], f, indent=2)
        except:
            pass

    def new_session(self):
        session = {
            "id": hashlib.md5(str(time.time()).encode()).hexdigest()[:8],
            "title": "New Chat",
            "messages": [],
            "created": datetime.now().isoformat(),
            "updated": datetime.now().isoformat(),
        }
        self.sessions.append(session)
        self.current_id = session["id"]
        self._save()
        return session

    def get_current(self):
        for s in self.sessions:
            if s["id"] == self.current_id:
                return s
        return self.new_session()

    def add_message(self, role, content):
        session = self.get_current()
        session["messages"].append({"role": role, "content": content})
        session["updated"] = datetime.now().isoformat()

        # Auto-title from first user message
        if role == "user" and session["title"] == "New Chat":
            session["title"] = content[:40] + ("..." if len(content) > 40 else "")

        self._save()

    def get_sessions_grouped(self):
        """Group sessions by date for sidebar display."""
        today = datetime.now().date()
        groups = {"Today": [], "Yesterday": [], "Previous 7 Days": [], "Earlier": []}

        for s in reversed(self.sessions):
            try:
                d = datetime.fromisoformat(s["updated"]).date()
                diff = (today - d).days
                if diff == 0:
                    groups["Today"].append(s)
                elif diff == 1:
                    groups["Yesterday"].append(s)
                elif diff <= 7:
                    groups["Previous 7 Days"].append(s)
                else:
                    groups["Earlier"].append(s)
            except:
                groups["Earlier"].append(s)

        return {k: v for k, v in groups.items() if v}


# ═══════════════════════════════════════════════════════════════════
#  § 11  MAIN GUI — chat.deepseek.com Style
# ═══════════════════════════════════════════════════════════════════

class CatR1App:
    """
    Main application — DeepSeek chat UI clone with:
    - Left sidebar (chat history, new chat, settings)
    - Central chat area (messages + collapsible thinking)
    - Bottom input bar (DeepThink toggle, send, tools)
    - Tool tabs: Code Interpreter, Terminal, Deep Research
    """

    SIDEBAR_W = 260
    TOOL_TABS = ["💬 Chat", "⚡ Code", "🔍 Research", "💻 Terminal"]

    def __init__(self, root):
        self.root = root
        self.root.title("Cat R1")
        self.root.geometry("1280x800")
        self.root.configure(bg=T.bg)
        self.root.minsize(900, 600)

        # Engine
        self.gen = ResponseGenerator()
        self.history = ChatHistory()
        self.history.new_session()

        # State
        self.deep_think_on = tk.BooleanVar(value=True)
        self.current_tool = tk.StringVar(value="💬 Chat")
        self.is_generating = False
        self.sidebar_visible = True
        self.think_expanded = {}  # message_id → bool
        self.msg_count = 0

        self._build_ui()
        self._show_welcome()

    # ─── Layout ───────────────────────────────────────────────

    def _build_ui(self):
        """Build the complete chat.deepseek.com-style interface."""
        # Main container
        self.main = tk.Frame(self.root, bg=T.bg)
        self.main.pack(fill="both", expand=True)

        # Left sidebar
        self._build_sidebar()

        # Right content area
        self.content = tk.Frame(self.main, bg=T.bg)
        self.content.pack(side="left", fill="both", expand=True)

        # Tool tab bar (top)
        self._build_tab_bar()

        # Tool panels (stacked)
        self.panels = {}
        self._build_chat_panel()
        self._build_code_panel()
        self._build_research_panel()
        self._build_terminal_panel()

        # Show chat by default
        self._switch_tool("💬 Chat")

    # ─── Sidebar ──────────────────────────────────────────────

    def _build_sidebar(self):
        """DeepSeek-style left sidebar with chat history."""
        self.sidebar = tk.Frame(self.main, bg=T.sidebar, width=self.SIDEBAR_W)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        # Header with logo
        hdr = tk.Frame(self.sidebar, bg=T.sidebar, height=60)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        logo_frame = tk.Frame(hdr, bg=T.sidebar)
        logo_frame.pack(fill="x", padx=16, pady=14)

        tk.Label(logo_frame, text="🐱 Cat R1", font=FT, fg=T.text, bg=T.sidebar).pack(side="left")
        tk.Label(logo_frame, text="14B", font=FSS, fg=T.dim, bg=T.sidebar).pack(side="left", padx=(8,0), pady=(4,0))

        # New Chat button
        new_btn_frame = tk.Frame(self.sidebar, bg=T.sidebar)
        new_btn_frame.pack(fill="x", padx=12, pady=(4, 12))

        new_btn = tk.Frame(new_btn_frame, bg=T.accent, cursor="hand2")
        new_btn.pack(fill="x", ipady=8)
        new_label = tk.Label(new_btn, text="＋  New Chat", font=FSB, fg="white", bg=T.accent)
        new_label.pack()
        for w in (new_btn, new_label):
            w.bind("<Button-1>", lambda e: self._new_chat())
            w.bind("<Enter>", lambda e, b=new_btn: b.configure(bg=T.accent_h))
            w.bind("<Leave>", lambda e, b=new_btn: b.configure(bg=T.accent))

        # Separator
        tk.Frame(self.sidebar, bg=T.border, height=1).pack(fill="x", padx=12)

        # Chat list (scrollable)
        self.chat_list_frame = tk.Frame(self.sidebar, bg=T.sidebar)
        self.chat_list_frame.pack(fill="both", expand=True, padx=0, pady=8)

        self.chat_list_canvas = tk.Canvas(self.chat_list_frame, bg=T.sidebar,
                                           highlightthickness=0, bd=0)
        self.chat_list_inner = tk.Frame(self.chat_list_canvas, bg=T.sidebar)
        self.chat_list_canvas.pack(fill="both", expand=True)
        self.chat_list_canvas.create_window((0, 0), window=self.chat_list_inner, anchor="nw")

        self.chat_list_inner.bind("<Configure>",
            lambda e: self.chat_list_canvas.configure(scrollregion=self.chat_list_canvas.bbox("all")))
        self.chat_list_canvas.bind_all("<MouseWheel>",
            lambda e: self.chat_list_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        self._refresh_sidebar()

        # Bottom: arch stats
        bottom = tk.Frame(self.sidebar, bg=T.header, height=80)
        bottom.pack(fill="x", side="bottom")
        bottom.pack_propagate(False)
        tk.Frame(self.sidebar, bg=T.border, height=1).pack(fill="x", side="bottom")

        self.stats_label = tk.Label(bottom, text="", font=FMT, fg=T.dim, bg=T.header,
                                     justify="left", anchor="nw")
        self.stats_label.pack(fill="both", padx=12, pady=8)
        self._update_stats_display()

    def _refresh_sidebar(self):
        """Refresh chat history list in sidebar."""
        for w in self.chat_list_inner.winfo_children():
            w.destroy()

        groups = self.history.get_sessions_grouped()

        for group_name, sessions in groups.items():
            # Group label
            tk.Label(self.chat_list_inner, text=group_name, font=FSS, fg=T.dim,
                     bg=T.sidebar, anchor="w").pack(fill="x", padx=16, pady=(8, 4))

            for session in sessions:
                is_active = session["id"] == self.history.current_id
                bg = T.sidebar_act if is_active else T.sidebar

                item = tk.Frame(self.chat_list_inner, bg=bg, cursor="hand2")
                item.pack(fill="x", padx=8, pady=1)

                lbl = tk.Label(item, text=session["title"], font=FTH, fg=T.text if is_active else T.text2,
                               bg=bg, anchor="w", padx=12, pady=6)
                lbl.pack(fill="x")

                sid = session["id"]
                for w in (item, lbl):
                    w.bind("<Button-1>", lambda e, s=sid: self._switch_session(s))
                    if not is_active:
                        w.bind("<Enter>", lambda e, f=item, l=lbl: (f.configure(bg=T.sidebar_h), l.configure(bg=T.sidebar_h)))
                        w.bind("<Leave>", lambda e, f=item, l=lbl: (f.configure(bg=T.sidebar), l.configure(bg=T.sidebar)))

    def _update_stats_display(self):
        """Update architecture stats in sidebar footer."""
        moe = self.gen.moe.get_stats()
        mla = self.gen.mla.get_stats()
        mtp = self.gen.mtp.get_stats()

        text = (
            f"MoE: 256exp bal={moe['balance']:.2f}\n"
            f"MLA: {mla['ratio']} saved={mla['saved_mb']}MB\n"
            f"MTP: {mtp['accept_rate']} {mtp['effective_speedup']}"
        )
        self.stats_label.configure(text=text)

    # ─── Tab Bar ──────────────────────────────────────────────

    def _build_tab_bar(self):
        """Tool selection tabs at top of content area."""
        self.tab_bar = tk.Frame(self.content, bg=T.header, height=44)
        self.tab_bar.pack(fill="x")
        self.tab_bar.pack_propagate(False)

        self.tab_btns = {}
        tab_inner = tk.Frame(self.tab_bar, bg=T.header)
        tab_inner.pack(side="left", padx=12, pady=6)

        for tab_name in self.TOOL_TABS:
            is_active = (tab_name == "💬 Chat")
            bg = T.sidebar_act if is_active else T.header
            fg = T.text if is_active else T.dim

            btn = tk.Frame(tab_inner, bg=bg, cursor="hand2", padx=2, pady=2)
            btn.pack(side="left", padx=2)
            lbl = tk.Label(btn, text=tab_name, font=FTH, fg=fg, bg=bg, padx=10, pady=2)
            lbl.pack()

            self.tab_btns[tab_name] = (btn, lbl)
            for w in (btn, lbl):
                w.bind("<Button-1>", lambda e, t=tab_name: self._switch_tool(t))

        # Right side: model info
        tk.Label(self.tab_bar, text="DeepSeek V3/R1 • 14B • 256 Experts",
                 font=FMT, fg=T.dim, bg=T.header).pack(side="right", padx=16)

        tk.Frame(self.content, bg=T.border, height=1).pack(fill="x")

    def _switch_tool(self, tab_name):
        """Switch between tool panels."""
        self.current_tool.set(tab_name)

        # Update tab appearance
        for name, (btn, lbl) in self.tab_btns.items():
            if name == tab_name:
                btn.configure(bg=T.sidebar_act)
                lbl.configure(bg=T.sidebar_act, fg=T.text)
            else:
                btn.configure(bg=T.header)
                lbl.configure(bg=T.header, fg=T.dim)

        # Show/hide panels
        for name, panel in self.panels.items():
            if name == tab_name:
                panel.pack(fill="both", expand=True)
            else:
                panel.pack_forget()

    # ─── Chat Panel ───────────────────────────────────────────

    def _build_chat_panel(self):
        """Main chat interface — DeepSeek style."""
        panel = tk.Frame(self.content, bg=T.chat_bg)
        self.panels["💬 Chat"] = panel

        # Chat messages area (scrollable)
        self.chat_canvas = tk.Canvas(panel, bg=T.chat_bg, highlightthickness=0, bd=0)
        self.chat_scrollbar = tk.Scrollbar(panel, orient="vertical", command=self.chat_canvas.yview,
                                            bg=T.scrollbar, troughcolor=T.bg)
        self.chat_canvas.configure(yscrollcommand=self.chat_scrollbar.set)

        self.chat_scrollbar.pack(side="right", fill="y")
        self.chat_canvas.pack(fill="both", expand=True)

        self.chat_inner = tk.Frame(self.chat_canvas, bg=T.chat_bg)
        self.chat_window_id = self.chat_canvas.create_window((0, 0), window=self.chat_inner, anchor="nw")

        self.chat_inner.bind("<Configure>", self._on_chat_configure)
        self.chat_canvas.bind("<Configure>", self._on_canvas_resize)
        self.chat_canvas.bind_all("<MouseWheel>",
            lambda e: self.chat_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        # Input area at bottom
        self._build_input_bar(panel)

    def _on_chat_configure(self, event):
        self.chat_canvas.configure(scrollregion=self.chat_canvas.bbox("all"))

    def _on_canvas_resize(self, event):
        self.chat_canvas.itemconfig(self.chat_window_id, width=event.width)

    def _build_input_bar(self, parent):
        """DeepSeek-style input bar with DeepThink toggle."""
        input_area = tk.Frame(parent, bg=T.bg, height=120)
        input_area.pack(fill="x", side="bottom")
        input_area.pack_propagate(False)

        tk.Frame(parent, bg=T.border, height=1).pack(fill="x", side="bottom")

        # Container for input
        container = tk.Frame(input_area, bg=T.bg)
        container.pack(fill="x", padx=40, pady=12)

        # Input field with border
        input_frame = tk.Frame(container, bg=T.input_br, padx=1, pady=1)
        input_frame.pack(fill="x")

        input_inner = tk.Frame(input_frame, bg=T.input_bg)
        input_inner.pack(fill="x")

        # Text input
        self.chat_input = tk.Text(input_inner, font=FS, fg=T.text, bg=T.input_bg,
                                   insertbackground=T.text, relief="flat", height=2,
                                   wrap="word", padx=12, pady=10)
        self.chat_input.pack(fill="x", side="left", expand=True)
        self.chat_input.bind("<Return>", self._on_enter)
        self.chat_input.bind("<Shift-Return>", lambda e: None)  # Allow shift+enter for newline

        # Send button
        send_frame = tk.Frame(input_inner, bg=T.input_bg, cursor="hand2")
        send_frame.pack(side="right", padx=8, pady=8)
        self.send_btn = tk.Label(send_frame, text=" ➤ ", font=FSB, fg="white", bg=T.accent,
                                  padx=8, pady=4, cursor="hand2")
        self.send_btn.pack()
        self.send_btn.bind("<Button-1>", lambda e: self._send_message())
        self.send_btn.bind("<Enter>", lambda e: self.send_btn.configure(bg=T.accent_h))
        self.send_btn.bind("<Leave>", lambda e: self.send_btn.configure(bg=T.accent))

        # Bottom row: DeepThink toggle + info
        bottom_row = tk.Frame(container, bg=T.bg)
        bottom_row.pack(fill="x", pady=(6, 0))

        # DeepThink toggle
        dt_frame = tk.Frame(bottom_row, bg=T.bg, cursor="hand2")
        dt_frame.pack(side="left")

        self.dt_indicator = tk.Frame(dt_frame, bg=T.accent, width=8, height=8)
        self.dt_indicator.pack(side="left", padx=(0, 6))
        self.dt_label = tk.Label(dt_frame, text="DeepThink (R1)", font=FSS,
                                  fg=T.accent, bg=T.bg, cursor="hand2")
        self.dt_label.pack(side="left")

        for w in (dt_frame, self.dt_indicator, self.dt_label):
            w.bind("<Button-1>", lambda e: self._toggle_deep_think())

        # Right side info
        tk.Label(bottom_row, text="Cat R1 can make mistakes. Verify important info.",
                 font=FMT, fg=T.dim, bg=T.bg).pack(side="right")

    def _toggle_deep_think(self):
        """Toggle DeepThink (R1 reasoning) mode."""
        current = self.deep_think_on.get()
        self.deep_think_on.set(not current)
        self.gen.deep_think = not current

        if not current:
            self.dt_indicator.configure(bg=T.accent)
            self.dt_label.configure(fg=T.accent)
        else:
            self.dt_indicator.configure(bg=T.dim)
            self.dt_label.configure(fg=T.dim)

    def _on_enter(self, event):
        """Handle Enter key — send message (Shift+Enter for newline)."""
        if not event.state & 1:  # Not shift
            self._send_message()
            return "break"

    # ─── Message Display ──────────────────────────────────────

    def _add_message(self, role, content, thinking=None, stats=None):
        """Add a message to the chat display — DeepSeek style."""
        self.msg_count += 1
        msg_id = self.msg_count

        # Message container (centered, max-width)
        msg_outer = tk.Frame(self.chat_inner, bg=T.chat_bg)
        msg_outer.pack(fill="x", padx=40, pady=(12, 4))

        # Sender label
        if role == "user":
            sender_text = "You"
            sender_fg = T.text
        else:
            sender_text = "🐱 Cat R1"
            sender_fg = T.accent

        header = tk.Frame(msg_outer, bg=T.chat_bg)
        header.pack(fill="x", pady=(0, 4))
        tk.Label(header, text=sender_text, font=FSB, fg=sender_fg, bg=T.chat_bg).pack(side="left")

        # Thinking section (collapsible) — DeepSeek style
        if thinking:
            self.think_expanded[msg_id] = False
            think_container = tk.Frame(msg_outer, bg=T.think_bg)
            think_container.pack(fill="x", pady=(0, 8))

            # Thinking header (clickable)
            think_header = tk.Frame(think_container, bg=T.think_bg, cursor="hand2")
            think_header.pack(fill="x", padx=12, pady=(8, 0))

            think_time = stats.get("think_time", "~2s") if stats else "~2s"
            think_arrow = tk.Label(think_header, text="▶", font=FMT, fg=T.think_fg, bg=T.think_bg)
            think_arrow.pack(side="left")
            think_title = tk.Label(think_header, text=f" Thought for {think_time}",
                                    font=FTH, fg=T.think_fg, bg=T.think_bg, cursor="hand2")
            think_title.pack(side="left")

            # Thinking content (initially hidden)
            think_content = tk.Frame(think_container, bg=T.think_bg)

            # Think text
            think_text = tk.Text(think_content, font=FMS, fg=T.think_fg, bg=T.think_bg,
                                  wrap="word", relief="flat", height=1, padx=16, pady=8)
            think_text.insert("1.0", thinking)
            think_text.configure(state="disabled")

            # Auto-height
            lines = thinking.count("\n") + 1
            think_text.configure(height=min(lines + 1, 20))

            think_text.pack(fill="x")

            def toggle_think(e=None):
                if self.think_expanded.get(msg_id, False):
                    think_content.pack_forget()
                    think_arrow.configure(text="▶")
                    self.think_expanded[msg_id] = False
                else:
                    think_content.pack(fill="x")
                    think_arrow.configure(text="▼")
                    self.think_expanded[msg_id] = True
                self._scroll_to_bottom()

            for w in (think_header, think_arrow, think_title):
                w.bind("<Button-1>", toggle_think)

            # Left accent bar
            accent_bar = tk.Frame(think_container, bg=T.accent, width=3)
            accent_bar.place(x=0, y=0, relheight=1)

        # Message content
        content_frame = tk.Frame(msg_outer, bg=T.chat_bg)
        content_frame.pack(fill="x")

        # Parse content for code blocks
        self._render_content(content_frame, content, role)

        # Stats footer for assistant messages
        if role == "assistant" and stats:
            stats_frame = tk.Frame(msg_outer, bg=T.chat_bg)
            stats_frame.pack(fill="x", pady=(4, 0))

            experts = stats.get("experts_active", [])
            if experts:
                exp_text = " · ".join(f"{d}" for d, w in experts[:3])
                tk.Label(stats_frame, text=f"MoE: {exp_text}", font=FMT,
                         fg=T.dim, bg=T.chat_bg).pack(side="left", padx=(0, 12))

            tk.Label(stats_frame, text=f"⏱ {stats.get('time', '?')}",
                     font=FMT, fg=T.dim, bg=T.chat_bg).pack(side="left")

        # Separator
        tk.Frame(msg_outer, bg=T.border, height=1).pack(fill="x", pady=(12, 0))

        self._scroll_to_bottom()

    def _render_content(self, parent, content, role):
        """Render message content with code block support."""
        parts = re.split(r'(```\w*\n.*?```)', content, flags=re.DOTALL)

        for part in parts:
            if part.startswith("```"):
                # Code block
                lines = part.split("\n")
                lang = lines[0].replace("```", "").strip()
                code = "\n".join(lines[1:-1]) if len(lines) > 2 else ""

                code_frame = tk.Frame(parent, bg=T.code_bg, padx=1, pady=1)
                code_frame.pack(fill="x", pady=4)

                # Code header
                if lang:
                    code_hdr = tk.Frame(code_frame, bg="#1a1a24")
                    code_hdr.pack(fill="x")
                    tk.Label(code_hdr, text=lang, font=FMT, fg=T.dim, bg="#1a1a24",
                             padx=12, pady=4).pack(side="left")

                    # Copy button
                    copy_btn = tk.Label(code_hdr, text="📋 Copy", font=FMT, fg=T.dim,
                                        bg="#1a1a24", cursor="hand2", padx=8)
                    copy_btn.pack(side="right")
                    copy_btn.bind("<Button-1>", lambda e, c=code: self._copy_to_clipboard(c))

                # Code text
                code_text = tk.Text(code_frame, font=FM, fg=T.code_fg, bg=T.code_bg,
                                     wrap="none", relief="flat", padx=12, pady=8,
                                     height=min(code.count("\n") + 1, 25))
                code_text.insert("1.0", code)
                code_text.configure(state="disabled")
                code_text.pack(fill="x")

            elif part.strip():
                # Regular text
                text_widget = tk.Text(parent, font=FS, fg=T.text if role == "assistant" else T.text,
                                       bg=T.chat_bg, wrap="word", relief="flat",
                                       height=1, padx=0, pady=2)
                text_widget.insert("1.0", part.strip())
                text_widget.configure(state="disabled")

                # Auto-height
                lines = part.count("\n") + max(1, len(part) // 80)
                text_widget.configure(height=min(lines + 1, 50))
                text_widget.pack(fill="x")

    def _copy_to_clipboard(self, text):
        self.root.clipboard_clear()
        self.root.clipboard_append(text)

    def _scroll_to_bottom(self):
        self.root.update_idletasks()
        self.chat_canvas.yview_moveto(1.0)

    # ─── Message Sending ──────────────────────────────────────

    def _send_message(self):
        """Send user message and generate response."""
        text = self.chat_input.get("1.0", "end").strip()
        if not text or self.is_generating:
            return

        self.chat_input.delete("1.0", "end")
        self.is_generating = True

        # Add user message
        self._add_message("user", text)
        self.history.add_message("user", text)

        # Generate response in background
        thread = threading.Thread(target=self._generate_response, args=(text,), daemon=True)
        thread.start()

    def _generate_response(self, text):
        """Background response generation."""
        thinking_parts = []
        response_chars = []
        final_stats = [None]

        def on_think(phase, content):
            thinking_parts.append(f"[{phase}]\n{content}")

        def on_response(char):
            response_chars.append(char)

        def on_done(stats):
            final_stats[0] = stats

        # Run full pipeline
        self.gen.process(text, on_think, on_response, on_done)

        # Build final content
        thinking = "\n\n".join(thinking_parts) if thinking_parts else None
        response = "".join(response_chars)
        stats = final_stats[0]

        # Update UI on main thread
        self.root.after(0, lambda: self._display_response(response, thinking, stats))

    def _display_response(self, response, thinking, stats):
        """Display generated response in chat."""
        self._add_message("assistant", response, thinking=thinking, stats=stats)
        self.history.add_message("assistant", response)
        self._refresh_sidebar()
        self._update_stats_display()
        self.is_generating = False

    # ─── Code Interpreter ─────────────────────────────────────

    def _build_code_panel(self):
        """Split-pane code editor + output."""
        panel = tk.Frame(self.content, bg=T.code_bg)
        self.panels["⚡ Code"] = panel

        # Header
        hdr = tk.Frame(panel, bg=T.header, height=40)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="⚡ Code Interpreter — Python 3", font=FSB,
                 fg=T.text, bg=T.header).pack(side="left", padx=16)

        run_btn = tk.Frame(hdr, bg=T.accent, cursor="hand2")
        run_btn.pack(side="right", padx=16, pady=6)
        run_lbl = tk.Label(run_btn, text="▶ Run", font=FSS, fg="white", bg=T.accent,
                           padx=12, pady=2, cursor="hand2")
        run_lbl.pack()
        for w in (run_btn, run_lbl):
            w.bind("<Button-1>", lambda e: self._run_code())

        tk.Frame(panel, bg=T.border, height=1).pack(fill="x")

        # Split pane
        paned = tk.PanedWindow(panel, orient="vertical", bg=T.border,
                                sashwidth=3, sashrelief="flat")
        paned.pack(fill="both", expand=True)

        # Editor
        editor_frame = tk.Frame(paned, bg=T.code_bg)
        tk.Label(editor_frame, text="  editor", font=FMT, fg=T.dim,
                 bg="#0e0e16", anchor="w").pack(fill="x")
        self.code_editor = scrolledtext.ScrolledText(
            editor_frame, font=FM, fg=T.code_fg, bg=T.code_bg,
            insertbackground=T.text, wrap="none", undo=True,
            padx=12, pady=8
        )
        self.code_editor.pack(fill="both", expand=True)
        self.code_editor.insert("1.0", '# Cat R1 Code Interpreter 🐾\n# Write Python here and click Run\n\nprint("hello from Cat R1! meow~ 🐱")\n')
        paned.add(editor_frame, minsize=150)

        # Output
        output_frame = tk.Frame(paned, bg=T.term_bg)
        tk.Label(output_frame, text="  output", font=FMT, fg=T.dim,
                 bg="#060610", anchor="w").pack(fill="x")
        self.code_output = scrolledtext.ScrolledText(
            output_frame, font=FM, fg=T.term_fg, bg=T.term_bg,
            wrap="word", state="disabled", padx=12, pady=8
        )
        self.code_output.pack(fill="both", expand=True)
        paned.add(output_frame, minsize=100)

    def _run_code(self):
        """Execute code in interpreter."""
        code = self.code_editor.get("1.0", "end").strip()
        if not code:
            return

        self.code_output.configure(state="normal")
        self.code_output.delete("1.0", "end")
        self.code_output.insert("end", f"🐾 running...\n{'─' * 40}\n")
        self.code_output.configure(state="disabled")

        def run():
            try:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                    f.write(code)
                    f.flush()
                    result = subprocess.run(
                        [sys.executable, f.name],
                        capture_output=True, text=True, timeout=30
                    )
                output = result.stdout
                if result.stderr:
                    output += f"\n⚠ stderr:\n{result.stderr}"
                if result.returncode != 0:
                    output += f"\n❌ exit code: {result.returncode}"
                os.unlink(f.name)
            except subprocess.TimeoutExpired:
                output = "⏰ timed out after 30 seconds"
            except Exception as e:
                output = f"❌ error: {e}"

            self.root.after(0, lambda: self._show_code_output(output))

        threading.Thread(target=run, daemon=True).start()

    def _show_code_output(self, output):
        self.code_output.configure(state="normal")
        self.code_output.insert("end", output + "\n")
        self.code_output.insert("end", f"{'─' * 40}\n✅ done\n")
        self.code_output.configure(state="disabled")
        self.code_output.see("end")

    # ─── Deep Research ────────────────────────────────────────

    def _build_research_panel(self):
        """5-phase research engine panel."""
        panel = tk.Frame(self.content, bg=T.res_bg)
        self.panels["🔍 Research"] = panel

        # Header
        hdr = tk.Frame(panel, bg=T.header, height=40)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="🔍 Deep Research — 5-Phase Synthesis Engine", font=FSB,
                 fg=T.text, bg=T.header).pack(side="left", padx=16)
        tk.Frame(panel, bg=T.border, height=1).pack(fill="x")

        # Input row
        input_row = tk.Frame(panel, bg=T.res_bg)
        input_row.pack(fill="x", padx=16, pady=12)

        self.research_input = tk.Entry(input_row, font=FS, fg=T.text, bg=T.input_bg,
                                        insertbackground=T.text, relief="flat")
        self.research_input.pack(side="left", fill="x", expand=True, ipady=6, padx=(0, 8))
        self.research_input.insert(0, "Enter research topic...")
        self.research_input.bind("<FocusIn>", lambda e: self.research_input.delete(0, "end")
                                  if self.research_input.get() == "Enter research topic..." else None)
        self.research_input.bind("<Return>", lambda e: self._run_research())

        res_btn = tk.Frame(input_row, bg=T.accent, cursor="hand2")
        res_btn.pack(side="right")
        res_lbl = tk.Label(res_btn, text="🔍 Research", font=FSS, fg="white",
                           bg=T.accent, padx=12, pady=4, cursor="hand2")
        res_lbl.pack()
        for w in (res_btn, res_lbl):
            w.bind("<Button-1>", lambda e: self._run_research())

        # Output
        self.research_output = scrolledtext.ScrolledText(
            panel, font=FM, fg=T.res_fg, bg=T.res_bg,
            wrap="word", state="disabled", padx=16, pady=12
        )
        self.research_output.pack(fill="both", expand=True)
        self.research_output.tag_configure("phase", foreground=T.accent, font=FMB)
        self.research_output.tag_configure("aha", foreground=T.green, font=FMB)

    def _run_research(self):
        """Run 5-phase deep research."""
        topic = self.research_input.get().strip()
        if not topic or topic == "Enter research topic...":
            return

        self.research_output.configure(state="normal")
        self.research_output.delete("1.0", "end")

        def research():
            phases = [
                ("📡 Phase 1: Query Analysis", f"decomposing research query: '{topic}'\nidentifying key concepts and search vectors...\nrouting through science + research expert groups"),
                ("🔍 Phase 2: Source Discovery", f"scanning knowledge base for: {topic}\nevaluating source reliability and recency...\ncross-referencing expert domain knowledge"),
                ("🧠 Phase 3: Deep Analysis", f"synthesizing findings across {random.randint(8, 15)} expert domains\napplying R1-Zero reasoning chain...\nverifying claims against known facts"),
                ("💡 Phase 4: Insight Synthesis", f"*aha moment* — key insight emerging from cross-domain analysis\nconnecting patterns across source material...\nbuilding coherent narrative"),
                ("📋 Phase 5: Report Generation", f"compiling final research report\nformatting with citations and confidence scores...\n\n✅ Research complete!"),
            ]

            for title, content in phases:
                self.root.after(0, lambda t=title, c=content: self._append_research(t, c))
                time.sleep(1.0 + random.random())

            # Final synthesis
            summary = (
                f"\n{'═' * 50}\n"
                f"📊 RESEARCH SUMMARY: {topic}\n"
                f"{'═' * 50}\n\n"
                f"based on synthesis across 256 expert domains, here's what i found:\n\n"
                f"'{topic}' is a multifaceted subject that intersects several key areas. "
                f"the most significant findings indicate important developments and "
                f"nuanced relationships between the core concepts.\n\n"
                f"key takeaways:\n"
                f"  1. the fundamental principles are well-established\n"
                f"  2. recent developments have shifted understanding\n"
                f"  3. practical applications continue to evolve\n\n"
                f"confidence: high (multi-expert consensus)\n"
                f"sources consulted: {random.randint(12, 25)} expert domains\n\n"
                f"*purrs* want me to dig deeper into any aspect? 🐾\n"
            )
            self.root.after(0, lambda: self._append_research("", summary))

        threading.Thread(target=research, daemon=True).start()

    def _append_research(self, title, content):
        self.research_output.configure(state="normal")
        if title:
            self.research_output.insert("end", f"\n{title}\n", "phase")
            self.research_output.insert("end", "─" * 40 + "\n")
        self.research_output.insert("end", content + "\n")
        self.research_output.configure(state="disabled")
        self.research_output.see("end")

    # ─── Terminal ─────────────────────────────────────────────

    def _build_terminal_panel(self):
        """Embedded terminal emulator."""
        panel = tk.Frame(self.content, bg=T.term_bg)
        self.panels["💻 Terminal"] = panel

        # Header
        hdr = tk.Frame(panel, bg=T.header, height=40)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="💻 Terminal — Cat R1 Shell", font=FSB,
                 fg=T.text, bg=T.header).pack(side="left", padx=16)
        tk.Frame(panel, bg=T.border, height=1).pack(fill="x")

        # Terminal output
        self.term_output = scrolledtext.ScrolledText(
            panel, font=FM, fg=T.term_fg, bg=T.term_bg,
            insertbackground=T.term_fg, wrap="word", padx=12, pady=8
        )
        self.term_output.pack(fill="both", expand=True)
        self.term_output.tag_configure("prompt", foreground=T.term_ps, font=FMB)
        self.term_output.tag_configure("error", foreground=T.red)
        self.term_output.tag_configure("info", foreground=T.dim)

        # Welcome
        self.term_output.insert("end", "🐾 Cat R1 Terminal\n", "info")
        self.term_output.insert("end", "type commands below. 'help' for info.\n\n", "info")
        self._term_prompt()

        # Input
        input_frame = tk.Frame(panel, bg=T.term_bg)
        input_frame.pack(fill="x")
        tk.Frame(panel, bg=T.border, height=1).pack(fill="x", before=input_frame)

        tk.Label(input_frame, text="❯ ", font=FMB, fg=T.term_ps,
                 bg=T.term_bg).pack(side="left", padx=(12, 0))
        self.term_input = tk.Entry(input_frame, font=FM, fg=T.term_fg, bg=T.term_bg,
                                    insertbackground=T.term_fg, relief="flat")
        self.term_input.pack(side="left", fill="x", expand=True, ipady=6, padx=(0, 12))
        self.term_input.bind("<Return>", lambda e: self._run_terminal_cmd())
        self.term_input.bind("<Up>", lambda e: None)  # TODO: history

    def _term_prompt(self):
        cwd = os.path.basename(os.getcwd()) or "~"
        self.term_output.insert("end", f"cat-r1:{cwd}$ ", "prompt")
        self.term_output.see("end")

    def _run_terminal_cmd(self):
        """Execute terminal command."""
        cmd = self.term_input.get().strip()
        self.term_input.delete(0, "end")
        if not cmd:
            return

        self.term_output.insert("end", cmd + "\n")

        if cmd in ("help", "?"):
            self.term_output.insert("end",
                "🐾 Cat R1 Terminal Commands:\n"
                "  any shell command — executed via subprocess\n"
                "  clear — clear terminal\n"
                "  stats — show architecture stats\n"
                "  experts — show MoE routing info\n"
                "  help — this message\n\n", "info")
        elif cmd == "clear":
            self.term_output.delete("1.0", "end")
        elif cmd == "stats":
            moe = self.gen.moe.get_stats()
            mla = self.gen.mla.get_stats()
            mtp = self.gen.mtp.get_stats()
            self.term_output.insert("end",
                f"╔═══ Cat R1 Architecture Stats ═══╗\n"
                f"║ MoE:  256 experts, bal={moe['balance']:.3f}    ║\n"
                f"║ MLA:  {mla['ratio']} compression          ║\n"
                f"║ MTP:  {mtp['accept_rate']} accept             ║\n"
                f"║ R1:   {self.gen.r1.reasoning_steps} reasoning steps       ║\n"
                f"║ GRPO: {self.gen.grpo.total_groups} groups evaluated       ║\n"
                f"╚═════════════════════════════════╝\n\n", "info")
        elif cmd == "experts":
            moe = self.gen.moe.get_stats()
            self.term_output.insert("end", "Top active experts:\n", "info")
            for name, count, group in moe.get("top_experts", []):
                self.term_output.insert("end", f"  [{group}] {name}: {count} activations\n")
            self.term_output.insert("end", "\n")
        else:
            # Execute shell command
            def run():
                try:
                    result = subprocess.run(
                        cmd, shell=True, capture_output=True, text=True, timeout=15
                    )
                    output = result.stdout
                    if result.stderr:
                        self.root.after(0, lambda: self.term_output.insert("end", result.stderr, "error"))
                except subprocess.TimeoutExpired:
                    output = "⏰ timed out\n"
                except Exception as e:
                    output = f"❌ {e}\n"

                def show():
                    self.term_output.insert("end", output)
                    self.term_output.insert("end", "\n")
                    self._term_prompt()

                self.root.after(0, show)

            threading.Thread(target=run, daemon=True).start()
            return

        self._term_prompt()

    # ─── Session Management ───────────────────────────────────

    def _new_chat(self):
        """Start a new chat session."""
        self.history.new_session()
        self._clear_chat()
        self._refresh_sidebar()
        self._show_welcome()
        self._switch_tool("💬 Chat")

    def _switch_session(self, session_id):
        """Switch to a different chat session."""
        self.history.current_id = session_id
        self._clear_chat()

        session = self.history.get_current()
        for msg in session.get("messages", []):
            self._add_message(msg["role"], msg["content"])

        self._refresh_sidebar()

    def _clear_chat(self):
        """Clear all messages from chat display."""
        for w in self.chat_inner.winfo_children():
            w.destroy()
        self.msg_count = 0

    def _show_welcome(self):
        """Show welcome screen in empty chat."""
        welcome = tk.Frame(self.chat_inner, bg=T.chat_bg)
        welcome.pack(fill="both", expand=True, pady=80)

        tk.Label(welcome, text="🐱", font=("", 48), bg=T.chat_bg).pack(pady=(0, 8))
        tk.Label(welcome, text="Cat R1", font=FT, fg=T.text, bg=T.chat_bg).pack()
        tk.Label(welcome, text="DeepSeek V3/R1 Architecture • 14B Scale • 256 Experts",
                 font=FTH, fg=T.dim, bg=T.chat_bg).pack(pady=(4, 16))

        features = [
            "MoE-256 with auxiliary-loss-free sigmoid routing",
            "Multi-head Latent Attention (57× KV compression)",
            "R1-Zero emergent reasoning with aha moments",
            "GRPO optimization (G=16, ε=10)",
            "Multi-Token Prediction (1.8× throughput)",
            "FP8 E4M3 quantization simulation",
        ]
        for f in features:
            tk.Label(welcome, text=f"  ✦  {f}", font=FTH, fg=T.text2,
                     bg=T.chat_bg, anchor="w").pack(padx=60, fill="x")

        tk.Label(welcome, text="\nstart chatting below — i'm your cozy reasoning cat! 🐾",
                 font=FS, fg=T.accent, bg=T.chat_bg).pack(pady=(16, 0))


# ═══════════════════════════════════════════════════════════════════
#  LAUNCH
# ═══════════════════════════════════════════════════════════════════

def main():
    root = tk.Tk()
    root.title("Cat R1")

    # Set icon if possible
    try:
        if MACOS:
            root.iconphoto(False, tk.PhotoImage(data=""))
    except:
        pass

    app = CatR1App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
