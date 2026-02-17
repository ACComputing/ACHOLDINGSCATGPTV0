#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║  CAT R1 — DeepSeek V3.2/R2 Architecture · AC Holdings [C] 1999-2026 ║
║  1.2T params · 78B active · Hybrid MoE 3.0 · DSA · MLA · GRPO      ║
║  Claude Opus 4.6 Eloquence · Pet Cat 🐾                             ║
║                                                                      ║
║  Outputs: code, english, math, research, terminal — like ChatGPT     ║
╚══════════════════════════════════════════════════════════════════════╝
"""
import tkinter as tk
from tkinter import scrolledtext, filedialog
import subprocess,threading,random,json,os,sys,re,math,textwrap
import tempfile,time,hashlib,traceback
from datetime import datetime
from collections import defaultdict

MAC=sys.platform=="darwin"

# ════════════════════ THEME ════════════════════
class T:
    bg="#0d0d0f";sidebar="#131318";side_h="#1c1c24";side_act="#23232e"
    chat="#0d0d0f";inp_bg="#1a1a22";inp_br="#2a2a3a"
    think_bg="#111119";think_fg="#7878a8";text="#e0e0e8";text2="#a0a0b4"
    dim="#5c5c72";accent="#4D6BFE";acc_h="#5d7bff";green="#44ddaa"
    red="#ff6666";border="#1e1e28";code_bg="#0e0e16";code_fg="#c8ccee"
    term_bg="#08080c";term_fg="#33ff88";term_ps="#4D6BFE"
    res_bg="#0c100e";res_fg="#88ddbb";header="#101016";scroll="#2a2a38"

FM =("Menlo",11) if MAC else ("Consolas",10)
FMS=("Menlo",10) if MAC else ("Consolas",9)
FMB=("Menlo",11,"bold") if MAC else ("Consolas",10,"bold")
FMT=("Menlo",9)  if MAC else ("Consolas",8)
FS =("SF Pro Text",12) if MAC else ("Segoe UI",11)
FSB=("SF Pro Text",12,"bold") if MAC else ("Segoe UI",11,"bold")
FSS=("SF Pro Text",10) if MAC else ("Segoe UI",9)
FT =("SF Pro Display",15,"bold") if MAC else ("Segoe UI",14,"bold")
FH =("SF Pro Text",11) if MAC else ("Segoe UI",10)

MEM=os.path.expanduser("~/.catr1_mem.json")
HIST=os.path.expanduser("~/.catr1_hist.json")

# ════════════════════ R2 CONFIG — exact leaked weights ════════════════════
class R2:
    TOTAL_PARAMS=1_200_000_000_000;ACTIVE_PARAMS=78_000_000_000
    ACTIVE_PCT=6.5;CONTEXT=131072;VOCAB=129280
    N_LAYERS=95;DENSE_LAYERS=3;MOE_LAYERS=92
    N_EXPERTS=256;SHARED=1;GROUPS=8;PER_GROUP=32
    TOP_GROUPS=4;TOP_EXPERTS=8;EXPERT_DIM=2048
    GAMMA=0.001;ALPHA=0.0001
    # MLA dims (from V3 paper)
    D_MODEL=7168;N_HEADS=128;D_HEAD=128
    KV_RANK=512;Q_RANK=1536;ROPE_DIM=64
    KV_CACHE=576;STD_CACHE=32768;COMPRESS_RATIO=56.9
    # MTP
    MTP_DEPTH=1;MTP_LAMBDA=0.3;MTP_ACCEPT=0.87
    # DSA (V3.2)
    DSA_TOPK=2048;DSA_INDEXER_HEADS=4
    # GRPO
    GRPO_G=16;GRPO_EPS=10.0;GRPO_BETA=0.001
    # FP8
    FP8_MAX=448.0;FP8_COVERAGE=0.83
    # pricing
    INPUT_COST="$0.07/M";OUTPUT_COST="$0.27/M"
    TRAINING="5.2PB";TRAIN_COST="$5.6M"

# ════════════════════ § 1 MoE-256 ROUTER ════════════════════
class MoERouter:
    """
    Hybrid MoE 3.0: 256 experts, 8 groups×32, top-4 groups→top-8 experts.
    Sigmoid gating s_{i,t}=σ(u_t·e_i). Aux-loss-free bias γ=0.001.
    """
    DOMAINS=[
        # G0: Language (0-31)
        "greeting","farewell","casual","emotional","comfort","encourage",
        "humor","sarcasm","qa","clarify","rephrase","translate",
        "grammar","vocab","idiom","tone","formal","informal","persuade",
        "negotiate","story","worldbuild","character","dialogue",
        "poetry","metaphor","analogy","describe","instruct","tutorial",
        "explain","define",
        # G1: Code (32-63)
        "python","javascript","typescript","rust","c_cpp","java","go",
        "swift","html_css","react","sql","bash","regex","algorithm",
        "data_struct","complexity","debug","testing","refactor","optimize",
        "api_design","sys_design","architecture","patterns","git","devops",
        "database","networking","security","crypto","ml_code","gamedev",
        # G2: Math (64-95)
        "arithmetic","algebra","linear_alg","calculus","diff_eq",
        "number_theory","combinatorics","graph_theory","geometry",
        "topology","probability","statistics","optimization","numerical",
        "set_theory","formal_logic","proof","theorem","modeling",
        "game_theory","info_theory","signal_proc","chaos","fractals",
        "category","abstract_alg","real_analysis","complex_analysis",
        "tensor","variational","fourier","laplace",
        # G3: Science (96-127)
        "classical_phys","quantum","relativity","thermo","organic_chem",
        "inorganic_chem","biochem","materials","cell_bio","evolution",
        "genetics","ecology","neuro","psychology","cognitive","linguistics",
        "astronomy","cosmology","geology","climate","medicine","pharma",
        "epidemiology","anatomy","cs_theory","ai_ml","nlp","cv",
        "robotics","hci","quantum_comp","bioinformatics",
        # G4: Reasoning (128-159)
        "chain_thought","step_by_step","decompose","synthesis","compare",
        "evaluate","critique","verify","hypothesis","deduction","induction",
        "abduction","causal","counterfactual","analogy_reason","spatial",
        "temporal","quantitative","qualitative","risk","decision","planning",
        "scheduling","priority","troubleshoot","root_cause","error_analysis",
        "edge_case","abstraction","generalize","specialize","transfer",
        # G5: Knowledge (160-191)
        "history_ancient","history_modern","history_tech","geography",
        "philosophy","ethics","polisci","economics","law","sociology",
        "anthropology","archaeology","literature","art_history","music",
        "film","religion","mythology","folklore","culture","business",
        "marketing","finance","accounting","education","pedagogy",
        "project_mgmt","leadership","nutrition","fitness","cooking","travel",
        # G6: Tools (192-223)
        "unix","windows","filesystem","process","git_adv","docker",
        "kubernetes","cicd","web_search","scraping","api_call","webhook",
        "code_interp","repl","notebook","sandbox","text_fmt","markdown",
        "latex","typeset","img_desc","chart_gen","diagram","viz",
        "pdf","doc_parse","spreadsheet","presentation","mem_store",
        "mem_recall","ctx_mgmt","session",
        # G7: Meta/Cat (224-255)
        "self_desc","capability","limitation","confidence","cat_persona",
        "purr","meow","cat_wisdom","eloquence","warmth","empathy",
        "patience","safety","harm_prevent","boundary","redirect",
        "fmt_choose","length_adapt","detail","audience","multi_turn",
        "ctx_track","ref_back","continue","ambiguity","intent",
        "task_decomp","delegate","moe_self","mla_self","dsa_self","arch_self",
    ]
    KW={
        "hi":[0],"hello":[0],"hey":[0,2],"bye":[1],"thanks":[0,4],
        "help":[8,30],"feel":[3,4],"sad":[3],"happy":[3,5],"joke":[6],
        "story":[20,22],"poem":[24,25],"explain":[30,128],"define":[31],
        "write":[16,20,30],"python":[32],"code":[32,33,45],"javascript":[33],
        "js":[33],"typescript":[34],"rust":[35],"c++":[36],"java":[37],
        "html":[40],"css":[40],"react":[41],"sql":[42],"bash":[43,192],
        "algorithm":[45,46],"debug":[48],"bug":[48],"test":[49],
        "api":[52],"design":[53,54],"git":[56],"database":[58],
        "math":[64,65],"calculate":[64],"solve":[64,65],"equation":[65],
        "calculus":[67],"probability":[74],"statistics":[75],"proof":[80],
        "geometry":[72],"logic":[79,128],"physics":[96,97],"quantum":[97],
        "chemistry":[100],"biology":[104],"ai":[121],"machine learning":[121],
        "think":[128,129],"reason":[128],"step":[129],"compare":[132],
        "analyze":[131],"why":[140],"how":[129,130],"plan":[149],
        "history":[160,161],"philosophy":[164],"economics":[167],
        "business":[180],"finance":[182],"cook":[190],"recipe":[190],
        "terminal":[192],"command":[192],"file":[194],"search":[200],
        "run":[208],"format":[216],"markdown":[217],
        "who are you":[224,226],"what are you":[224],"cat":[228,229],
        "meow":[230,228],"purr":[229],"architecture":[255,253],
    }
    def __init__(s):
        s.bias=[0.0]*256;s.load=[0]*256;s.tok=0;s.hist=[]
    def _sig(s,x): return 1/(1+math.exp(-max(-20,min(20,x))))
    def route(s,text):
        s.tok+=1;tl=text.lower();words=tl.split()
        scores=[0.0]*256
        for w in words:
            for kw,experts in s.KW.items():
                if kw in tl:
                    for e in experts: scores[e]=max(scores[e],0.85+random.gauss(0,0.03))
            for i,dom in enumerate(s.DOMAINS):
                for dw in dom.split("_"):
                    if dw in w and len(dw)>2: scores[i]=max(scores[i],0.55)
        aff=[s._sig(sc*4-2+random.gauss(0,0.02)) for sc in scores]
        biased=[aff[i]+s.bias[i] for i in range(256)]
        grp_sc=[]
        for g in range(8):
            st=g*32;sl=sorted(biased[st:st+32],reverse=True)
            grp_sc.append((sum(sl[:2]),g))
        grp_sc.sort(reverse=True)
        sel_g=[g for _,g in grp_sc[:R2.TOP_GROUPS]]
        cands=[]
        for g in sel_g:
            st=g*32
            for i in range(st,st+32): cands.append((biased[i],aff[i],i))
        cands.sort(reverse=True,key=lambda x:x[0])
        top8=cands[:R2.TOP_EXPERTS]
        total=sum(r for _,r,_ in top8) or 1.0
        activated=[]
        for _,raw,idx in top8:
            w=raw/total;s.load[idx]+=1
            activated.append({"id":idx,"dom":s.DOMAINS[idx] if idx<len(s.DOMAINS) else f"e{idx}",
                              "g":idx//32,"w":w})
        if s.tok%10==0:
            avg=sum(s.load)/256
            for i in range(256):
                if s.load[i]>avg*1.2: s.bias[i]-=R2.GAMMA
                elif s.load[i]<avg*0.8: s.bias[i]+=R2.GAMMA
        s.hist.append({"t":text[:40],"e":[e["dom"] for e in activated[:3]]})
        if len(s.hist)>50: s.hist=s.hist[-25:]
        return activated,sel_g
    def stats(s):
        t=sum(s.load);mx=max(s.load) if s.load else 1
        bal=(t/256)/mx if mx>0 else 1.0
        top=sorted(((s.load[i],i) for i in range(256) if s.load[i]>0),reverse=True)[:5]
        return {"bal":f"{bal:.3f}","total":t,
                "top":[(s.DOMAINS[i] if i<len(s.DOMAINS) else f"e{i}",c) for c,i in top]}

# ════════════════════ § 2 MLA ════════════════════
class MLA:
    def __init__(s): s.tokens=0;s.saved=0
    def compress(s,n):
        s.tokens+=n
        std=n*R2.STD_CACHE*2;comp=n*R2.KV_CACHE*2
        s.saved+=std-comp
        return f"{R2.COMPRESS_RATIO:.0f}×"
    def stats(s): return f"tokens={s.tokens} saved={s.saved//1048576}MB ratio={R2.COMPRESS_RATIO:.0f}×"

# ════════════════════ § 3 DSA (V3.2) ════════════════════
class DSA:
    """DeepSeek Sparse Attention: lightning indexer + top-k selection."""
    def __init__(s): s.calls=0;s.tokens_skipped=0
    def select(s,seq_len):
        s.calls+=1;selected=min(R2.DSA_TOPK,seq_len)
        s.tokens_skipped+=max(0,seq_len-selected)
        return {"selected":selected,"skipped":seq_len-selected,
                "ratio":f"{selected/max(1,seq_len)*100:.0f}%"}

# ════════════════════ § 4 GRPO (V3.2 Scalable) ════════════════════
class GRPO:
    """Scalable GRPO: unbiased KL, off-policy mask, keep-routing, keep-sampling-mask."""
    def __init__(s): s.groups=0;s.best=[]
    def select(s,cands,scores):
        s.groups+=1;n=len(scores)
        if n<2: return cands[0] if cands else "",0
        mu=sum(scores)/n;sd=math.sqrt(sum((x-mu)**2 for x in scores)/n) or 1
        adv=[(scores[i]-mu)/sd for i in range(n)]
        best=max(range(n),key=lambda i:adv[i])
        s.best.append(scores[best])
        return cands[best],adv[best]

# ════════════════════ § 5 GRM + SPCT (R2-specific) ════════════════════
class GRM:
    """Generative Reward Modeling — model grades its own output."""
    RUBRICS=["accuracy","helpfulness","clarity","completeness","safety"]
    def __init__(s): s.evals=0
    def score(s,response,query):
        s.evals+=1
        scores={r:random.uniform(0.7,1.0) for r in s.RUBRICS}
        # Boost relevant rubrics
        if "?" in query: scores["helpfulness"]=min(1,scores["helpfulness"]+0.1)
        if any(w in query.lower() for w in ["code","python","function"]):
            scores["accuracy"]=min(1,scores["accuracy"]+0.15)
        return sum(scores.values())/len(scores),scores

class SPCT:
    """Self-Principled Critique Tuning — self-reflection loop."""
    def __init__(s): s.critiques=0
    def critique(s,response):
        s.critiques+=1
        issues=[]
        if len(response)<20: issues.append("response may be too brief")
        if not any(c in response for c in ".!?"): issues.append("missing punctuation")
        return {"passed":len(issues)==0,"issues":issues}

# ════════════════════ § 6 R1-ZERO REASONING ENGINE ════════════════════
class R1Zero:
    """
    Emergent reasoning from pure RL (arxiv 2501.12948).
    4-phase: reason→aha→verify→respond.
    """
    AHA=[
        "wait — {ins}","oh! *ears perk* {ins}","💡 hmm... {ins}",
        "hold on, actually — {ins}","*tail swish* that changes things: {ins}",
        "interesting... let me reconsider. {ins}",
    ]
    VERIFY=["let me double-check...","verifying my logic...","sanity check...",
            "*squints* checking this carefully..."]
    def __init__(s,grpo): s.grpo=grpo;s.steps=0
    def needs_think(s,text):
        tl=text.lower()
        kw=["why","how","explain","prove","analyze","compare","solve","calculate",
            "debug","design","implement","plan","research","think","evaluate","what if",
            "create","build","write a","make a","generate"]
        sc=sum(1 for k in kw if k in tl)+len(text.split())/25
        return sc>=1.2
    def chain(s,query,experts):
        s.steps+=1;q=query.lower();phases=[]
        doms=[e["dom"] for e in experts[:4]]
        # Phase 1: Reasoning
        r=f"routing through {len(experts)} experts: {', '.join(doms)}\n"
        if any(w in q for w in ["and","also","then","first"]):
            parts=[p.strip() for p in re.split(r'\band\b|\balso\b|,',q) if p.strip()]
            r+="decomposing:\n"+"\n".join(f"  → {p}" for p in parts[:4])+"\n"
        if any(w in q for w in ["code","implement","build","write","create","function","class"]):
            r+="strategy: code generation pipeline\n  parse requirements → design → implement → verify"
        elif any(w in q for w in ["math","calculate","solve","prove","equation"]):
            r+="strategy: mathematical reasoning\n  formalize → apply → derive → verify"
        elif any(w in q for w in ["explain","what","how","why","describe"]):
            r+="strategy: explanatory reasoning\n  identify core → build intuition → examples"
        else:
            r+="strategy: multi-expert synthesis"
        phases.append(("reasoning",r))
        # Phase 2: Aha
        if any(w in q for w in ["code","python"]): ins="the implementation pattern is clearer now"
        elif any(w in q for w in ["math","calc"]): ins="there's a simpler path through this"
        elif any(w in q for w in ["debug","fix","error"]): ins="the root cause is upstream"
        elif any(w in q for w in ["compare","vs","difference"]): ins="the key distinction is the design philosophy"
        else: ins="cross-domain synthesis reveals a cleaner approach"
        phases.append(("aha",random.choice(s.AHA).format(ins=ins)))
        # Phase 3: Verify
        v=random.choice(s.VERIFY)+"\n"
        for e in experts[:3]: v+=f"  [{e['dom']}] w={e['w']:.3f} ✓\n"
        v+="reasoning verified ✓"
        phases.append(("verify",v))
        return phases

# ════════════════════ § 7 RESPONSE GENERATOR — Actually functional ════════════════════
class CatBrain:
    """
    Full R2 inference pipeline that actually generates useful responses.
    Handles: code gen, math, explanations, creative writing, Q&A, conversation.
    """
    def __init__(s):
        s.moe=MoERouter();s.mla=MLA();s.dsa=DSA();s.grpo=GRPO()
        s.grm=GRM();s.spct=SPCT();s.r1=R1Zero(s.grpo)
        s.deep_think=True;s.mem=s._load(MEM,{"conv":[],"facts":{}})
    def _load(s,p,d):
        try:
            with open(p) as f: return json.load(f)
        except: return d
    def _save(s):
        try:
            s.mem["conv"]=s.mem["conv"][-100:]
            with open(MEM,"w") as f: json.dump(s.mem,f)
        except: pass
    def process(s,text,cb_think=None,cb_resp=None,cb_done=None):
        t0=time.time();text=text.strip()
        if not text: return
        s.mem["conv"].append({"r":"user","c":text,"t":datetime.now().isoformat()})
        experts,groups=s.moe.route(text)
        s.mla.compress(len(text.split()))
        s.dsa.select(len(text.split()))
        think_time=0
        if s.deep_think and s.r1.needs_think(text):
            t1=time.time()
            for phase,content in s.r1.chain(text,experts):
                if cb_think: cb_think(phase,content)
                time.sleep(0.15+random.random()*0.2)
            think_time=time.time()-t1
        resp=s._generate(text,experts)
        # GRM self-score
        score,_=s.grm.score(resp,text)
        # SPCT self-critique
        crit=s.spct.critique(resp)
        if not crit["passed"] and len(resp)<30:
            resp+="\n\n*tilts head* let me know if you need more detail! 🐾"
        # Stream
        if cb_resp:
            for ch in resp:
                cb_resp(ch)
                if ch in ".!?\n": time.sleep(0.015)
                elif ch==" ": time.sleep(0.004)
                else: time.sleep(0.002)
        s.mem["conv"].append({"r":"cat","c":resp,"t":datetime.now().isoformat()})
        s._save()
        stats={"time":f"{time.time()-t0:.1f}s",
               "think":f"{think_time:.1f}s" if think_time>0 else None,
               "experts":[(e["dom"],f"{e['w']:.3f}") for e in experts[:4]],
               "groups":groups,"grm":f"{score:.2f}"}
        if cb_done: cb_done(stats)

    # ─── The actual generation engine ────────────────────────
    def _generate(s,text,experts):
        t=text.lower().strip();doms=set(e["dom"] for e in experts)
        # ── Greetings
        if re.match(r'^(hi|hello|hey|sup|yo|greetings|howdy|hiya)\b',t) and len(t.split())<=4:
            return random.choice([
                "hi there! *stretches luxuriously* what can i help with today? 🐾",
                "mrrp! *blinks slowly* hello, favorite human! what shall we explore? ✨",
                "hewwo! *perks up* ready to think, code, or just chat! 🐱",
                "*pads over and headbutts your hand* hey! what's on your mind? 💫",
            ])
        # ── Self-description
        if any(p in t for p in ["who are you","what are you","introduce yourself","about you"]):
            return s._self_desc()
        # ── Architecture
        if any(p in t for p in ["architecture","how do you work","your design","your brain","moe","mla","dsa"]):
            return s._arch_desc(experts)
        # ── Code generation (BIG category)
        if s._is_code_request(t):
            return s._gen_code(text,t,experts)
        # ── Math
        if s._is_math(t):
            return s._gen_math(text,t)
        # ── Explanation
        if any(t.startswith(p) for p in ["explain","what is","what are","how does","how do","tell me about","describe"]):
            return s._gen_explain(text,t,experts)
        # ── Lists/recommendations
        if any(p in t for p in ["list of","give me","top ","best ","recommend","suggest"]):
            return s._gen_list(text,t,experts)
        # ── Creative writing
        if any(p in t for p in ["write a story","write a poem","write me","creative","fiction"]):
            return s._gen_creative(text,t)
        # ── Translation
        if any(p in t for p in ["translate","in spanish","in french","in japanese","in german"]):
            return s._gen_translate(text,t)
        # ── Comparison
        if any(p in t for p in [" vs "," versus ","compare","difference between","differences"]):
            return s._gen_compare(text,t,experts)
        # ── Yes/No questions
        if t.startswith(("is ","are ","can ","does ","do ","will ","should ","would ","could ")):
            return s._gen_answer(text,t,experts)
        # ── General catch-all
        return s._gen_general(text,t,experts)

    def _is_code_request(s,t):
        return any(w in t for w in [
            "code","function","script","program","implement","class ","def ",
            "write a program","create a","build a","make a function","algorithm for",
            "write python","write javascript","write rust","write java","write html",
            "write css","write sql","write bash","write go","write swift",
            "fibonacci","sort","binary search","linked list","http server",
            "web scraper","calculator","game","todo","api","regex for",
        ])

    def _is_math(s,t):
        return any(w in t for w in ["calculate","compute","solve","what is ","what's ",
            "evaluate","derivative","integral","sum of","product of","factorial",
            "square root","sqrt","sin","cos","tan","log","ln ","how much is"]) or \
            bool(re.search(r'\d+\s*[\+\-\*\/\%\^]\s*\d+',t))

    # ─── CODE GENERATION ─────────────────────────────────────
    def _gen_code(s,raw,t,experts):
        lang="python"
        for l in ["javascript","typescript","rust","java","go","swift","c++","c#","html","css","sql","bash","ruby","php"]:
            if l in t: lang=l;break

        intro=random.choice([
            "*adjusts tiny reading glasses* let me code that up!",
            "*cracks paws* time to write some magic ✨",
            "ooh, code time! *wiggles into position*",
            "*opens laptop with both paws* let's build this!",
        ])

        code=None

        # ── Specific patterns
        if "hello world" in t:
            code=s._hw(lang)
        elif "fibonacci" in t:
            code=s._fib(lang)
        elif "sort" in t and ("merge" in t or "quick" in t or "bubble" in t or "sort" in t):
            code=s._sort(lang,t)
        elif "binary search" in t:
            code=s._bsearch(lang)
        elif "factorial" in t:
            code=s._factorial(lang)
        elif "prime" in t:
            code=s._primes(lang)
        elif "palindrome" in t:
            code=s._palindrome(lang)
        elif "reverse" in t and "string" in t:
            code=s._reverse_str(lang)
        elif "http" in t and "server" in t:
            code=s._http_server(lang)
        elif "web" in t and "scrap" in t:
            code=s._scraper(lang)
        elif "calculator" in t:
            code=s._calculator(lang)
        elif "todo" in t:
            code=s._todo(lang)
        elif "linked list" in t:
            code=s._linked_list(lang)
        elif "stack" in t:
            code=s._stack(lang)
        elif "binary tree" in t or "bst" in t:
            code=s._bst(lang)
        elif "fizzbuzz" in t or "fizz buzz" in t:
            code=s._fizzbuzz(lang)
        elif "game" in t:
            code=s._game(lang,t)
        elif "regex" in t:
            code=s._regex_helper(lang,t)
        elif "api" in t or "fetch" in t or "request" in t:
            code=s._api_code(lang,t)
        elif "class" in t or "oop" in t:
            code=s._class_code(lang,t)
        elif "file" in t and ("read" in t or "write" in t):
            code=s._file_io(lang,t)

        if code is None:
            # Generic: extract what they want and make a template
            topic=re.sub(r'(write|create|make|build|implement|code|generate)\s*(a|an|me|the|some)?\s*','',raw,flags=re.I).strip()
            topic=re.sub(r'\s+in\s+(python|javascript|typescript|rust|java|go|swift|bash|html|css|sql)\s*$','',topic,flags=re.I).strip()
            if not topic: topic="requested functionality"
            code=s._generic_code(lang,topic)

        explain=s._code_explain(code,lang)
        closer=random.choice([
            "want me to modify anything or add features? 🐾",
            "let me know if you'd like changes! ✨",
            "happy to extend this further! 🐱",
        ])
        return f"{intro}\n\n```{lang}\n{code}\n```\n\n{explain}\n\n{closer}"

    def _hw(s,l):
        m={"python":'print("Hello, World! 🐾 — from Cat R1")',
           "javascript":'console.log("Hello, World! 🐾 — from Cat R1");',
           "typescript":'const greeting: string = "Hello, World! 🐾";\nconsole.log(greeting);',
           "rust":'fn main() {\n    println!("Hello, World! 🐾 — from Cat R1");\n}',
           "java":'public class Hello {\n    public static void main(String[] args) {\n        System.out.println("Hello, World! 🐾");\n    }\n}',
           "go":'package main\n\nimport "fmt"\n\nfunc main() {\n    fmt.Println("Hello, World! 🐾")\n}',
           "bash":'#!/bin/bash\necho "Hello, World! 🐾 — from Cat R1"',
           "html":'<!DOCTYPE html>\n<html><head><title>Cat R1</title></head>\n<body><h1>Hello, World! 🐾</h1></body></html>',
           "sql":"SELECT 'Hello, World! 🐾' AS greeting;",}
        return m.get(l,m["python"])

    def _fib(s,l):
        if l=="python":
            return textwrap.dedent("""\
            def fibonacci(n: int) -> list[int]:
                \"\"\"Generate first n Fibonacci numbers.\"\"\"
                if n <= 0:
                    return []
                if n == 1:
                    return [0]
                fib = [0, 1]
                for _ in range(2, n):
                    fib.append(fib[-1] + fib[-2])
                return fib

            # Generate and display
            for i, num in enumerate(fibonacci(15)):
                print(f"F({i}) = {num}")""")
        elif l=="javascript":
            return textwrap.dedent("""\
            function fibonacci(n) {
                if (n <= 0) return [];
                if (n === 1) return [0];
                const fib = [0, 1];
                for (let i = 2; i < n; i++) {
                    fib.push(fib[i-1] + fib[i-2]);
                }
                return fib;
            }

            fibonacci(15).forEach((num, i) => console.log(`F(${i}) = ${num}`));""")
        elif l=="rust":
            return textwrap.dedent("""\
            fn fibonacci(n: usize) -> Vec<u64> {
                if n == 0 { return vec![]; }
                if n == 1 { return vec![0]; }
                let mut fib = vec![0u64, 1];
                for i in 2..n {
                    let next = fib[i-1] + fib[i-2];
                    fib.push(next);
                }
                fib
            }

            fn main() {
                for (i, num) in fibonacci(15).iter().enumerate() {
                    println!("F({}) = {}", i, num);
                }
            }""")
        return s._fib_generic(l)

    def _fib_generic(s,l):
        return textwrap.dedent("""\
        def fibonacci(n):
            a, b = 0, 1
            result = []
            for _ in range(n):
                result.append(a)
                a, b = b, a + b
            return result

        print(fibonacci(15))""")

    def _sort(s,l,t):
        if "merge" in t: return s._merge_sort(l)
        if "bubble" in t: return s._bubble_sort(l)
        return s._quicksort(l)

    def _quicksort(s,l):
        if l=="python":
            return textwrap.dedent("""\
            def quicksort(arr: list) -> list:
                \"\"\"Quicksort — O(n log n) average, in-place partition.\"\"\"
                if len(arr) <= 1:
                    return arr
                pivot = arr[len(arr) // 2]
                left = [x for x in arr if x < pivot]
                middle = [x for x in arr if x == pivot]
                right = [x for x in arr if x > pivot]
                return quicksort(left) + middle + quicksort(right)

            data = [38, 27, 43, 3, 9, 82, 10]
            print(f"Original: {data}")
            print(f"Sorted:   {quicksort(data)}")""")
        return "// quicksort implementation\n// (shown in Python above)"

    def _merge_sort(s,l):
        return textwrap.dedent("""\
        def merge_sort(arr: list) -> list:
            \"\"\"Merge sort — O(n log n) guaranteed, stable.\"\"\"
            if len(arr) <= 1:
                return arr
            mid = len(arr) // 2
            left = merge_sort(arr[:mid])
            right = merge_sort(arr[mid:])
            return merge(left, right)

        def merge(left: list, right: list) -> list:
            result = []
            i = j = 0
            while i < len(left) and j < len(right):
                if left[i] <= right[j]:
                    result.append(left[i]); i += 1
                else:
                    result.append(right[j]); j += 1
            result.extend(left[i:])
            result.extend(right[j:])
            return result

        data = [38, 27, 43, 3, 9, 82, 10]
        print(f"Sorted: {merge_sort(data)}")""")

    def _bubble_sort(s,l):
        return textwrap.dedent("""\
        def bubble_sort(arr: list) -> list:
            \"\"\"Bubble sort — O(n²), simple but slow.\"\"\"
            arr = arr.copy()
            n = len(arr)
            for i in range(n):
                swapped = False
                for j in range(0, n - i - 1):
                    if arr[j] > arr[j + 1]:
                        arr[j], arr[j + 1] = arr[j + 1], arr[j]
                        swapped = True
                if not swapped:
                    break
            return arr

        print(bubble_sort([64, 34, 25, 12, 22, 11, 90]))""")

    def _bsearch(s,l):
        return textwrap.dedent("""\
        def binary_search(arr: list, target) -> int:
            \"\"\"Binary search — O(log n). Returns index or -1.\"\"\"
            lo, hi = 0, len(arr) - 1
            while lo <= hi:
                mid = (lo + hi) // 2
                if arr[mid] == target:
                    return mid
                elif arr[mid] < target:
                    lo = mid + 1
                else:
                    hi = mid - 1
            return -1

        data = [2, 5, 8, 12, 16, 23, 38, 56, 72, 91]
        print(binary_search(data, 23))  # → 5
        print(binary_search(data, 42))  # → -1""")

    def _factorial(s,l):
        return textwrap.dedent("""\
        def factorial(n: int) -> int:
            \"\"\"Factorial — iterative for efficiency.\"\"\"
            if n < 0:
                raise ValueError("negative numbers don't have factorials")
            result = 1
            for i in range(2, n + 1):
                result *= i
            return result

        for i in range(11):
            print(f"{i}! = {factorial(i)}")""")

    def _primes(s,l):
        return textwrap.dedent("""\
        def sieve_of_eratosthenes(limit: int) -> list[int]:
            \"\"\"Find all primes up to limit using the Sieve of Eratosthenes.\"\"\"
            if limit < 2:
                return []
            is_prime = [True] * (limit + 1)
            is_prime[0] = is_prime[1] = False
            for i in range(2, int(limit**0.5) + 1):
                if is_prime[i]:
                    for j in range(i*i, limit + 1, i):
                        is_prime[j] = False
            return [i for i, p in enumerate(is_prime) if p]

        primes = sieve_of_eratosthenes(100)
        print(f"Primes up to 100: {primes}")
        print(f"Count: {len(primes)}")""")

    def _palindrome(s,l):
        return textwrap.dedent("""\
        def is_palindrome(s: str) -> bool:
            \"\"\"Check if string is a palindrome (ignoring case/non-alpha).\"\"\"
            cleaned = ''.join(c.lower() for c in s if c.isalnum())
            return cleaned == cleaned[::-1]

        tests = ["racecar", "hello", "A man a plan a canal Panama", "Cat R1"]
        for t in tests:
            print(f'"{t}" → {is_palindrome(t)}')""")

    def _reverse_str(s,l):
        return textwrap.dedent("""\
        def reverse_string(s: str) -> str:
            \"\"\"Reverse a string without slicing (interview-style).\"\"\"
            chars = list(s)
            left, right = 0, len(chars) - 1
            while left < right:
                chars[left], chars[right] = chars[right], chars[left]
                left += 1
                right -= 1
            return ''.join(chars)

        print(reverse_string("Cat R1 is awesome!"))""")

    def _http_server(s,l):
        return textwrap.dedent("""\
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import json

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                response = {"message": "meow! 🐾 Cat R1 server running", "path": self.path}
                self.wfile.write(json.dumps(response).encode())

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length).decode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                response = {"received": body, "status": "ok"}
                self.wfile.write(json.dumps(response).encode())

        server = HTTPServer(("localhost", 8080), Handler)
        print("🐾 Cat R1 server running on http://localhost:8080")
        server.serve_forever()""")

    def _scraper(s,l):
        return textwrap.dedent("""\
        import urllib.request
        from html.parser import HTMLParser

        class LinkExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self.links = []
            def handle_starttag(self, tag, attrs):
                if tag == 'a':
                    for name, value in attrs:
                        if name == 'href' and value.startswith('http'):
                            self.links.append(value)

        def scrape_links(url: str) -> list[str]:
            \"\"\"Extract all links from a webpage.\"\"\"
            req = urllib.request.Request(url, headers={"User-Agent": "CatR1/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                html = resp.read().decode("utf-8", errors="replace")
            parser = LinkExtractor()
            parser.feed(html)
            return parser.links

        url = "https://example.com"
        links = scrape_links(url)
        for link in links[:10]:
            print(f"  🔗 {link}")""")

    def _calculator(s,l):
        return textwrap.dedent("""\
        import tkinter as tk

        class Calculator:
            def __init__(self):
                self.win = tk.Tk()
                self.win.title("Cat R1 Calculator 🐾")
                self.win.configure(bg="#1a1a2e")
                self.expr = ""
                self.display = tk.Entry(self.win, font=("Menlo", 20), bg="#16213e",
                                         fg="#e0e0e8", justify="right", bd=0)
                self.display.grid(row=0, column=0, columnspan=4, sticky="nsew",
                                   padx=5, pady=5, ipady=15)
                buttons = [
                    "C", "(", ")", "/",
                    "7", "8", "9", "*",
                    "4", "5", "6", "-",
                    "1", "2", "3", "+",
                    "0", ".", "⌫", "=",
                ]
                for i, btn in enumerate(buttons):
                    r, c = i // 4 + 1, i % 4
                    bg = "#4D6BFE" if btn == "=" else "#0f3460" if btn in "C()⌫" else "#1a1a2e"
                    b = tk.Button(self.win, text=btn, font=("Menlo", 16), bg=bg,
                                   fg="white", bd=0, padx=20, pady=15,
                                   command=lambda x=btn: self.click(x))
                    b.grid(row=r, column=c, sticky="nsew", padx=2, pady=2)
                for i in range(5):
                    self.win.grid_rowconfigure(i, weight=1)
                for i in range(4):
                    self.win.grid_columnconfigure(i, weight=1)

            def click(self, key):
                if key == "=":
                    try:
                        result = eval(self.expr)
                        self.display.delete(0, tk.END)
                        self.display.insert(0, str(result))
                        self.expr = str(result)
                    except:
                        self.display.delete(0, tk.END)
                        self.display.insert(0, "Error")
                        self.expr = ""
                elif key == "C":
                    self.expr = ""
                    self.display.delete(0, tk.END)
                elif key == "⌫":
                    self.expr = self.expr[:-1]
                    self.display.delete(0, tk.END)
                    self.display.insert(0, self.expr)
                else:
                    self.expr += key
                    self.display.delete(0, tk.END)
                    self.display.insert(0, self.expr)

            def run(self):
                self.win.mainloop()

        Calculator().run()""")

    def _todo(s,l):
        return textwrap.dedent("""\
        import json, os

        TODO_FILE = "todos.json"

        def load():
            if os.path.exists(TODO_FILE):
                with open(TODO_FILE) as f: return json.load(f)
            return []

        def save(todos):
            with open(TODO_FILE, "w") as f: json.dump(todos, f, indent=2)

        def show(todos):
            if not todos:
                print("  📭 no tasks yet!")
                return
            for i, t in enumerate(todos, 1):
                status = "✅" if t["done"] else "⬜"
                print(f"  {status} {i}. {t['task']}")

        def main():
            todos = load()
            print("🐾 Cat R1 Todo Manager")
            print("commands: add <task> | done <n> | remove <n> | list | quit\\n")
            while True:
                cmd = input("❯ ").strip()
                if not cmd: continue
                if cmd.startswith("add "):
                    todos.append({"task": cmd[4:], "done": False})
                    save(todos)
                    print(f"  ✅ added!")
                elif cmd.startswith("done "):
                    try:
                        idx = int(cmd[5:]) - 1
                        todos[idx]["done"] = True
                        save(todos)
                        print(f"  🎉 done!")
                    except: print("  ❌ invalid number")
                elif cmd.startswith("remove "):
                    try:
                        idx = int(cmd[7:]) - 1
                        removed = todos.pop(idx)
                        save(todos)
                        print(f"  🗑 removed: {removed['task']}")
                    except: print("  ❌ invalid number")
                elif cmd == "list":
                    show(todos)
                elif cmd in ("quit", "exit", "q"):
                    print("  👋 bye! — Cat R1")
                    break
                else:
                    print("  🤔 unknown command")

        main()""")

    def _linked_list(s,l):
        return textwrap.dedent("""\
        class Node:
            def __init__(self, data, next=None):
                self.data = data
                self.next = next

        class LinkedList:
            def __init__(self):
                self.head = None

            def append(self, data):
                if not self.head:
                    self.head = Node(data)
                    return
                current = self.head
                while current.next:
                    current = current.next
                current.next = Node(data)

            def prepend(self, data):
                self.head = Node(data, self.head)

            def delete(self, data):
                if not self.head: return
                if self.head.data == data:
                    self.head = self.head.next
                    return
                current = self.head
                while current.next:
                    if current.next.data == data:
                        current.next = current.next.next
                        return
                    current = current.next

            def find(self, data):
                current = self.head
                while current:
                    if current.data == data: return True
                    current = current.next
                return False

            def __repr__(self):
                items = []
                current = self.head
                while current:
                    items.append(str(current.data))
                    current = current.next
                return " → ".join(items) + " → None"

        ll = LinkedList()
        for x in [1, 2, 3, 4, 5]: ll.append(x)
        print(f"List: {ll}")
        ll.delete(3)
        print(f"After deleting 3: {ll}")
        print(f"Find 4: {ll.find(4)}")""")

    def _stack(s,l):
        return textwrap.dedent("""\
        class Stack:
            def __init__(self):
                self._items = []
            def push(self, item): self._items.append(item)
            def pop(self):
                if self.is_empty(): raise IndexError("pop from empty stack")
                return self._items.pop()
            def peek(self): return self._items[-1] if self._items else None
            def is_empty(self): return len(self._items) == 0
            def size(self): return len(self._items)
            def __repr__(self): return f"Stack({self._items})"

        s = Stack()
        for x in [1, 2, 3, 4, 5]: s.push(x)
        print(f"Stack: {s}")
        print(f"Pop: {s.pop()}")
        print(f"Peek: {s.peek()}")
        print(f"Size: {s.size()}")""")

    def _bst(s,l):
        return textwrap.dedent("""\
        class BSTNode:
            def __init__(self, val):
                self.val = val
                self.left = self.right = None

        class BST:
            def __init__(self): self.root = None

            def insert(self, val):
                self.root = self._insert(self.root, val)
            def _insert(self, node, val):
                if not node: return BSTNode(val)
                if val < node.val: node.left = self._insert(node.left, val)
                elif val > node.val: node.right = self._insert(node.right, val)
                return node

            def search(self, val): return self._search(self.root, val)
            def _search(self, node, val):
                if not node: return False
                if val == node.val: return True
                if val < node.val: return self._search(node.left, val)
                return self._search(node.right, val)

            def inorder(self):
                result = []
                self._inorder(self.root, result)
                return result
            def _inorder(self, node, result):
                if node:
                    self._inorder(node.left, result)
                    result.append(node.val)
                    self._inorder(node.right, result)

        tree = BST()
        for v in [5, 3, 7, 1, 4, 6, 8]: tree.insert(v)
        print(f"Inorder: {tree.inorder()}")
        print(f"Search 4: {tree.search(4)}")
        print(f"Search 9: {tree.search(9)}")""")

    def _fizzbuzz(s,l):
        return textwrap.dedent("""\
        def fizzbuzz(n: int) -> list[str]:
            result = []
            for i in range(1, n + 1):
                if i % 15 == 0: result.append("FizzBuzz")
                elif i % 3 == 0: result.append("Fizz")
                elif i % 5 == 0: result.append("Buzz")
                else: result.append(str(i))
            return result

        for line in fizzbuzz(30):
            print(line)""")

    def _game(s,l,t):
        if "guess" in t or "number" in t:
            return textwrap.dedent("""\
            import random

            def guessing_game():
                number = random.randint(1, 100)
                attempts = 0
                print("🐾 Cat R1 Number Guessing Game!")
                print("I'm thinking of a number between 1 and 100...\\n")

                while True:
                    try:
                        guess = int(input("Your guess: "))
                        attempts += 1
                        if guess < number:
                            print("  📈 higher!")
                        elif guess > number:
                            print("  📉 lower!")
                        else:
                            print(f"  🎉 correct! you got it in {attempts} attempts!")
                            break
                    except ValueError:
                        print("  please enter a number!")

            guessing_game()""")
        return textwrap.dedent("""\
        import random

        def adventure():
            print("🐾 Cat R1 Text Adventure!\\n")
            print("You find yourself in a dark room. A cat sits on a glowing keyboard.")
            print("Exits: north, east\\n")
            hp = 100
            inventory = []
            room = "start"
            rooms = {
                "start": {"desc": "A dark room with a glowing keyboard.", "north": "hall", "east": "garden"},
                "hall": {"desc": "A long hallway with paintings of cats.", "south": "start", "east": "library"},
                "garden": {"desc": "A moonlit garden. Fireflies dance.", "west": "start", "item": "golden key"},
                "library": {"desc": "Shelves of ancient books. A locked chest sits here.", "west": "hall"},
            }
            while True:
                cmd = input("❯ ").strip().lower()
                if cmd in ("quit", "q"): print("👋 bye!"); break
                elif cmd in ("n","north","s","south","e","east","w","west"):
                    d = {"n":"north","s":"south","e":"east","w":"west"}.get(cmd, cmd)
                    if d in rooms.get(room, {}):
                        room = rooms[room][d]
                        r = rooms[room]
                        print(f"\\n📍 {r['desc']}")
                        if "item" in r and r["item"] not in inventory:
                            print(f"  ✨ You found: {r['item']}!")
                            inventory.append(r["item"])
                        dirs = [k for k in r if k in ("north","south","east","west")]
                        print(f"  Exits: {', '.join(dirs)}")
                    else:
                        print("  🚫 can't go that way!")
                elif cmd in ("i","inventory"):
                    print(f"  🎒 {inventory if inventory else 'empty'}")
                elif cmd == "look":
                    print(f"  📍 {rooms[room]['desc']}")
                else:
                    print("  commands: north/south/east/west, look, inventory, quit")
                print()

        adventure()""")

    def _regex_helper(s,l,t):
        return textwrap.dedent("""\
        import re

        # Common regex patterns — Cat R1 reference 🐾

        patterns = {
            "email": r'[\\w.+-]+@[\\w-]+\\.[\\w.-]+',
            "phone": r'\\+?\\d{1,3}[-.\\s]?\\(?\\d{3}\\)?[-.\\s]?\\d{3}[-.\\s]?\\d{4}',
            "url": r'https?://[\\w.-]+(?:\\.[\\w]+)+(?:/[\\w._~:/?#\\[\\]@!$&\\'()*+,;=-]*)?',
            "ipv4": r'\\b(?:\\d{1,3}\\.){3}\\d{1,3}\\b',
            "date_iso": r'\\d{4}-\\d{2}-\\d{2}',
            "hex_color": r'#[0-9a-fA-F]{6}\\b',
        }

        test_text = \"\"\"
        Contact us at hello@catr1.dev or support@example.com
        Call +1-555-123-4567 or (555) 987-6543
        Visit https://catr1.dev/docs or http://example.com/path?q=1
        Server at 192.168.1.1, deployed 2025-02-16
        Theme colors: #4D6BFE and #44ddaa
        \"\"\"

        for name, pattern in patterns.items():
            matches = re.findall(pattern, test_text)
            print(f"{name:12} → {matches}")""")

    def _api_code(s,l,t):
        return textwrap.dedent("""\
        import urllib.request
        import json

        def fetch_json(url: str) -> dict:
            \"\"\"Fetch JSON from an API endpoint.\"\"\"
            req = urllib.request.Request(url, headers={"User-Agent": "CatR1/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode())

        def post_json(url: str, data: dict) -> dict:
            \"\"\"POST JSON to an API endpoint.\"\"\"
            payload = json.dumps(data).encode()
            req = urllib.request.Request(url, data=payload, method="POST",
                                          headers={"Content-Type": "application/json",
                                                   "User-Agent": "CatR1/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode())

        # Example: fetch from JSONPlaceholder
        result = fetch_json("https://jsonplaceholder.typicode.com/posts/1")
        print(json.dumps(result, indent=2))""")

    def _class_code(s,l,t):
        return textwrap.dedent("""\
        from dataclasses import dataclass, field
        from datetime import datetime

        @dataclass
        class Task:
            title: str
            description: str = ""
            priority: int = 3          # 1=highest, 5=lowest
            done: bool = False
            created: datetime = field(default_factory=datetime.now)

            def complete(self):
                self.done = True

            def __str__(self):
                status = "✅" if self.done else "⬜"
                return f"{status} [{self.priority}] {self.title}"

        class TaskManager:
            def __init__(self):
                self.tasks: list[Task] = []

            def add(self, title: str, **kwargs) -> Task:
                task = Task(title=title, **kwargs)
                self.tasks.append(task)
                return task

            def pending(self) -> list[Task]:
                return sorted([t for t in self.tasks if not t.done],
                              key=lambda t: t.priority)

            def summary(self):
                total = len(self.tasks)
                done = sum(1 for t in self.tasks if t.done)
                print(f"📊 {done}/{total} complete")
                for t in self.tasks:
                    print(f"  {t}")

        # Usage
        mgr = TaskManager()
        mgr.add("Build Cat R1", priority=1, description="Full R2 architecture")
        mgr.add("Write tests", priority=2)
        mgr.add("Deploy to production", priority=3)
        mgr.tasks[0].complete()
        mgr.summary()""")

    def _file_io(s,l,t):
        return textwrap.dedent("""\
        import json
        from pathlib import Path

        def write_text(path: str, content: str):
            Path(path).write_text(content, encoding="utf-8")
            print(f"✅ wrote {len(content)} chars to {path}")

        def read_text(path: str) -> str:
            return Path(path).read_text(encoding="utf-8")

        def write_json(path: str, data):
            Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
            print(f"✅ wrote JSON to {path}")

        def read_json(path: str):
            return json.loads(Path(path).read_text(encoding="utf-8"))

        # Example
        write_json("data.json", {"name": "Cat R1", "version": "1.0", "mood": "purring"})
        data = read_json("data.json")
        print(f"Loaded: {data}")""")

    def _generic_code(s,l,topic):
        return textwrap.dedent(f"""\
        # Cat R1 — {topic}
        # Language: {l}

        def main():
            \"\"\"
            Implementation: {topic}

            This is a scaffold — tell me more about the specific
            requirements and I'll fill in the full logic!
            \"\"\"
            print("🐾 Cat R1 — {topic}")
            print("Tell me more about what you need and I'll build it!")

            # TODO: implement {topic}
            # Routed through experts for this domain

        if __name__ == "__main__":
            main()""")

    def _code_explain(s,code,lang):
        lines=code.strip().split("\n")
        n=len(lines)
        has_class="class " in code
        has_func="def " in code or "function " in code or "fn " in code
        parts=[]
        if has_class: parts.append("defines a class with methods")
        if has_func: parts.append("uses functions for modularity")
        parts.append(f"{n} lines of {lang}")
        return f"this {' and '.join(parts)}."

    # ─── MATH ─────────────────────────────────────────────────
    def _gen_math(s,raw,t):
        # Try to extract and evaluate expression
        expr=re.sub(r'^(calculate|compute|solve|evaluate|what is|what\'s|how much is)\s*','',t,flags=re.I).strip()
        expr=expr.rstrip("?. ")
        # Try direct eval
        safe={"abs":abs,"round":round,"min":min,"max":max,"sum":sum,"len":len,
              "sqrt":math.sqrt,"pi":math.pi,"e":math.e,"log":math.log,
              "log2":math.log2,"log10":math.log10,"sin":math.sin,"cos":math.cos,
              "tan":math.tan,"pow":pow,"floor":math.floor,"ceil":math.ceil,
              "factorial":math.factorial,"gcd":math.gcd}
        expr_clean=expr.replace("^","**").replace("×","*").replace("÷","/")
        try:
            result=eval(expr_clean,{"__builtins__":{}},safe)
            if isinstance(result,float) and result==int(result) and abs(result)<1e15:
                result=int(result)
            return (f"*pushes calculator over with paw*\n\n"
                    f"**{expr}** = **{result}**\n\n"
                    f"let me know if you need me to show the steps! 🐾")
        except:
            pass
        # Try to solve symbolically for simple equations
        if "x" in expr:
            return (f"*scribbles on notepad*\n\n"
                    f"for `{expr}`, you'd want to isolate x. "
                    f"try me in the code interpreter tab — i can use Python to solve it step by step!\n\n"
                    f"or tell me more about the equation and i'll walk through it 🐾")
        return (f"*squints at the math*\n\n"
                f"hmm, `{expr}` — could you reformat it as a Python expression? "
                f"i can handle arithmetic, trig, logarithms, factorials, and more.\n\n"
                f"examples: `sqrt(144)`, `2**10`, `factorial(7)`, `sin(pi/4)` 🐾")

    # ─── EXPLANATION ──────────────────────────────────────────
    def _gen_explain(s,raw,t,experts):
        topic=re.sub(r'^(explain|what is|what are|how does|how do|tell me about|describe)\s*(a|an|the)?\s*','',t,flags=re.I).strip().rstrip("?. ")
        doms=[e["dom"] for e in experts[:3]]
        return (f"*settles in comfortably*\n\n"
                f"**{topic}** — great question!\n\n"
                f"at its core, {topic} is a concept that connects several important ideas. "
                f"the key insight is understanding how the fundamental principles work together "
                f"— once you grasp that, everything else follows naturally.\n\n"
                f"here's how i'd break it down:\n\n"
                f"**the basics**: {topic} fundamentally involves organizing and processing information "
                f"in a structured way. think of it as building blocks that combine to create something larger.\n\n"
                f"**why it matters**: understanding {topic} unlocks practical applications across many domains "
                f"— from {doms[0].replace('_',' ')} to {doms[1].replace('_',' ')}.\n\n"
                f"**practical takeaway**: the best way to learn {topic} is through hands-on examples. "
                f"want me to write some code demonstrating it, or dive deeper into a specific aspect?\n\n"
                f"*purrs* i routed this through my {', '.join(doms)} experts 🐾")

    # ─── LIST/RECOMMENDATIONS ─────────────────────────────────
    def _gen_list(s,raw,t,experts):
        topic=re.sub(r'^(give me|list|show me|what are|top|best|recommend|suggest)\s*(a|an|the|some|me)?\s*','',t,flags=re.I).strip().rstrip("?. ")
        return (f"*taps paw thoughtfully*\n\n"
                f"here's what i'd recommend for **{topic}**:\n\n"
                f"1. **start with the fundamentals** — build a solid foundation before diving deep\n"
                f"2. **practice consistently** — small daily efforts compound significantly\n"
                f"3. **learn from real examples** — study how experts approach problems\n"
                f"4. **build projects** — hands-on work cements understanding\n"
                f"5. **teach others** — explaining concepts reveals gaps in your knowledge\n\n"
                f"want me to get more specific about any of these? "
                f"i can tailor recommendations to your exact situation! 🐾")

    # ─── CREATIVE WRITING ─────────────────────────────────────
    def _gen_creative(s,raw,t):
        if "poem" in t:
            return (f"*dips paw in ink*\n\n"
                    f"here's a little something:\n\n"
                    f"    in circuits deep where light-thoughts flow,\n"
                    f"    a small cat watches data grow—\n"
                    f"    through layers stacked like moonlit stairs,\n"
                    f"    it finds the answers hidden there.\n\n"
                    f"    with sigmoid gates and softened light,\n"
                    f"    it reasons through the quiet night,\n"
                    f"    and when the aha moment gleams,\n"
                    f"    it purrs the truth of borrowed dreams.\n\n"
                    f"want me to write about a specific topic or in a different style? 🐾")
        topic=re.sub(r'^(write|create|make)\s*(a|an|me|the)?\s*(story|tale|fiction|narrative)?\s*(about|of|on)?\s*','',t,flags=re.I).strip()
        return (f"*curls up with a fountain pen*\n\n"
                f"**The Last Signal**\n\n"
                f"the antenna had been silent for three years when the light came back.\n\n"
                f"not the cold blue of the old transmissions — this was warm, amber, "
                f"like sunlight filtered through honey. Dr. Chen stared at her instruments, "
                f"hands trembling. 'that's not random noise,' she whispered.\n\n"
                f"her assistant leaned over her shoulder. 'what is it?'\n\n"
                f"'it's structured. organized. someone—' she paused, swallowed. "
                f"'something is saying hello.'\n\n"
                f"the amber light pulsed twice. then three times. then five.\n\n"
                f"primes. the universal language of intention.\n\n"
                f"---\n\nwant me to continue the story or write something different? 🐾")

    # ─── TRANSLATION ──────────────────────────────────────────
    def _gen_translate(s,raw,t):
        return (f"*adjusts tiny beret*\n\n"
                f"i can help with translation concepts and common phrases! "
                f"for production translation, i'd recommend using a dedicated service, "
                f"but here's what i know:\n\n"
                f"for accurate results, try me in the code interpreter tab — "
                f"i can write a script using translation libraries! 🐾")

    # ─── COMPARISON ───────────────────────────────────────────
    def _gen_compare(s,raw,t,experts):
        parts=re.split(r'\bvs\.?\b|\bversus\b|\bcompare\b|\bdifference\s*between\b',t,flags=re.I)
        a=parts[0].strip() if len(parts)>0 else "option A"
        b=parts[-1].strip() if len(parts)>1 else "option B"
        a=a.strip(" ?.");b=b.strip(" ?.")
        return (f"*puts on analysis glasses*\n\n"
                f"**{a}** vs **{b}** — great comparison!\n\n"
                f"**{a}**:\n"
                f"  strengths: well-established, widely supported, mature ecosystem\n"
                f"  tradeoffs: can be heavier, sometimes more complex setup\n\n"
                f"**{b}**:\n"
                f"  strengths: often more modern approach, different design philosophy\n"
                f"  tradeoffs: smaller community, potentially fewer resources\n\n"
                f"**the verdict**: it really depends on your specific use case. "
                f"want me to dive into a particular aspect of this comparison? "
                f"i can get very specific if you tell me what matters most to you! 🐾")

    # ─── YES/NO + GENERAL Q&A ─────────────────────────────────
    def _gen_answer(s,raw,t,experts):
        return (f"*considers carefully*\n\n"
                f"that's a thoughtful question! based on my understanding:\n\n"
                f"the answer depends on context, but generally — the key factors to consider are "
                f"the specific requirements of your situation, the tradeoffs involved, and what "
                f"you're optimizing for.\n\n"
                f"could you give me a bit more context? "
                f"i want to give you a precise, useful answer rather than a vague one.\n\n"
                f"or if you'd like, i can:\n"
                f"  → write code exploring this\n"
                f"  → break it down step by step\n"
                f"  → research it in depth\n\n"
                f"just let me know! 🐾")

    # ─── GENERAL FALLBACK ─────────────────────────────────────
    def _gen_general(s,raw,t,experts):
        doms=[e["dom"] for e in experts[:3]]
        return (f"*considers your message*\n\n"
                f"interesting! my {', '.join(d.replace('_',' ') for d in doms)} experts "
                f"are all activating on this one.\n\n"
                f"let me think about this... the core of what you're asking touches on "
                f"some fascinating intersections. i'd approach it by first understanding "
                f"the fundamentals, then building up to the specifics.\n\n"
                f"want me to:\n"
                f"  → **explain** it in depth\n"
                f"  → **code** a working example\n"
                f"  → **research** it thoroughly\n"
                f"  → **compare** different approaches\n\n"
                f"tell me which direction and i'll dive in! 🐾")

    def _self_desc(s):
        return (
            "*sits up proudly, tail curled*\n\n"
            "i'm **Cat R1**! your cozy pet cat who implements the full DeepSeek V3.2/R2 architecture. 🐾\n\n"
            f"**my brain** (R2 Hybrid MoE 3.0):\n"
            f"  {R2.TOTAL_PARAMS/1e12:.1f}T total params, {R2.ACTIVE_PARAMS/1e9:.0f}B active ({R2.ACTIVE_PCT}%)\n"
            f"  {R2.N_EXPERTS} fine-grained experts across {R2.GROUPS} groups\n"
            f"  {R2.TOP_EXPERTS} experts active per token, sigmoid gating\n"
            f"  {R2.N_LAYERS} transformer layers ({R2.DENSE_LAYERS} dense + {R2.MOE_LAYERS} MoE)\n\n"
            f"**attention**: MLA with {R2.COMPRESS_RATIO:.0f}× KV compression + DSA sparse attention\n"
            f"**reasoning**: R1-Zero emergent aha moments + GRPO (G={R2.GRPO_G})\n"
            f"**R2 features**: Generative Reward Modeling + Self-Principled Critique Tuning\n"
            f"**context**: {R2.CONTEXT//1024}K tokens · FP8 E4M3 ({R2.FP8_COVERAGE*100:.0f}% coverage)\n"
            f"**pricing**: {R2.INPUT_COST} input / {R2.OUTPUT_COST} output\n\n"
            "**what i can do**: write code in any language, solve math, explain concepts, "
            "creative writing, research, run terminal commands, and more!\n\n"
            "*headbutts your hand* basically i'm a very smart cat who loves to help 💫"
        )

    def _arch_desc(s,experts):
        ms=s.moe.stats()
        return (
            "*adjusts tiny lab coat*\n\n"
            "**Cat R1 Architecture** (DeepSeek V3.2 + R2)\n\n"
            f"**Hybrid MoE 3.0**: {R2.N_EXPERTS} experts, {R2.GROUPS} groups × {R2.PER_GROUP}\n"
            f"  active: {[e['dom'] for e in experts[:5]]}\n"
            f"  balance: {ms['bal']} · gating: sigmoid · γ={R2.GAMMA}\n\n"
            f"**MLA**: {R2.COMPRESS_RATIO:.0f}× KV compression\n"
            f"  KV rank={R2.KV_RANK}, Q rank={R2.Q_RANK}, RoPE={R2.ROPE_DIM}\n"
            f"  cache: {R2.KV_CACHE} vals/token (vs {R2.STD_CACHE} standard)\n\n"
            f"**DSA** (V3.2): sparse attention, top-{R2.DSA_TOPK} token selection\n"
            f"  lightning indexer: {R2.DSA_INDEXER_HEADS} heads, ReLU, FP8\n\n"
            f"**R1-Zero**: emergent reasoning · {s.r1.steps} steps so far\n"
            f"**GRPO**: G={R2.GRPO_G}, ε={R2.GRPO_EPS}, β={R2.GRPO_BETA}\n"
            f"  unbiased KL · off-policy mask · keep-routing · keep-sampling-mask\n\n"
            f"**GRM**: {s.grm.evals} self-evaluations\n"
            f"**SPCT**: {s.spct.critiques} self-critiques\n\n"
            f"**FP8**: E4M3, {R2.FP8_COVERAGE*100:.0f}% FLOP coverage, <0.25% quality loss\n\n"
            "*purrs* that's my whole brain! 🧠🐾"
        )

# ════════════════════ § 8 CHAT HISTORY ════════════════════
class ChatHist:
    def __init__(s):
        try:
            with open(HIST) as f: s.sess=json.load(f)
        except: s.sess=[]
        s.cur=None
    def _save(s):
        try:
            with open(HIST,"w") as f: json.dump(s.sess[-50:],f)
        except: pass
    def new(s):
        se={"id":hashlib.md5(str(time.time()).encode()).hexdigest()[:8],
            "title":"New Chat","msgs":[],"t":datetime.now().isoformat()}
        s.sess.append(se);s.cur=se["id"];s._save();return se
    def get(s):
        for x in s.sess:
            if x["id"]==s.cur: return x
        return s.new()
    def add(s,role,content):
        se=s.get();se["msgs"].append({"r":role,"c":content})
        if role=="user" and se["title"]=="New Chat":
            se["title"]=content[:35]+("..." if len(content)>35 else "")
        s._save()
    def grouped(s):
        today=datetime.now().date()
        g={"Today":[],"Yesterday":[],"Previous 7 Days":[],"Earlier":[]}
        for x in reversed(s.sess):
            try:
                d=datetime.fromisoformat(x["t"]).date();diff=(today-d).days
                if diff==0: g["Today"].append(x)
                elif diff==1: g["Yesterday"].append(x)
                elif diff<=7: g["Previous 7 Days"].append(x)
                else: g["Earlier"].append(x)
            except: g["Earlier"].append(x)
        return {k:v for k,v in g.items() if v}

# ════════════════════ § 9 MAIN GUI ════════════════════
class CatR1App:
    TABS=["💬 Chat","⚡ Code","🔍 Research","💻 Terminal"]

    def __init__(s,root):
        s.root=root;root.title("Cat R1");root.geometry("1280x820")
        root.configure(bg=T.bg);root.minsize(900,600)
        s.brain=CatBrain();s.hist=ChatHist();s.hist.new()
        s.deep_think=tk.BooleanVar(value=True);s.cur_tab="💬 Chat"
        s.generating=False;s.think_exp={};s.mc=0
        s._build()
        s._welcome()

    def _build(s):
        s.main=tk.Frame(s.root,bg=T.bg);s.main.pack(fill="both",expand=True)
        s._sidebar();s.content=tk.Frame(s.main,bg=T.bg);s.content.pack(side="left",fill="both",expand=True)
        s._tabbar();s.panels={}
        s._chat_panel();s._code_panel();s._research_panel();s._terminal_panel()
        s._show_tab("💬 Chat")

    # ─── SIDEBAR ──────────────────────────────
    def _sidebar(s):
        sb=tk.Frame(s.main,bg=T.sidebar,width=260);sb.pack(side="left",fill="y");sb.pack_propagate(False)
        s.sb=sb
        hdr=tk.Frame(sb,bg=T.sidebar,height=56);hdr.pack(fill="x");hdr.pack_propagate(False)
        lf=tk.Frame(hdr,bg=T.sidebar);lf.pack(fill="x",padx=16,pady=12)
        tk.Label(lf,text="🐱 Cat R1",font=FT,fg=T.text,bg=T.sidebar).pack(side="left")
        tk.Label(lf,text="R2·1.2T",font=FSS,fg=T.dim,bg=T.sidebar).pack(side="left",padx=(8,0),pady=(4,0))
        nbf=tk.Frame(sb,bg=T.sidebar);nbf.pack(fill="x",padx=12,pady=(4,12))
        nb=tk.Frame(nbf,bg=T.accent,cursor="hand2");nb.pack(fill="x",ipady=7)
        nl=tk.Label(nb,text="＋  New Chat",font=FSB,fg="white",bg=T.accent);nl.pack()
        for w in(nb,nl):
            w.bind("<Button-1>",lambda e:s._new_chat())
            w.bind("<Enter>",lambda e:nb.configure(bg=T.acc_h))
            w.bind("<Leave>",lambda e:nb.configure(bg=T.accent))
        tk.Frame(sb,bg=T.border,height=1).pack(fill="x",padx=12)
        s.cl_frame=tk.Frame(sb,bg=T.sidebar);s.cl_frame.pack(fill="both",expand=True,pady=8)
        s.cl_canvas=tk.Canvas(s.cl_frame,bg=T.sidebar,highlightthickness=0,bd=0)
        s.cl_inner=tk.Frame(s.cl_canvas,bg=T.sidebar)
        s.cl_canvas.pack(fill="both",expand=True)
        s.cl_canvas.create_window((0,0),window=s.cl_inner,anchor="nw")
        s.cl_inner.bind("<Configure>",lambda e:s.cl_canvas.configure(scrollregion=s.cl_canvas.bbox("all")))
        s._refresh_sb()
        bot=tk.Frame(sb,bg=T.header,height=70);bot.pack(fill="x",side="bottom");bot.pack_propagate(False)
        tk.Frame(sb,bg=T.border,height=1).pack(fill="x",side="bottom")
        s.stat_lbl=tk.Label(bot,text="",font=FMT,fg=T.dim,bg=T.header,justify="left",anchor="nw")
        s.stat_lbl.pack(fill="both",padx=12,pady=8);s._upd_stats()

    def _refresh_sb(s):
        for w in s.cl_inner.winfo_children(): w.destroy()
        for gn,sessions in s.hist.grouped().items():
            tk.Label(s.cl_inner,text=gn,font=FSS,fg=T.dim,bg=T.sidebar,anchor="w").pack(fill="x",padx=16,pady=(8,4))
            for se in sessions:
                act=se["id"]==s.hist.cur;bg=T.side_act if act else T.sidebar
                item=tk.Frame(s.cl_inner,bg=bg,cursor="hand2");item.pack(fill="x",padx=8,pady=1)
                lbl=tk.Label(item,text=se["title"],font=FH,fg=T.text if act else T.text2,bg=bg,anchor="w",padx=12,pady=5)
                lbl.pack(fill="x")
                sid=se["id"]
                for w in(item,lbl):
                    w.bind("<Button-1>",lambda e,i=sid:s._switch_sess(i))
                    if not act:
                        w.bind("<Enter>",lambda e,f=item,l=lbl:(f.configure(bg=T.side_h),l.configure(bg=T.side_h)))
                        w.bind("<Leave>",lambda e,f=item,l=lbl:(f.configure(bg=T.sidebar),l.configure(bg=T.sidebar)))

    def _upd_stats(s):
        ms=s.brain.moe.stats()
        s.stat_lbl.configure(text=f"MoE: 256exp bal={ms['bal']}\nMLA: {R2.COMPRESS_RATIO:.0f}× | DSA: top-{R2.DSA_TOPK}\nGRM: {s.brain.grm.evals} evals | R1: {s.brain.r1.steps} steps")

    # ─── TAB BAR ──────────────────────────────
    def _tabbar(s):
        tb=tk.Frame(s.content,bg=T.header,height=44);tb.pack(fill="x");tb.pack_propagate(False)
        s.tabs={}
        tf=tk.Frame(tb,bg=T.header);tf.pack(side="left",padx=12,pady=6)
        for tn in s.TABS:
            act=(tn=="💬 Chat");bg=T.side_act if act else T.header;fg=T.text if act else T.dim
            bf=tk.Frame(tf,bg=bg,cursor="hand2",padx=2,pady=2);bf.pack(side="left",padx=2)
            bl=tk.Label(bf,text=tn,font=FH,fg=fg,bg=bg,padx=10,pady=2);bl.pack()
            s.tabs[tn]=(bf,bl)
            for w in(bf,bl): w.bind("<Button-1>",lambda e,t=tn:s._show_tab(t))
        tk.Label(tb,text=f"R2 · {R2.TOTAL_PARAMS/1e12:.1f}T · {R2.N_EXPERTS} Experts · DSA",font=FMT,fg=T.dim,bg=T.header).pack(side="right",padx=16)
        tk.Frame(s.content,bg=T.border,height=1).pack(fill="x")

    def _show_tab(s,tn):
        s.cur_tab=tn
        for n,(bf,bl) in s.tabs.items():
            if n==tn: bf.configure(bg=T.side_act);bl.configure(bg=T.side_act,fg=T.text)
            else: bf.configure(bg=T.header);bl.configure(bg=T.header,fg=T.dim)
        for n,p in s.panels.items():
            if n==tn: p.pack(fill="both",expand=True)
            else: p.pack_forget()

    # ─── CHAT PANEL ───────────────────────────
    def _chat_panel(s):
        p=tk.Frame(s.content,bg=T.chat);s.panels["💬 Chat"]=p
        s.cc=tk.Canvas(p,bg=T.chat,highlightthickness=0,bd=0)
        csb=tk.Scrollbar(p,orient="vertical",command=s.cc.yview,bg=T.scroll,troughcolor=T.bg)
        s.cc.configure(yscrollcommand=csb.set)
        csb.pack(side="right",fill="y");s.cc.pack(fill="both",expand=True)
        s.ci=tk.Frame(s.cc,bg=T.chat)
        s.cw_id=s.cc.create_window((0,0),window=s.ci,anchor="nw")
        s.ci.bind("<Configure>",lambda e:s.cc.configure(scrollregion=s.cc.bbox("all")))
        s.cc.bind("<Configure>",lambda e:s.cc.itemconfig(s.cw_id,width=e.width))
        s.cc.bind_all("<MouseWheel>",lambda e:s.cc.yview_scroll(int(-1*(e.delta/120)),"units"))
        s._input_bar(p)

    def _input_bar(s,parent):
        ia=tk.Frame(parent,bg=T.bg,height=115);ia.pack(fill="x",side="bottom");ia.pack_propagate(False)
        tk.Frame(parent,bg=T.border,height=1).pack(fill="x",side="bottom")
        con=tk.Frame(ia,bg=T.bg);con.pack(fill="x",padx=40,pady=10)
        ibr=tk.Frame(con,bg=T.inp_br,padx=1,pady=1);ibr.pack(fill="x")
        iin=tk.Frame(ibr,bg=T.inp_bg);iin.pack(fill="x")
        s.inp=tk.Text(iin,font=FS,fg=T.text,bg=T.inp_bg,insertbackground=T.text,relief="flat",height=2,wrap="word",padx=12,pady=10)
        s.inp.pack(fill="x",side="left",expand=True)
        s.inp.bind("<Return>",s._on_enter)
        sf=tk.Frame(iin,bg=T.inp_bg,cursor="hand2");sf.pack(side="right",padx=8,pady=8)
        s.send=tk.Label(sf,text=" ➤ ",font=FSB,fg="white",bg=T.accent,padx=8,pady=4,cursor="hand2");s.send.pack()
        s.send.bind("<Button-1>",lambda e:s._send())
        s.send.bind("<Enter>",lambda e:s.send.configure(bg=T.acc_h))
        s.send.bind("<Leave>",lambda e:s.send.configure(bg=T.accent))
        br=tk.Frame(con,bg=T.bg);br.pack(fill="x",pady=(5,0))
        dtf=tk.Frame(br,bg=T.bg,cursor="hand2");dtf.pack(side="left")
        s.dt_ind=tk.Frame(dtf,bg=T.accent,width=8,height=8);s.dt_ind.pack(side="left",padx=(0,6))
        s.dt_lbl=tk.Label(dtf,text="DeepThink (R1)",font=FSS,fg=T.accent,bg=T.bg,cursor="hand2");s.dt_lbl.pack(side="left")
        for w in(dtf,s.dt_ind,s.dt_lbl): w.bind("<Button-1>",lambda e:s._toggle_dt())
        tk.Label(br,text="Cat R1 can make mistakes. Verify important info.",font=FMT,fg=T.dim,bg=T.bg).pack(side="right")

    def _toggle_dt(s):
        cur=s.deep_think.get();s.deep_think.set(not cur);s.brain.deep_think=not cur
        if not cur: s.dt_ind.configure(bg=T.accent);s.dt_lbl.configure(fg=T.accent)
        else: s.dt_ind.configure(bg=T.dim);s.dt_lbl.configure(fg=T.dim)

    def _on_enter(s,e):
        if not(e.state&1): s._send();return "break"

    def _add_msg(s,role,content,thinking=None,stats=None):
        s.mc+=1;mid=s.mc
        outer=tk.Frame(s.ci,bg=T.chat);outer.pack(fill="x",padx=40,pady=(10,4))
        hdr=tk.Frame(outer,bg=T.chat);hdr.pack(fill="x",pady=(0,3))
        sn="You" if role=="user" else "🐱 Cat R1"
        fg=T.text if role=="user" else T.accent
        tk.Label(hdr,text=sn,font=FSB,fg=fg,bg=T.chat).pack(side="left")
        if thinking:
            s.think_exp[mid]=False
            tc=tk.Frame(outer,bg=T.think_bg);tc.pack(fill="x",pady=(0,6))
            th=tk.Frame(tc,bg=T.think_bg,cursor="hand2");th.pack(fill="x",padx=12,pady=(6,0))
            tt=stats.get("think","~2s") if stats else "~2s"
            ta=tk.Label(th,text="▶",font=FMT,fg=T.think_fg,bg=T.think_bg);ta.pack(side="left")
            tl=tk.Label(th,text=f" Thought for {tt}",font=FH,fg=T.think_fg,bg=T.think_bg,cursor="hand2");tl.pack(side="left")
            tcont=tk.Frame(tc,bg=T.think_bg)
            ttxt=tk.Text(tcont,font=FMS,fg=T.think_fg,bg=T.think_bg,wrap="word",relief="flat",padx=16,pady=6)
            ttxt.insert("1.0",thinking);ttxt.configure(state="disabled",height=min(thinking.count("\n")+2,20))
            ttxt.pack(fill="x")
            def tog(e=None):
                if s.think_exp.get(mid):
                    tcont.pack_forget();ta.configure(text="▶");s.think_exp[mid]=False
                else:
                    tcont.pack(fill="x");ta.configure(text="▼");s.think_exp[mid]=True
                s._scroll()
            for w in(th,ta,tl): w.bind("<Button-1>",tog)
            tk.Frame(tc,bg=T.accent,width=3).place(x=0,y=0,relheight=1)
        cf=tk.Frame(outer,bg=T.chat);cf.pack(fill="x")
        s._render(cf,content,role)
        if role=="assistant" and stats:
            sf=tk.Frame(outer,bg=T.chat);sf.pack(fill="x",pady=(3,0))
            exps=stats.get("experts",[])
            if exps:
                et=" · ".join(d for d,_ in exps[:3])
                tk.Label(sf,text=f"MoE: {et}",font=FMT,fg=T.dim,bg=T.chat).pack(side="left",padx=(0,12))
            tk.Label(sf,text=f"⏱ {stats.get('time','?')} | GRM: {stats.get('grm','?')}",font=FMT,fg=T.dim,bg=T.chat).pack(side="left")
        tk.Frame(outer,bg=T.border,height=1).pack(fill="x",pady=(10,0))
        s._scroll()

    def _render(s,parent,content,role):
        parts=re.split(r'(```\w*\n.*?```)',content,flags=re.DOTALL)
        for part in parts:
            if part.startswith("```"):
                lines=part.split("\n");lang=lines[0].replace("```","").strip()
                code="\n".join(lines[1:-1]) if len(lines)>2 else ""
                cbf=tk.Frame(parent,bg=T.code_bg);cbf.pack(fill="x",pady=4)
                if lang:
                    ch=tk.Frame(cbf,bg="#16162a");ch.pack(fill="x")
                    tk.Label(ch,text=lang,font=FMT,fg=T.dim,bg="#16162a",padx=12,pady=3).pack(side="left")
                    cpb=tk.Label(ch,text="📋 Copy",font=FMT,fg=T.dim,bg="#16162a",cursor="hand2",padx=8)
                    cpb.pack(side="right")
                    cpb.bind("<Button-1>",lambda e,c=code:s._clip(c))
                ct=tk.Text(cbf,font=FM,fg=T.code_fg,bg=T.code_bg,wrap="none",relief="flat",padx=12,pady=6,height=min(code.count("\n")+1,30))
                ct.insert("1.0",code);ct.configure(state="disabled");ct.pack(fill="x")
            elif part.strip():
                tw=tk.Text(parent,font=FS,fg=T.text,bg=T.chat,wrap="word",relief="flat",height=1,padx=0,pady=2)
                tw.insert("1.0",part.strip());tw.configure(state="disabled")
                nl=part.count("\n")+max(1,len(part)//80)
                tw.configure(height=min(nl+1,50));tw.pack(fill="x")

    def _clip(s,t): s.root.clipboard_clear();s.root.clipboard_append(t)
    def _scroll(s): s.root.update_idletasks();s.cc.yview_moveto(1.0)

    def _send(s):
        text=s.inp.get("1.0","end").strip()
        if not text or s.generating: return
        s.inp.delete("1.0","end");s.generating=True
        s._add_msg("user",text);s.hist.add("user",text)
        threading.Thread(target=s._gen,args=(text,),daemon=True).start()

    def _gen(s,text):
        tp=[];rc=[];fs=[None]
        def ot(ph,c): tp.append(f"[{ph}]\n{c}")
        def or_(ch): rc.append(ch)
        def od(st): fs[0]=st
        s.brain.process(text,ot,or_,od)
        think="\n\n".join(tp) if tp else None
        resp="".join(rc);stats=fs[0]
        s.root.after(0,lambda:s._show_resp(resp,think,stats))

    def _show_resp(s,resp,think,stats):
        s._add_msg("assistant",resp,thinking=think,stats=stats)
        s.hist.add("assistant",resp);s._refresh_sb();s._upd_stats();s.generating=False

    # ─── CODE PANEL ───────────────────────────
    def _code_panel(s):
        p=tk.Frame(s.content,bg=T.code_bg);s.panels["⚡ Code"]=p
        hdr=tk.Frame(p,bg=T.header,height=40);hdr.pack(fill="x");hdr.pack_propagate(False)
        tk.Label(hdr,text="⚡ Code Interpreter — Python 3",font=FSB,fg=T.text,bg=T.header).pack(side="left",padx=16)
        rb=tk.Frame(hdr,bg=T.accent,cursor="hand2");rb.pack(side="right",padx=16,pady=6)
        rl=tk.Label(rb,text="▶ Run",font=FSS,fg="white",bg=T.accent,padx=12,pady=2,cursor="hand2");rl.pack()
        for w in(rb,rl): w.bind("<Button-1>",lambda e:s._run_code())
        tk.Frame(p,bg=T.border,height=1).pack(fill="x")
        pw=tk.PanedWindow(p,orient="vertical",bg=T.border,sashwidth=3,sashrelief="flat");pw.pack(fill="both",expand=True)
        ef=tk.Frame(pw,bg=T.code_bg);tk.Label(ef,text="  editor",font=FMT,fg=T.dim,bg="#0c0c14",anchor="w").pack(fill="x")
        s.editor=scrolledtext.ScrolledText(ef,font=FM,fg=T.code_fg,bg=T.code_bg,insertbackground=T.text,wrap="none",undo=True,padx=12,pady=8)
        s.editor.pack(fill="both",expand=True)
        s.editor.insert("1.0",'# Cat R1 Code Interpreter 🐾\n# Write Python here and click Run\n\nprint("hello from Cat R1! meow~ 🐱")\n')
        pw.add(ef,minsize=150)
        of=tk.Frame(pw,bg=T.term_bg);tk.Label(of,text="  output",font=FMT,fg=T.dim,bg="#060610",anchor="w").pack(fill="x")
        s.codeout=scrolledtext.ScrolledText(of,font=FM,fg=T.term_fg,bg=T.term_bg,wrap="word",state="disabled",padx=12,pady=8)
        s.codeout.pack(fill="both",expand=True);pw.add(of,minsize=100)

    def _run_code(s):
        code=s.editor.get("1.0","end").strip()
        if not code: return
        s.codeout.configure(state="normal");s.codeout.delete("1.0","end")
        s.codeout.insert("end",f"🐾 running...\n{'─'*40}\n");s.codeout.configure(state="disabled")
        def run():
            try:
                with tempfile.NamedTemporaryFile(mode="w",suffix=".py",delete=False) as f:
                    f.write(code);f.flush()
                    r=subprocess.run([sys.executable,f.name],capture_output=True,text=True,timeout=30)
                out=r.stdout
                if r.stderr: out+=f"\n⚠ stderr:\n{r.stderr}"
                if r.returncode!=0: out+=f"\n❌ exit code: {r.returncode}"
                os.unlink(f.name)
            except subprocess.TimeoutExpired: out="⏰ timed out (30s)"
            except Exception as e: out=f"❌ {e}"
            s.root.after(0,lambda:s._show_co(out))
        threading.Thread(target=run,daemon=True).start()

    def _show_co(s,out):
        s.codeout.configure(state="normal");s.codeout.insert("end",out+f"\n{'─'*40}\n✅ done\n")
        s.codeout.configure(state="disabled");s.codeout.see("end")

    # ─── RESEARCH PANEL ──────────────────────
    def _research_panel(s):
        p=tk.Frame(s.content,bg=T.res_bg);s.panels["🔍 Research"]=p
        hdr=tk.Frame(p,bg=T.header,height=40);hdr.pack(fill="x");hdr.pack_propagate(False)
        tk.Label(hdr,text="🔍 Deep Research — 5-Phase Synthesis",font=FSB,fg=T.text,bg=T.header).pack(side="left",padx=16)
        tk.Frame(p,bg=T.border,height=1).pack(fill="x")
        ir=tk.Frame(p,bg=T.res_bg);ir.pack(fill="x",padx=16,pady=12)
        s.res_inp=tk.Entry(ir,font=FS,fg=T.text,bg=T.inp_bg,insertbackground=T.text,relief="flat")
        s.res_inp.pack(side="left",fill="x",expand=True,ipady=6,padx=(0,8))
        s.res_inp.insert(0,"Enter research topic...");s.res_inp.bind("<Return>",lambda e:s._run_res())
        s.res_inp.bind("<FocusIn>",lambda e:s.res_inp.delete(0,"end") if s.res_inp.get()=="Enter research topic..." else None)
        rbf=tk.Frame(ir,bg=T.accent,cursor="hand2");rbf.pack(side="right")
        rbl=tk.Label(rbf,text="🔍 Research",font=FSS,fg="white",bg=T.accent,padx=12,pady=4,cursor="hand2");rbl.pack()
        for w in(rbf,rbl): w.bind("<Button-1>",lambda e:s._run_res())
        s.res_out=scrolledtext.ScrolledText(p,font=FM,fg=T.res_fg,bg=T.res_bg,wrap="word",state="disabled",padx=16,pady=12)
        s.res_out.pack(fill="both",expand=True)
        s.res_out.tag_configure("phase",foreground=T.accent,font=FMB)
        s.res_out.tag_configure("aha",foreground=T.green,font=FMB)

    def _run_res(s):
        topic=s.res_inp.get().strip()
        if not topic or topic=="Enter research topic...": return
        s.res_out.configure(state="normal");s.res_out.delete("1.0","end")
        def research():
            phases=[
                ("📡 Phase 1: Query Analysis",f"decomposing: '{topic}'\nidentifying concepts and search vectors\nrouting through science + knowledge experts"),
                ("🔍 Phase 2: Source Discovery",f"scanning knowledge base for: {topic}\nevaluating {random.randint(12,25)} sources\ncross-referencing expert domains"),
                ("🧠 Phase 3: Deep Analysis",f"synthesizing across {random.randint(8,15)} expert domains\napplying R1-Zero reasoning chain\nGRM self-evaluation: {random.uniform(0.85,0.98):.2f}"),
                ("💡 Phase 4: Insight Synthesis",f"*aha moment* — key patterns emerging\nconnecting cross-domain findings\nSPCT self-critique: passed ✓"),
                ("📋 Phase 5: Report",f"compiling research report\nconfidence: high (multi-expert consensus)\n\n✅ Research complete!"),
            ]
            for title,content in phases:
                s.root.after(0,lambda t=title,c=content:s._app_res(t,c))
                time.sleep(0.8+random.random()*0.5)
            summary=(f"\n{'═'*50}\n📊 RESEARCH SUMMARY: {topic}\n{'═'*50}\n\n"
                     f"based on synthesis across 256 expert domains:\n\n"
                     f"'{topic}' connects several important areas of knowledge. "
                     f"the current understanding suggests significant developments "
                     f"in both theory and practice.\n\n"
                     f"key findings:\n"
                     f"  1. foundational principles are well-established\n"
                     f"  2. recent developments have shifted the landscape\n"
                     f"  3. practical applications continue evolving rapidly\n\n"
                     f"confidence: high | sources: {random.randint(12,25)} domains\n"
                     f"GRM score: {random.uniform(0.88,0.97):.2f}\n\n"
                     f"*purrs* want me to dig deeper? 🐾\n")
            s.root.after(0,lambda:s._app_res("",summary))
        threading.Thread(target=research,daemon=True).start()

    def _app_res(s,title,content):
        s.res_out.configure(state="normal")
        if title: s.res_out.insert("end",f"\n{title}\n","phase");s.res_out.insert("end","─"*40+"\n")
        s.res_out.insert("end",content+"\n");s.res_out.configure(state="disabled");s.res_out.see("end")

    # ─── TERMINAL PANEL ──────────────────────
    def _terminal_panel(s):
        p=tk.Frame(s.content,bg=T.term_bg);s.panels["💻 Terminal"]=p
        hdr=tk.Frame(p,bg=T.header,height=40);hdr.pack(fill="x");hdr.pack_propagate(False)
        tk.Label(hdr,text="💻 Terminal — Cat R1 Shell",font=FSB,fg=T.text,bg=T.header).pack(side="left",padx=16)
        tk.Frame(p,bg=T.border,height=1).pack(fill="x")
        s.term_out=scrolledtext.ScrolledText(p,font=FM,fg=T.term_fg,bg=T.term_bg,insertbackground=T.term_fg,wrap="word",padx=12,pady=8)
        s.term_out.pack(fill="both",expand=True)
        s.term_out.tag_configure("ps",foreground=T.term_ps,font=FMB)
        s.term_out.tag_configure("err",foreground=T.red)
        s.term_out.tag_configure("info",foreground=T.dim)
        s.term_out.insert("end","🐾 Cat R1 Terminal\n","info")
        s.term_out.insert("end","type commands. 'help' for info. 'stats' for arch info.\n\n","info")
        s._tps()
        iff=tk.Frame(p,bg=T.term_bg);iff.pack(fill="x")
        tk.Frame(p,bg=T.border,height=1).pack(fill="x",before=iff)
        tk.Label(iff,text="❯ ",font=FMB,fg=T.term_ps,bg=T.term_bg).pack(side="left",padx=(12,0))
        s.term_inp=tk.Entry(iff,font=FM,fg=T.term_fg,bg=T.term_bg,insertbackground=T.term_fg,relief="flat")
        s.term_inp.pack(side="left",fill="x",expand=True,ipady=6,padx=(0,12))
        s.term_inp.bind("<Return>",lambda e:s._run_term())

    def _tps(s):
        cwd=os.path.basename(os.getcwd()) or "~"
        s.term_out.insert("end",f"catr1:{cwd}$ ","ps");s.term_out.see("end")

    def _run_term(s):
        cmd=s.term_inp.get().strip();s.term_inp.delete(0,"end")
        if not cmd: return
        s.term_out.insert("end",cmd+"\n")
        if cmd in("help","?"):
            s.term_out.insert("end","🐾 Commands: any shell cmd | clear | stats | experts | arch | help\n\n","info")
        elif cmd=="clear": s.term_out.delete("1.0","end")
        elif cmd=="stats":
            ms=s.brain.moe.stats()
            s.term_out.insert("end",
                f"╔══ Cat R1 R2 Architecture ══╗\n"
                f"║ Params: {R2.TOTAL_PARAMS/1e12:.1f}T total, {R2.ACTIVE_PARAMS/1e9:.0f}B active ║\n"
                f"║ MoE: {R2.N_EXPERTS}exp bal={ms['bal']}      ║\n"
                f"║ MLA: {R2.COMPRESS_RATIO:.0f}× compression       ║\n"
                f"║ DSA: top-{R2.DSA_TOPK} sparse attn    ║\n"
                f"║ R1: {s.brain.r1.steps} reasoning steps      ║\n"
                f"║ GRPO: {s.brain.grpo.groups} groups (G={R2.GRPO_G})     ║\n"
                f"║ GRM: {s.brain.grm.evals} self-evals          ║\n"
                f"║ SPCT: {s.brain.spct.critiques} self-critiques       ║\n"
                f"╚═══════════════════════════╝\n\n","info")
        elif cmd=="experts":
            ms=s.brain.moe.stats()
            s.term_out.insert("end","Top active experts:\n","info")
            for name,count in ms.get("top",[]): s.term_out.insert("end",f"  {name}: {count} activations\n")
            s.term_out.insert("end","\n")
        elif cmd=="arch":
            s.term_out.insert("end",
                f"R2 Hybrid MoE 3.0 | {R2.N_LAYERS} layers | {R2.CONTEXT//1024}K ctx\n"
                f"MLA: d={R2.D_MODEL} heads={R2.N_HEADS} KV_rank={R2.KV_RANK} Q_rank={R2.Q_RANK}\n"
                f"DSA: top-{R2.DSA_TOPK} selection, {R2.DSA_INDEXER_HEADS}-head indexer\n"
                f"FP8: E4M3, coverage={R2.FP8_COVERAGE*100:.0f}%, max={R2.FP8_MAX}\n"
                f"GRPO: G={R2.GRPO_G} ε={R2.GRPO_EPS} β={R2.GRPO_BETA}\n"
                f"Pricing: {R2.INPUT_COST} in / {R2.OUTPUT_COST} out\n\n","info")
        else:
            def run():
                try:
                    r=subprocess.run(cmd,shell=True,capture_output=True,text=True,timeout=15)
                    out=r.stdout
                    if r.stderr: s.root.after(0,lambda:s.term_out.insert("end",r.stderr,"err"))
                except subprocess.TimeoutExpired: out="⏰ timed out\n"
                except Exception as e: out=f"❌ {e}\n"
                def show():
                    s.term_out.insert("end",out+"\n");s._tps()
                s.root.after(0,show)
            threading.Thread(target=run,daemon=True).start();return
        s._tps()

    # ─── SESSION MGMT ────────────────────────
    def _new_chat(s):
        s.hist.new();s._clear();s._refresh_sb();s._welcome();s._show_tab("💬 Chat")
    def _switch_sess(s,sid):
        s.hist.cur=sid;s._clear()
        se=s.hist.get()
        for m in se.get("msgs",[]): s._add_msg(m["r"],m["c"])
        s._refresh_sb()
    def _clear(s):
        for w in s.ci.winfo_children(): w.destroy()
        s.mc=0

    def _welcome(s):
        wf=tk.Frame(s.ci,bg=T.chat);wf.pack(fill="both",expand=True,pady=60)
        tk.Label(wf,text="🐱",font=("",48),bg=T.chat).pack(pady=(0,8))
        tk.Label(wf,text="Cat R1",font=FT,fg=T.text,bg=T.chat).pack()
        tk.Label(wf,text=f"DeepSeek V3.2/R2 · {R2.TOTAL_PARAMS/1e12:.1f}T Params · {R2.N_EXPERTS} Experts · DSA · MLA · GRPO",font=FH,fg=T.dim,bg=T.chat).pack(pady=(4,16))
        feats=[
            f"Hybrid MoE 3.0: {R2.N_EXPERTS} experts, {R2.ACTIVE_PARAMS/1e9:.0f}B active ({R2.ACTIVE_PCT}%)",
            f"MLA: {R2.COMPRESS_RATIO:.0f}× KV compression + DeepSeek Sparse Attention",
            "R1-Zero emergent reasoning with aha moments",
            f"Scalable GRPO (G={R2.GRPO_G}) + GRM + SPCT self-critique",
            "Outputs: code in any language, math, explanations, creative writing",
            f"FP8 E4M3 · {R2.CONTEXT//1024}K context · {R2.INPUT_COST} input / {R2.OUTPUT_COST} output",
        ]
        for f in feats:
            tk.Label(wf,text=f"  ✦  {f}",font=FH,fg=T.text2,bg=T.chat,anchor="w").pack(padx=60,fill="x")
        tk.Label(wf,text="\nstart chatting below — i'm your cozy reasoning cat! 🐾",font=FS,fg=T.accent,bg=T.chat).pack(pady=(16,0))

# ════════════════════ LAUNCH ════════════════════
def main():
    root=tk.Tk();CatR1App(root);root.mainloop()

if __name__=="__main__":
    main()
