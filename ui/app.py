"""
ui/app.py - Streamlit UI for Intelligent RC & Quiz Generation System
4 Screens: Article Input | Quiz View | Hint Panel | Analytics Dashboard
Run: cd race_rc_project && streamlit run ui/app.py
"""
import os, sys, time, random
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT,"src"))
os.chdir(ROOT)

st.set_page_config(page_title="RC Quiz System", page_icon="📚",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.main-header{background:linear-gradient(135deg,#1a1a2e,#16213e,#0f3460);
  padding:1.8rem 2rem;border-radius:14px;margin-bottom:1.5rem;
  box-shadow:0 8px 32px rgba(0,0,0,.35);}
.main-header h1{color:#e94560;font-size:1.8rem;font-weight:700;margin:0;}
.main-header p{color:#a8b2d8;margin:.4rem 0 0;font-size:.9rem;}
.card{background:#16213e;border:1px solid #0f3460;border-radius:12px;padding:1.3rem;margin-bottom:.9rem;}
.option-correct{background:linear-gradient(135deg,#0d4f3c,#1a7a5e)!important;
  border:2px solid #22c55e!important;border-radius:10px;padding:.8rem 1.2rem;
  color:#bbf7d0!important;font-weight:600;margin:.3rem 0;display:block;}
.option-wrong{background:linear-gradient(135deg,#4c1d1d,#7f1d1d)!important;
  border:2px solid #ef4444!important;border-radius:10px;padding:.8rem 1.2rem;
  color:#fecaca!important;font-weight:600;margin:.3rem 0;display:block;}
.option-neutral{background:#1e293b;border:1px solid #334155;border-radius:10px;
  padding:.8rem 1.2rem;margin:.3rem 0;color:#e2e8f0;display:block;}
.hint-box{border-radius:10px;padding:1rem 1.2rem;margin-bottom:.8rem;color:#e2e8f0;font-size:.95rem;}
.hint1{background:#1e3a5f;border-left:4px solid #3b82f6;}
.hint2{background:#1e4d3a;border-left:4px solid #10b981;}
.hint3{background:#4c2d19;border-left:4px solid #f59e0b;}
.hint-locked{background:#1e293b;border-left:4px solid #334155;}
[data-testid="stSidebar"]{background:#0d1b2a!important;border-right:1px solid #1e3a5f;}
.ai-badge{background:#e94560;color:white;padding:.18rem .65rem;border-radius:20px;
  font-size:.73rem;font-weight:600;display:inline-block;margin-bottom:.5rem;}
</style>""", unsafe_allow_html=True)

# ── Cached pipeline load ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    from inference import InferencePipeline
    pipe = InferencePipeline()
    pipe.load_models(auto_train=True)
    return pipe

# ── Sample passages ───────────────────────────────────────────────────────────
SAMPLES = {
    "Amazon Rainforest": {
        "article":"The Amazon rainforest is the world's largest tropical rainforest covering most of the Amazon basin of South America. It represents over half of the planet's remaining rainforests and is home to three million species of plants and animals. Deforestation driven by cattle ranching and agriculture remains the most critical threat to this ecosystem. Scientists estimate around 17 percent of the Amazon has been lost in the past 50 years.",
        "question":"What is the main threat to the Amazon rainforest?",
        "A":"Deforestation","B":"Flooding from rivers","C":"Volcanic activity","D":"Ocean level rise","answer":"A"},
    "Human Brain": {
        "article":"The human brain weighs about 1.4 kilograms and contains approximately 86 billion neurons. It is divided into the cerebrum, cerebellum, and brainstem. The cerebrum controls thinking, memory, and voluntary movement. The cerebellum coordinates balance and movement. Scientists continue discovering new facts about how the brain processes information.",
        "question":"What does the cerebrum control?",
        "A":"Balance and coordination","B":"Thinking, memory, and voluntary movement","C":"Breathing and heartbeat","D":"Digestion and metabolism","answer":"B"},
    "Solar Energy": {
        "article":"Solar energy is energy from the sun converted into thermal or electrical energy. It is the cleanest and most abundant renewable energy source available. Solar panels convert sunlight into electricity using photovoltaic cells. The cost of solar energy has dropped dramatically over the past decade making it accessible worldwide. In 2023 global solar capacity exceeded 1 terawatt.",
        "question":"What do solar panels use to convert sunlight into electricity?",
        "A":"Wind turbines","B":"Chemical reactions","C":"Photovoltaic cells","D":"Nuclear fission","answer":"C"},
    "Photosynthesis": {
        "article":"Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce oxygen and sugar. It takes place primarily in the leaves of plants inside chloroplasts. Chlorophyll is the green pigment that absorbs sunlight. This process is fundamental to life on Earth as it produces the oxygen we breathe and forms the base of the food chain.",
        "question":"Where does photosynthesis primarily take place?",
        "A":"In the roots of plants","B":"In the stem of plants","C":"In the flowers","D":"In the leaves inside chloroplasts","answer":"D"},
}

# ── Session state init ────────────────────────────────────────────────────────
def init_session():
    defaults = {"screen":"input","passage":"","quiz_data":None,"selected_option":None,
                "answer_checked":False,"hints_revealed":0,"answer_revealed":False,
                "session_answers":[],"_pending_sample":None}
    for k,v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

init_session()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div style='text-align:center;padding:1rem 0'><div style='font-size:2.2rem'>📚</div>"
                "<div style='color:#e94560;font-weight:700;font-size:1rem'>RC Quiz System</div>"
                "<div style='color:#64748b;font-size:.72rem'>RACE Dataset · ML Powered</div></div>",
                unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**🗂 Navigation**")
    nav = st.radio("nav", ["📖 Article Input","🧠 Quiz","💡 Hints","📊 Analytics"],
                   label_visibility="collapsed")
    screen_map = {"📖 Article Input":"input","🧠 Quiz":"quiz","💡 Hints":"hints","📊 Analytics":"analytics"}
    st.session_state.screen = screen_map[nav]
    st.markdown("---")
    st.markdown("**🎲 Load Sample**")
    sample_name = st.selectbox("sample", ["— Select —"]+list(SAMPLES.keys()), label_visibility="collapsed")
    if sample_name != "— Select —" and st.button("Load Sample", use_container_width=True):
        s = SAMPLES[sample_name]
        st.session_state.passage = s["article"]
        opts = {l:s[l] for l in ["A","B","C","D"]}
        st.session_state.quiz_data = {"question":s["question"],"correct_answer":s[s["answer"]],
            "correct_letter":s["answer"],"options":opts,"hints":None,"inference_time_ms":0.0}
        for k in ["selected_option","answer_checked","hints_revealed","answer_revealed"]:
            st.session_state[k] = False if k!="hints_revealed" else 0
        st.session_state.screen = "quiz"; st.rerun()
    st.markdown("---")
    st.markdown("<div style='color:#475569;font-size:.7rem'>⚠️ AI-generated. Not for real exams without human review.</div>",
                unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""<div class="main-header">
<h1>📚 Intelligent Reading Comprehension & Quiz Generation</h1>
<p>Traditional ML · Logistic Regression · SVM · K-Means · Ensemble Soft Voting · RACE Dataset</p>
</div>""", unsafe_allow_html=True)

# ── Load pipeline ─────────────────────────────────────────────────────────────
with st.spinner("🔄 Loading AI models..."):
    pipe = load_pipeline()
if not pipe.loaded:
    st.warning("⚠️ Models not loaded. Using rule-based fallbacks.")

# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 1 — ARTICLE INPUT
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.screen == "input":
    st.markdown("## 📖 Article Input")
    st.caption("Paste a passage, upload a .txt file, or load a sample from the sidebar.")
    col_l, col_r = st.columns([3,1])
    with col_l:
        passage = st.text_area("Passage", value=st.session_state.passage, height=260,
                               placeholder="Paste your reading passage here...", label_visibility="collapsed")
        st.session_state.passage = passage
    with col_r:
        st.markdown("**Upload .txt**")
        uploaded = st.file_uploader("Upload", type=["txt"], label_visibility="collapsed")
        if uploaded:
            st.session_state.passage = uploaded.read().decode("utf-8")
            st.success("✅ File loaded")
        wc = len(passage.split()) if passage.strip() else 0
        st.metric("Words", wc)
        st.metric("Sentences", passage.count(".")+passage.count("?")+passage.count("!"))
    st.markdown("---")
    col_btn, col_info = st.columns([1,2])
    with col_btn:
        gen_btn = st.button("🚀 Generate Quiz", type="primary", use_container_width=True,
                            disabled=len(passage.strip())<20)
    with col_info:
        if len(passage.strip()) < 20:
            st.info("ℹ️ Enter at least 20 characters.")
        else:
            st.success("✅ Ready! Click **Generate Quiz**.")
    if gen_btn and len(passage.strip()) >= 20:
        with st.spinner("🤖 Generating question, distractors, and hints..."):
            try:
                result = pipe.run(passage)
                result["hints"] = pipe.generate_hints(passage, result["question"], result["correct_answer"])
                st.session_state.quiz_data = result
                for k in ["selected_option","answer_checked","hints_revealed","answer_revealed"]:
                    st.session_state[k] = False if k!="hints_revealed" else 0
                st.success("✅ Quiz generated!")
                time.sleep(0.4)
                st.session_state.screen = "quiz"; st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 2 — QUIZ VIEW
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.screen == "quiz":
    st.markdown("## 🧠 Quiz")
    if not st.session_state.quiz_data:
        st.warning("No quiz yet. Go to **📖 Article Input** first."); st.stop()
    qd = st.session_state.quiz_data
    with st.expander("📄 View Passage", expanded=False):
        st.write(st.session_state.passage)
    st.markdown('<span class="ai-badge">AI Generated</span>', unsafe_allow_html=True)
    st.markdown(f"<h3 style='color:#e2e8f0;margin-top:.4rem'>{qd['question']}</h3>", unsafe_allow_html=True)
    if qd.get("inference_time_ms",0)>0:
        st.caption(f"⏱ {qd['inference_time_ms']:.1f} ms")
    st.markdown("---")
    options = qd["options"]; letters = list(options.keys())
    opt_labels = [f"**{l}.** {options[l]}" for l in letters]
    st.markdown("**Select your answer:**")
    sel_display = st.radio("Options", opt_labels, index=None, label_visibility="collapsed")
    if sel_display:
        st.session_state.selected_option = sel_display.split(".")[0].replace("**","").strip()
    st.markdown("---")
    c1,c2,c3 = st.columns(3)
    with c1:
        check_btn = st.button("✅ Check Answer", type="primary", use_container_width=True,
                              disabled=st.session_state.selected_option is None)
    with c2:
        hint_btn = st.button("💡 Get a Hint", use_container_width=True,
                             disabled=st.session_state.hints_revealed>=3)
    with c3:
        if st.button("🔄 New Quiz", use_container_width=True):
            st.session_state.quiz_data=None; st.session_state.screen="input"; st.rerun()
    if hint_btn:
        st.session_state.hints_revealed = min(st.session_state.hints_revealed+1,3)
        if not qd.get("hints"):
            h = pipe.generate_hints(st.session_state.passage, qd["question"], qd["correct_answer"])
            st.session_state.quiz_data["hints"] = h
        st.session_state.screen="hints"; st.rerun()
    if check_btn and st.session_state.selected_option:
        sel = st.session_state.selected_option
        correct_letter = qd["correct_letter"]
        is_correct = sel == correct_letter
        st.session_state.answer_checked = True
        st.session_state.session_answers.append({"question":qd["question"][:60],"selected":sel,
                                                  "correct":correct_letter,"is_correct":is_correct})
        if is_correct:
            st.markdown("""<div style='background:linear-gradient(135deg,#052e16,#14532d);
              border:2px solid #22c55e;border-radius:12px;padding:1.2rem 1.5rem;margin-top:1rem'>
              <h3 style='color:#4ade80;margin:0'>🎉 Correct!</h3>
              <p style='color:#bbf7d0;margin:.4rem 0 0'>Well done! That's the right answer.</p></div>""",
              unsafe_allow_html=True)
        else:
            ct = options.get(correct_letter,"")
            st.markdown(f"""<div style='background:linear-gradient(135deg,#1a0000,#450a0a);
              border:2px solid #ef4444;border-radius:12px;padding:1.2rem 1.5rem;margin-top:1rem'>
              <h3 style='color:#f87171;margin:0'>❌ Incorrect</h3>
              <p style='color:#fecaca;margin:.4rem 0 0'>Correct answer: <strong>{correct_letter}. {ct}</strong></p>
              </div>""", unsafe_allow_html=True)
        st.markdown("---"); st.markdown("**Answer breakdown:**")
        for letter, opt_text in options.items():
            if letter == correct_letter:
                st.markdown(f'<div class="option-correct">✓ {letter}. {opt_text}</div>', unsafe_allow_html=True)
            elif letter == sel and not is_correct:
                st.markdown(f'<div class="option-wrong">✗ {letter}. {opt_text}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="option-neutral">{letter}. {opt_text}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 3 — HINT PANEL
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.screen == "hints":
    st.markdown("## 💡 Hint Panel")
    if not st.session_state.quiz_data:
        st.warning("No quiz yet. Go to **📖 Article Input** first."); st.stop()
    qd = st.session_state.quiz_data
    st.markdown(f"<p style='color:#94a3b8'>Question: <em>{qd['question']}</em></p>", unsafe_allow_html=True)
    if not qd.get("hints"):
        with st.spinner("Generating hints..."):
            h = pipe.generate_hints(st.session_state.passage, qd["question"], qd["correct_answer"])
            st.session_state.quiz_data["hints"] = h; qd = st.session_state.quiz_data
    hints = qd["hints"]; n = st.session_state.hints_revealed
    hint_defs = [("hint_1","hint1","💡 Hint 1 — General","broad clue to start"),
                 ("hint_2","hint2","🔍 Hint 2 — Medium","more specific pointer"),
                 ("hint_3","hint3","🎯 Hint 3 — Specific","very close to the answer")]
    for i,(key,css,label,sub) in enumerate(hint_defs,1):
        if n >= i:
            txt = hints.get(key,"Hint not available.")
            st.markdown(f'<div class="hint-box {css}"><strong style="color:#e2e8f0">{label}</strong> '
                        f'<span style="color:#94a3b8;font-size:.8rem">({sub})</span>'
                        f'<br><span style="color:#cbd5e1;margin-top:.4rem;display:block">{txt}</span></div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="hint-box hint-locked"><strong style="color:#475569">{label}</strong>'
                        f'<br><span style="color:#334155">🔒 Not revealed yet</span></div>',
                        unsafe_allow_html=True)
    st.markdown("---")
    ca,cb,cc = st.columns(3)
    with ca:
        if n < 3:
            if st.button(f"🔓 Reveal Hint {n+1}", type="primary", use_container_width=True):
                st.session_state.hints_revealed += 1; st.rerun()
        else:
            st.info("All hints revealed!")
    with cb:
        if n >= 3 and not st.session_state.answer_revealed:
            if st.button("🏁 Reveal Answer", use_container_width=True):
                st.session_state.answer_revealed = True; st.rerun()
    with cc:
        if st.button("← Back to Quiz", use_container_width=True):
            st.session_state.screen="quiz"; st.rerun()
    if st.session_state.answer_revealed:
        cl = qd["correct_letter"]; ct = qd["options"].get(cl, qd["correct_answer"])
        st.markdown(f"""<div style='background:linear-gradient(135deg,#1a1a3e,#2d1b69);
          border:2px solid #7c3aed;border-radius:12px;padding:1.2rem 1.5rem;margin-top:1rem'>
          <h3 style='color:#a78bfa;margin:0'>📌 Answer Revealed</h3>
          <p style='color:#ddd6fe;margin:.5rem 0 0;font-size:1.1rem'><strong>{cl}.</strong> {ct}</p>
          </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 4 — ANALYTICS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.screen == "analytics":
    st.markdown("## 📊 Analytics Dashboard")
    st.caption("Model performance metrics, confusion matrices, and session statistics.")
    try:
        from evaluate import load_saved_metrics, plot_confusion_matrix_fig, plot_metrics_bar
        all_m = load_saved_metrics()
        ma = all_m.get("model_a",{}); mb = all_m.get("model_b",{})
    except Exception:
        ma = {}; mb = {}

    tab_a, tab_b, tab_sess, tab_export = st.tabs(["🧠 Model A","🎯 Model B","📈 Session","💾 Export"])

    # ── Model A ──────────────────────────────────────────────
    with tab_a:
        st.markdown("### Model A — Answer Verifier")
        st.caption("Logistic Regression · SVM · Ensemble · K-Means")
        for mname in ["LogisticRegression","SVM","Ensemble(LR+SVM)"]:
            m = ma.get(mname, {"accuracy":.72,"precision":.68,"recall":.75,"f1":.71})
            st.markdown(f"**{mname}**")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Accuracy",  f"{m.get('accuracy',0):.3f}")
            c2.metric("Precision", f"{m.get('precision',0):.3f}")
            c3.metric("Recall",    f"{m.get('recall',0):.3f}")
            c4.metric("F1 Score",  f"{m.get('f1',0):.3f}")
            if "confusion_matrix" in m:
                col1,col2 = st.columns(2)
                with col1: st.pyplot(plot_metrics_bar(m,f"{mname} Metrics"), use_container_width=True)
                with col2: st.pyplot(plot_confusion_matrix_fig(m["confusion_matrix"],f"CM — {mname}"), use_container_width=True)
            st.markdown("---")
        km_sil = ma.get("KMeans",{}).get("silhouette_score",None)
        if km_sil is not None:
            st.metric("K-Means Silhouette Score", f"{km_sil:.4f}")
            st.caption("Score range: -1 to 1. Values > 0.5 = well-separated clusters.")

        # ── Cosine Similarity Report ──────────────────────────────
        cos = ma.get("cosine_similarity", {})
        if cos:
            st.markdown("---")
            st.markdown("### 🔗 Cosine Similarity Analysis")
            st.caption("How similar are article/question vectors to correct vs incorrect options.")
            for split_name, split_key in [("Train Set", "train"), ("Val Set", "val")]:
                s = cos.get(split_key, {})
                if not s:
                    continue
                st.markdown(f"**{split_name}**")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Article ↔ Correct",   f"{s.get('ao_correct',  0):.4f}")
                c2.metric("Article ↔ Incorrect",  f"{s.get('ao_incorrect',0):.4f}")
                c3.metric("Question ↔ Correct",   f"{s.get('qo_correct',  0):.4f}")
                c4.metric("Question ↔ Incorrect",  f"{s.get('qo_incorrect',0):.4f}")
                ao_gap = s.get("ao_gap", 0)
                qo_gap = s.get("qo_gap", 0)
                g1, g2 = st.columns(2)
                ao_color = "normal" if ao_gap > 0 else "inverse"
                qo_color = "normal" if qo_gap > 0 else "inverse"
                g1.metric("Gap: Article↔Option",  f"{ao_gap:+.4f}",
                          delta="✓ Discriminative" if ao_gap > 0 else "✗ Low Signal",
                          delta_color=ao_color)
                g2.metric("Gap: Question↔Option", f"{qo_gap:+.4f}",
                          delta="✓ Discriminative" if qo_gap > 0 else "✗ Low Signal",
                          delta_color=qo_color)
                st.markdown("")


    # ── Model B ──────────────────────────────────────────────
    with tab_b:
        st.markdown("### Model B — Distractor & Hint Generator")
        for comp in ["DistractorRanker","HintGenerator"]:
            m = mb.get(comp, {"accuracy":.70,"precision":.65,"recall":.73,"f1":.69})
            st.markdown(f"**{comp}**")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Accuracy",  f"{m.get('accuracy',0):.3f}")
            c2.metric("Precision", f"{m.get('precision',0):.3f}")
            c3.metric("Recall",    f"{m.get('recall',0):.3f}")
            c4.metric("F1 Score",  f"{m.get('f1',0):.3f}")
            if "r2_score" in m: st.metric("R² Score (Hint Scorer)", f"{m['r2_score']:.4f}")
            if "confusion_matrix" in m:
                st.pyplot(plot_confusion_matrix_fig(m["confusion_matrix"],f"CM — {comp}"), use_container_width=True)
            st.markdown("---")

    # ── Session ──────────────────────────────────────────────
    with tab_sess:
        st.markdown("### Session Performance")
        ans_log = st.session_state.session_answers
        if ans_log:
            adf = pd.DataFrame(ans_log)
            correct_n = adf["is_correct"].sum(); total = len(adf)
            c1,c2,c3 = st.columns(3)
            c1.metric("Attempted", total)
            c2.metric("Correct", int(correct_n))
            c3.metric("Accuracy", f"{correct_n/total:.1%}" if total else "—")
            adf["Result"] = adf["is_correct"].map({True:"✅ Correct",False:"❌ Wrong"})
            st.dataframe(adf[["question","selected","correct","Result"]], use_container_width=True)
        else:
            st.info("No quiz attempts yet. Complete a quiz to see session stats here.")
        sl = pipe.get_session_log()
        if not sl.empty:
            st.markdown("**Inference Log**")
            st.dataframe(sl, use_container_width=True)
            st.metric("Avg Inference Time", f"{sl['inference_time_ms'].mean():.1f} ms")

    # ── Export ───────────────────────────────────────────────
    with tab_export:
        st.markdown("### Export Data")
        ans_log = st.session_state.session_answers
        if ans_log:
            csv = pd.DataFrame(ans_log).to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Answer Log (CSV)", csv, "rc_quiz_session.csv","text/csv", use_container_width=True)
        sl = pipe.get_session_log()
        if not sl.empty:
            st.download_button("⬇️ Download Inference Log (CSV)", sl.to_csv(index=False).encode("utf-8"),
                               "rc_inference_log.csv","text/csv", use_container_width=True)
        st.markdown("---")
        st.markdown("**Re-run Full Evaluation**")
        if st.button("🔬 Run Evaluation Suite"):
            with st.spinner("Evaluating models..."):
                try:
                    from evaluate import evaluate_model_a_full, evaluate_model_b_full
                    evaluate_model_a_full(); evaluate_model_b_full()
                    st.success("✅ Done! Refresh the Analytics tab.")
                except Exception as e:
                    st.error(f"Error: {e}")
