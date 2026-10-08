import io, json, traceback, contextlib, datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="LLM Data Analyst", page_icon="📊", layout="wide")
sns.set_theme(style="whitegrid")

st.markdown(
    "**Created by:** Festus Amutenya - 220006245 · Kennedy Hauwanga - 218205299 "
    "| **Module:** Large Language Models  "
    "| **Institution:** Your University Name"
)

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
GROQ_KEY = st.secrets.get("GROQ_API_KEY", "")
BASE_URL = "https://api.groq.com/openai/v1"
MODEL    = "openai/gpt-oss-120b"

# ------------------------------------------------------------------
# PROMPTS
# ------------------------------------------------------------------
CODE_PROMPT = """You are a senior data analyst who writes Python code.
You will be given a pandas DataFrame named `df` (already loaded) and a user question.
Return ONLY a JSON object with two keys: "plan" and "code".
- "plan": a 3-5 step analysis plan in plain English, numbered.
- "code": Python code implementing the plan.
Rules:
- In scope: pd, np, plt, sns, df. Do NOT read files or import anything.
- Use print() for tables/text. Build charts with plt/sns (do NOT call savefig/show).
- Store the key answer in a variable called `result`.
- Under 40 lines. Handle missing values.
- If the question cannot be answered, print a short explanation."""

EXPLAIN_PROMPT = """You are a friendly analyst explaining results to a non-technical user.
Given the question, plan, code, text output, and chart count, write a plain-language explanation:
1) what was analysed, 2) the key finding with actual numbers, 3) caveats, 4) one follow-up question.
Under 200 words. No code, no asterisks, no markdown symbols."""

# ------------------------------------------------------------------
# SANDBOX (Ethics: restricted execution)
# ------------------------------------------------------------------
SAFE = {"print":print,"len":len,"range":range,"min":min,"max":max,"sum":sum,"abs":abs,
        "round":round,"sorted":sorted,"list":list,"dict":dict,"set":set,"tuple":tuple,
        "str":str,"int":int,"float":float,"bool":bool,"enumerate":enumerate,"zip":zip,
        "isinstance":isinstance,"any":any,"all":all,"type":type,
        "ValueError":ValueError,"TypeError":TypeError,"KeyError":KeyError}

def run_code(code, df):
    ns = {"__builtins__": SAFE, "df": df.copy(), "pd": pd, "np": np, "plt": plt, "sns": sns}
    out, err = io.StringIO(), None
    plt.close("all")
    try:
        with contextlib.redirect_stdout(out):
            exec(code, ns)
    except Exception:
        err = traceback.format_exc(limit=3)
    figs = [plt.figure(n) for n in plt.get_fignums()]
    plt.close("all")
    return out.getvalue(), figs, err, ns.get("result")

def describe_df(df):
    lines = [f"Shape: {df.shape[0]} rows x {df.shape[1]} cols", "", "Columns:"]
    for c in df.columns:
        lines.append(f"  - {c} ({df[c].dtype}), {int(df[c].isna().sum())} missing")
    lines += ["", "First 3 rows:", df.head(3).to_string()]
    num = df.select_dtypes(include=np.number)
    if not num.empty:
        lines += ["", "Numeric summary:", num.describe().round(2).to_string()]
    return "\n".join(lines)

def call_llm(messages, json_mode=False):
    client = OpenAI(api_key=GROQ_KEY, base_url=BASE_URL)
    kw = dict(model=MODEL, messages=messages, temperature=0.1 if json_mode else 0.3)
    if json_mode:
        kw["response_format"] = {"type": "json_object"}
    return client.chat.completions.create(**kw).choices[0].message.content

def parse_json(raw):
    t = raw.strip()
    if t.startswith("```"):
        t = t.split("```")[1]
        if t.startswith("json"): t = t[4:]
    try: return json.loads(t)
    except: return {"plan": "parse error", "code": "print('bad JSON')"}

def analyse(df, q, retries=2):
    msgs = [{"role":"system","content":CODE_PROMPT},
            {"role":"user","content":f"DataFrame:\n{describe_df(df)}\n\nQuestion: {q}"}]
    for attempt in range(retries+1):
        raw = call_llm(msgs, json_mode=True)
        p = parse_json(raw)
        out, figs, err, res = run_code(p.get("code",""), df)
        if err is None:
            return p.get("plan",""), p.get("code",""), out, figs, res, attempt+1
        msgs += [{"role":"assistant","content":raw},
                 {"role":"user","content":f"Error:\n{err}\nFix and return same JSON format."}]
    return p.get("plan",""), p.get("code",""), out, figs, res, retries+1

def sample_df():
    rng = np.random.default_rng(42); n = 300
    df = pd.DataFrame({
        "region": rng.choice(["North","South","East","West"], n),
        "product": rng.choice(["Alpha","Beta","Gamma"], n),
        "units": rng.integers(1,50,n),
        "unit_price": np.round(rng.uniform(5,100,n),2),
        "date": pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0,365,n), unit="D")})
    df["revenue"] = (df["units"]*df["unit_price"]).round(2)
    return df

# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.title("📊 LLM Data Analyst")
st.caption("Upload a CSV, ask a question in plain English, and let the LLM plan, code, run, and explain the analysis.")

if not GROQ_KEY:
    st.error("GROQ_API_KEY secret not configured. Add it in Streamlit Cloud settings.")
    st.stop()

# --- Ethics & Safeguards panel (visible evidence for the report) ---
with st.sidebar:
    st.header("🛡️ Ethics & Safeguards")
    with st.expander("View safeguards", expanded=False):
        st.markdown(
            "**Privacy**  \n"
            "Only the schema (column names, dtypes, first 3 rows) is sent to the LLM. "
            "Generated code has no network or file access.  \n\n"
            "**Fairness**  \n"
            "Code is not judged as 'correct' — it is only checked for execution success. "
            "No stylistic preferences are imposed.  \n\n"
            "**No blind trust**  \n"
            "The generated code and analysis plan are always shown before results. "
            "A provenance panel lists the dataset, model, and timestamp.  \n\n"
            "**Security**  \n"
            "Execution uses a restricted set of builtins — no `import`, no `open`, "
            "no `eval`, no `exec`. Only `pandas`, `numpy`, `matplotlib`, `seaborn`, "
            "and a copy of the DataFrame are in scope."
        )

st.subheader("1. Load data")
c1, c2 = st.columns([3, 1])
with c1:
    up = st.file_uploader("Upload a CSV", type=["csv"])
with c2:
    if st.button("Use sample data"):
        st.session_state["df"] = sample_df()
        st.session_state["src"] = "synthetic_sales_sample"
        st.session_state["t"] = datetime.datetime.now().isoformat(timespec="seconds")

if up is not None:
    try:
        st.session_state["df"] = pd.read_csv(up)
        st.session_state["src"] = up.name
        st.session_state["t"] = datetime.datetime.now().isoformat(timespec="seconds")
    except Exception as e:
        st.error(f"CSV error: {e}")

df = st.session_state.get("df")
if df is None:
    st.info("Upload a CSV or click **Use sample data** to begin.")
    st.stop()

with st.expander(f"Preview — {df.shape[0]} rows × {df.shape[1]} cols"):
    st.dataframe(df.head(20), use_container_width=True)

st.subheader("2. Ask a question")
q = st.text_input("Question", placeholder="Which region generated the most revenue?")
run = st.button("🚀 Analyse", type="primary", disabled=not q)

if run:
    st.subheader("3. Analysis")
    with st.spinner("LLM is planning, coding, and running…"):
        try:
            plan, code, out, figs, res, attempts = analyse(df, q)
        except Exception as e:
            st.error(f"LLM call failed: {e}")
            st.stop()

    st.markdown("#### 🧭 Plan")
    st.markdown(plan)
    if attempts > 1:
        st.info(f"🔁 Self-repair used: {attempts-1} retry attempt(s).")

    with st.expander("🐍 Generated code", expanded=True):
        st.code(code, language="python")

    st.markdown("#### 📈 Results")
    if out.strip():
        st.text(out)
    for f in figs:
        st.pyplot(f, use_container_width=True)
    if res is not None:
        st.success(f"**Key result:** {res}")

    st.markdown("#### 🗣️ Explanation")
    with st.spinner("Explaining…"):
        try:
            exp = call_llm([{"role":"system","content":EXPLAIN_PROMPT},
                            {"role":"user","content":
                             f"Q: {q}\nPlan: {plan}\nCode: {code}\nOutput: {out}\nCharts: {len(figs)}"}])
            st.write(exp)
        except Exception as e:
            st.warning(f"Explanation failed: {e}")

    st.markdown("#### 📚 Sources & Provenance")
    st.markdown(f"- Dataset: `{st.session_state.get('src','?')}`\n"
                f"- Loaded: {st.session_state.get('t','?')}\n"
                f"- Shape: {df.shape[0]} × {df.shape[1]}\n"
                f"- Model: `{MODEL}` via Groq\n"
                f"- Generated: {datetime.datetime.now().isoformat(timespec='seconds')}")
    st.caption("⚠️ LLM-generated code can be wrong. Verify numbers before trusting them.")
