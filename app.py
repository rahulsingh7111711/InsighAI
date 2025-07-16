import streamlit as st
import pandas as pd
import altair as alt
import google.generativeai as genai
import re

# ------------------ Streamlit Setup ------------------
st.set_page_config(page_title="InsightAI", layout="wide")
st.title("📊 InsightAI - Your AI-Powered Data Analyst")

# ------------------ Gemini Configuration ------------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Try Gemini models in priority order
AVAILABLE_MODELS = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]

def _get_gemini_model():
    for name in AVAILABLE_MODELS:
        try:
            return genai.GenerativeModel(name)
        except Exception:
            continue
    raise RuntimeError("❌ No supported Gemini model found.")

_model = _get_gemini_model()

# ------------------ Helpers ------------------

def build_dataset_context(df: pd.DataFrame, max_rows: int = 10) -> str:
    rows, cols = df.shape
    dtypes_str = df.dtypes.astype(str).to_dict()
    sample_csv = df.head(max_rows).to_csv(index=False)
    return (
        f"Dataset shape: {rows} rows x {cols} columns.\n"
        f"Column dtypes: {dtypes_str}\n\n"
        f"Top {min(max_rows, rows)} rows (CSV):\n{sample_csv}\n"
    )

def ask_ai(question: str, df: pd.DataFrame) -> str:
    context = build_dataset_context(df)
    prompt = f"""
You are a helpful data analyst. You are given a dataset summary and a user question.

{context}

User question: \"{question}\"

Instructions:
1. Answer using the data shown (mention if more rows are needed).
2. Provide a short reasoning.
3. If a chart would help, include a suggestion in the format:
   CHART: <chart_type> | X=<column> | Y=<column or aggregation(column)> | GROUP=<column or None>
"""
    try:
        response = _model.generate_content(prompt)
        return response.text or "(No response text returned.)"
    except Exception as e:
        return f"AI error: {e}"

def extract_chart_instruction(text):
    match = re.search(r'CHART:\s*(\w+)\s*\|\s*X=(\w+)\s*\|\s*Y=([\w\(\)]+)\s*\|\s*GROUP=(\w+|None)', text)
    if match:
        chart_type, x, y, group = match.groups()
        return {
            "type": chart_type.lower(),
            "x": x,
            "y": y,
            "group": None if group == "None" else group
        }
    return None

# ------------------ File Upload ------------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### 🧾 Data Preview")
    st.dataframe(df.head())

    with st.expander("🔍 Show column names"):
        st.write(list(df.columns))

    question = st.text_input("Ask a question about your data (e.g., 'Average TENUR by CUST_STATE')")

    if question:
        with st.spinner("🔎 Analyzing with Gemini..."):
            answer = ask_ai(question, df)

        st.markdown("### 🧠 InsightAI Says:")
        st.write(answer)

        # ------------- Try to Extract Chart -------------
        instruction = extract_chart_instruction(answer)
        if instruction:
            st.markdown("### 📊 AI-Suggested Visualization")

            try:
                chart_data = df.copy()

                # Handle aggregation (e.g., mean(EMI))
                agg_match = re.match(r'(\w+)\((\w+)\)', instruction["y"])
                if agg_match:
                    agg_func, y_col = agg_match.groups()
                    if agg_func.lower() == "mean":
                        chart_data = chart_data.groupby(instruction["x"], as_index=False)[y_col].mean()
                        y_column = y_col
                    else:
                        raise ValueError(f"Unsupported aggregation: {agg_func}")
                else:
                    y_column = instruction["y"]

                if instruction["group"]:
                    chart_data[instruction["group"]] = df[instruction["group"]]

                chart_type = instruction["type"]
                x = instruction["x"]
                y = y_column
                group = instruction["group"]

                if chart_type == "bar":
                    chart = alt.Chart(chart_data).mark_bar().encode(
                        x=x,
                        y=y,
                        color=group if group else alt.value("steelblue")
                    ).properties(width=700, height=400)
                elif chart_type == "line":
                    chart = alt.Chart(chart_data).mark_line().encode(
                        x=x,
                        y=y,
                        color=group if group else alt.value("steelblue")
                    ).properties(width=700, height=400)
                elif chart_type == "scatter":
                    chart = alt.Chart(chart_data).mark_circle(size=60).encode(
                        x=x,
                        y=y,
                        color=group if group else alt.value("steelblue"),
                        tooltip=[x, y]
                    ).interactive().properties(width=700, height=400)
                else:
                    st.info(f"⚠️ Chart type '{chart_type}' is not supported yet.")
                    chart = None

                if chart:
                    st.altair_chart(chart)

            except Exception as e:
                st.error(f"❌ Could not generate chart: {e}")
