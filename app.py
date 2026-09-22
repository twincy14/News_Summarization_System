"""
Module 6: Demo Interface (Streamlit Web App)

Interactive browser UI for News Summarization & ROUGE Benchmarking:
- Custom article input & abstractive summary generation
- Support for Executive Key Takeaways (Structured) & Standard Paragraph formats
- Real-time ROUGE evaluation vs user reference summary
- Preset dataset sample explorer (CNN/DailyMail)
"""

import time
import streamlit as st

from preprocess import preprocess_article
from summarizer import NewsSummarizer
from evaluate import compute_rouge
from dataset import load_news_dataset

# Page configuration
st.set_page_config(
    page_title="AI News Summarization Engine",
    page_icon="📰",
    layout="wide"
)

# Custom CSS for rich aesthetics
st.markdown("""
    <style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2563EB;
    }
    .metric-lbl {
        font-size: 0.9rem;
        font-weight: 600;
        color: #475569;
    }
    .summary-box {
        background-color: #F8FAFC;
        border-left: 5px solid #2563EB;
        padding: 20px;
        border-radius: 8px;
        font-size: 1.05rem;
        line-height: 1.6;
        color: #0F172A;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_summarizer():
    """Cache the summarization engine across Streamlit sessions."""
    return NewsSummarizer(model_name="facebook/bart-large-cnn")


def render_rouge_metrics(rouge_res: dict):
    """Render ROUGE metrics in stylized columns."""
    st.markdown("### 📊 ROUGE Evaluation Scores")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">ROUGE-1 (Unigram Overlap)</div>
                <div class="metric-val">{rouge_res['rouge1']['fmeasure']:.4f}</div>
                <small style="color: #64748B;">P: {rouge_res['rouge1']['precision']} | R: {rouge_res['rouge1']['recall']}</small>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">ROUGE-2 (Bigram Overlap)</div>
                <div class="metric-val">{rouge_res['rouge2']['fmeasure']:.4f}</div>
                <small style="color: #64748B;">P: {rouge_res['rouge2']['precision']} | R: {rouge_res['rouge2']['recall']}</small>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">ROUGE-L (Longest Sequence)</div>
                <div class="metric-val">{rouge_res['rougeL']['fmeasure']:.4f}</div>
                <small style="color: #64748B;">P: {rouge_res['rougeL']['precision']} | R: {rouge_res['rougeL']['recall']}</small>
            </div>
        """, unsafe_allow_html=True)


def main():
    st.markdown('<div class="main-header">📰 AI News Summarization Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Abstractive News Summarization powered by HuggingFace BART & ROUGE Evaluation</div>', unsafe_allow_html=True)

    # Sidebar parameters
    st.sidebar.header("⚙️ Generation Settings")
    
    summary_mode = st.sidebar.radio(
        "Summary Output Format",
        ["📌 Executive Key Takeaways (Structured)", "📝 Standard Abstractive Paragraph"],
        index=0
    )

    max_len = st.sidebar.slider("Max Summary Length (tokens)", min_value=30, max_value=250, value=150, step=10)
    min_len = st.sidebar.slider("Min Summary Length (tokens)", min_value=10, max_value=100, value=30, step=5)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Model Specs:**")
    st.sidebar.info("Model: `facebook/bart-large-cnn`\nDataset: `CNN/DailyMail v3.0.0`")

    # Tabs
    tab1, tab2 = st.tabs(["✍️ Single Article Summarizer", "📚 CNN/DailyMail Dataset Benchmark"])

    # Tab 1: Single Article Input
    with tab1:
        st.subheader("Paste & Summarize Custom Article")
        
        article_input = st.text_area(
            "News Article Text",
            height=240,
            placeholder="Paste your news article text here..."
        )
        
        reference_input = st.text_area(
            "Reference Summary (Optional for ROUGE Scoring)",
            height=90,
            placeholder="Paste reference summary here to calculate ROUGE scores live..."
        )
        
        if st.button("🚀 Generate Summary", type="primary", use_container_width=True):
            if not article_input.strip():
                st.warning("Please enter or paste a news article first.")
            else:
                with st.spinner("Processing article & generating BART summary..."):
                    # Module 2: Preprocess
                    clean_text = preprocess_article(article_input)
                    
                    # Module 3: Summarization
                    summarizer = get_summarizer()
                    start = time.time()
                    
                    if "Structured" in summary_mode:
                        generated_summary = summarizer.generate_structured_summary(
                            article_input, max_length=max_len, min_length=min_len
                        )
                    else:
                        generated_summary = summarizer.generate_summary(
                            clean_text, max_length=max_len, min_length=min_len
                        )
                    duration = time.time() - start
                    
                st.markdown("### 📝 Generated Summary")
                st.markdown(f'<div class="summary-box">{generated_summary}</div>', unsafe_allow_html=True)
                st.caption(f"⚡ Generation completed in {duration:.2f} seconds | Preprocessed input length: {len(clean_text.split())} words")

                # Module 4: Evaluate if reference provided
                if reference_input.strip():
                    rouge_res = compute_rouge(generated_summary, reference_input.strip())
                    st.write("")
                    render_rouge_metrics(rouge_res)

    # Tab 2: Dataset Explorer
    with tab2:
        st.subheader("Benchmark Pre-loaded CNN/DailyMail Samples")
        
        sample_count = st.slider("Number of samples to fetch", min_value=1, max_value=10, value=5)
        
        if "dataset_samples" not in st.session_state or len(st.session_state["dataset_samples"]) != sample_count:
            if st.button("📥 Load Dataset Samples"):
                with st.spinner("Fetching samples from HuggingFace..."):
                    st.session_state["dataset_samples"] = load_news_dataset(split="test", sample_size=sample_count)

        if "dataset_samples" in st.session_state:
            samples = st.session_state["dataset_samples"]
            sample_options = [f"Article #{i+1} ({samples[i]['article'][:60]}...)" for i in range(len(samples))]
            
            selected_idx = st.selectbox("Select Sample Article", range(len(sample_options)), format_func=lambda i: sample_options[i])
            selected_sample = samples[selected_idx]

            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown("**Original Raw Article:**")
                st.text_area("Article Content", selected_sample["article"], height=200, disabled=True)
            
            with col_right:
                st.markdown("**Human Reference Summary:**")
                st.text_area("Ground Truth Summary", selected_sample["reference_summary"], height=200, disabled=True)

            if st.button("⚡ Summarize Sample Article", type="primary"):
                with st.spinner("Processing article & evaluating..."):
                    clean_text = preprocess_article(selected_sample["article"])
                    summarizer = get_summarizer()
                    
                    if "Structured" in summary_mode:
                        gen_summary = summarizer.generate_structured_summary(selected_sample["article"], max_length=max_len, min_length=min_len)
                    else:
                        gen_summary = summarizer.generate_summary(clean_text, max_length=max_len, min_length=min_len)
                        
                    rouge_res = compute_rouge(gen_summary, selected_sample["reference_summary"])

                st.markdown("### 📝 Generated BART Summary")
                st.markdown(f'<div class="summary-box">{gen_summary}</div>', unsafe_allow_html=True)
                st.write("")
                render_rouge_metrics(rouge_res)


if __name__ == "__main__":
    main()
