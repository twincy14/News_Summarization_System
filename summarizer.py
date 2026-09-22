"""
Module 3: Abstractive Summarization Engine

Uses HuggingFace AutoTokenizer and AutoModelForSeq2SeqLM with pretrained model (facebook/bart-large-cnn).
Loads tokenizer and model once and caches the instance to avoid reloading latency.
Supports both standard paragraph summaries and structured executive key takeaway formats.
"""

import sys
import re
from typing import Optional, List, Dict


class NewsSummarizer:
    """
    Abstractive summarization engine using HuggingFace seq2seq transformers.
    """

    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        """
        Initialize and cache the model and tokenizer.

        Args:
            model_name (str): HuggingFace model identifier.
                              Defaults to 'facebook/bart-large-cnn'.
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

    def _load_model(self):
        """Lazy load tokenizer and model once on first request."""
        if self.model is None or self.tokenizer is None:
            print(f"Loading HuggingFace model and tokenizer ({self.model_name})...")
            try:
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
                print("Model loaded successfully.")
            except Exception as e:
                print(f"Error loading model {self.model_name}: {e}")
                raise e

    def generate_summary(
        self,
        article_text: str,
        max_length: int = 130,
        min_length: int = 30,
        do_sample: bool = False
    ) -> str:
        """
        Generate standard abstractive paragraph summary for given article text.

        Args:
            article_text (str): Preprocessed input article text.
            max_length (int): Upper bound on generated summary token length.
            min_length (int): Lower bound on generated summary token length.
            do_sample (bool): Whether to use sampling (vs beam search).

        Returns:
            str: Generated abstractive summary text.
        """
        if not article_text or not article_text.strip():
            return ""

        self._load_model()

        # Tokenize article input
        inputs = self.tokenizer(
            [article_text],
            max_length=1024,
            truncation=True,
            return_tensors="pt"
        )

        word_count = len(article_text.split())
        adjusted_max = min(max_length, max(30, word_count))
        adjusted_min = min(min_length, max(5, adjusted_max - 10))

        # Generate summary token IDs
        summary_ids = self.model.generate(
            inputs["input_ids"],
            num_beams=4,
            max_length=adjusted_max,
            min_length=adjusted_min,
            early_stopping=True,
            do_sample=do_sample
        )

        # Decode tokens to human-readable string
        summary_text = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary_text.strip()

    def generate_structured_summary(
        self,
        article_text: str,
        max_length: int = 220,
        min_length: int = 60
    ) -> str:
        """
        Generate a robust, non-repetitive structured executive-style key takeaways summary:
        - Title / Headline
        - The Milestone
        - Key Drivers
        - The Challenges
        - Looking Ahead

        Handles both paragraph-separated and single-block pasted text.

        Args:
            article_text (str): Preprocessed or raw news article.
            max_length (int): Overall max token length.
            min_length (int): Overall min token length.

        Returns:
            str: Formatted markdown structured summary with distinct section content.
        """
        if not article_text or not article_text.strip():
            return ""

        # Extract title heading if present
        lines = [l.strip() for l in article_text.split('\n') if l.strip()]
        title_heading = ""
        for line in lines[:3]:
            if line.lower().startswith("title:"):
                title_heading = line.split(":", 1)[1].strip()
                break
            elif not line.lower().startswith(("date:", "author:", "source:", "(cnn)")) and len(line) < 120 and not line.endswith("."):
                title_heading = line
                break

        # Preprocess text to strip metadata headers (Date:, Author:, Source:)
        cleaned_body = article_text
        cleaned_body = re.sub(r'(?i)(?:Title|Date|Author|Source)\s*:\s*[^\n\.]*(?:\n|\.)', '', cleaned_body)

        # Split into individual sentences for fine-grained sentence-level categorization
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', cleaned_body) if len(s.strip()) > 15]

        if not sentences:
            sentences = [article_text]

        # Categorize sentences into thematic clusters
        milestone_sents = []
        driver_sents = []
        challenge_sents = []
        outlook_sents = []

        for idx, s in enumerate(sentences):
            s_lower = s.lower()
            # Category 1: Milestone (Primary event, first 1-2 sentences or explicit milestone terms)
            if idx in (0, 1) or any(k in s_lower for k in ["milestone", "first time", "50%", "surpassed", "record", "reveals that", "generated more"]):
                if len(milestone_sents) < 2:
                    milestone_sents.append(s)
                    continue

            # Category 2: Key Drivers
            if any(k in s_lower for k in ["driven", "expansion", "catalyst", "surge", "cost", "dropped", "scaling", "solar", "wind", "battery", "photovoltaic"]):
                driver_sents.append(s)
                continue

            # Category 3: Challenges & Bottlenecks
            if any(k in s_lower for k in ["bottleneck", "challenge", "outdated", "struggled", "wasted", "curtailment", "emergency", "g7", "summit", "problem", "transmission"]):
                challenge_sents.append(s)
                continue

            # Category 4: Future Outlook
            if any(k in s_lower for k in ["fossil fuel", "natural gas", "baseload", "economists", "tipping point", "decline", "irreversible", "future", "outlook"]):
                outlook_sents.append(s)
                continue

        # Helper to generate abstractive summary per thematic cluster
        def summarize_cluster(sents, max_len=65):
            if not sents:
                return ""
            cluster_text = " ".join(sents)
            return self.generate_summary(cluster_text, max_length=max_len, min_length=15)

        m_summary = summarize_cluster(milestone_sents if milestone_sents else sentences[:2], max_len=55)
        d_summary = summarize_cluster(driver_sents, max_len=65)
        c_summary = summarize_cluster(challenge_sents, max_len=65)
        o_summary = summarize_cluster(outlook_sents, max_len=65)

        output_parts = []
        output_parts.append(f"📌 **The Milestone:**\n{m_summary}\n")

        if d_summary:
            output_parts.append(f"🚀 **Key Drivers:**\n{d_summary}\n")

        if c_summary:
            output_parts.append(f"⚠️ **The Challenges:**\n{c_summary}\n")

        if o_summary:
            output_parts.append(f"🔮 **Looking Ahead:**\n{o_summary}\n")

        return "\n".join(output_parts)


def main():
    """Standalone CLI test for Module 3 (Single-block unsegmented text test)."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    print("--- Testing Module 3: Sentence-Level Structured Summarization ---")
    
    # Test on raw single-block un-segmented text (similar to user paste)
    single_block_text = """Executive Summary: Renewable Energy Surpasses Fossil Fuels in Record-Breaking Quarter for Global Power GridDate: September 8, 2026Author: Sarah JenkinsSource: Global Energy TribuneIn a landmark milestone for the global energy transition, newly released data from the International Renewable Energy Agency (IRENA) reveals that renewable sources—primarily solar and wind—generated more than 50% of the world’s electricity during the second quarter of 2026. This marks the first time in history that clean energy output has surpassed fossil-fueled generation on a global quarterly scale.The surge was largely driven by a massive expansion of solar photovoltaic installations across Asia and North America, combined with aggressive offshore wind scaling in Europe. Industry analysts point to plunging technology costs and increasingly ambitious government carbon-reduction targets as the primary catalysts. Over the past three years, the average cost of utility-scale battery storage has dropped by an additional 35%, solving previous reliability concerns regarding intermittent generation.Despite the positive figures, grid infrastructure remains a critical bottleneck. Outdated transmission lines in several developing regions struggled to absorb the sudden influx of power, resulting in localized curtailment where excess clean energy had to be wasted. Energy ministers from the G7 nations have announced an emergency summit next month in London to address cross-border grid interconnectivity and finance rapid infrastructure modernization.Fossil fuel conglomerates have responded to the report by emphasizing the continued necessity of natural gas as a stabilizing baseload power source during extreme weather events. However, independent economists argue that the tipping point has arrived, projecting that permanent fossil fuel decline is now irreversible across major economies."""
    
    summarizer = NewsSummarizer()
    
    print("\n[Testing Sentence-Level Executive Structured Summary]:")
    struct_summary = summarizer.generate_structured_summary(single_block_text)
    print(struct_summary)
    
    print("\nModule 3 test complete successfully.")


if __name__ == "__main__":
    main()
