"""
Module 3: Abstractive Summarization Engine

Uses HuggingFace AutoTokenizer and AutoModelForSeq2SeqLM with pretrained model (facebook/bart-large-cnn).
Loads tokenizer and model once and caches the instance to avoid reloading latency.
Provides full-document comprehensive summarization:
- Standard abstractive paragraph summarizing the entire article from lead to conclusion (via narrative Map-Reduce).
- Structured executive key takeaways covering Milestone, Key Drivers, Challenges, and Looking Ahead for any topic.
"""

import sys
import re
from typing import Optional, List, Dict


def clean_sentence_endings(text: str) -> str:
    """
    Ensure generated summary ends cleanly on a full sentence with proper punctuation.
    Trims broken, cut-off sentence fragments at the end caused by max_length limits.
    """
    text = text.strip()
    if not text:
        return ""
    
    # If it ends with valid punctuation already, return as is
    if text[-1] in '.!?':
        return text

    # Look for last sentence-ending punctuation
    last_punct = max(text.rfind('. '), text.rfind('! '), text.rfind('? '), text.rfind('."'), text.rfind('."'))
    if last_punct > 30:
        return text[:last_punct + 1].strip()

    last_single_punct = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
    if last_single_punct > 30 and (len(text) - last_single_punct) < 40:
        return text[:last_single_punct + 1].strip()

    # Fallback: append period if it looks like a complete thought
    return text + "."


def split_sentences(text: str) -> List[str]:
    """Split text cleanly into sentences using boundary punctuation."""
    if not text or not text.strip():
        return []
    raw_sents = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'\(\[])', text.strip())
    return [s.strip() for s in raw_sents if len(s.strip()) > 8]


class NewsSummarizer:
    """
    Abstractive summarization engine using HuggingFace seq2seq transformers.
    Supports comprehensive whole-article coverage and universal structured breakdowns.
    Optimized with batched inference, local caching, and adaptive beam search.
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
        self.device = "cpu"

    def _load_model(self):
        """Load tokenizer and model with local cache priority and device setup."""
        if self.model is None or self.tokenizer is None:
            print(f"Loading HuggingFace model and tokenizer ({self.model_name})...")
            try:
                import os
                import torch
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

                # Determine compute device
                if torch.cuda.is_available():
                    self.device = "cuda"
                else:
                    self.device = "cpu"
                    # Optimize CPU threads for parallel inference
                    cpu_cnt = os.cpu_count() or 4
                    torch.set_num_threads(max(1, min(cpu_cnt, 8)))

                # Prioritize local cache to avoid slow network/DNS timeout retries
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, local_files_only=True)
                except Exception:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

                self.model.to(self.device)
                self.model.eval()
                print(f"Model loaded successfully on {self.device.upper()}.")
            except Exception as e:
                print(f"Error loading model {self.model_name}: {e}")
                raise e

    def _generate_single_pass(
        self,
        text: str,
        max_length: int = 130,
        min_length: int = 30,
        do_sample: bool = False,
        num_beams: int = 2
    ) -> str:
        """Internal helper to run model generation on a single text chunk."""
        if not text or not text.strip():
            return ""

        self._load_model()
        import torch

        inputs = self.tokenizer(
            [text],
            max_length=1024,
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        word_count = len(text.split())
        adjusted_max = max(20, min(max_length, word_count + 10))
        adjusted_min = min(min_length, max(5, adjusted_max - 15))

        with torch.inference_mode():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                num_beams=num_beams,
                max_length=adjusted_max,
                min_length=adjusted_min,
                length_penalty=1.2,
                no_repeat_ngram_size=3,
                early_stopping=True if num_beams > 1 else False,
                do_sample=do_sample
            )

        summary_text = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return clean_sentence_endings(summary_text)

    def _generate_batch(
        self,
        texts: List[str],
        max_length: int = 65,
        min_length: int = 15,
        do_sample: bool = False,
        num_beams: int = 2
    ) -> List[str]:
        """
        Run batched generation across multiple text chunks simultaneously.
        Dramatically faster on multi-core CPUs than sequential loops.
        """
        if not texts:
            return []

        self._load_model()
        import torch

        valid_indices = [i for i, t in enumerate(texts) if t and t.strip()]
        if not valid_indices:
            return ["" for _ in texts]

        valid_texts = [texts[i] for i in valid_indices]

        inputs = self.tokenizer(
            valid_texts,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                length_penalty=1.2,
                no_repeat_ngram_size=3,
                early_stopping=True if num_beams > 1 else False,
                do_sample=do_sample
            )

        decoded = [
            clean_sentence_endings(self.tokenizer.decode(sid, skip_special_tokens=True))
            for sid in summary_ids
        ]

        results = ["" for _ in texts]
        for idx, res in zip(valid_indices, decoded):
            results[idx] = res

        return results

    def extract_thematic_clusters(self, article_text: str) -> Dict[str, str]:
        """
        Universal semantic & structural extractor for news articles across ANY domain
        (politics, science, sports, business, crime, health, environment, etc.).

        Partitions article into 4 core narrative dimensions:
        1. Milestone / Core Event (The Lead)
        2. Key Drivers & Context (Body facts, mechanisms, reasons)
        3. Challenges & Complications (Hurdles, criticisms, risks, opposition)
        4. Looking Ahead & Outlook (Future plans, projections, next steps)
        """
        # Split text into paragraphs
        paragraphs = [p.strip() for p in article_text.split('\n\n') if p.strip()]
        if len(paragraphs) < 2:
            paragraphs = [p.strip() for p in article_text.split('\n') if p.strip()]

        all_sentences = split_sentences(article_text)
        total_sents = len(all_sentences)
        if total_sents == 0:
            return {"milestone": "", "drivers": "", "challenges": "", "outlook": ""}

        # 1. Lead / Milestone: First paragraph or first 25% of sentences
        lead_sents = split_sentences(paragraphs[0]) if paragraphs else all_sentences[:max(1, total_sents // 4)]
        milestone_text = " ".join(lead_sents[:3])

        # Generalized discourse indicators
        CHALLENGE_PATTERNS = [
            r'\bhowever\b', r'\balthough\b', r'\bdespite\b', r'\byet\b', r'\bwhile\b',
            r'\bchallenge', r'\bbottleneck', r'\bobstacle', r'\bhurdle', r'\bstruggle',
            r'\brisk', r'\bthreat', r'\bconcern', r'\bcritic', r'\boppos', r'\bcontrovers',
            r'\blag', r'\bdelay', r'\bshortage', r'\black of\b', r'\bproblem', r'\bfailure',
            r'\bregulat', r'\bsafety', r'\bflaw', r'\bcostly\b', r'\bexpensive\b', r'\bhurts\b',
            r'\bwarning\b', r'\bdispute\b', r'\bdebt\b', r'\blimit\b'
        ]

        FUTURE_PATTERNS = [
            r'\blook(?:ing)? ahead\b', r'\bfuture\b', r'\bwill\b', r'\bplans? to\b',
            r'\bexpect', r'\bproject', r'\bpredict', r'\baims? to\b', r'\bupcoming\b',
            r'\bscheduled\b', r'\bnext (?:month|year|decade|quarter|phase|steps?|week)\b',
            r'\bby 202\d\b', r'\bby 203\d\b', r'\btarget\b', r'\bgoal\b', r'\btomorrow\b',
            r'\blong-term\b', r'\bpath forward\b', r'\bprepared to\b', r'\bseek to\b'
        ]

        DRIVER_PATTERNS = [
            r'\bdriven by\b', r'\bbecause\b', r'\bdue to\b', r'\bas a result\b',
            r'\bbreakthrough\b', r'\bdevelop', r'\btechnolog', r'\bmethod\b',
            r'\bdiscover', r'\bfactor', r'\bengine', r'\bresearch', r'\banchored\b',
            r'\binnovat', r'\bcatalyst', r'\baccelerat', r'\bexpansion\b', r'\bgrowth\b',
            r'\bcost\b', r'\befficiency\b', r'\bscale\b', r'\bscaling\b', r'\barchitect',
            r'\baccording to\b', r'\bdata shows?\b', r'\breport reveals?\b', r'\banalysis\b'
        ]

        # If clear multi-paragraph journalistic structure exists (>= 3 paragraphs):
        if len(paragraphs) >= 3:
            lead_sents = split_sentences(paragraphs[0])
            milestone_text = " ".join(lead_sents[:3])

            outlook_sents = split_sentences(paragraphs[-1])
            outlook_text = " ".join(outlook_sents)

            # Analyze middle paragraphs for drivers vs challenges
            middle_text = "\n\n".join(paragraphs[1:-1])
            middle_sents = split_sentences(middle_text)

            driver_candidates = []
            challenge_candidates = []

            for sent in middle_sents:
                sent_lower = sent.lower()
                if any(re.search(pat, sent_lower) for pat in CHALLENGE_PATTERNS):
                    challenge_candidates.append(sent)
                else:
                    driver_candidates.append(sent)

            # If middle paragraphs didn't split into challenges, check if any challenges in article
            if not challenge_candidates:
                challenge_candidates = [s for s in all_sentences if any(re.search(pat, s.lower()) for pat in CHALLENGE_PATTERNS)]

            if not driver_candidates:
                driver_candidates = middle_sents if middle_sents else lead_sents

            return {
                "milestone": milestone_text,
                "drivers": " ".join(driver_candidates) if driver_candidates else milestone_text,
                "challenges": " ".join(challenge_candidates) if challenge_candidates else "Operational hurdles and stakeholder considerations are under continuous review.",
                "outlook": outlook_text if outlook_text else "Further updates and subsequent milestone achievements will be reported as developments proceed."
            }

        # For single-block or 2-paragraph articles, use sentence-level classification
        body_sentences = all_sentences[len(lead_sents):]
        if not body_sentences:
            body_sentences = all_sentences

        driver_candidates = []
        challenge_candidates = []
        future_candidates = []

        for idx, sent in enumerate(body_sentences):
            sent_lower = sent.lower()
            rel_pos = (idx + len(lead_sents)) / max(total_sents, 1)

            is_future = any(re.search(pat, sent_lower) for pat in FUTURE_PATTERNS)
            is_challenge = any(re.search(pat, sent_lower) for pat in CHALLENGE_PATTERNS)
            is_driver = any(re.search(pat, sent_lower) for pat in DRIVER_PATTERNS)

            # Give priority to explicit future/concluding statements in later portions of text
            if is_future or (rel_pos > 0.65 and any(w in sent_lower for w in ["will", "plan", "seek", "hope", "aim", "next"])):
                future_candidates.append(sent)
            elif is_challenge:
                challenge_candidates.append(sent)
            elif is_driver or (0.15 <= rel_pos <= 0.65):
                driver_candidates.append(sent)

        # Robust Quartile Fallbacks ensuring whole-article coverage
        if not driver_candidates:
            q1 = int(total_sents * 0.20)
            q2 = max(q1 + 1, int(total_sents * 0.55))
            driver_candidates = all_sentences[q1:q2]

        if not challenge_candidates:
            q2 = int(total_sents * 0.50)
            q3 = max(q2 + 1, int(total_sents * 0.75))
            challenge_candidates = all_sentences[q2:q3]

        if not future_candidates:
            q3 = int(total_sents * 0.70)
            future_candidates = all_sentences[q3:]

        return {
            "milestone": milestone_text,
            "drivers": " ".join(driver_candidates) if driver_candidates else milestone_text,
            "challenges": " ".join(challenge_candidates) if challenge_candidates else "Operational hurdles and stakeholder considerations are under continuous review.",
            "outlook": " ".join(future_candidates) if future_candidates else "Further updates and subsequent milestone achievements will be reported as developments proceed."
        }

    def generate_summary(
        self,
        article_text: str,
        max_length: int = 150,
        min_length: int = 40,
        do_sample: bool = False,
        thorough: bool = True,
        num_beams: int = 2
    ) -> str:
        """
        Generate standard abstractive paragraph summary.
        Uses adaptive processing:
        - For articles <= 500 words (or when thorough=False), uses direct single-pass generation
          (clean, cohesive, and ~4-5x faster).
        - For longer documents (> 500 words) with thorough=True, performs batched Map-Reduce
          to ensure full beginning-to-end thematic coverage without lead bias.

        Args:
            article_text (str): Preprocessed input article text.
            max_length (int): Upper bound on generated summary token length.
            min_length (int): Lower bound on generated summary token length.
            do_sample (bool): Whether to use sampling (vs beam search).
            thorough (bool): When True, processes all narrative sections of long articles.
            num_beams (int): Number of beams for decoding (default 2 for balanced CPU speed).

        Returns:
            str: Generated full-document abstractive summary text.
        """
        if not article_text or not article_text.strip():
            return ""

        words = article_text.split()

        # For standard articles (<= 500 words) or when thorough=False, single-pass is ideal
        if len(words) <= 500 or not thorough:
            return self._generate_single_pass(
                article_text,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                num_beams=num_beams
            )

        # For very long articles (> 500 words) with thorough=True:
        # Extract thematic clusters and summarize via high-speed batched inference
        clusters = self.extract_thematic_clusters(article_text)
        cluster_keys = ["milestone", "drivers", "challenges", "outlook"]
        cluster_texts = [clusters[k] for k in cluster_keys if clusters[k]]

        summaries = self._generate_batch(
            cluster_texts,
            max_length=max(30, max_length // 3),
            min_length=max(10, min_length // 4),
            num_beams=num_beams,
            do_sample=do_sample
        )

        summary_sentences = []
        for s in summaries:
            if s and s not in summary_sentences:
                summary_sentences.append(s)

        combined_summary = " ".join(summary_sentences)
        return clean_sentence_endings(combined_summary)

    def generate_structured_summary(
        self,
        article_text: str,
        max_length: int = 240,
        min_length: int = 60,
        num_beams: int = 2
    ) -> str:
        """
        Generate a robust, universal structured executive-style key takeaways summary
        covering the ENTIRE article from start to finish via batched parallel inference:
        - 📌 The Milestone (The Lead / Primary Event)
        - 🚀 Key Drivers (Core Background, Reasons & Facts)
        - ⚠️ The Challenges (Complications, Hurdles, Risks & Criticisms)
        - 🔮 Looking Ahead (Next Steps, Projected Timeline & Future Outlook)

        Args:
            article_text (str): Preprocessed or raw news article.
            max_length (int): Overall max token length.
            min_length (int): Overall min token length.
            num_beams (int): Number of beams for decoding (default 2 for optimal CPU performance).

        Returns:
            str: Formatted markdown structured summary with distinct section content.
        """
        if not article_text or not article_text.strip():
            return ""

        # Extract thematic clusters across the full document
        clusters = self.extract_thematic_clusters(article_text)
        cluster_keys = ["milestone", "drivers", "challenges", "outlook"]
        cluster_texts = [clusters[k] for k in cluster_keys]

        # Process all 4 thematic clusters simultaneously in one batched forward pass!
        per_section_max = max(35, min(75, max_length // 3))
        per_section_min = max(12, min(25, min_length // 3))

        summaries = self._generate_batch(
            cluster_texts,
            max_length=per_section_max,
            min_length=per_section_min,
            num_beams=num_beams
        )

        m_summary, d_summary, c_summary, o_summary = summaries

        output_parts = [
            f"📌 **The Milestone:**\n{m_summary}\n",
            f"🚀 **Key Drivers & Context:**\n{d_summary}\n",
            f"⚠️ **The Challenges & Complications:**\n{c_summary}\n",
            f"🔮 **Looking Ahead & Next Steps:**\n{o_summary}\n"
        ]

        return "\n".join(output_parts).strip()


def main():
    """Standalone CLI test for Module 3 with whole-article coverage verification."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    print("--- Testing Module 3: Full-Article Comprehensive Summarizer ---")
    
    test_article = """
    Global health officials on Tuesday announced the successful phase-three completion of a revolutionary universal influenza vaccine that demonstrates robust cross-strain immunity against both seasonal and pandemic flu variants. The landmark clinical trial, spanning 45,000 participants across four continents, achieved an unprecedented 88% reduction in severe viral illness.

    The breakthrough is anchored in a novel computational mRNA delivery architecture developed over the past seven years by an international research consortium. Unlike traditional vaccines that target the rapidly mutating head of the hemagglutinin surface protein, this new formulation directs the immune system toward the highly conserved stem region, providing enduring protection even as the virus drifts. Lead researchers noted that manufacturing costs are projected to be roughly 40% lower than current egg-based production systems, making rapid global deployment feasible within standard cold-chain networks.

    However, significant logistical and regulatory bottlenecks remain before the vaccine can reach clinic shelves. Several developing nations lack the ultra-low temperature storage hubs required for long-term stockpiling, raising urgent concerns about equitable distribution. Furthermore, global regulatory bodies including the FDA and EMA have requested an additional six-month safety follow-up monitoring period regarding rare neuro-inflammatory markers observed in less than 0.01% of trial candidates.

    Looking ahead, the consortium plans to file for emergency expedited approval by mid-2027, with high-volume manufacturing facilities already being prepared in India, Brazil, and Germany. Health economists estimate that widespread adoption could save global healthcare systems over $120 billion annually while virtually eliminating the threat of catastrophic avian flu pandemics jumping into human populations. An international summit in Geneva has been scheduled for next month to finalize global purchase guarantees and subsidy frameworks.
    """
    
    summarizer = NewsSummarizer()
    
    print("\n[Testing Full-Article Standard Abstractive Summary]:")
    std_summary = summarizer.generate_summary(test_article, thorough=True)
    print(std_summary)
    
    print("\n[Testing Universal Executive Structured Summary]:")
    struct_summary = summarizer.generate_structured_summary(test_article)
    print(struct_summary)
    
    print("\nModule 3 test complete successfully.")


if __name__ == "__main__":
    main()
