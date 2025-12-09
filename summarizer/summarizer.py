"""Text summarization using transformers."""

from typing import Optional
import re

import torch
from transformers import pipeline


class Summarizer:
    """Offline text summarizer using BART model."""

    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        """
        Initialize summarizer with specified model.

        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self.pipe: Optional[pipeline] = None

    def _load_model(self) -> None:
        """Lazy load the summarization model (downloads on first use)."""
        if self.pipe is None:
            print(f"📥 Loading summarization model (first run may download ~1.6GB)...")
            self.pipe = pipeline(
                "summarization",
                model=self.model_name,
                device=-1,  # CPU
                framework="pt"  # Force PyTorch
            )
            print("✅ Model loaded successfully")

    def _clean_transcript(self, text: str) -> str:
        """Remove sponsor segments and clean transcript."""
        # Common sponsor phrases to filter out
        sponsor_patterns = [
            r'this video is sponsored by.*?\.',
            r'thanks to.*?for sponsoring.*?\.',
            r'today\'s sponsor is.*?\.',
            r'special thanks to.*?for supporting.*?\.',
            r'coursera sponsored.*?\.',
        ]

        cleaned = text
        for pattern in sponsor_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

        return cleaned.strip()

    def summarize(
        self,
        text: str,
        max_length: int = 250,
        min_length: int = 100
    ) -> str:
        """
        Summarize text using BART model.

        Args:
            text: Input text to summarize
            max_length: Maximum length of summary
            min_length: Minimum length of summary

        Returns:
            Summarized text
        """
        self._load_model()

        # Clean the transcript first
        text = self._clean_transcript(text)

        # Handle empty or very short text
        word_count = len(text.split())
        if word_count < 50:
            return text

        # Warn if transcript is extremely long (>20k words = ~2+ hour video)
        if word_count > 20000:
            print(f"   ⚠️  Very long transcript ({word_count} words) - summarizing first 15,000 words to avoid memory issues")
            text = " ".join(text.split()[:15000])
            word_count = 15000

        # For long texts, use improved chunking strategy
        if word_count > 1024:
            return self._summarize_long_text(text, max_length, min_length)

        # Standard summarization
        try:
            summary = self.pipe(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                truncation=True
            )[0]['summary_text']
            return summary
        except Exception as e:
            print(f"Error in summarization: {e}")
            # Fallback: return first sentences
            sentences = text.split('.')[:5]
            return '. '.join(sentences) + '.'

    def _summarize_long_text(
        self,
        text: str,
        max_length: int,
        min_length: int
    ) -> str:
        """
        Summarize long text using hierarchical summarization.

        Args:
            text: Long input text
            max_length: Maximum length per chunk summary
            min_length: Minimum length per chunk summary

        Returns:
            Combined summary
        """
        # Split into ~800 word chunks (BART's sweet spot)
        chunk_size = 800
        words = text.split()

        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            if len(chunk.split()) > 50:
                chunks.append(chunk)

        print(f"   Processing {len(chunks)} chunks for summarization...")

        # First pass: Summarize each chunk
        chunk_summaries = []
        failed = 0

        for i, chunk in enumerate(chunks):
            try:
                # Generate more detailed summary for this chunk
                summary = self.pipe(
                    chunk,
                    max_length=200,  # Increased from 150
                    min_length=80,   # Increased from 50
                    do_sample=False,
                    truncation=True
                )[0]['summary_text']

                chunk_summaries.append(summary)

                if (i + 1) % 3 == 0:
                    print(f"   Progress: {i + 1}/{len(chunks)} chunks processed")

            except Exception as e:
                failed += 1
                if failed <= 2:  # Only show first 2 errors
                    print(f"   Warning: Chunk {i+1} failed: {str(e)[:80]}")
                continue

        if not chunk_summaries:
            print("   Error: All chunks failed, using extractive fallback")
            return self._extractive_summary(text, max_length)

        print(f"   ✓ Successfully processed {len(chunk_summaries)}/{len(chunks)} chunks")

        # Second pass: Combine and create final summary
        combined = " ".join(chunk_summaries)

        if len(combined.split()) > 400:  # Increased threshold
            try:
                print("   Creating final comprehensive summary...")
                final_summary = self.pipe(
                    combined,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False,
                    truncation=True
                )[0]['summary_text']
                return final_summary
            except Exception as e:
                print(f"   Warning: Final summarization failed, returning combined chunks")
                # Return more of the combined text
                combined_words = combined.split()
                if len(combined_words) > max_length:
                    return " ".join(combined_words[:max_length])
                return combined

        return combined

    def _extractive_summary(self, text: str, max_words: int) -> str:
        """Fallback: Simple extractive summary using first sentences."""
        sentences = text.split('.')
        summary = []
        word_count = 0

        for sentence in sentences:
            words = sentence.split()
            if word_count + len(words) > max_words:
                break
            summary.append(sentence.strip())
            word_count += len(words)

        return '. '.join(summary) + '.'




# """Text summarization using transformers."""

# from typing import Optional
# import re

# import torch
# from transformers import pipeline


# class Summarizer:
#     """Offline text summarizer using BART model."""

#     def __init__(self, model_name: str = "facebook/bart-large-cnn"):
#         """
#         Initialize summarizer with specified model.

#         Args:
#             model_name: HuggingFace model identifier
#         """
#         self.model_name = model_name
#         self.pipe: Optional[pipeline] = None

#     def _load_model(self) -> None:
#         """Lazy load the summarization model (downloads on first use)."""
#         if self.pipe is None:
#             print(f"📥 Loading summarization model (first run may download ~1.6GB)...")
#             self.pipe = pipeline(
#                 "summarization",
#                 model=self.model_name,
#                 device=-1,  # CPU
#                 framework="pt"  # Force PyTorch
#             )
#             print("✅ Model loaded successfully")

#     def _clean_transcript(self, text: str) -> str:
#         """Remove sponsor segments and clean transcript."""
#         # Common sponsor phrases to filter out
#         sponsor_patterns = [
#             r'this video is sponsored by.*?\.',
#             r'thanks to.*?for sponsoring.*?\.',
#             r'today\'s sponsor is.*?\.',
#             r'special thanks to.*?for supporting.*?\.',
#             r'coursera sponsored.*?\.',
#         ]

#         cleaned = text
#         for pattern in sponsor_patterns:
#             cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

#         return cleaned.strip()

#     def summarize(
#         self,
#         text: str,
#         max_length: int = 250,
#         min_length: int = 100
#     ) -> str:
#         """
#         Summarize text using BART model.

#         Args:
#             text: Input text to summarize
#             max_length: Maximum length of summary
#             min_length: Minimum length of summary

#         Returns:
#             Summarized text
#         """
#         self._load_model()

#         # Clean the transcript first
#         text = self._clean_transcript(text)

#         # Handle empty or very short text
#         word_count = len(text.split())
#         if word_count < 50:
#             return text

#         # For long texts, use improved chunking strategy
#         if word_count > 1024:
#             return self._summarize_long_text(text, max_length, min_length)

#         # Standard summarization
#         try:
#             summary = self.pipe(
#                 text,
#                 max_length=max_length,
#                 min_length=min_length,
#                 do_sample=False,
#                 truncation=True
#             )[0]['summary_text']
#             return summary
#         except Exception as e:
#             print(f"Error in summarization: {e}")
#             # Fallback: return first sentences
#             sentences = text.split('.')[:5]
#             return '. '.join(sentences) + '.'

#     def _summarize_long_text(
#         self,
#         text: str,
#         max_length: int,
#         min_length: int
#     ) -> str:
#         """
#         Summarize long text using hierarchical summarization.

#         Args:
#             text: Long input text
#             max_length: Maximum length per chunk summary
#             min_length: Minimum length per chunk summary

#         Returns:
#             Combined summary
#         """
#         # Split into ~800 word chunks (BART's sweet spot)
#         chunk_size = 800
#         words = text.split()

#         chunks = []
#         for i in range(0, len(words), chunk_size):
#             chunk = " ".join(words[i:i + chunk_size])
#             if len(chunk.split()) > 50:
#                 chunks.append(chunk)

#         print(f"   Processing {len(chunks)} chunks for summarization...")

#         # First pass: Summarize each chunk
#         chunk_summaries = []
#         failed = 0

#         for i, chunk in enumerate(chunks):
#             try:
#                 # Generate summary for this chunk
#                 summary = self.pipe(
#                     chunk,
#                     max_length=150,
#                     min_length=50,
#                     do_sample=False,
#                     truncation=True
#                 )[0]['summary_text']

#                 chunk_summaries.append(summary)

#                 if (i + 1) % 3 == 0:
#                     print(f"   Progress: {i + 1}/{len(chunks)} chunks processed")

#             except Exception as e:
#                 failed += 1
#                 if failed <= 2:  # Only show first 2 errors
#                     print(f"   Warning: Chunk {i+1} failed: {str(e)[:80]}")
#                 continue

#         if not chunk_summaries:
#             print("   Error: All chunks failed, using extractive fallback")
#             return self._extractive_summary(text, max_length)

#         print(f"   ✓ Successfully processed {len(chunk_summaries)}/{len(chunks)} chunks")

#         # Second pass: Combine and create final summary
#         combined = " ".join(chunk_summaries)

#         if len(combined.split()) > 300:
#             try:
#                 print("   Creating final comprehensive summary...")
#                 final_summary = self.pipe(
#                     combined,
#                     max_length=max_length,
#                     min_length=min_length,
#                     do_sample=False,
#                     truncation=True
#                 )[0]['summary_text']
#                 return final_summary
#             except Exception as e:
#                 print(f"   Warning: Final summarization failed, returning combined chunks")
#                 return combined

#         return combined

#     def _extractive_summary(self, text: str, max_words: int) -> str:
#         """Fallback: Simple extractive summary using first sentences."""
#         sentences = text.split('.')
#         summary = []
#         word_count = 0

#         for sentence in sentences:
#             words = sentence.split()
#             if word_count + len(words) > max_words:
#                 break
#             summary.append(sentence.strip())
#             word_count += len(words)

#         return '. '.join(summary) + '.'
