"""
data_chunk_processor.py – Scalable Complaint Data Chunk Loader & Cleaner (B5W6)
------------------------------------------------------------------------------
Processes large-scale complaint datasets safely in chunks and provides:
  • Filtering by relevant products
  • Removal of missing or empty narratives
  • Minimal text cleaning for embedding readiness
  • Detailed diagnostics on reasons for row drops
  • Option to return the cleaned DataFrame

Author: Nabil Mohamed
"""

import os
import re
import pandas as pd
from tqdm import tqdm


class ComplaintChunkProcessor:
    """
    Processes large CFPB complaint datasets safely in chunks with robust filtering, cleaning, and diagnostics.
    """

    def __init__(self, filepath: str, output_path: str, chunk_size: int = 100_000):
        if not isinstance(filepath, str) or not isinstance(output_path, str):
            raise TypeError("❌ Both filepath and output_path must be strings.")
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"❌ Cannot find file at: {filepath}")

        self.filepath = filepath
        self.output_path = output_path
        self.chunk_size = chunk_size
        self.allowed_products = [
            "Credit card",
            "Personal loan",
            "Buy Now, Pay Later",
            "Savings account",
            "Money transfer, virtual currency",
        ]

        # Track granular drop reasons
        self.stats = {
            "chunks_processed": 0,
            "rows_loaded": 0,
            "rows_kept": 0,
            "rows_dropped_total": 0,
            "rows_dropped_no_narrative": 0,
            "rows_dropped_wrong_product": 0,
            "products_found": {},
        }

        self.cleaned_df = None  # Store final DataFrame

    def clean_text(self, text: str) -> str:
        if pd.isna(text):
            return ""
        text = re.sub(r"\s+", " ", str(text)).strip()
        text = re.sub(r'[^A-Za-z0-9.,!?\'" ]+', "", text)
        return text.lower()

    def process_chunks(self, return_dataframe: bool = True) -> pd.DataFrame:
        """
        Processes complaint data in chunks, saves cleaned CSV, and optionally returns DataFrame.

        Args:
            return_dataframe (bool): Whether to return cleaned DataFrame (default = True).

        Returns:
            pd.DataFrame: Cleaned complaint dataset (optional).
        """
        filtered_data = []

        try:
            for chunk in tqdm(
                pd.read_csv(self.filepath, chunksize=self.chunk_size),
                desc="🚀 Processing Chunks",
            ):
                self.stats["chunks_processed"] += 1
                self.stats["rows_loaded"] += len(chunk)

                # Track pre-filter size
                original_chunk_size = len(chunk)

                # Filter 1: Allowed products
                chunk = chunk[chunk["Product"].isin(self.allowed_products)]
                dropped_wrong_product = original_chunk_size - len(chunk)
                self.stats["rows_dropped_wrong_product"] += dropped_wrong_product

                # Filter 2: Valid narratives
                pre_narrative_size = len(chunk)
                chunk = chunk.dropna(subset=["Consumer complaint narrative"])
                chunk = chunk[chunk["Consumer complaint narrative"].str.strip() != ""]
                dropped_no_narrative = pre_narrative_size - len(chunk)
                self.stats["rows_dropped_no_narrative"] += dropped_no_narrative

                # Clean narratives
                chunk["Consumer complaint narrative"] = chunk[
                    "Consumer complaint narrative"
                ].apply(self.clean_text)

                # Update product counts
                product_counts = chunk["Product"].value_counts().to_dict()
                for product, count in product_counts.items():
                    self.stats["products_found"][product] = (
                        self.stats["products_found"].get(product, 0) + count
                    )

                self.stats["rows_kept"] += len(chunk)
                filtered_data.append(chunk)

        except pd.errors.ParserError as e:
            raise RuntimeError(f"❌ CSV parsing failed: {e}")
        except Exception as e:
            raise RuntimeError(f"❌ Unexpected error while processing chunks: {e}")

        if filtered_data:
            try:
                self.cleaned_df = pd.concat(filtered_data, ignore_index=True)
                self.cleaned_df.to_csv(self.output_path, index=False)

                self.stats["rows_dropped_total"] = (
                    self.stats["rows_loaded"] - self.stats["rows_kept"]
                )

                self._print_summary()

                if return_dataframe:
                    return self.cleaned_df

            except Exception as e:
                raise RuntimeError(f"❌ Failed to save cleaned dataset: {e}")
        else:
            print("⚠️ No matching complaints found. No file saved.")
            return pd.DataFrame()  # Return empty DataFrame safely

    def _print_summary(self):
        """
        Prints detailed, engaging summary of the filtering process.
        """
        print("\n✅ Complaint Data Cleaning Complete")
        print(f"• Chunks processed:           {self.stats['chunks_processed']:,}")
        print(f"• Total rows loaded:          {self.stats['rows_loaded']:,}")
        print(f"• Rows retained (clean):      {self.stats['rows_kept']:,}")
        print(f"• Rows dropped (total):       {self.stats['rows_dropped_total']:,}")

        print("\n❌ Drop Reasons:")
        print(
            f"• Rows dropped (wrong product):     {self.stats['rows_dropped_wrong_product']:,}"
        )
        print(
            f"• Rows dropped (no narrative):      {self.stats['rows_dropped_no_narrative']:,}"
        )

        print("\n📊 Product Distribution (Filtered):")
        for product, count in self.stats["products_found"].items():
            print(f"• {product}: {count:,} complaints")

        print(f"\n📄 Cleaned dataset saved to: {self.output_path}")
