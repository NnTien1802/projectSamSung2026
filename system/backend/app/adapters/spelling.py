"""M1 inference; loads the class and collator from the ORIGINAL locked evaluation source.

Input must be `article_segmented`, like the training dataset. This adapter does
NOT silently segment raw Vietnamese, highlight word spans, or correct mistakes.
"""
from contextlib import nullcontext
from ._source import ROOT, import_original


class SpellingAdapter:
    def __init__(self, device=None):
        import torch
        from transformers import AutoTokenizer
        from json import loads

        source = ROOT / "module1" / "scripts" / "16_evaluate_phobert_test.py"
        original = import_original(source, "_project_samsung_original_m1_eval_v1")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model_dir = ROOT / "module1" / "models" / "module1_phobert_mil_full"
        checkpoint_path = model_dir / "best_model.pt"
        threshold_path = ROOT / "module1" / "results" / "module1_phobert_mil_full" / "threshold_tuning" / "best_threshold.json"
        # Trusted local checkpoint, identical loading policy to original evaluation.
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:  # older torch
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        threshold_data = loads(threshold_path.read_text(encoding="utf-8"))
        for key in ("model_name", "text_field", "max_length", "stride", "max_chunks_per_document", "state_dict"):
            if key not in checkpoint:
                raise RuntimeError(f"M1 checkpoint missing {key}")
        locked = ("vinai/phobert-base-v2", "article_segmented", 256, 64, 3)
        actual = (checkpoint["model_name"], checkpoint["text_field"], int(checkpoint["max_length"]),
                  int(checkpoint["stride"]), int(checkpoint["max_chunks_per_document"]))
        if actual != locked:
            raise RuntimeError(f"M1 checkpoint configuration mismatch: {actual}")
        self.threshold = float(threshold_data["selected_threshold"])
        if abs(self.threshold - 0.57) > 1e-9:
            raise RuntimeError(f"M1 threshold mismatch: {self.threshold}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer", use_fast=False)
        self.collator = original.DocumentCollator(
            self.tokenizer, max_length=256, stride=64, max_chunks=3
        )
        self.model = original.PhoBERTMIL(checkpoint["model_name"])
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.to(self.device)
        self.model.eval()
        self.torch = torch

    def predict(self, article_segmented: str) -> dict:
        if not isinstance(article_segmented, str) or not article_segmented.strip():
            raise ValueError("article_segmented is required (preserve underscores from word segmentation)")
        # Reuse evaluation collator exactly. Metadata is irrelevant for inference,
        # but original collator expects all of these fields.
        row = {
            "text": article_segmented.strip(), "label": 0, "source_id": "online",
            "variant_id": "online", "severity": "unknown", "error_count": 0,
            "error_types": [], "error_sources": [],
        }
        batch = self.collator([row])
        torch = self.torch
        with torch.inference_mode():
            amp = torch.amp.autocast(device_type="cuda", dtype=torch.float16) if self.device.type == "cuda" else nullcontext()
            with amp:
                logits = self.model(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                    doc_indices=batch["doc_indices"].to(self.device),
                    num_documents=1,
                )
            probability = float(torch.sigmoid(logits).float().cpu()[0].item())
        pred = int(probability >= self.threshold)
        return {
            "probability_error": probability, "threshold": self.threshold,
            "prediction": pred, "label": "HAS_ERROR" if pred else "NO_ERROR",
            "chunk_count": int(batch["input_ids"].shape[0]),
            "warnings": ["document_level_only: does not identify or fix individual words"],
        }
