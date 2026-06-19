import sys

class Preprocess:
    def __init__(self):
        print("[DEBUG] Preprocess __init__ called", file=sys.stderr, flush=True)
        self.model = None

    def load(self, local_file_name):
        print(f"[DEBUG] load() called with file: {local_file_name}", file=sys.stderr, flush=True)
        import joblib
        self.model = joblib.load(local_file_name)
        print("[DEBUG] Model loaded successfully", file=sys.stderr, flush=True)
        return self.model

    def preprocess(self, body, state, collect_custom_statistics_fn=None):
        text = body.get("text", "")
        if isinstance(text, str):
            text = [text]
        return list(text)

    def process(self, data, state, collect_custom_statistics_fn=None):
        preds = self.model.predict(data)
        proba = self.model.predict_proba(data)
        return preds, proba

    def postprocess(self, data, state, collect_custom_statistics_fn=None):
        preds, proba = data
        classes = [str(c) for c in self.model.classes_]
        out = []
        for i, label in enumerate(preds):
            row = proba[i]
            out.append({
                "label": str(label),
                "score": float(max(row)),
                "scores": {c: float(p) for c, p in zip(classes, row)},
            })
        return {"predictions": out}