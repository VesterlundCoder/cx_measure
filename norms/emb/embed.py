"""Sentence embeddings with an ONNX sentence-transformer (mean pooling, L2-normalised)."""
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


class Embedder:
    def __init__(self, model_path, tokenizer_path, max_len=128, pooling="mean"):
        self.pooling = pooling
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(tokenizer_path)
        self.tok.enable_truncation(max_len)
        self.tok.enable_padding()
        self.inputs = [i.name for i in self.sess.get_inputs()]

    def encode(self, texts, batch=64):
        out = []
        for i in range(0, len(texts), batch):
            enc = self.tok.encode_batch(list(texts[i:i + batch]))
            ids = np.array([e.ids for e in enc], np.int64)
            mask = np.array([e.attention_mask for e in enc], np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = np.zeros_like(ids)
            h = self.sess.run(None, feed)[0]
            if self.pooling == "cls":
                v = h[:, 0]
            else:
                m = mask[..., None].astype(np.float32)
                v = (h * m).sum(1) / m.sum(1)
            out.append(v / np.linalg.norm(v, axis=1, keepdims=True))
        return np.vstack(out)
