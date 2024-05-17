# hyperseg
Code for 
-
- [Unsupervised Dialogue Topic Segmentation in Hyperdimensional Space](https://arxiv.org/abs/2308.10464) (Interspeech 2023)
- [Unsupervised Extractive Dialogue Summarization In Hyperdimensional Space](https://arxiv.org/abs/2405.09765) (ICASSP 2024)


HyperSeg:
```python
segmenter = HyperSegSegmenter()
res = segmenter.segment(["sentence 1", "sentence 2"])
```

HyperSum:
```python
summarizer = Summarizer()
res = summarizer.summarize(["sentence 1", "sentence 2"])
```
