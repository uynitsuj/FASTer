Custom implementation for DCT (Discrete Cosine Transform) early stopping technique to decode only the first few DCT coefficients from PaLi-Gemma rather than waiting for a full action chunk inference to complete. This can cut down inference to action time by roughly half since around the first 3-4 frequency coefficients of the DCT are typically enough for a coarse action token reconstruction.

# FAST: Efficient Action Tokenization for Vision-Language-Action Models

This is a modified repo for the [FAST action tokenizer](https://www.pi.website/research/fast).

The action tokenizer maps any sequence of robot actions into a sequence of dense, discrete **action tokens** for training autoregressive VLA models.
