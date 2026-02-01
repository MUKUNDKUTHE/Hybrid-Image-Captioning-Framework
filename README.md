Hybrid Image Captioning Framework

A hybrid image captioning framework that integrates the BLIP-2 vision-language model with a large language model (LLM) to generate image captions. The system explores different parameter settings and improves caption quality through BLEU-based refining and optimization.

Features

Initial caption generation using BLIP-2 for visual feature extraction
Caption refinement with an LLM
Parameter exploration for quality improvement
BLEU score-based optimization for better captions
Batch caption generation support

Tech Stack
Python, BLIP-2 (vision-language model), LLM integration, BLEU evaluation metrics, Transformers

How It Works

Image is preprocessed for model input
BLIP-2 generates an initial caption
Initial caption is fed to an LLM for refinement
BLEU score is computed to evaluate caption quality
Parameters are tuned to optimize BLEU score
