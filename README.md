# Face-Generation-from-Speech
11-785 Project (Spring 19)

-------------------
## Imp Links
- https://blog.insightdatascience.com/generating-custom-photo-realistic-faces-using-ai-d170b1b59255
- https://medium.com/@animeshsk3/t2f-text-to-face-generation-using-deep-learning-b3b6ba5a5a93

#### Datasets:
VGGFace2, Voxceleb2, Voxceleb1 (Used only for X-Vector training)


This work uses X-Vector Speaker Embeddings, with Deepsphere face Embeddings to train a joint embedding network using the N-Pair Loss.
The obtained embeddings are used to generate face images conditioned on provided speaker embeddings shifted to a joint embedding space.


##Code Elements
1. Face Embedding Extraction from Pre-trained DeepSphere Model
2. Kaldi VoxCeleb X-Vector Extraction
3. Joint Embedding Network using MLP
4. Conditional DC GAN for Image Synthesis with Scaling Loss
