# Transfer Knowledge from CLIP to Lightweight CNN model using VL2Lite architecture
[VL2Lite](https://github.com/jsjangAI/VL2Lite) introduced by Jang et al. 2025<sup>1</sup> is a Vision-Language Knowledge Distillation framework designed to transfer rich, multi-modal semantic knowledge from large-scale foundation models into lightweight, compute-efficient convolutional vision backbones.

                  ┌───────────────────────────────┐
                  │   Teacher: CLIP (ViT-B/32)    │  [Frozen]
                  └───────┬──────────────┬────────┘
                          │              │
       Visual Embedding z_t              │ Text Embeddings T
                          │              │ ("a photo of a {class}")
                          ▼              ▼
                     ┌─────────┐   ┌───────────┐
                     │ MSE Loss│   │  KL-Div   │
                     └────▲────┘   └─────▲─────┘
                          │              │
       Projected Embed z_s│              │ Logit Similarities
                          │              │
                  ┌───────┴──────────────┴────────┐
                  │    Connector (MLP Head)       │
                  └───────────────▲───────────────┘
                                  │ Features
                  ┌───────────────┴───────────────┐
                  │  Student: ResNet (18/34/50)   │ ──► Task Loss (Cross-Entropy)
                  └───────────────────────────────┘






Standard knowledge distillation traditionally distills classification logits from a unimodal vision teacher. In contrast, VL2Lite leverages both visual features and linguistic semantics from a dual-encoder model (CLIP):

1. *Teacher Backbone:* A pre-trained, frozen CLIP model (openai/clip-vit-base-patch32).
2. *Student Backbone:* Lightweight convolutional architectures (ResNet-18, ResNet-34, or ResNet-50).
3. *Connector (Projection Head):* A 2-layer Multi-Layer Perceptron (MLP) that maps the student's internal feature representation into CLIP’s 512-dimensional metric space.
4. *Text Anchors:* Pre-computed, fixed text embeddings of target class labels generated through prompt engineering ("a photo of a {class}").

Mathematical Formulation & Training MechanicsThe distillation process is governed by a tripartite loss function modulated by a Dynamic Linear Ramp Schedule.
1. Classification Task Loss ($L_{\text{task}}$)Standard multi-class Cross-Entropy on the student's direct task predictions:

   $$
   L_{\text{task}} = -\sum_{c=1}^{C} y_c \log(\hat{y}_c)
   $$

4. Visual Distillation Loss ($L_{\text{visual}}$)Forces the projected student visual embeddings ($\hat{z}_{\text{student}}$) to align with the teacher's visual embeddings ($z_{\text{teacher}}$) using Mean Squared Error (MSE):

   $$
   L_{\text{visual}} = \Vert{}\hat{z}_{\text{student}} - z_{\text{teacher}}\Vert{}_2^2
   $$

6. Linguistic Distillation Loss ($L_{\text{text}}$)Calculates cosine similarity logits between visual embeddings and all class text anchors $T$, scaled by temperature parameter $\tau = 0.07$. The Kullback-Leibler (KL) Divergence minimizes the distance between student and teacher similarity distributions:$$P_{\text{student}} = \text{softmax}\left(\frac{\hat{z}_{\text{student}} T^\top}{\tau}\right), \quad P_{\text{teacher}} = \text{softmax}\left(\frac{z_{\text{teacher}} T^\top}{\tau}\right)$$$$L_{\text{text}} = \tau^2 \cdot D_{\text{KL}}(P_{\text{student}} \parallel P_{\text{teacher}})$$


7. Dynamic Linear Ramp Weighting ScheduleDuring early epochs, the student relies heavily on multi-modal distillation to structure its feature manifold. As training progresses through the ramp ratio $r = 0.125$ (12.5% of total epochs), distillation weight fades and the direct task loss takes full precedence:$$w_{\text{task}}(t) = \min\left(1.0, \max\left(0.0, \frac{t}{T \cdot r}\right)\right)$$

## Reference
1. J. Jang, C. Ma and B. Lee, "[VL2Lite: Task-Specific Knowledge Distillation from Large Vision-Language Models to Lightweight Networks](https://openaccess.thecvf.com/content/CVPR2025/papers/Jang_VL2Lite_Task-Specific_Knowledge_Distillation_from_Large_Vision-Language_Models_to_Lightweight_CVPR_2025_paper.pdf)," 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Nashville, TN, USA, 2025, pp. 30073-30083, doi: 10.1109/CVPR52734.2025.02799.
