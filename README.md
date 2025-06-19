# VesselGPT: Autoregressive Modeling of Vascular Geometry

We introduce an autoregressive method for synthesizing anatomical trees. Our approach first embeds vessel structures into a learned discrete vocabulary using a VQ-VAE architecture, then models their generation autoregressively with a GPT-2 model. This method effectively captures intricate geometries and branching patterns, enabling realistic vascular tree synthesis.
Comprehensive qualitative and quantitative evaluations reveal that our technique achieves high-fidelity tree reconstruction with compact discrete representations. Moreover, our B-spline representation of vessel cross-sections preserves critical morphological details that are often overlooked in previous' methods parameterizations. To the best of our knowledge, this work is the first to generate blood vessels in an autoregressive manner.

![teaser](Fig1.png)

[Full Paper](https://arxiv.org/pdf/2505.13318v1)


# Code 
...