# TEA (Test-time Embedding Adjustment) Integration Guide

## Overview

This is a guide to integrate TEA into the ReVersion pipeline. For installing ReVersion, please refer to the [ReVersion README](https://github.com/ziqihuangg/ReVersion).

TEA (Test-time Embedding Adjustment) is a technique that improves subject personalization by adjusting text embeddings at inference time. It works by interpolating between the original prompt embeddings and target prompt embeddings using spherical linear interpolation (SLERP) and norm adjustment.

The qualitative results can be found in the `evaluation_massive` folder. 

![Results without TEA](evaluation_massive/carved_by/cat%20%3CR%3E%20carrot%20in%20the%20garden.png)
*ReVersion: Cat carved by carrot in the garden*

![Results with TEA](evaluation_massive/carved_by_tea/cat%20%3CR%3E%20carrot%20in%20the%20garden.png)
*ReVersion with TEA: Cat carved by carrot in the garden*

![Results without TEA](evaluation_massive/carved_by/dog%20%3CR%3E%20paper%20in%20the%20library.png)
*ReVersion: Dog carved by paper in the library*

![Results with TEA](evaluation_massive/carved_by_tea/dog%20%3CR%3E%20paper%20in%20the%20library.png)
*ReVersion with TEA: Dog carved by paper in the library*

![Results without TEA](evaluation_massive/carved_by/dog%20%3CR%3E%20glass%20in%20the%20park.png)
*ReVersion: Dog carved by glass in the park*

![Results with TEA](evaluation_massive/carved_by_tea/dog%20%3CR%3E%20glass%20in%20the%20park.png)
*ReVersion with TEA: Dog carved by glass in the park*

![Results without TEA](evaluation_massive/carved_by/dog%20%3CR%3E%20jade%20in%20the%20desert.png)
*ReVersion: Dog carved by jade in the desert*

![Results with TEA](evaluation_massive/carved_by_tea/dog%20%3CR%3E%20jade%20in%20the%20desert.png)
*ReVersion with TEA: Dog carved by jade in the desert*

## Reproduce the results

After installing ReVersion and download the pretrained models in their repository, you can reproduce the results by running the following commands:

```bash
bash gen_massive.sh
```

The script includes the inference script to generate the images with the trained personalization model and the TEA-enabled pipeline (see the `inference.py` and `inference_tea.py` scripts).

The scripts also includes the evaluation script to calculate the CLIP alignment score. The results have been run on A100 80G GPU. 

## Core Concept

TEA addresses the challenge of subject personalization by:
1. **Direction Adjustment**: Using SLERP to smoothly interpolate between prompt directions
2. **Magnitude Control**: Adjusting embedding norms to balance between original and target characteristics
3. **Test-time Adaptation**: No additional training required - works at inference time
