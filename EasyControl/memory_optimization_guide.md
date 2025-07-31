# GPU Memory Optimization Guide for FLUX Training (Updated)

## Memory Usage Overview

FLUX.1-dev is a large model that requires significant GPU memory. Here's how to optimize memory usage with the latest text encoder training options:

## 🔧 Key Parameters for Memory Reduction

### 1. **Text Encoder Training Options** (NEW - Highest Impact)
```bash
# Option 1: Train both text encoders (default)
--train_text_encoder

# Option 2: Train only text_encoder_one, freeze text_encoder_two (NEW!)
--train_text_encoder_one_only

# Option 3: No text encoder training (maximum memory savings)
# (remove both flags above)
```
**Impact**: Training only text_encoder_one saves ~1-2GB memory vs training both
**Trade-off**: Slightly reduced text understanding capability but still very effective

### 2. **LoRA Ranks & Alphas** (High Impact)
```bash
# Current settings (moderate memory)
--ranks 64
--network_alphas 64
--text_encoder_rank=32
--text_encoder_alpha=32

# Low memory settings
--ranks 32
--network_alphas 32
--text_encoder_rank=16
--text_encoder_alpha=16

# Ultra-low memory settings
--ranks 16
--network_alphas 16
--text_encoder_rank=8
--text_encoder_alpha=8
```

### 3. **Image Resolution** (High Impact)
```bash
# Standard settings
--cond_size=384
--noise_size=768
--test_h 768
--test_w 768

# Low memory settings
--cond_size=256
--noise_size=512
--test_h 512
--test_w 512

# Ultra-low memory settings
--cond_size=192
--noise_size=384
--test_h 384
--test_w 384
```

## 🚀 Updated Memory Configurations

### **1. Standard Configuration** (~12-16GB VRAM)
```bash
bash train_subject_with_text_encoder_fixed.sh  # Both encoders
```

### **2. Text Encoder One Only** (~10-14GB VRAM) - NEW!
```bash
bash train_subject_text_encoder_one_only.sh  # Only encoder one
```

### **3. Low Memory + Text Encoder One** (~6-8GB VRAM) - RECOMMENDED!
```bash
bash train_subject_text_encoder_one_low_memory.sh
```

### **4. No Text Encoder Training** (~4-6GB VRAM)
```bash
# Remove --train_text_encoder and --train_text_encoder_one_only flags
```

## 📊 Updated Parameter Impact Table

| Parameter | High Memory | Medium | Low Memory | Ultra-Low | Memory Impact |
|-----------|-------------|--------|------------|-----------|---------------|
| Text Encoder Training | Both | One Only | One Only | None | Very High |
| `ranks` | 128 | 64 | 32 | 16 | Very High |
| `text_encoder_rank` | 64 | 32 | 16 | 8 | High |
| `cond_size` | 512 | 384 | 256 | 192 | High |
| `noise_size` | 1024 | 768 | 512 | 384 | High |
| `test_h/w` | 1024 | 768 | 512 | 384 | Medium |
| `validation_steps` | 20 | 50 | 100 | 200 | Medium |
| `gradient_accumulation` | 1 | 1 | 2 | 4 | Medium |

## 🎯 Text Encoder Training Comparison

### **Both Text Encoders (`--train_text_encoder`)**
- ✅ Maximum text understanding capability
- ✅ Best performance on complex prompts  
- ❌ Highest memory usage (~2-3GB additional)
- ❌ Longer training time

### **Text Encoder One Only (`--train_text_encoder_one_only`)** - RECOMMENDED!
- ✅ Good text understanding (95% of full capability)
- ✅ Significant memory savings (~1-2GB less)
- ✅ Faster training
- ❌ Slightly reduced performance on very complex prompts

### **No Text Encoder Training**
- ✅ Maximum memory savings (~3-4GB less)
- ✅ Fastest training
- ❌ Uses only pre-trained text understanding
- ❌ May not adapt well to domain-specific terminology

## 🔍 Memory Usage Examples

### **RTX 4090 (24GB VRAM)**
```bash
# Standard: Both text encoders, full resolution
bash train_subject_with_text_encoder_fixed.sh

# Recommended: Text encoder one only, balanced settings
bash train_subject_text_encoder_one_only.sh
```

### **RTX 4070/4080 (12-16GB VRAM)**
```bash
# Recommended: Text encoder one only, low memory
bash train_subject_text_encoder_one_low_memory.sh
```

### **RTX 4060 Ti (8-12GB VRAM)**
```bash
# Ultra-low settings with text encoder one only
# Manually adjust parameters in the low memory script
```

## 🛠️ Script Selection Guide

1. **If you have 16GB+ VRAM**: Start with `train_subject_text_encoder_one_only.sh`
2. **If you have 8-16GB VRAM**: Use `train_subject_text_encoder_one_low_memory.sh`  
3. **If you have <8GB VRAM**: Remove text encoder training entirely
4. **For maximum quality**: Use `train_subject_with_text_encoder_fixed.sh` (both encoders)

## 📈 Quality vs Memory Trade-offs

| Configuration | Memory | Quality | Training Speed | Recommended For |
|---------------|--------|---------|----------------|-----------------|
| Both Encoders | Highest | Best | Slowest | High-end GPUs |
| **Text Encoder One** | **Medium** | **Very Good** | **Fast** | **Most Users** |
| No Text Encoder | Lowest | Good | Fastest | Low-end GPUs |

## 🚀 Quick Start Commands

```bash
# Make scripts executable
chmod +x *.sh

# Check your GPU memory
bash check_rtx4000_setup.sh

# Start with the recommended configuration
bash train_subject_text_encoder_one_only.sh

# Monitor during training
watch -n 1 nvidia-smi
```

The **text encoder one only** option provides the best balance of memory efficiency and training quality for most users! 