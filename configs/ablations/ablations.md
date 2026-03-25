configs/experiment/ablations/
├── Phase 1: Projector (frozen LLM, ASR, 50k steps each)
│   ├── p1a_conv1d.yaml
│   ├── p1b_concat.yaml
│   ├── p1c_mlp.yaml
│   ├── p1d_transformer.yaml
│   └── p1e_qformer.yaml
├── Phase 2: CTC compression (frozen LLM, ASR, 50k steps)
│   ├── p2a_no_compress.yaml        # baseline
│   ├── p2b_ctc_avg.yaml            # avg + conv1d
│   ├── p2c_ctc_weighted.yaml       # weighted + conv1d
│   └── p2d_ctc_avg_mlp.yaml        # avg + mlp (CTC does all compression)
├── Phase 3: LLM training (ASR, 50k steps)
│   ├── p3a_frozen.yaml             # no LoRA
│   ├── p3b_lora_r8.yaml
│   ├── p3c_lora_r16.yaml
│   └── p3d_lora_r32.yaml
├── Phase 4: Encoder scale (LoRA r=16, ASR, 50k steps)
│   ├── p4a_encoder_base.yaml       # 73M
│   └── p4b_encoder_large.yaml      # 250M
└── Phase 5: Task / curriculum (LoRA r=16, 100k steps)
    ├── p5a_asr_only.yaml
    ├── p5b_st_only.yaml
    ├── p5c_cot_scratch.yaml
    └── p5d_curriculum_cot.yaml      # resume from p5a