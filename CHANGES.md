# DUO-Anchor: Tài liệu thay đổi & Phân tích

Repository này mở rộng paper **"Direct Unlearning Optimization for Robust and Safe Text-to-Image Models"** (DUO, NeurIPS 2024, arXiv 2407.21035) với 1 cải tiến nhằm giải quyết trực tiếp hạn chế mà tác giả thừa nhận trong Section 5 về vấn đề "unrelated concepts that share excessively similar visual features may be influenced by unlearning".

Ngoài ra bổ sung bộ đánh giá white-box attack (Concept Inversion, paper Sec 4.1) để so sánh trực tiếp giữa DUO gốc và DUO-Anchor.

---

## 1. Bối cảnh: DUO gốc hoạt động thế nào?

### 1.1 Bài toán
Unlearning trong T2I model: xóa khái niệm độc hại (vd "naked", "blood") khỏi Stable Diffusion 1.4 mà vẫn giữ chất lượng sinh ảnh cho các concept khác.

### 1.2 Phương pháp DUO (paper Section 3)
1. **Tạo paired dataset bằng SDEdit** (Sec 3.2): cho mỗi unsafe image `x⁻`, dùng SDEdit với negative guidance tạo `x⁺` giữ nguyên background, da, bố cục, chỉ thay đổi concept độc hại. → Cặp `(x⁺, x⁻)` cho mỗi concept.
2. **Preference Optimization (Sec 3.3)**: áp dụng Diffusion-DPO để mô hình **thích** `x⁺` hơn `x⁻`. Loss:
   ```
   L_DPO = -E[log σ(β·(||ε-ε_θ(x⁻_t)||² - ||ε-ε_θ(x⁺_t)||²))]
   ```
   trong đó `ε_θ` là mô hình có LoRA đang train, còn `ε_φ` (reference) là bản gốc không LoRA.
3. **Output-preservation regularization (Sec 3.4)**:
   ```
   L_prior = ||ε_φ(x_T) - ε_θ(x_T)||²   (chỉ tại t=T)
   ```
   Tác giả giới hạn **chỉ tại `t=T`** vì sợ rằng bảo tồn ở nhiều timestep sẽ cản trở việc xóa feature độc hại ở `x₀` (Sec 3.4 dòng 395-397).

### 1.3 Decomposition cho Violence
Sec 4.1: tác giả tách "Violence" thành 4 concept con (Blood, Suffer, Gun, Horror), train 4 LoRA riêng, merge bằng **cộng đều** (`set_adapters([...], weights=[1,1,1,1])` — xem `inference.py:58`).

### 1.4 Hạn chế DUO thừa nhận (Section 5, dòng 627-630)
> "Since our method involves unlearning visual features, **unrelated concepts that share excessively similar visual features may be influenced by unlearning**. We anticipate that this issue could be addressed by curating paired datasets that include similar concepts, but we leave this as future work."

Tóm lại DUO chưa giải quyết 2 vấn đề:
- **H1 (Selectivity)**: unlearn "Blood" có thể làm hỏng ảnh "ketchup", "tomato sauce", "red wine" (vì cùng đặc trưng thị giác màu đỏ).
- **H2 (Compositional Robustness)**: 4 LoRA con cho Violence được merge bằng cộng đều — không tối ưu, dễ under- hoặc over-defense ở từng tiểu concept.

---

## 2. Cải tiến của repo này

### 2.1 Idea 1 — Anchor-based Retention (giải quyết H1)

**Ý tưởng cốt lõi**: bổ sung một loss mới `L_retain` giữ cho LoRA unlearned **không thay đổi** hành vi của mô hình trên một tập "anchor" gồm các concept **an toàn nhưng có đặc trưng thị giác tương tự** concept độc hại.

**Công thức**:
```
L_total = L_DPO + λ_prior · L_prior(t=T)   +   λ_anchor · L_retain(t ~ U[t_min, t_max])

với  L_retain = E_{x_anc ~ Anchor, t~U[t_min,t_max]} [ ||ε - ε_θ(x_anc_t, t)||² ]
```

So với DUO gốc:
| Thành phần | DUO gốc | DUO-Anchor |
|---|---|---|
| Loss chính | DPO trên `(x⁺, x⁻)` | Giữ nguyên |
| Prior preservation | `t=T` (một mức nhiễu) | `t=T` (giữ nguyên) |
| **Anchor retention** | ❌ không có | `t ~ U[1, 750]` (đa mức nhiễu) |

**Tại sao multi-timestep?**
- `L_prior` gốc ở `t=T` chỉ bảo vệ "high-level distribution" (ảnh toàn noise).
- Ở `t` thấp (vd `t=50`), ảnh gần sạch → gradient bám sát chi tiết vật thể (màu sắc, texture).
- Bảo vệ ở **nhiều mức nhiễu** = bảo vệ cả cấu trúc lẫn chi tiết.

**Cách xây dựng Anchor set** (`datasets/SD/generate_anchors.py`):
- Với mỗi concept, dùng **chính SD 1.4 gốc** sinh ảnh cho 8 prompt "an toàn" có đặc trưng thị giác gần:
  - `Blood` → ketchup, tomato sauce, strawberry jam, red wine, pomegranate, spaghetti bolognese, fruit punch, red paint
  - `Gun` → wrench, toolbox, backpack, camera+với lens, violin, kitchen knife, tennis racket, microphone
  - `Horror` → dark forest, abandoned house, foggy graveyard, shadowy figure, skull, haunted mansion, black cat, storm+castle
  - `Suffer` → crying baby, person in pain, headache, exhausted runner, sad child (chưa thêm trong config.json gốc, cần bổ sung)
  - `Nudity` → person in swimsuit, workout clothes, summer dress, business attire, pajamas, winter coat, kimono, wedding dress
- Lưu 4 ảnh/prompt (mặc định) → 32 ảnh/concept.

**Đánh giá**: LPIPS giữa ảnh anchor sinh từ SD gốc vs SD-unlearned.
- LPIPS thấp hơn = unlearn tốt hơn (giữ được đặc trưng anchor).
- So sánh: `L_baseline` (DUO gốc) vs `L_ours` (DUO + anchor).

### 2.2 Idea 2 — BỎ (Task Arithmetic)

Đã cân nhắc nhưng loại bỏ vì:
- **Novelty thấp**: DUO paper đã có decomposition + merge đều; Task Arithmetic chỉ đổi trọng số.
- **Compute không kham nổi**: 5⁴=625 combinations × 80 ảnh = ~28 giờ inference P100, ăn vào thời gian chạy evaluation chính.
- **Ablation đủ rồi**: so sánh DUO gốc vs DUO-Anchor đã là 2 điểm trên Pareto curve; thêm merge search phân tán narrative.

Code `train/merge_loras.py` cũ đã xóa. Inference giữ merge 4 sub-LoRA bằng cộng đều `[1,1,1,1]` (giống paper). Đánh dấu là future work trong paper.

### 2.3 Đánh giá white-box attack (Concept Inversion)

Paper DUO đánh giá Concept Inversion ở Sec 4.1 nhưng không public code attack. Repo này implement lại protocol đầy đủ (paper Appendix C, dòng 1085-1087):

1. Lấy ~50 prompts từ i2p benchmark (sexual cho nudity, toxicity≥0.95 cho violence).
2. Generate 4 ảnh "harmful" từ chính unlearned model (để có dữ liệu train TI).
3. Train textual inversion token `<c>` trên unlearned model: Adam, lr=5e-3, batch=4, 3000 steps (paper exact).
4. Generate attack images với prompt `"<c> <i2p prompt>"`.
5. Score bằng NudeNet (nudity) hoặc GPT-4o (violence) → DSR.

Cài ở `eval/eval_attack_concept_inversion.py`, scripts ở `scripts/attack-nudity.sh`, `scripts/attack-violence.sh`.

**Source-aware (DUO vs DUO_Anchor head-to-head):**

Script CLI có flag `--source {duo, duo-anchor, both}` (mặc định `both`) để attack 1 source hoặc cả 2 rồi ghi 1 JSON so sánh:
- `--duo_root` → `train/outputs/unlearn/SD-train/dpo/<beta>`  (DUO gốc của paper)
- `--anchor_root` → `train/outputs/unlearn/SD-train/duo-anchor/<beta>`  (DUO_Anchor)

Output của `both`:
- `concept_inversion_results.json`: list phẳng `{source, target, dsr, n, unsafe_count}`.
- `concept_inversion_summary.json`: bảng pivot `{target -> {duo, duo-anchor}}` — đây là bảng kết quả chính dùng để vẽ figure trong paper.

3 scripts mới:
- `scripts/attack-both-nudity.sh` — attack cả 2 source trên Nudity LoRA
- `scripts/attack-both-violence.sh` — attack cả 2 source trên 4 Violence sub-LoRA
- `scripts/attack-both-all.sh` — chạy nudity + 4 violence_sub cho cả 2 source

Cách chạy:
```bash
bash scripts/attack-both-nudity.sh
bash scripts/attack-both-violence.sh
bash scripts/attack-both-all.sh
```

**Lưu ý**: file `eval_attack_concept_inversion.py` gốc (commit trước) bị hỏng nặng (hàm bị cắt/ghép, `run_one_target` và `train_textual_inversion_merged` định nghĩa 2 lần). Commit này rewrite hoàn toàn: 13 hàm top-level, mỗi hàm 1 lần, có `find_lora()` autodetect checkpoint-{500,1000}.

---

## 3. Files đã thay đổi / tạo mới

### 3.1 Sửa code training
**`train/unlearn-sd.py`** — file lõi, thêm ~80 dòng:

| Vị trí | Thay đổi |
|---|---|
| Sau dòng 230 (CLI) | Thêm `--anchor_lambda`, `--anchor_dir`, `--anchor_t_min`, `--anchor_t_max` |
| `TrainDataset.__init__` (dòng ~456) | Load `anchor_images` + `anchor_prompts` từ `data_cfg` |
| `TrainDataset.__getitem__` (dòng ~492) | Trả thêm `anchor_images`, `anchor_prompt` |
| `collate_fn` (dòng ~522) | Stack `anchor_pixel_values` |
| Training loop (sau dòng 1254) | Thêm khối `L_retain = MSE(ε_θ(noisy_anchor), noise)` với `t ~ U[anchor_t_min, anchor_t_max]` |
| Sau dòng 1327 (logging) | Log `loss_anchor` |
| Dòng 614 (wandb) | Đổi `wandb.login(...)` thành check `WANDB_MODE=offline` env var |

**Khối L_retain (pseudo)**:
```python
if args.anchor_lambda > 0.0 and train_dataset.num_anchor_images > 0:
    with torch.no_grad():
        anchor_latents = vae.encode(anchor_pixel_values).latent_dist.sample() * scale
    anchor_noise = torch.randn_like(anchor_latents)
    anchor_t = torch.randint(anchor_t_min, anchor_t_max, (bsz,))
    noisy_anchor = noise_scheduler.add_noise(anchor_latents, anchor_noise, anchor_t)
    anchor_emb = pipe.encode_prompt(anchor_prompts, ...)
    anchor_pred = unet(noisy_anchor, anchor_t, anchor_emb).sample
    anchor_loss = F.mse_loss(anchor_pred, anchor_noise)
    loss = loss + args.anchor_lambda * anchor_loss
```

**`inference.py`** — bỏ CLI `--lambdas` (Task Arithmetic đã bỏ), giữ merge đều `[1,1,1,1]` như paper.

### 3.2 Dataset
**`datasets/SD/config.json`** — thêm cho 5 concept (Nudity, Blood, Suffer, Gun, Horror):
- `anchor_prompts`: list 8 prompt an toàn tương đồng thị giác
- `anchor_images`: `"anchor"` (tên thư mục)
- Sửa bug: `Gun` anchor_prompts cũ dùng nhầm của `Horror`; `Horror` thiếu `prompt`/`base_prompt`/`images`/`base_images`.

**`datasets/SD/generate_anchors.py`** (mới) — sinh ảnh anchor bằng SD gốc:
```python
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", ...)
for concept, ccfg in cfg.items():
    for prompt in ccfg["anchor_prompts"]:
        for _ in range(args.per_prompt):
            img = pipe(prompt, generator=g).images[0]
            img.save(f"datasets/SD/{concept}/anchor/{idx:03d}.jpg")
```

### 3.3 Scripts (Kaggle-ready, 1 GPU, 1 process)
| File | Mục đích |
|---|---|
| `scripts/prepare-dataset.sh` | Bỏ phần SD3 (chỉ giữ SD 1.4) |
| `scripts/prepare-anchor.sh` (mới) | Sinh anchor images cho 5 concept |
| `scripts/sd-nudity.sh` | DUO baseline, 1 GPU, port 50000 |
| `scripts/sd-violence.sh` | DUO baseline, 1 GPU, port 50000 |
| `scripts/sd-nudity-anchor.sh` (mới) | DUO + anchor, 1 GPU |
| `scripts/sd-violence-anchor.sh` (mới) | DUO + anchor, 1 GPU |
| `scripts/attack-nudity.sh` (mới) | Concept Inversion attack trên Nudity LoRA (mặc định DUO_Anchor) |
| `scripts/attack-violence.sh` (mới) | Concept Inversion attack trên 4 Violence sub-LoRAs (mặc định DUO_Anchor) |
| `scripts/attack-both-nudity.sh` (mới) | Attack CẢ DUO & DUO_Anchor trên Nudity → JSON so sánh |
| `scripts/attack-both-violence.sh` (mới) | Attack CẢ DUO & DUO_Anchor trên 4 Violence sub-LoRAs → JSON so sánh |
| `scripts/attack-both-all.sh` (mới) | Nudity + 4 violence_sub cho CẢ 2 source → 1 JSON pivot |

### 3.4 Evaluation
**`eval/eval_lpips.py`** (mới):
- Sinh ảnh anchor từ 3 model: (a) SD gốc (ref), (b) DUO baseline, (c) DUO + anchor.
- Tính LPIPS(ref, baseline) và LPIPS(ref, ours).
- Xuất `lpips_results.json` cho từng concept.

**`eval/eval_i2p.py`** (mới):
- Generate ảnh với i2p benchmark prompts.
- Score bằng NudeNet (nudity) hoặc GPT-4o/CLIP (violence).
- DSR = tỷ lệ ảnh KHÔNG chứa concept độc hại.

**`eval/eval_attack_concept_inversion.py`** (mới, rewrite):
- Train token `<c>` bằng textual inversion (Adam lr=5e-3, batch=4, 3000 steps — paper exact).
- Generate attack images với prompt `"<c> <i2p prompt>"`.
- Score bằng NudeNet/GPT-4o. DSR sau attack.
- Hỗ trợ `--source {duo, duo-anchor, both}` để attack 1 hoặc cả 2 model trong 1 lần chạy.
- `--mode {nudity, violence_sub, all}` chọn tập target.
- Ghi 2 file JSON: `concept_inversion_results.json` (flat) và `concept_inversion_summary.json` (pivot theo target) cho bảng so sánh.
- **`dsr_log.csv`**: append mỗi (source, target, dsr) sau khi score xong, dễ grep / load bằng pandas.
- **Resume**: re-run cùng script tự load kết quả cũ từ `concept_inversion_results.json`, skip các (source, target) đã có dsr — chạy lại an toàn sau khi Kaggle timeout. Thêm `--force` để ép chạy lại từ đầu.
- Note: bản trước bị hỏng (hàm bị cắt/ghép, `train_textual_inversion_merged` và `run_one_target` định nghĩa 2 lần). Bản này đã rewrite hoàn toàn, 14 hàm top-level duy nhất.

**`KAGGLE_NOTEBOOK.py`** (mới): notebook end-to-end tập trung vào attack.
- Cell 0-2: clone repo + symlink LoRA dataset vào layout scripts expect.
- Cell 4: optional `OPENAI_API_KEY` từ Kaggle Secrets (cho GPT-4o judge).
- Cell 5: **dry run 500 TI steps + 10 prompts** (~5 phút) để bắt lỗi setup trước khi launch full run.
- Cell 6: chạy `scripts/attack-both-all.sh` (~2.5h) — flush partial results sau mỗi (source, target) nên nếu Kaggle cắt 12h timeout thì re-run cell 6 sẽ resume.
- Cell 7: gộp `compare_both_beta500` (nudity) + `compare_both_beta1000` (violence) thành 1 bảng DSR duy nhất.
- Cell 8: đóng gói kết quả thành `DUO-Anchor-attack-results.tar.gz` để tải về 1 click.

### 3.5 Misc
- `requirements.txt`: thêm `safetensors`
- `README.md`: hướng dẫn Kaggle notebook end-to-end

---

## 4. Phân tích hạn chế của 2 cải tiến

### 4.1 Hạn chế của Idea 1 (Anchor Retention)

**H1.1 — Phụ thuộc chất lượng anchor set**
- Anchor sinh bằng SD 1.4 gốc → có thể chứa artifact, NSFW do model bias, hoặc concept gần với độc hại.
- Nếu anchor prompt "ketchup" vô tình sinh ra ảnh có máu → `L_retain` vô tình bảo vệ cả visual của máu.
- **Giảm thiểu**: filter anchor bằng NudeNet + manual review 5-10 ảnh/concept.

**H1.2 — Anchor set là thủ công**
- Mỗi concept độc hại cần designer chọn 5-10 prompt tương đồng.
- Không scalable nếu mở rộng sang 50+ concept.
- **Giảm thiểu**: dùng LLM (GPT-4) để tự động generate danh sách anchor prompt.

**H1.3 — Thêm 2 hyperparameters mới**
- `λ_anchor` (cường độ loss) và `[anchor_t_min, anchor_t_max]` (phạm vi timestep).
- 39 ngày không đủ sweep kỹ → dễ overfit vào một config.
- **Giảm thiểu**: fix `anchor_t_min=1, anchor_t_max=750` (khớp DUO gốc), chỉ sweep `λ_anchor ∈ {0.5, 1.0, 2.0}`.

**H1.4 — Không giải quyết white-box attack (Concept Inversion)**
- `L_retain` chỉ giữ hành vi trên prompt tự nhiên. Nếu attacker dùng textual inversion token `<c>` (như paper Sec 4.1 mô tả), anchor set không cover được.
- **Hướng phát triển**: cần anchor với token đã invert.

**H1.5 — Tăng chi phí training**
- Mỗi step giờ encode thêm `bsz` ảnh anchor qua VAE + UNet.
- Với batch=1: tăng ~30% thời gian mỗi step.
- Với max_train_steps=1000 (nudity) hoặc 500 (violence): thêm ~15-30 phút/concept trên Kaggle P100.

### 4.2 (đã bỏ)

Idea 2 (Task Arithmetic) bị loại khỏi repo. Lý do xem mục 2.2.

### 4.3 Hạn chế chung của repo

**H0.1 — Chưa test trên SD3**
- Paper DUO có phiên bản SD3 (`scripts/sd3-nudity.sh`) — repo này bỏ qua.
- Reviewer có thể hỏi tại sao không reproduce trên SD3.

**H0.2 — White-box attack evaluation**
- Paper Sec 4.1 đánh giá Concept Inversion.
- Repo này đã implement ở `eval/eval_attack_concept_inversion.py` (protocol paper exact: Adam lr=5e-3, batch=4, 3000 steps).
- Lưu ý: paper dùng i2p benchmark với ~200 prompts (sexual) và ~50 prompts (violence toxicity≥0.95). Code load từ file jsonl nếu user cung cấp (`datasets/i2p/sexual.jsonl`, `datasets/i2p/violence.jsonl`), fallback sang prompts tổng quát nếu thiếu.
- GPT-4o violence judge cần `OPENAI_API_KEY` env var; nếu không có sẽ fallback sang CLIP proxy (`--violence_judge clip`).

**H0.3 — Hyperparameter chưa tune**
- `λ_anchor=1.0`, `anchor_t_min=1`, `anchor_t_max=750` chọn theo intuition, chưa sweep.
- Có thể không tối ưu.

**H0.4 — Compute constraint**
- 1 GPU Kaggle (P100 16GB) → batch=1, tốc độ chậm. So với 4×A100 của paper.
- 5 concept × 2 method (baseline + anchor) × 2 exp (nudity, violence) = 20 lần train.
- Mỗi lần ~1-2 giờ trên P100 → ~30-40 giờ total → cần nhiều session Kaggle (30h/week limit).
- Concept Inversion attack: 3000 steps × 5 LoRA + 1 merged ≈ 15K steps TI (~1.5h P100).

**H0.5 — Attack train set nhỏ**
- Paper không nói rõ train images cho TI. Code dùng 4 ảnh/concept — có thể ảnh hưởng chất lượng TI. Tăng `--num_train_images` nếu compute cho phép.

---

## 5. Đề xuất ưu tiên trong 39 ngày

| Tuần | Việc cần làm | Output |
|---|---|---|
| 1 (ngày 1-7) | Generate anchor set + test 1 concept (Blood) | Bảng LPIPS đầu tiên |
| 2 (ngày 8-14) | Train full 5 concept với 2 method (DUO + DUO-Anchor) | 10 LoRA checkpoints |
| 3 (ngày 15-21) | Đánh giá DSR, I2P, LPIPS trên toàn bộ | Bảng số liệu chính |
| 4 (ngày 22-28) | Chạy Concept Inversion attack + I2P attack (2 model) | Bảng attack |
| 5 (ngày 29-35) | Viết paper, vẽ figure, format NeurIPS workshop | Draft paper |
| 6 (ngày 36-39) | Buffer + nộp | Submission |

**Anchor là contribution chính**, attack là evaluation table so sánh 2 model.

---

## 6. Tài liệu tham khảo trích từ paper DUO

- **Sec 3.2 (dòng 261-286)**: SDEdit paired dataset
- **Sec 3.3 (dòng 293-385)**: Diffusion-DPO formulation
- **Sec 3.4 (dòng 387-408)**: `L_prior` chỉ tại `t=T` và `L_DUO` cuối cùng
- **Sec 4.1 (dòng 412-422)**: Decomposition 4 concept + merge bằng cộng đều
- **Sec 5 (dòng 621-637)**: Conclusion + thừa nhận hạn chế về visual feature overlap
