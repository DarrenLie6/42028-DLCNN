# ── Smoke test (python -m src.training.metrics) ────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)
    B, C, H, W = 2, 4, 8, 8
    m = SegmentationMetrics(device="cpu")

    # ── Test 1: Perfect prediction ─────────────────────────────────────────────
    targets = torch.randint(0, 4, (B, H, W))
    logits  = torch.zeros(B, C, H, W)
    for cls in range(C):
        logits[:, cls][targets == cls] = 10.0   # spike the correct class

    m.update(logits, targets)
    out = m.compute()

    print("── Test 1: Perfect prediction ──")
    for k, v in out.items():
        print(f"  {k:<25} {v:.4f}")

    assert out["mean_iou"]        > 0.99,  "mean_iou should be ~1.0 for perfect preds"
    assert out["mean_f1"]         > 0.99,  "mean_f1 should be ~1.0 for perfect preds"
    assert out["iou/Background"] == 0.0,   "Background is masked out → IoU always 0"
    assert out["f1/Background"]  == 0.0,   "Background is masked out → F1 always 0"
    assert out["iou/Intact"]      > 0.99,  "Intact IoU should be ~1.0"
    assert out["iou/Damaged"]     > 0.99,  "Damaged IoU should be ~1.0"
    assert out["iou/Destroyed"]   > 0.99,  "Destroyed IoU should be ~1.0"
    print("  ✅ Passed\n")

    # ── Test 2: reset() zeroes the matrix ─────────────────────────────────────
    m.reset()
    assert m.conf_matrix.sum().item() == 0, "reset() should zero the matrix"
    print("── Test 2: reset() zeroes the matrix ──")
    print("  ✅ Passed\n")

    # ── Test 3: All-wrong prediction (predicts class 0 for everything) ─────────
    targets_fg = torch.randint(1, 4, (B, H, W))   # only foreground classes
    logits_bad  = torch.zeros(B, C, H, W)
    logits_bad[:, 0] = 10.0                        # always predicts Background

    m.reset()
    m.update(logits_bad, targets_fg)
    out_bad = m.compute()

    print("── Test 3: All-wrong prediction ──")
    for k, v in out_bad.items():
        print(f"  {k:<25} {v:.4f}")

    assert out_bad["mean_iou"] < 0.01, "mean_iou should be ~0.0 for all-wrong preds"
    assert out_bad["mean_f1"]  < 0.01, "mean_f1 should be ~0.0 for all-wrong preds"
    print("  ✅ Passed\n")

    # ── Test 4: Multi-batch accumulation is exact ──────────────────────────────
    m.reset()
    targets_batch = torch.randint(0, 4, (B, H, W))
    logits_batch  = torch.zeros(B, C, H, W)
    for cls in range(C):
        logits_batch[:, cls][targets_batch == cls] = 10.0

    # Feed same batch 3 times — result must be identical to feeding it once
    m.update(logits_batch, targets_batch)
    m.update(logits_batch, targets_batch)
    m.update(logits_batch, targets_batch)
    out_multi = m.compute()

    m.reset()
    m.update(logits_batch, targets_batch)
    out_single = m.compute()

    print("── Test 4: Multi-batch accumulation ──")
    for k in out_multi:
        print(f"  {k:<25} multi={out_multi[k]:.4f}  single={out_single[k]:.4f}")
        assert abs(out_multi[k] - out_single[k]) < 1e-5, f"Mismatch on {k}"
    print("  ✅ Passed\n")

    print("🎉  All smoke tests passed.")