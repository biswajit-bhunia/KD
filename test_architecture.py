"""
Smoke test for the dual-domain architecture refactor.
Tests: imports, forward passes, output shapes, KD compatibility, param counts.
"""

import torch
import sys

def test_imports():
    print("1. Testing imports...")
    from models.fusion import GatedFusion
    from models.kd import MultiLevelKD
    from models.teacher import TeacherModel, SemanticTeacher, ForensicTeacher
    from models.student import StudentModel, SemanticStudentBranch, ForensicStudentCNN
    from models.grl import GradientReversal
    from models.gen_classifier import GeneratorClassifier
    from losses.losses import ClassificationLoss, KDLoss, SupConLoss
    from losses.fedprox import FedProxLoss
    from features.forensic import build_forensic_stack
    print("   ✓ All imports successful")

def test_teacher_forward():
    print("\n2. Testing teacher forward pass...")
    from models.teacher import TeacherModel
    
    teacher = TeacherModel(pretrained=False, embed_dim=256, num_classes=2)
    x_rgb = torch.randn(2, 3, 256, 256)
    x_for = torch.randn(2, 12, 256, 256)
    
    out = teacher(x_rgb, x_for)
    
    assert "semantic_feat" in out, "Missing key: semantic_feat"
    assert "forensic_feat" in out, "Missing key: forensic_feat"
    assert "embedding" in out, "Missing key: embedding"
    assert "logits" in out, "Missing key: logits"
    
    assert out["semantic_feat"].shape == (2, 256), f"semantic_feat shape: {out['semantic_feat'].shape}"
    assert out["forensic_feat"].shape == (2, 256), f"forensic_feat shape: {out['forensic_feat'].shape}"
    assert out["embedding"].shape == (2, 256), f"embedding shape: {out['embedding'].shape}"
    assert out["logits"].shape == (2, 2), f"logits shape: {out['logits'].shape}"
    
    print(f"   ✓ Teacher forward OK — shapes: sem={out['semantic_feat'].shape}, "
          f"for={out['forensic_feat'].shape}, emb={out['embedding'].shape}, "
          f"logits={out['logits'].shape}")

def test_student_forward():
    print("\n3. Testing student forward pass...")
    from models.student import StudentModel
    
    student = StudentModel(embed_dim=256, num_classes=2, pretrained=False)
    x_rgb = torch.randn(2, 3, 256, 256)
    x_for = torch.randn(2, 12, 256, 256)
    
    out = student(x_rgb, x_for)
    
    assert "semantic_feat" in out, "Missing key: semantic_feat"
    assert "forensic_feat" in out, "Missing key: forensic_feat"
    assert "embedding" in out, "Missing key: embedding"
    assert "logits" in out, "Missing key: logits"
    
    assert out["semantic_feat"].shape == (2, 256), f"semantic_feat shape: {out['semantic_feat'].shape}"
    assert out["forensic_feat"].shape == (2, 256), f"forensic_feat shape: {out['forensic_feat'].shape}"
    assert out["embedding"].shape == (2, 256), f"embedding shape: {out['embedding'].shape}"
    assert out["logits"].shape == (2, 2), f"logits shape: {out['logits'].shape}"
    
    print(f"   ✓ Student forward OK — shapes: sem={out['semantic_feat'].shape}, "
          f"for={out['forensic_feat'].shape}, emb={out['embedding'].shape}, "
          f"logits={out['logits'].shape}")

def test_kd_compatibility():
    print("\n4. Testing KD compatibility...")
    from models.teacher import TeacherModel
    from models.student import StudentModel
    from models.kd import MultiLevelKD
    
    teacher = TeacherModel(pretrained=False)
    student = StudentModel(pretrained=False)
    kd = MultiLevelKD(temperature=4.0)
    
    x_rgb = torch.randn(2, 3, 256, 256)
    x_for = torch.randn(2, 12, 256, 256)
    
    with torch.no_grad():
        t_out = teacher(x_rgb, x_for)
    s_out = student(x_rgb, x_for)
    
    losses = kd(s_out, t_out)
    
    assert "semantic" in losses
    assert "forensic" in losses
    assert "embedding" in losses
    assert "logits" in losses
    assert "total" in losses
    
    for k, v in losses.items():
        assert torch.isfinite(v), f"KD loss '{k}' is not finite: {v}"
        print(f"   ✓ KD loss '{k}': {v.item():.4f}")
    
    print("   ✓ Multi-level KD compatible")

def test_forensic_stack():
    print("\n5. Testing forensic stack integration...")
    from features.forensic import build_forensic_stack
    
    x_rgb = torch.randn(2, 3, 256, 256)
    x_for = build_forensic_stack(x_rgb)
    
    assert x_for.shape == (2, 12, 256, 256), f"Forensic stack shape: {x_for.shape}"
    print(f"   ✓ Forensic stack shape: {x_for.shape}")

def test_param_counts():
    print("\n6. Testing parameter counts...")
    from models.teacher import TeacherModel
    from models.student import StudentModel
    
    teacher = TeacherModel(pretrained=False)
    student = StudentModel(pretrained=False)
    
    t_params = sum(p.numel() for p in teacher.parameters())
    s_params = sum(p.numel() for p in student.parameters())
    
    print(f"   Teacher: {t_params:,} params")
    print(f"   Student: {s_params:,} params")
    print(f"   Ratio:   {s_params/t_params*100:.1f}%")
    
    # Student should be in 3-5M range
    assert 2_000_000 < s_params < 6_000_000, f"Student params {s_params:,} outside expected 2-6M range"
    print(f"   ✓ Student params within expected range")

def test_fusion():
    print("\n7. Testing gated fusion...")
    from models.fusion import GatedFusion
    
    fusion = GatedFusion(feat_dim=256)
    a = torch.randn(4, 256)
    b = torch.randn(4, 256)
    
    out = fusion(a, b)
    assert out.shape == (4, 256), f"Fusion output shape: {out.shape}"
    
    # Check gate is producing values in [0, 1] range
    with torch.no_grad():
        concat = torch.cat([a, b], dim=1)
        alpha = fusion.gate_net(concat)
        assert alpha.min() >= 0 and alpha.max() <= 1, "Gate values outside [0, 1]"
    
    print(f"   ✓ Fusion output shape: {out.shape}, gate in [0, 1]")

def test_backward():
    print("\n8. Testing backward pass (gradient flow)...")
    from models.student import StudentModel
    from losses.losses import ClassificationLoss
    
    student = StudentModel(pretrained=False)
    cls_loss = ClassificationLoss()
    
    x_rgb = torch.randn(2, 3, 256, 256)
    x_for = torch.randn(2, 12, 256, 256)
    labels = torch.tensor([0, 1])
    
    out = student(x_rgb, x_for)
    loss = cls_loss(out["logits"], labels)
    loss.backward()
    
    # Check gradients exist for key parameters
    has_grad = 0
    total = 0
    for name, p in student.named_parameters():
        if p.requires_grad:
            total += 1
            if p.grad is not None:
                has_grad += 1
    
    print(f"   ✓ Backward OK — {has_grad}/{total} parameters have gradients")
    assert has_grad == total, f"Missing gradients: {total - has_grad} parameters"

if __name__ == "__main__":
    try:
        test_imports()
        test_teacher_forward()
        test_student_forward()
        test_kd_compatibility()
        test_forensic_stack()
        test_param_counts()
        test_fusion()
        test_backward()
        
        print(f"\n{'='*60}")
        print(f"  ALL TESTS PASSED ✓")
        print(f"{'='*60}\n")
    except Exception as e:
        print(f"\n  ✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
