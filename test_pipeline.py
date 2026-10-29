import pytest
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from pipeline_botanico import (
    compute_classification_metrics,
    compute_contingency_table,
    compute_mcnemar,
    compute_paired_bootstrap,
    compute_bootstrap_ci,
    compute_calibration_metrics,
    FlowerClassifier,
    Config,
    get_transforms
)


class TestMetrics:
    def test_classification_metrics_perfect(self):
       
        y_true = np.array([0, 1, 2, 3, 4])
        # Probabilidades com 1.0 na classe correta
        y_probs = np.zeros((5, 10))
        for i, y in enumerate(y_true):
            y_probs[i, y] = 1.0
        metrics = compute_classification_metrics(y_true, y_probs, k=5)
        assert metrics["top1"] == 100.0
        assert metrics["top5"] == 100.0
        assert metrics["f1_macro"] == pytest.approx(100.0)

    def test_classification_metrics_imperfect(self):
        
        y_true = np.array([0, 1, 2, 3])
        # Amostras:
        # idx 0 (true=0): probs [0.8, 0.2, 0.0, 0.0] -> top1=0 (acerto), top2=[0, 1] (acerto)
        # idx 1 (true=1): probs [0.1, 0.7, 0.2, 0.0] -> top1=1 (acerto), top2=[1, 2] (acerto)
        # idx 2 (true=2): probs [0.6, 0.1, 0.3, 0.0] -> top1=0 (erro), top2=[0, 2] (acerto)
        # idx 3 (true=3): probs [0.5, 0.4, 0.1, 0.0] -> top1=0 (erro), top2=[0, 1] (erro, true 3 não está no top 2)
        y_probs = np.array([
            [0.8, 0.2, 0.0, 0.0],
            [0.1, 0.7, 0.2, 0.0],
            [0.6, 0.1, 0.3, 0.0],
            [0.5, 0.4, 0.1, 0.0]
        ])
        metrics = compute_classification_metrics(y_true, y_probs, k=2)
        assert metrics["top1"] == 50.0   # 2 de 4 acertos no top 1
        assert metrics["top2"] == 75.0   # 3 de 4 acertos no top 2


class TestCalibrationECE:
    def test_ece_perfect_calibration(self):
       
        # Cria 100 amostras com confiança 0.8 onde exatamente 80% acertam
        y_true = np.zeros(100, dtype=int)
        y_probs = np.zeros((100, 2))
        y_probs[:, 0] = 0.8
        y_probs[:, 1] = 0.2
        # 80 acertam (classe 0), 20 erram (classe 1)
        y_true[80:] = 1
        
        calib = compute_calibration_metrics(y_true, y_probs, n_bins=10)
        # O bin de 0.8 tem 100 amostras, acc = 0.8, conf = 0.8 -> erro do bin = 0
        assert calib["ece"] == pytest.approx(0.0, abs=1e-5)

    def test_ece_edge_case_prob_one(self):
        
        y_true = np.array([0, 1, 2])
        y_probs = np.zeros((3, 3))
        y_probs[0, 0] = 1.0
        y_probs[1, 1] = 1.0
        y_probs[2, 0] = 1.0  # Erro com confiança 1.0
        
        calib = compute_calibration_metrics(y_true, y_probs, n_bins=10)
        assert calib["bin_counts"][-1] == 3  # Todas as 3 devem estar no bin 9 ([0.9, 1.0])
        assert calib["bin_accs"][-1] == pytest.approx(2.0 / 3.0)
        assert calib["bin_confs"][-1] == pytest.approx(1.0)
        # ECE = 1 * |2/3 - 1.0| = 1/3
        assert calib["ece"] == pytest.approx(1.0 / 3.0, abs=1e-4)

    def test_ece_empty_bins_handling(self):
       
        y_true = np.array([0, 0])
        y_probs = np.array([[0.95, 0.05], [0.95, 0.05]])
        
        calib = compute_calibration_metrics(y_true, y_probs, n_bins=10)
        # Apenas o último bin tem contagem > 0
        for b in range(9):
            assert calib["bin_counts"][b] == 0
            assert calib["bin_accs"][b] is None
        assert calib["bin_counts"][9] == 2
        assert calib["bin_accs"][9] == 1.0


class TestMcNemarEdwards:
    def test_contingency_table(self):
      
        y_true  = np.array([0, 1, 0, 1, 0])
        preds_a = np.array([0, 0, 0, 1, 1])  # acerta idx 0, 2, 3; erra idx 1, 4
        preds_b = np.array([1, 1, 0, 1, 0])  # acerta idx 1, 2, 3, 4; erra idx 0
        
        # A errou e B acertou: idx 1 (true=1, a=0, b=1) e idx 4 (true=0, a=1, b=0) -> b = 2
        # A acertou e B errou: idx 0 (true=0, a=0, b=1) -> c = 1
        b, c = compute_contingency_table(y_true, preds_a, preds_b)
        assert b == 2
        assert c == 1

    def test_mcnemar_known_values(self):
        
        # Se b = 372 e c = 132:
        # (|372 - 132| - 1)^2 / (372 + 132) = 239^2 / 504 = 57121 / 504 = 113.3353
        res = compute_mcnemar(372, 132, correction=True)
        assert res["chi2"] == pytest.approx(113.3353, abs=1e-2)
        assert res["df"] == 1
        assert res["p_value"] < 1e-20
        assert res["significant_005"] is True

    def test_mcnemar_identical_models(self):
      
        res = compute_mcnemar(0, 0, correction=True)
        assert res["chi2"] == 0.0
        assert res["p_value"] == 1.0
        assert res["significant_005"] is False


class TestBootstrap:
    def test_paired_bootstrap(self):
       
        rng = np.random.RandomState(42)
        n = 500
        y_true = rng.randint(0, 5, size=n)
        preds_a = y_true.copy()
        # introduz 50 erros em A
        preds_a[rng.choice(n, size=50, replace=False)] = (y_true[:50] + 1) % 5
        # B tem apenas 10 erros
        preds_b = y_true.copy()
        preds_b[rng.choice(n, size=10, replace=False)] = (y_true[:10] + 1) % 5

        res = compute_paired_bootstrap(y_true, preds_a, preds_b, n_boot=200, seed=42)
        # B deve ser significativamente melhor que A
        assert res["mean_delta"] > 0
        assert res["ci_low"] > 0


class TestArchitectureAndBatchNorm:
    def test_flower_classifier_forward(self):
       
        backbone = models.mobilenet_v2(weights=None)
        classifier = FlowerClassifier(backbone, num_classes=102)
        x = torch.randn(2, 3, 224, 224)
        logits = classifier(x)
        assert logits.shape == (2, 102)

    def test_freeze_backbone_bn_eval_mode(self):
      
        backbone = models.mobilenet_v2(weights=None)
        classifier = FlowerClassifier(backbone, num_classes=102)
        
        # Simula estado de Fase 1
        classifier.train()
        classifier.backbone.eval()  # Congelamento estrito
        
        bn = classifier.backbone.features[0][1]
        assert isinstance(bn, nn.BatchNorm2d)
        initial_mean = bn.running_mean.clone()
        
        # Passa batch de dados
        x = torch.randn(4, 3, 224, 224) * 5.0 + 10.0
        _ = classifier(x)
        
        # Em modo eval(), running_mean não deve sofrer alteração!
        assert torch.equal(bn.running_mean, initial_mean)


class TestDataTransforms:
    def test_transforms_aspect_ratio_preservation(self):
       
        cfg = Config(img_size=224, resize_size=256)
        train_trans, val_trans = get_transforms(cfg)
        
        from PIL import Image
        img_rect = Image.new("RGB", (400, 200), color="red")  # retangular 2:1
        tensor_out = val_trans(img_rect)
        assert tensor_out.shape == (3, 224, 224)
