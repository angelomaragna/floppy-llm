import torch

from floppy_llm.model import EXPECTED_PARAMS, TinyGPT, count_params


def test_exact_param_budget():
    model = TinyGPT()
    n = count_params(model)
    assert n == EXPECTED_PARAMS == 358960, f"got {n}, expected {EXPECTED_PARAMS}"


def test_forward_shape():
    model = TinyGPT()
    model.eval()
    with torch.no_grad():
        logits = model(torch.zeros(2, 16, dtype=torch.long))
    assert logits.shape == (2, 16, 512)


def test_weight_tying_is_shared():
    model = TinyGPT()
    assert model.lm_head.weight is model.tok_emb.weight


def test_expected_params_math():
    # 57,344 emb + 2*(50,176 attn + 100,352 ffn + 224 norms) + 112 final norm
    assert EXPECTED_PARAMS == 512 * 112 + 2 * (50176 + 100352 + 224) + 112
