"""MedDecider NLI runtime with synthetic doubles; never clinical evidence."""

import copy
import json
import math
from dataclasses import replace

import pytest

from openmed.cli.main import main
from openmed.clinical.brief import build_clinical_brief
from openmed.clinical.nli import verify
from openmed.clinical.nli_backends import EncoderNLIBackend, LocalNLIError
from openmed.clinical.nli_gate import NLIThresholds
from openmed.clinical.nli_qualification import (
    NLIQualificationPolicy,
    bind_qualified_nli,
    qualify_local_nli,
)
from openmed.core.models import ModelLoader
from openmed.eval.nli_error_slices import CLINICAL_NLI_PHENOMENA
from openmed.torch.meddecider_nli import (
    MEDDECIDER_NLI_LABEL_MAPPING,
    MedDeciderReadout,
    meddecider_nli_prompt,
    read_meddecider_artifact,
)
from tests.unit.clinical.test_brief import fixture_context

THRESHOLDS = NLIThresholds(
    calibration_id="synthetic-docs-only", calibration_method="synthetic-fixture"
)
_STATEMENT = "Statement: "


def synthetic_label(premise, claim):
    """Rule standing in for a model: exact claim, negated claim, otherwise none."""
    claim = claim.strip().rstrip(".")
    if claim in premise:
        return "entailment"
    if claim.replace(" not ", " ") in premise:
        return "contradiction"
    return "neutral"


def fake_readout(strength=6.0, bias=0.0, temperature=1.0):
    """Readout whose token ids are the prompt bytes, scored by a synthetic rule."""

    def letter_logits(ids):
        message = bytes(ids).decode("utf-8")
        premise = message.split("\n\nQuestion: ", 1)[0]
        claim = message.split(_STATEMENT, 1)[1].split("\nOptions:\n", 1)[0]
        label = synthetic_label(premise, claim)
        option = {
            "entailment": "Yes, it follows",
            "neutral": "It might be true",
            "contradiction": "No, the text",
        }[label]
        shown = message.split("\nOptions:\n", 1)[1].splitlines()[:3]
        logits = [strength if option in line else 0.0 for line in shown]
        logits[0] += bias
        return logits

    return MedDeciderReadout(
        lambda message: list(message.encode("utf-8")),
        letter_logits,
        temperature=temperature,
    )


class FakeMedDeciderLoader:
    """Injected loader returning the synthetic readout's classifier mapping."""

    def __init__(self, readout=None):
        self.readout = readout or fake_readout()
        self.calls = []

    def load_local_sequence_classifier(self, model_ref, *, revision, runtime):
        self.calls.append(runtime)
        assert runtime == "meddecider"
        return self.readout.classifier()


def encoder(loader=None, thresholds=THRESHOLDS):
    return EncoderNLIBackend(
        "synthetic-meddecider-artifact",
        runtime="meddecider",
        label_mapping=MEDDECIDER_NLI_LABEL_MAPPING,
        thresholds=thresholds,
        loader=loader or FakeMedDeciderLoader(),
    )


def write_artifact(root, *, adapter_license="apache-2.0", base_license="apache-2.0"):
    adapter, base = root / "adapter", root / "base"
    adapter.mkdir(parents=True)
    base.mkdir()
    (adapter / "adapter_config.json").write_text('{"peft_type": "LORA"}')
    (adapter / "adapter_model.safetensors").write_bytes(b"synthetic adapter")
    (adapter / "meddecider_config.json").write_text(
        json.dumps({"temperature": {"choice": 1.25}})
    )
    (adapter / "README.md").write_text(f"---\nlicense: {adapter_license}\n---\n")
    (base / "README.md").write_text(f"---\nlicense: {base_license}\n---\n")
    (base / "config.json").write_text("{}")
    (base / "model.safetensors").write_bytes(b"synthetic weights")
    return root


def test_prompt_layout_is_fixed():
    assert meddecider_nli_prompt(" Premise. ", " Claim. ", (0, 1, 2)) == (
        "Premise.\n\nQuestion: Based only on the text above, is this statement "
        "true? Statement: Claim.\nOptions:\nA. Yes, it follows from the text\n"
        "B. It might be true, but the text does not say\n"
        "C. No, the text contradicts it\n\n"
        "Answer with the letter of the correct option only."
    )
    assert "A. No, the text contradicts it" in meddecider_nli_prompt(
        "P", "H", (2, 1, 0)
    )


def test_readout_averages_orders_and_applies_temperature():
    readout = fake_readout(strength=0.0, bias=2.0)
    probabilities = readout.probabilities(readout.encode("Text.", "Claim."))
    assert math.isclose(sum(probabilities), 1.0)
    # A pure letter-A preference cancels across the two option orders.
    assert math.isclose(probabilities[0], probabilities[2])
    sharp = fake_readout(temperature=1.0)
    soft = fake_readout(temperature=3.0)
    pair = ("The cough is resolved.", "The cough is resolved.")
    assert (
        sharp.probabilities(sharp.encode(*pair))[0]
        > soft.probabilities(soft.encode(*pair))[0]
        > 1 / 3
    )


def test_classifier_logits_recover_probabilities():
    readout = fake_readout()
    classifier = readout.classifier()
    encoded = classifier["tokenizer"]("A b.", "A b.", truncation=False)
    logits = classifier["model"](**encoded).logits[0].tolist()
    total = sum(math.exp(value) for value in logits)
    expected = readout.probabilities(encoded["meddecider_orders"])
    for value, probability in zip(logits, expected):
        assert math.isclose(math.exp(value) / total, probability, rel_tol=1e-9)
    with pytest.raises(ValueError):
        classifier["tokenizer"]("A b.", "A b.", truncation=True)


def test_encoder_backend_runs_meddecider_through_verify():
    results = verify(
        ["The cough is resolved.", "The cough is not resolved.", "A rash is present."],
        "Synthetic note. The cough is resolved.",
        backend=encoder(),
    )
    assert [row["label"] for row in results] == [
        "entailment",
        "contradiction",
        "abstention",
    ]
    assert {row["backend_id"] for row in results} == {"local-encoder"}


def test_overlong_pair_abstains_without_truncation():
    backend = encoder()
    result = backend.predict("Synthetic premise. " * 40, "Synthetic premise.")
    assert result["label"] == "abstention"


def synthetic_split(split):
    rows = []
    for i, slice_name in enumerate(CLINICAL_NLI_PHENOMENA):
        premise = f"Synthetic {split} finding {i} is resolved."
        for label, claim in (
            ("entailment", premise),
            ("contradiction", f"Synthetic {split} finding {i} is not resolved."),
            ("neutral", f"Synthetic {split} item {i} is present."),
        ):
            rows.append(
                {
                    "id": f"{split}-{i}-{label}",
                    "group_id": f"{split}-{i}",
                    "premise": premise,
                    "hypothesis": claim,
                    "gold_label": label,
                    "slices": [slice_name],
                }
            )
    return {
        "provenance": {"kind": "synthetic", "reference": f"synthetic-{split}"},
        "records": rows,
    }


def test_qualification_and_binding_use_meddecider_runtime(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "weights.bin").write_bytes(b"synthetic")
    records = {split: synthetic_split(split) for split in ("dev", "eval")}
    options = dict(
        label_mapping=MEDDECIDER_NLI_LABEL_MAPPING,
        development=records["dev"],
        evaluation=records["eval"],
        policy=NLIQualificationPolicy(min_per_class=1),
        runtime="meddecider",
    )
    loader = FakeMedDeciderLoader()
    receipt = qualify_local_nli(artifact, **options, loader=loader).to_dict()
    assert receipt["status"] == "synthetic_only"
    assert receipt["binding"]["runtime"] == "meddecider"
    assert set(loader.calls) == {"meddecider"}

    # Binding mechanics only: the declared kind is flipped on synthetic data,
    # as in the existing qualification tests.
    declared = copy.deepcopy(options)
    for name in ("development", "evaluation"):
        declared[name]["provenance"]["kind"] = "caller_supplied"
    qualified = qualify_local_nli(artifact, **declared, loader=loader)
    assert qualified.to_dict()["status"] == "qualified"
    callback = bind_qualified_nli(qualified, artifact, **declared, loader=loader)
    result, context = fixture_context()
    context = replace(context, nli_predict=callback, thresholds=callback.thresholds)
    brief = build_clinical_brief(result, model="extractive", context=context)
    assert brief.refusal_reason is None


def test_cli_accepts_meddecider_runtime(tmp_path, capfd):
    paths = {}
    for name, value in (
        ("labels", MEDDECIDER_NLI_LABEL_MAPPING),
        ("policy", {"min_per_class": 1}),
    ):
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(value))
    artifact = write_artifact(tmp_path / "artifact", adapter_license="cc-by-nc-4.0")
    code = main(
        [
            "nli-qualify",
            "--artifact",
            str(artifact),
            "--label-mapping",
            str(paths["labels"]),
            "--runtime",
            "meddecider",
            "--json",
        ]
    )
    assert code == 1
    payload = json.loads(capfd.readouterr().out)["data"]
    assert payload["binding"]["runtime"] == "meddecider"
    assert payload["reasons"] == ["development_unavailable", "evaluation_unavailable"]


def test_artifact_validation_and_licence_gate(tmp_path):
    artifact = write_artifact(tmp_path / "ok")
    info = read_meddecider_artifact(artifact)
    assert info["temperature"] == 1.25
    for name, kwargs in (
        ("adapter-nc", {"adapter_license": "cc-by-nc-4.0"}),
        ("base-other", {"base_license": "other"}),
    ):
        with pytest.raises(ValueError, match="licence is not permissive"):
            read_meddecider_artifact(write_artifact(tmp_path / name, **kwargs))
    (artifact / "adapter" / "adapter_model.safetensors").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        read_meddecider_artifact(artifact)
    broken = write_artifact(tmp_path / "temperature")
    (broken / "adapter" / "meddecider_config.json").write_text(
        '{"temperature": {"choice": 0}}'
    )
    with pytest.raises(ValueError, match="temperature"):
        read_meddecider_artifact(broken)


def test_loader_refuses_hub_reference_and_non_permissive_artifact(
    tmp_path, monkeypatch
):
    # Validation happens before any optional runtime import.
    monkeypatch.setattr("openmed.core.models.HF_AVAILABLE", True)
    loader = ModelLoader()
    with pytest.raises(ValueError, match="local directory"):
        loader.load_local_sequence_classifier(
            "synthetic/remote-reference", revision="a" * 40, runtime="meddecider"
        )
    artifact = write_artifact(tmp_path / "nc", adapter_license="cc-by-nc-4.0")
    with pytest.raises(ValueError, match="licence"):
        loader.load_local_sequence_classifier(str(artifact), runtime="meddecider")
    backend = EncoderNLIBackend(
        str(artifact),
        runtime="meddecider",
        label_mapping=MEDDECIDER_NLI_LABEL_MAPPING,
        thresholds=THRESHOLDS,
    )
    with pytest.raises(LocalNLIError, match="checkpoint is unavailable"):
        backend.predict_scores("Synthetic premise.", "Synthetic claim.")


def test_lora_merge_matches_low_rank_update(tmp_path):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    from openmed.torch.meddecider_nli import merge_lora_adapter

    torch.manual_seed(0)
    model = torch.nn.Sequential()
    model.add_module("proj", torch.nn.Linear(4, 3, bias=False))
    original = model.proj.weight.detach().clone()
    down, up = torch.randn(2, 4), torch.randn(3, 2)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": 2, "lora_alpha": 4, "bias": "none"})
    )
    safetensors.save_file(
        {
            "base_model.model.proj.lora_A.weight": down,
            "base_model.model.proj.lora_B.weight": up,
        },
        str(adapter / "adapter_model.safetensors"),
    )
    merge_lora_adapter(model, adapter)
    assert torch.allclose(model.proj.weight, original + 2.0 * (up @ down), atol=1e-6)
    (adapter / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": 2, "lora_alpha": 4, "use_dora": True})
    )
    with pytest.raises(ValueError, match="unsupported"):
        merge_lora_adapter(model, adapter)


PLAIN_LORA = {
    "peft_type": "LORA",
    "task_type": "CAUSAL_LM",
    "r": 2,
    "lora_alpha": 4,
    "bias": "none",
    "init_lora_weights": True,
    "target_modules": ["proj"],
    "lora_dropout": 0.0,
    "use_dora": False,
    "use_rslora": False,
    "alora_invocation_tokens": None,
    "layer_replication": None,
    "rank_pattern": {},
    "alpha_pattern": {},
    "modules_to_save": None,
}


def test_lora_scale_admits_plain_lora():
    from openmed.torch.meddecider_nli import lora_scale

    assert lora_scale(PLAIN_LORA) == 2.0


@pytest.mark.parametrize(
    "change",
    [
        {"alora_invocation_tokens": [10, 11]},
        {"layer_replication": [[0, 2], [1, 3]]},
        {"use_dora": True},
        {"use_rslora": True},
        {"rank_pattern": {"proj": 4}},
        {"alpha_pattern": {"proj": 8}},
        {"modules_to_save": ["lm_head"]},
        {"trainable_token_indices": [5]},
        {"target_parameters": ["proj.weight"]},
        {"init_lora_weights": "pissa"},
        {"bias": "all"},
        {"peft_type": "LOHA"},
        {"task_type": "SEQ_CLS"},
        {"r": 0},
        {"r": True},
        {"r": 2.0},
        {"lora_alpha": 0},
        {"lora_alpha": float("nan")},
        {"lora_alpha": float("inf")},
        {"synthetic_future_option": True},
    ],
)
def test_lora_scale_refuses_everything_outside_plain_lora(change):
    from openmed.torch.meddecider_nli import lora_scale

    with pytest.raises(ValueError, match="unsupported"):
        lora_scale({**PLAIN_LORA, **change})


def _adapter(tmp_path, tensors, **config):
    safetensors = pytest.importorskip("safetensors.torch")
    adapter = tmp_path / "adapter"
    adapter.mkdir(exist_ok=True)
    (adapter / "adapter_config.json").write_text(json.dumps({**PLAIN_LORA, **config}))
    safetensors.save_file(tensors, str(adapter / "adapter_model.safetensors"))
    return adapter


def _two_layer_model(torch):
    model = torch.nn.Sequential()
    model.add_module("first", torch.nn.Linear(4, 3, bias=False))
    model.add_module("proj", torch.nn.Linear(4, 3, bias=False))
    return model


@pytest.mark.parametrize(
    "case", ["rank_mismatch", "nonfinite", "extra_tensor", "missing_up", "shape"]
)
def test_lora_merge_validates_everything_before_mutating(case, tmp_path):
    torch = pytest.importorskip("torch")
    from openmed.torch.meddecider_nli import merge_lora_adapter

    torch.manual_seed(0)
    model = _two_layer_model(torch)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    tensors = {
        "base_model.model.first.lora_A.weight": torch.randn(2, 4),
        "base_model.model.first.lora_B.weight": torch.randn(3, 2),
        "base_model.model.proj.lora_A.weight": torch.randn(2, 4),
        "base_model.model.proj.lora_B.weight": torch.randn(3, 2),
    }
    config = {}
    if case == "rank_mismatch":
        config["r"] = 32  # declared rank disagrees with the rank-2 tensors
    elif case == "nonfinite":
        tensors["base_model.model.proj.lora_B.weight"][0, 0] = float("nan")
    elif case == "extra_tensor":
        tensors["base_model.model.proj.lora_magnitude_vector"] = torch.ones(3)
    elif case == "missing_up":
        del tensors["base_model.model.proj.lora_B.weight"]
    else:
        tensors["base_model.model.proj.lora_A.weight"] = torch.randn(2, 5)
    with pytest.raises(ValueError):
        merge_lora_adapter(model, _adapter(tmp_path, tensors, **config))
    for name, value in model.named_parameters():
        assert torch.equal(value, before[name]), "weights changed before refusal"


def test_lora_merge_rounds_low_precision_weights_once(tmp_path):
    torch = pytest.importorskip("torch")
    from openmed.torch.meddecider_nli import merge_lora_adapter

    model = torch.nn.Sequential()
    model.add_module("proj", torch.nn.Linear(1, 1, bias=False))
    model = model.to(torch.bfloat16)
    with torch.no_grad():
        model.proj.weight.fill_(1.0)
    # 1.0 + 0.00392 rounds to 1.0078125 in bfloat16 when accumulated in float32;
    # rounding the delta to bfloat16 first would leave the weight at 1.0.
    tensors = {
        "base_model.model.proj.lora_A.weight": torch.ones(1, 1),
        "base_model.model.proj.lora_B.weight": torch.full((1, 1), 0.00392),
    }
    merge_lora_adapter(model, _adapter(tmp_path, tensors, r=1, lora_alpha=1))
    assert model.proj.weight.dtype == torch.bfloat16
    assert model.proj.weight.item() == 1.0078125


def test_cli_hides_runtime_failure_details(tmp_path, monkeypatch, capfd):
    from openmed.cli._output import CliError

    def leaky_loader(directory, *, device=None):
        raise CliError("SYNTHETIC-PRIVATE-CANARY", code="synthetic_canary_code")

    monkeypatch.setattr("openmed.core.models.HF_AVAILABLE", True)
    monkeypatch.setattr(
        "openmed.torch.meddecider_nli.load_meddecider_classifier", leaky_loader
    )
    files = {
        "labels": MEDDECIDER_NLI_LABEL_MAPPING,
        "policy": {"min_per_class": 1},
        "development": synthetic_split("dev"),
        "evaluation": synthetic_split("eval"),
    }
    paths = {}
    for name, value in files.items():
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(value))
    args = ["nli-qualify", "--artifact", str(write_artifact(tmp_path / "a"))]
    args += ["--runtime", "meddecider", "--json"]
    for flag, name in (
        ("--label-mapping", "labels"),
        ("--policy", "policy"),
        ("--development", "development"),
        ("--evaluation", "evaluation"),
    ):
        args += [flag, str(paths[name])]
    assert main(args) == 1
    captured = capfd.readouterr()
    payload = json.loads(captured.out)["data"]
    assert payload["reasons"] == ["local_inference_unavailable"]
    assert "CANARY" not in captured.out + captured.err
    assert "synthetic_canary_code" not in captured.out + captured.err
