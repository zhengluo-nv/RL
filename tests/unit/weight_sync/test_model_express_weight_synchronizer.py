# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, call, create_autospec, patch

import pytest

from nemo_rl.weight_sync.factory import create_weight_synchronizer
from nemo_rl.weight_sync.model_express import ModelExpressWeightSynchronizer


def _sync():
    policy = MagicMock()
    policy.cfg = {
        "megatron_cfg": {"enabled": True},
        "offload_policy_before_refit": False,
    }
    policy.initialize_model_express.return_value = {"trainer-0": object()}
    generation = MagicMock()
    generation.cfg = {"model_name": "test/model"}
    control = MagicMock()
    control.create_trainer_mesh.return_value = SimpleNamespace(mesh_id="mesh-1")
    control.create_weight_version.return_value = SimpleNamespace(
        version_id="version-1",
        ref=SimpleNamespace(version_id="version-1"),
    )
    control.get_weight_version.return_value = SimpleNamespace(
        state=SimpleNamespace(value="READY")
    )
    return (
        ModelExpressWeightSynchronizer(
            policy=policy,
            generation=generation,
            control_client=control,
            payload_format=SimpleNamespace(value="FULL_TENSOR"),
        ),
        policy,
        generation,
        control,
    )


def test_sync_runs_complete_version_lifecycle():
    sync, policy, generation, control = _sync()

    sync.init_communicator()
    sync.sync_weights(timer=MagicMock(time=lambda _name: nullcontext()))

    policy.initialize_model_express.assert_called_once_with(server_url=None)
    generation.initialize_model_express.assert_called_once_with(server_url=None)
    control.create_weight_version.assert_called_once()
    assert control.create_weight_version.call_args.kwargs["trainer_mesh_id"] == "mesh-1"
    assert control.create_weight_version.call_args.kwargs["version_number"] == 1
    policy.publish_model_express_version.assert_called_once()
    generation.update_weights_from_model_express.assert_called_once()
    control.delete_weight_version.assert_called_once_with("version-1")
    policy.release_model_express_version.assert_called_once()
    assert sync.is_stale is False


def test_sync_preserves_redundant_publishers_in_trainer_mesh():
    sync, policy, _generation, control = _sync()
    metadata = object()
    policy.initialize_model_express.return_value = {
        "trainer-0": metadata,
        "trainer-1": metadata,
    }

    sync.init_communicator()
    sync.sync_weights()

    control.create_trainer_mesh.assert_called_once_with(
        model_name="test/model",
        idempotency_key=sync._run_id,
        workers={"trainer-0": metadata, "trainer-1": metadata},
    )


def test_sync_retires_and_releases_version_when_update_fails():
    sync, policy, generation, control = _sync()
    sync.init_communicator()
    generation.update_weights_from_model_express.side_effect = RuntimeError(
        "apply failed"
    )

    with pytest.raises(RuntimeError, match="apply failed"):
        sync.sync_weights()

    assert control.mock_calls[-1] == call.delete_weight_version("version-1")
    policy.release_model_express_version.assert_called_once()
    assert sync.is_stale is True


def test_retry_after_failed_transfer_uses_fresh_idempotency_key():
    sync, _policy, generation, control = _sync()
    sync.init_communicator()
    generation.update_weights_from_model_express.side_effect = [
        RuntimeError("apply failed"),
        None,
    ]

    with pytest.raises(RuntimeError, match="apply failed"):
        sync.sync_weights()
    sync.sync_weights()

    first, second = control.create_weight_version.call_args_list
    assert first.kwargs["version_number"] == 1
    assert second.kwargs["version_number"] == 1
    assert first.kwargs["idempotency_key"].split(":")[1] == "1"
    assert second.kwargs["idempotency_key"].split(":")[1] == "1"
    assert first.kwargs["idempotency_key"] != second.kwargs["idempotency_key"]
    assert sync.is_stale is False


def test_sync_requires_ready_version_before_generator_update():
    sync, policy, generation, control = _sync()
    sync.init_communicator()
    control.get_weight_version.return_value = SimpleNamespace(
        state=SimpleNamespace(value="STAGING")
    )

    with pytest.raises(RuntimeError, match="did not become READY"):
        sync.sync_weights()

    generation.update_weights_from_model_express.assert_not_called()
    control.delete_weight_version.assert_called_once_with("version-1")
    policy.release_model_express_version.assert_called_once()


@pytest.mark.parametrize(
    ("generation_backend", "colocated", "megatron_enabled"),
    [
        ("vllm", True, True),
        ("vllm", False, False),
        ("sglang", False, True),
        ("megatron", False, True),
        ("dynamo", False, True),
    ],
)
def test_factory_rejects_unsupported_model_express_topologies(
    generation_backend: str, colocated: bool, megatron_enabled: bool
):
    sync, policy, generation, _control = _sync()
    policy.cfg["megatron_cfg"]["enabled"] = megatron_enabled
    generation.cfg["refit_transport"] = "model_express"

    with pytest.raises(ValueError, match="non-colocated vLLM"):
        create_weight_synchronizer(
            policy=policy,
            generation=generation,
            generation_backend=generation_backend,
            colocated=colocated,
        )


def test_factory_routes_model_express():
    policy = MagicMock()
    policy.cfg = {
        "megatron_cfg": {"enabled": True},
        "offload_policy_before_refit": False,
    }
    generation = MagicMock()
    generation.cfg = {
        "model_name": "test/model",
        "refit_transport": "model_express",
        "refit_cfg": SimpleNamespace(
            model_express=SimpleNamespace(server_url="mx-server:50051")
        ),
    }

    with patch(
        "nemo_rl.weight_sync.model_express.ModelExpressWeightSynchronizer"
    ) as constructor:
        result = create_weight_synchronizer(
            policy=policy,
            generation=generation,
            generation_backend="vllm",
            colocated=False,
        )

    assert result is constructor.return_value
    constructor.assert_called_once_with(
        policy=policy,
        generation=generation,
        server_url="mx-server:50051",
    )


def test_shutdown_deletes_mesh_before_closing_control_client():
    sync, _policy, _generation, control = _sync()
    sync.init_communicator()

    sync.shutdown()
    sync.shutdown()

    control.delete_trainer_mesh.assert_called_once_with("mesh-1")
    assert control.mock_calls[-3:] == [
        call.delete_trainer_mesh("mesh-1"),
        call.close(),
        call.close(),
    ]


def test_failed_retirement_does_not_release_in_place_tensors():
    sync, policy, _generation, control = _sync()
    sync.init_communicator()
    control.delete_weight_version.side_effect = RuntimeError("active reader lease")

    with pytest.raises(RuntimeError, match="active reader lease"):
        sync.sync_weights()

    policy.release_model_express_version.assert_not_called()
    assert sync.is_stale


def test_synchronizer_matches_public_model_express_api():
    mx = pytest.importorskip("modelexpress_rl")
    binding = mx.TrainerTensorsMetadata(
        logical_shard_id="partition-a", metadata_endpoint="trainer:6555"
    )
    policy = MagicMock()
    policy.initialize_model_express.return_value = {"trainer-0": binding}
    generation = MagicMock()
    generation.cfg = {"model_name": "test/model"}
    control = create_autospec(mx.ModelExpressControlClient, instance=True)
    control.create_trainer_mesh.return_value = mx.TrainerMesh(
        mesh_id="mesh-1",
        model_name="test/model",
        generation=1,
        workers={"trainer-0": binding},
    )
    version = mx.WeightVersion(
        version_id="version-1",
        model_name="test/model",
        payload_format=mx.WeightPayloadFormat.FULL_TENSOR,
        layout_signature="layout-a",
        state=mx.WeightVersionState.READY,
        created_at_unix_ms=0,
        trainer_mesh_id="mesh-1",
        trainer_mesh_generation=1,
        version_number=1,
    )
    control.create_weight_version.return_value = version
    control.get_weight_version.return_value = version
    sync = ModelExpressWeightSynchronizer(
        policy=policy,
        generation=generation,
        control_client=control,
        payload_format=mx.WeightPayloadFormat.FULL_TENSOR,
    )

    sync.init_communicator()
    sync.sync_weights()
    sync.shutdown()

    generation.update_weights_from_model_express.assert_called_once_with(version.ref)
    control.create_weight_version.assert_called_once_with(
        model_name="test/model",
        idempotency_key=f"{sync._run_id}:1:1",
        payload_format=mx.WeightPayloadFormat.FULL_TENSOR,
        trainer_mesh_id="mesh-1",
        version_number=1,
    )
