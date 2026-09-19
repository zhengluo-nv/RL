# Weight Refit: Choosing a Transport

Weight refit copies updated policy weights into the rollout model. Choose the
topology first, then select one transport with
`policy.generation.refit_transport`.

## Pick a Transport

| Topology | `refit_transport` | Transport | Use when |
|---|---|---|---|
| `colocated.enabled: true` | `null` | CUDA IPC | Policy and rollout workers share GPUs; vLLM uses ZMQ plus CUDA IPC handles and SGLang uses Ray CUDA IPC. |
| `colocated.enabled: true` | `mcore` | MCore wake-reshard | Megatron policy and generation share GPUs. |
| `colocated.enabled: false` | `null` | NCCL broadcast | vLLM and Megatron use the default full-weight collective; SGLang uses its own weight-update group. |
| `colocated.enabled: false` | `mcore` | MCore native refit | Megatron policy sends directly to Megatron generation through an MCore copy service. |
| `colocated.enabled: false` | `nccl_reshard` | NCCL reshard | Provides the best performance for large models (>100s B models). |
| `colocated.enabled: false` | `vllm_zmq_sparse` | Sparse delta over ZeroMQ | The link is bandwidth-limited and workers can reach a relay over TCP. |
| `colocated.enabled: false` | `vllm_s3_sparse` | Sparse delta through S3 | Workers communicate through shared object storage. |
| `colocated.enabled: false` | `nixl` | NIXL checkpoint engine | The cluster has a fast UCX/RDMA fabric for full-weight refit. |
| `colocated.enabled: false` | `model_express` | ModelExpress versioned NIXL refit | A Megatron policy publishes native shards to vLLM through an MX trainer mesh. |

`null` is the default. For non-colocated Megatron generation it selects the
packed collective; Megatron's native refit must be selected explicitly with
`mcore`. The sparse transports read only `refit_cfg.sparse`; NIXL reads only
`refit_cfg.nixl`. Because one selector chooses the transport, sparse delta and
NIXL cannot both be active.

## ModelExpress

The ModelExpress integration supports GRPO with a Megatron policy and
non-colocated vLLM generation. Install the same ModelExpress revision in the
controller, Megatron worker, and vLLM worker environments, and run a matching MX
server. This integration requires the public trainer-mesh APIs
(`create_trainer_mesh`, `TrainerTensorsMetadata`, and mesh-backed weight versions)
and generator staging APIs available on ModelExpress main. Older source-slot
APIs and the removed `ModelExpressVllmWorker` bootstrap hook are not used.

```yaml
policy:
  offload_policy_before_refit: false
  generation:
    backend: vllm
    colocated:
      enabled: false
    refit_transport: model_express
    refit_cfg:
      model_express:
        server_url: mx-server:8001  # null uses MX server environment settings
        staging_buffer_bytes: null  # null stages the complete local update
        staging_buffers_count: 1  # use 2 to overlap bounded reads and installation
```

For bounded streaming, set `staging_buffer_bytes` to a positive capacity in
bytes per receive arena. ModelExpress owns transfer planning, receiver buffers,
verification, and vLLM installation. The optional arena count must be positive;
two arenas can overlap reads and installation when bounded streaming is enabled.
The framework uses the standard vLLM worker and initializes the MX generator
client through its existing collective RPC extension.

Each trainer binds its native Megatron tensors and returns its worker identity
and `TrainerTensorsMetadata`. The controller registers every publisher in one
trainer mesh, including redundant data-parallel replicas. Each synchronization
creates a mesh-backed `FULL_TENSOR` weight version, publishes every trainer
shard, checks readiness, and waits for every vLLM engine to finish installation
before retiring the version and releasing trainer publications. The mesh is
deleted when the synchronizer shuts down.

Megatron publication explicitly uses `IN_PLACE`: NIXL reads the existing live
parameter storage. Keep training quiesced until the synchronous refit returns;
the trainer needs stable tensor addresses and must not mutate source weights
while generators read them. MX owns its listeners through `MX_WORKER_HOST`,
`MX_METADATA_PORT`, and `MX_WORKER_GRPC_PORT`; configure reachable addresses and
non-conflicting port bases in the deployment manifest.

Generated FP8 KV-cache scales and policy offload before refit are unsupported.
GPU correctness and performance depend on the model, precision, and topology
and require a separate end-to-end run; focused unit tests do not qualify them.

## vLLM Reload API

For non-colocated vLLM using the default NCCL transport, set
`policy.generation.vllm_cfg.refit_with_reload_api: true` to install refitted
weights through vLLM's native `reload_weights` API. The flag is off by default.
Use this path when vLLM's post-load processing is required after every refit,
for example `flashinfer_trtllm` MoE models where `process_weights_after_loading`
reshapes expert weight buffers.

Early measurements from this PR show that reload refit can be faster for Llama
8B and Qwen3 32B, while Qwen3 30B-A3B was about 44% slower. MXFP8 reload refit
was about 2x slower in the measured run and used higher peak memory. Leave
additional KV-cache headroom for MXFP8, for example by lowering
`policy.generation.vllm_cfg.gpu_memory_utilization`.

The reload API path currently supports only `policy.generation.backend: vllm`,
`policy.generation.colocated.enabled: false`, and
`policy.generation.refit_transport: null`. It is explicitly unsupported with
`nccl_reshard`, ModelOpt quantization (`policy.generation.quant_cfg`), sparse
refit transports, and checkpoint-engine/NIXL transports. Eagle/MTP draft
weights refitted from the trainer are not supported yet; use the legacy loader
for those cases.

## Constraints

| Transport | Generation backend | Policy backend | Quantization and MoE |
|---|---|---|---|
| Colocated CUDA IPC | vLLM or SGLang | DTensor or Megatron | Uses the generation backend's standard loader. |
| NCCL | vLLM or Megatron | DTensor or Megatron | Uses the standard full-weight loader. |
| NCCL + reload API | vLLM | DTensor or Megatron | Default non-colocated NCCL only. Rejects colocated refit, `nccl_reshard`, sparse delta, NIXL/checkpoint-engine transports, ModelOpt quantization (`real_quant` or `quant_cfg`), and Eagle/MTP trainer-refit draft weights. |
| MCore native refit | Megatron | Megatron | Supports MCore's configured copy-service backend. |
| SGLang NCCL weight-update group | SGLang | Megatron | Trainer rank 0 broadcasts finalized HF tensors to the engine leaders. |
| NCCL reshard | vLLM or Megatron | Megatron | Supports BF16 and the documented FP8/MXFP8 combinations; see the [NCCL Reshard Refit design](../design-docs/nccl-reshard-refit.md). |
| Sparse delta | vLLM | Megatron | BF16/FP16, unquantized rollout only. |
| NIXL, full weights | vLLM | DTensor or Megatron | Supports the standard full-weight FP8 loader. DTensor FP8 KV-cache scale transfer is not yet supported. |
| NIXL, sharded experts | vLLM | DTensor or Megatron | Unquantized BF16/FP16 Triton MoE only; FP8/MXFP8 and dynamic expert placement are rejected. |

Non-colocated SGLang generation is supported with a Megatron policy and
`refit_transport: null`; it creates its own NCCL weight-update group. The NIXL
restrictions are on the generation backend; both Megatron and DTensor policy
workers can send weights. Sparse delta is currently limited to GRPO. NIXL is
initialized by the GRPO and distillation setup paths; PPO currently requires
colocated generation.

## Minimal Configuration

Colocated vLLM and SGLang refit need no transport configuration:

```yaml
policy:
  generation:
    colocated:
      enabled: true
    refit_transport: null
```

For non-colocated NCCL, change the topology and leave the selector unset:

```yaml
policy:
  offload_policy_before_refit: false
  generation:
    colocated:
      enabled: false
    refit_transport: null
```

Large quantized exports can temporarily need more memory than training itself.
Set `offload_policy_before_refit: true` to drop completed gradient buffers before
the collective export. The same lifecycle can also move the optimizer and clear
Transformer Engine workspaces:

```yaml
policy:
  offload_policy_before_refit: true
  offload_optimizer_for_refit: true
  megatron_cfg:
    fp8_cfg:
      enabled: true
      force_clear_fp8_caches: true
  generation:
    colocated:
      enabled: false
    refit_transport: null
```

This option requires the Megatron policy backend. It applies to non-colocated
vLLM collective and `nccl_reshard` transports, plus non-colocated Megatron
generation. Unsupported combinations fail during synchronizer setup. It is
disabled by default because CPU offload adds transfer overhead when the export
already fits in trainer GPU memory.

For native MCore refit, select it explicitly:

```yaml
policy:
  generation:
    backend: megatron
    refit_transport: mcore
    mcore_generation_config:
      refit_backend: nccl  # gloo | nccl | nccl_m2n (nvshmem is broken; see #3646)
```

For NCCL reshard with Megatron policy training and vLLM generation:

```yaml
policy:
  generation:
    colocated:
      enabled: false
    refit_transport: nccl_reshard
```

`offload_policy_before_refit` also works with this transport. The reshard moves
only parameters, so releasing gradient buffers, optimizer state, and caches
before transfer is safe.

For sparse delta, select one data plane and configure its scope:

```yaml
policy:
  generation:
    colocated:
      enabled: false
    refit_transport: vllm_zmq_sparse  # or vllm_s3_sparse
    refit_cfg:
      sparse:
        delta_compression:
          encoding: xor
        storage:
          s3_bucket: null  # required for vllm_s3_sparse
```

For NIXL, select the checkpoint engine and configure its scope:

```yaml
policy:
  generation:
    colocated:
      enabled: false
    refit_transport: nixl
    refit_cfg:
      nixl:
        update_weights_bucket_memory_ratio: 0.05
        device: cuda
        backend_name: UCX
        release_after_refit: false
        shard_expert_weights: false
```

## Learn More

- [NCCL Reshard Refit](../design-docs/nccl-reshard-refit.md) describes its
  requirements, architecture, and shard-to-shard transfer.
- [Sparse Delta Refit](../design-docs/sparse-delta-refit.md) explains baseline,
  compression, ZeroMQ, and S3 behavior.
- [Checkpoint-Engine Refit](checkpoint-engine-refit.md) covers NIXL setup,
  performance tuning, FP8, sharded experts, and fault tolerance.
- [Checkpoint Engines](../design-docs/checkpoint-engines.md) describes the
  checkpoint-engine protocol and implementation.
- [Training and Generation Backends](../about/backends.md) summarizes backend
  compatibility.
