# Multi-Stream Synchronization for Host Compute Overlap

**Date:** 2026-06-08  
**Status:** Draft — for team discussion  
**Authors:** @JRosenkranz
**Resolves:** ProgramExecutionRFC Unresolved Question #1 (stream events)

---

## Problem Statement

Today, `launchJobPlan` submits all operations (host compute, DMA, device compute) to a single stream in FIFO order. The device sits idle during every host compute + DMA phase. For tiled workloads with hundreds of iterations, this idle time can become a bottleneck.

**Insight:** Host compute for iteration N+1 does not depend on device compute from iteration N — it only needs `LogicalAddresses` known at launch time. This enables overlapping host compute preparation with device execution.

**Goal:** A general-purpose inter-stream synchronization mechanism that allows `launchJobPlan` to distribute operations across multiple streams based on compiler-provided hints, minimizing device idle time.

## Stream Topology

Two streams with a fixed producer/consumer relationship:

```
Stream A (Host Compute):  [HC_0][Signal_0][HC_1][Signal_1]...[HC_N][Signal_N]
                              ↓               ↓                    ↓
Stream B (DMA + Compute): [Wait_0][DMA_0][Comp_0][Wait_1][DMA_1][Comp_1]...[Wait_N][DMA_N][Comp_N]
```

With backpressure (lookahead K=2):

```
Stream A: [HC_0][Sig_0][HC_1][Sig_1][RevWait_0][HC_2][Sig_2][RevWait_1]...
Stream B: [Wait_0][DMA_0][Comp_0][RevSig_0][Wait_1][DMA_1][Comp_1][RevSig_1]...
```

The mechanism is general-purpose — any stream can signal, any stream can wait. The first consumer is program correction, but the primitive is workload-agnostic and supports N-stream topologies.

---

## Option 1: Host Callbacks (No New Primitives)

### Concept

Use the existing `RuntimeOperationHostCallback` to signal readiness. The callback dynamically injects dependent operations into Stream B. No new types or Scheduler changes required.

### Runtime API

Uses existing primitives only:
- `RuntimeOperationHostCallback` — enqueued on Stream A as the host compute operation
- `registerHostCallback` — attached to operations for completion notification

### Implementation Sketch: `launchJobPlan`

```cpp
void launchJobPlan(const JobPlan& job_plan, const std::vector<at::Tensor>& args) {
  auto device = c10::Device(c10::DeviceType::PrivateUse1, -1);
  auto hostStream = getStreamFromPool(device, /*priority=*/0);   // Stream A
  auto computeStream = getCurrentStream(device);                  // Stream B

  LaunchContext ctx{args};
  std::mutex inject_mu;

  for (const auto& step : job_plan.steps) {
    if (step->streamHint() == StreamHint::HostCompute) {
      // Enqueue host compute on Stream A
      auto op = step->construct(ctx);

      // Attach callback: when host compute finishes, inject DMA+Compute into Stream B
      auto pending_ops = buildDependentOps(job_plan, step, ctx);
      op->registerHostCallback([&inject_mu, &computeStream, ops = std::move(pending_ops)]
                               (void*) {
        std::lock_guard<std::mutex> lock(inject_mu);
        computeStream.getRuntimeHandle()->launchOperation(ops);
      }, nullptr);

      hostStream.getRuntimeHandle()->launchOperation(std::move(op));
    }
    // DMA + Compute ops are NOT enqueued here — they're injected from the callback
  }
}
```

### Backpressure (Host Callbacks)

```cpp
// Limit lookahead to K using a counting semaphore
std::counting_semaphore<K> slots(K);

// Before starting next host compute iteration:
slots.acquire();  // blocks host thread if K iterations are in-flight

// In the callback attached to the Compute op (after it completes):
compute_op->registerHostCallback([&slots](void*) { slots.release(); }, nullptr);
```

### Characteristics

| Property | Assessment |
|----------|-----------|
| New runtime types | None |
| Scheduler changes | None |
| Pre-enqueue all ops | **No** — Stream B ops injected dynamically from callback context |
| Thread safety | Mutex required for cross-thread injection into Stream B |
| Backpressure | Blocking semaphore on host thread — cannot express in-stream |
| Testability | **Non-deterministic** — operations appear on Stream B at callback-firing time, making timing-dependent behavior harder to reproduce in tests |
| Profiling | Operations appear "out of thin air" on Stream B — not visible at submission time |

---

## Option 2: Events (New Synchronization Primitives) — Recommended

### Concept

Introduce `Event`, `EventSignalOp`, `EventWaitOp` as `RuntimeOperation` subclasses. All operations are pre-enqueued on both streams upfront. The Scheduler handles blocked streams by skipping them.

### Runtime API (flex layer)

```cpp
class Event {
public:
  Event() : signaled_(false), poisoned_(false) {}
  void signal() { signaled_.store(true, std::memory_order_release); }
  void poison(RuntimeError err) { error_ = err; poisoned_.store(true, std::memory_order_release); }
  bool isSignaled() const { return signaled_.load(std::memory_order_acquire); }
  bool isPoisoned() const { return poisoned_.load(std::memory_order_acquire); }
  RuntimeError getError() const { return error_; }
private:
  std::atomic<bool> signaled_;
  std::atomic<bool> poisoned_;
  RuntimeError error_;
};

class EventSignalOp : public RuntimeOperation {
public:
  EventSignalOp(std::shared_ptr<Event> event) : event_(std::move(event)) {}
  std::shared_ptr<Event> getEvent() const { return event_; }
private:
  std::shared_ptr<Event> event_;
};

class EventWaitOp : public RuntimeOperation {
public:
  EventWaitOp(std::shared_ptr<Event> event) : event_(std::move(event)) {}
  std::shared_ptr<Event> getEvent() const { return event_; }
  bool isReady() const { return event_->isSignaled(); }
  bool isPoisoned() const { return event_->isPoisoned(); }
private:
  std::shared_ptr<Event> event_;
};
```

### Stream Assignment via JobPlan Hints

Operations carry a `StreamHint` that tells `launchJobPlan` which stream to target:

```cpp
enum class StreamHint {
  Default,       // no preference — goes to the "current" stream
  HostCompute,   // should run on the host-compute stream
  DMACompute,    // should run on the DMA+device-compute stream
};
```

Synchronization points are explicit steps in the JobPlan. They implement `construct()` like all other steps, producing their RuntimeOperation via the standard factory pattern:

```cpp
// LaunchContext extension — event registry for per-launch state
struct LaunchContext {
  const std::vector<at::Tensor>& inputs_outputs;

  std::shared_ptr<Event> getOrCreateEvent(int event_id) {
    auto [it, _] = events_.try_emplace(event_id, std::make_shared<Event>());
    return it->second;
  }

private:
  std::unordered_map<int, std::shared_ptr<Event>> events_;
};

class JobPlanStepEventSignal : public JobPlanStep {
public:
  JobPlanStepEventSignal(int event_id, StreamHint hint) : event_id_(event_id) {
    setStreamHint(hint);
  }
  std::unique_ptr<RuntimeOperation> construct(LaunchContext& ctx) const override {
    return std::make_unique<EventSignalOp>(ctx.getOrCreateEvent(event_id_));
  }
private:
  int event_id_;
};

class JobPlanStepEventWait : public JobPlanStep {
public:
  JobPlanStepEventWait(int event_id, StreamHint hint) : event_id_(event_id) {
    setStreamHint(hint);
  }
  std::unique_ptr<RuntimeOperation> construct(LaunchContext& ctx) const override {
    return std::make_unique<EventWaitOp>(ctx.getOrCreateEvent(event_id_));
  }
private:
  int event_id_;
};
```

Event steps use the same `construct()` interface as all other `JobPlanStep` subclasses. The shared `Event` object is managed by `LaunchContext` — two steps with the same `event_id` get the same `Event` instance. Since `LaunchContext` is created fresh per launch, Events are automatically fresh each time. The `Event` class intentionally omits a `reset()` method — freshness is guaranteed by construction, not mutation.

The `JobPlanBuilder` places signal/wait steps based on data dependencies derived from SpyreCode command types. `launchJobPlan` is a uniform dispatcher — no type-specific branching needed.

### Implementation Sketch: `launchJobPlan`

```cpp
void launchJobPlan(const JobPlan& job_plan, const std::vector<at::Tensor>& args) {
  auto device = c10::Device(c10::DeviceType::PrivateUse1, -1);

  // Check if multi-stream is needed
  bool needs_multi_stream = std::any_of(job_plan.steps.begin(), job_plan.steps.end(),
      [](const auto& s) { return s->streamHint() == StreamHint::HostCompute; });

  if (!needs_multi_stream) {
    // Fast path: single-stream as today
    auto stream = getCurrentStream(device);
    stream.launch(job_plan, args);
    return;
  }

  // Multi-stream path — both streams created eagerly at device initialization
  auto hostStream = getHostComputeStream(device);                 // Stream A
  auto computeStream = getCurrentStream(device);                  // Stream B (default PyTorch stream)

  LaunchContext ctx{args};
  std::vector<std::unique_ptr<RuntimeOperation>> host_ops, compute_ops;

  // Uniform construction loop — no type-specific branching
  // Event steps produce EventSignalOp/EventWaitOp via construct().
  // Steps sharing the same event_id get the same Event from LaunchContext.
  for (const auto& step : job_plan.steps) {
    auto& target = (step->streamHint() == StreamHint::HostCompute) ? host_ops : compute_ops;
    target.push_back(step->construct(ctx));
  }

  // Submit both streams — everything is pre-enqueued
  hostStream.getRuntimeHandle()->launchOperation(host_ops);
  computeStream.getRuntimeHandle()->launchOperation(compute_ops);
}
```

**Stream lifecycle:** Both streams are created eagerly at device initialization. The default PyTorch stream (ID 0) maps to the compute stream, following the existing pattern. The host-compute stream is a second well-known stream created by `RuntimeContext` at construction and registered in the torch-spyre stream pool during `initializeStreamPoolImpl()`. `SpyreStream::Synchronize()` waits on both internal streams.

### Backpressure (CUDA-Style Reverse Events)

Backpressure uses the same event primitive bidirectionally:

```cpp
// For tiled execution with lookahead K:
for (int i = 0; i < num_iterations; i++) {
  // Forward: Stream A signals, Stream B waits
  // (placed by JobPlanBuilder as JobPlanStepEventSignal/Wait)

  // Reverse (backpressure): Stream B signals after compute, Stream A waits before HC_{i+K}
  // Only activates when i >= K
}
```

The lookahead window K is determined by the number of pre-allocated correction buffers. This mirrors the standard CUDA double/triple-buffering pattern:

```
CUDA equivalent:
  cudaStreamWaitEvent(producerStream, consumed[slot]);  // backpressure wait
  launchProducer(producerStream, buffer[slot]);
  cudaEventRecord(produced[slot], producerStream);      // forward signal
  cudaStreamWaitEvent(consumerStream, produced[slot]);  // forward wait
  launchConsumer(consumerStream, buffer[slot]);
  cudaEventRecord(consumed[slot], consumerStream);      // backpressure signal
```

### Scheduler Changes (flex layer)

**Note:** This extends the RuntimeStream.md contract, which currently states "the runtime does not check for stream-to-stream dependencies" and delegates cross-stream coordination to `registerHostCallback`. Events promote inter-stream synchronization from a user-managed callback pattern to a Scheduler-native primitive. `registerHostCallback` remains available for ad-hoc coordination.

Streams using `EventSignalOp`/`EventWaitOp` must use `STRICT_ORDERING` mode. In `OP_ORDERING` mode, the Scheduler issues operations without waiting for prior completions — a signal could fire before preceding device work actually completes, violating synchronization semantics.

The Scheduler must be updated to handle `EventSignalOp` and `EventWaitOp` without deadlocking. The required behavioral changes:

1. **Round-robin across streams:** The Scheduler must not drain one stream before moving to another. It must cycle across all active streams, giving each a chance to make progress.

2. **EventWaitOp check sequence:** When a stream's head operation is an `EventWaitOp`, the Scheduler checks in order:
   - `isReady()` (signaled) → consume the WaitOp, proceed to next operation on this stream
   - `isPoisoned()` → put this stream in error state (cross-stream error propagation)
   - Neither → skip this stream (still blocked), move to next stream

3. **Process signal ops immediately:** When a stream's head operation is an `EventSignalOp`, the Scheduler marks the event as signaled (one atomic store) and moves on. This is a zero-cost operation with no hardware involvement.

4. **Consume wait ops on ready:** When a previously-blocked `EventWaitOp` becomes ready (event is signaled), the Scheduler consumes it and proceeds to the next operation on that stream.

5. **Deadlock detection:** If a full pass across all streams makes no progress (every stream is blocked on unsignaled waits), the Scheduler reports an error rather than hanging. This indicates a circular dependency bug in the plan.

6. **Error propagation via poisoned events:** When a stream enters error state, any events it was responsible for signaling are poisoned. Streams waiting on a poisoned event immediately enter error state themselves, preventing indefinite hangs.

### Error Handling

Sticky error model with cross-stream propagation:

| Scenario | Behavior |
|----------|----------|
| Host compute fails on Stream A | Stream A enters error state. EventSignalOp never fires. Event is poisoned. Stream B's EventWaitOp detects poison → Stream B enters error state. |
| DMA/Compute fails on Stream B | Stream B enters error state. Reverse EventSignalOp never fires. Reverse event is poisoned. Stream A's reverse EventWaitOp detects poison → Stream A enters error state. |
| Circular dependency (bug) | All streams blocked, no progress. Scheduler detects deadlock and reports. |

Errors surface to torch-spyre via `synchronize()` on either stream.

### Characteristics

| Property | Assessment |
|----------|-----------|
| New runtime types | 3 classes (~80 lines: Event, EventSignalOp, EventWaitOp) |
| Scheduler changes | Round-robin + skip-on-blocked + deadlock detection |
| Pre-enqueue all ops | **Yes** — matches existing `SpyreStream::launch()` pattern |
| Thread safety | Atomics only (in Scheduler); no mutexes at torch-spyre level |
| Backpressure | Expressed declaratively in-stream via reverse events |
| Testability | **Fully deterministic** — same plan always produces same execution order. All operations visible at submission time, making tests reproducible and debuggable |
| Profiling | All operations visible in trace at submission time |

---

## CUDA Comparison

| CUDA | Option 2 (Events) | Option 1 (Host Callbacks) |
|------|-------------------|--------------------------|
| `cudaStream_t` | `RuntimeStream*` | `RuntimeStream*` |
| `cudaEvent_t` | `Event` (shared_ptr) | N/A — no equivalent |
| `cudaEventCreate()` | `std::make_shared<Event>()` | N/A |
| `cudaEventRecord(event, stream)` | Enqueue `EventSignalOp(event)` on stream | N/A |
| `cudaStreamWaitEvent(stream, event)` | Enqueue `EventWaitOp(event)` on stream | N/A — inject ops from callback instead |
| `cudaLaunchHostFunc(stream, fn)` | `RuntimeOperationHostCallback` | Primary synchronization mechanism |
| `cudaStreamSynchronize()` | `RuntimeStream::synchronize()` | `RuntimeStream::synchronize()` |
| Backpressure (double-buffer + reverse events) | Reverse `EventSignalOp`/`EventWaitOp` pairs | `std::counting_semaphore` in callbacks |

**Key difference from CUDA:** Our events are software-only (atomic flags checked by the Scheduler), not hardware-backed. The API is designed to be forward-compatible — when future hardware supports native event record/wait in the command stream, the implementation swaps from software polling to hardware signals without changing the interface.

---

## Testing Strategy

### Unit Tests (flex-runtime level)

| Test | Validates |
|------|-----------|
| Event signal/wait basic | Signal on one stream unblocks wait on another |
| Event ordering | Operations after EventWaitOp don't execute until signal |
| Multiple events | Independent event pairs don't interfere |
| Backpressure | Reverse events stall producer when K slots are in-flight |
| Deadlock detection | Scheduler reports error when all streams are mutually blocked |
| Error propagation | Poisoned event fails the waiting stream |
| Event per-launch freshness | Fresh LaunchContext per launch means events are always new (no reset needed) |
| Single-stream fallback | EventSignalOp/EventWaitOp on the same stream work (signal already set when wait reached) |

### Integration Tests (torch-spyre level)

| Test | Validates |
|------|-----------|
| Multi-stream program correction | Tiled workload with host compute on Stream A, DMA+compute on Stream B produces correct results |
| Single-iteration multi-stream | Non-tiled JobPlan with host compute still works correctly |
| Fallback path | JobPlan without HostCompute hints uses single-stream (no regression) |
| Correctness under overlap | Results identical whether multi-stream or single-stream (determinism) |
| Backpressure correctness | With K=1, execution is effectively serial; results still correct |

### Performance Validation

| Metric | Measurement |
|--------|-------------|
| Device idle time reduction | Profile trace: compare device utilization single-stream vs multi-stream |
| Overhead of events | Micro-benchmark: throughput of Signal+Wait pairs vs. no synchronization |
| Backpressure tuning | Sweep K values, measure throughput vs. memory usage |

### Testability: Option 1 vs Option 2

**Option 1 (Host Callbacks):** Tests must account for non-deterministic timing. Operations appear on Stream B only when callbacks fire, which depends on thread scheduling. Reproducing failures requires careful synchronization in test harnesses. Race conditions between callback execution and stream queries make assertions fragile.

**Option 2 (Events):** Tests are fully deterministic. All operations are submitted upfront — the execution order is a function of the plan, not thread scheduling. A given JobPlan always produces the same stream contents, making tests reproducible, debuggable, and CI-stable.

---

## Recommendation

**Option 2 (Events)** is recommended for the following reasons:

| Criterion | Winner | Rationale |
|-----------|--------|-----------|
| Reusability | Events | General primitive; any stream can signal/wait any other |
| Pre-enqueue model | Events | Matches existing `SpyreStream::launch()` pattern |
| CUDA familiarity | Events | 1:1 mapping to `cudaEventRecord`/`cudaStreamWaitEvent` |
| Backpressure | Events | Declarative in-stream; no host-thread blocking |
| Testability | Events | Deterministic execution order; reproducible tests |
| Profiling | Events | Full static trace visibility |
| Consumer complexity | Events | Declarative signal/wait vs. manual callback wiring with mutexes |
| Upfront cost | Callbacks | No new types — but pays ongoing complexity tax at every use site |
| Hardware forward-compat | Events | API unchanged when hardware events arrive |

## Scope

**Building:**
1. flex-runtime: `Event`, `EventSignalOp`, `EventWaitOp` + Scheduler changes + error propagation
2. torch-spyre (JobPlan): `JobPlanStepEventSignal`, `JobPlanStepEventWait` + `StreamHint` on all steps
3. torch-spyre (launchJobPlan): Multi-stream dispatch logic
4. torch-spyre (JobPlanBuilder): Placement of signal/wait steps from SpyreCode commands

**Not building (explicit non-goals):**
- Hardware-backed events (future chip work)
- Tiled execution with cross-iteration events (see Future section below)
- Cross-JobPlan pipelining (separate future design)
- Stream priorities for event scheduling
- More than 2 streams (design supports N, first implementation is 2)

---

## Future: Tiled Execution

The initial implementation targets non-tiled workloads where all operations are constructed in a single pass with one `LaunchContext`. When tiled execution is added (iterating N times over a JobPlan with per-tile tensor offsets), cross-iteration events become necessary for backpressure.

**The problem:** Backpressure requires events that span iterations — `Compute_i` signals, `HC_{i+K}` waits. If `LaunchContext` is created fresh per iteration (as required for adjusted tensor addresses), events cannot be shared across iterations through the current `LaunchContext` registry alone.

**Planned approach:** Introduce a `TiledLaunchContext` that owns the cross-iteration event registry and spans the entire tiled launch. Per-iteration `LaunchContext` gains a pointer to it plus an `iteration_index`. Event IDs are iteration-qualified (e.g., `base_id * N + iteration`) so each iteration gets unique events. `JobPlanStepEventWait` gains an `iteration_offset` field to reference events from earlier iterations.

This approach:
- Keeps `LaunchContext` per-iteration (backward-compatible, tensor addresses still vary)
- Maintains the "pre-enqueue all ops" model (full iteration loop completes before submission)
- Eliminates the need for `Event::reset()` — each iteration uses a fresh event by ID
- Requires no changes to the flex-layer `Event`/`EventSignalOp`/`EventWaitOp` primitives

---

## References

- [EventSynchronizationRFC.md](/tmp/devel/src/runtime_sprint_planning/EventSynchronizationRFC.md) — original RFC proposing events for program correction pipelining
- [RuntimeStream.md](/tmp/devel/src/flex/docs/RFCs/RuntimeStream.md) — stream-based execution model, host callback synchronization discussion
- [ProgramExecutionSpec.md] — unresolved question #1 (stream events) that this design resolves
- CUDA Programming Guide: Stream Synchronization, Events
