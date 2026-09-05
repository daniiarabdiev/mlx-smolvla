# LeRobot async-inference pickle — unsent follow-up draft

Reviewed 2026-09-05. Nothing has been submitted upstream.

## Existing report and reporting route

This is already reported in [LeRobot issue #3047](https://github.com/huggingface/lerobot/issues/3047),
which was open when checked and links a proposed fix in
[PR #3048](https://github.com/huggingface/lerobot/pull/3048). Do not file a duplicate
public issue. The existing report includes a reproduction and covers both
server and client deserialization.

The current [LeRobot security policy](https://github.com/huggingface/lerobot/security/policy)
requires reproduction on an exact current-main commit and private reporting
through GitHub's vulnerability-reporting form or `security@huggingface.co`.
This review inspected v0.6.1; it did not reproduce an exploit on current main.
The draft below is a follow-up basis, not a submission-ready new vulnerability
report. Before sending it, establish whether the existing report/fix already
covers current main and follow upstream's disclosure policy.

## Draft follow-up

**Subject:** v0.6.1 async-inference pickle exposure and migration guidance

LeRobot v0.6.1's async-inference transport accepts unauthenticated pickle
payloads over gRPC. In the pinned
[policy server](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/async_inference/policy_server.py),
`SendPolicyInstructions` deserializes `PolicySetup.data` before its type check;
`SendObservations` deserializes the reassembled observation bytes. The server
uses `add_insecure_port` without peer authentication. A peer that can reach the
port can supply a pickle that executes code with the server process's
privileges before application validation (CWE-502). Network reachability is the
precondition; loopback restricts exposure to local peers but does not make
untrusted pickle safe.

For a minimal reproduction pointer, see the already public
[issue #3047 proof of concept](https://github.com/huggingface/lerobot/issues/3047).
Its sequence connects to an owned test server, calls `Ready`, then supplies a
crafted `PolicySetup` to `SendPolicyInstructions`. We have not run that exploit
or connected to any third-party server during this documentation review.

We would welcome guidance on the supported migration path for v0.6.1 clients.
A versioned JSON envelope plus safetensors, or NumPy tensor serialization with
pickle disabled, would remove executable object reconstruction. Validate shapes,
dtypes, payload sizes, and metadata before constructing runtime objects. If an
allowlisting unpickler is considered as an interim measure, its complete object
and reconstruction surface needs dedicated review; it should not be presented
as equivalent to replacing pickle. Peer authentication remains a separate
transport concern.

This project currently restricts compatible serving to loopback by default and
requires explicit trust for remote serving. These controls limit exposure;
they do not fix the upstream wire format. Its existing security boundaries are
documented in the [project security policy](../../.github/SECURITY.md) and
[architecture](../ARCHITECTURE.md#lerobot-061-async-inference-wire-contract).
