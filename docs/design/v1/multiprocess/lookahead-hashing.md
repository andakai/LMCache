# Lookahead chunk identities

A chunk covering KV tokens `[s, e)` can depend on accepted tokens through
`e + H`. The IPC key carries the complete packed token sequence, the KV range
`start/end`, and `num_hash_lookahead_tokens = H`. The dependency tokens identify
KV; they are not additional KV to transfer. A positive-H key is rejected if
its token sequence does not reach `end + H`.

For chunk size C, the hasher emits only
`floor(max(0, N - H) / C)` hashes. It hashes windows `[i*C, (i+1)*C + H)` in
a rolling prefix chain. An explicit `end` bounds the KV range, while dependency
tokens can lie past it. List and packed-buffer entry points produce identical
digests. Session hashing uses the same windows incrementally and refuses an
explicit end that lacks dependencies.

H=0 preserves the legacy chain exactly. H>0 salts the initial prefix with
`LMCacheLookahead`, schema version 1, C and H (packed uint32 fields). Thus old
keys and different dependency lengths cannot alias. Engine adapters must also
isolate model/draft-state configurations through their model namespace; H
alone does not identify a draft model or physical layout.

Example: C=4, H=1, N=8 permits one chunk. Appending token 9 makes the second
chunk's key available, but only KV tokens 5–8 are stored. Changing token 9
changes that key and every later chained key, while preserving the first key.

The IPC field is present in both msgspec and gRPC schemas. Old payloads decode
as H=0. An old server may ignore a new field, so a client must negotiate support
before sending H>0; adding the field alone does not make an old server safe.
The non-blend server advertises `lookahead_hashes_v1`; scheduler and worker
adapters check it at initialization and on recovery before accepting a server
as healthy. CacheBlend's non-prefix matching rejects positive-H keys and does
not advertise support.

The vLLM MP adapter resolves H from engine configuration, namespaces the model
with the model/draft/parallel/dtype identity, and sends the same H from scheduler
and workers. Store metadata caps KV at `min(computed, allocated, N-H, policy)`
while retaining all accepted token bytes in the IPC key. Client-generated and
server-generated events carry the suffix separately. Existing preemption and
store completion ownership are retained; no new finish-time store is queued.
In-process integrations and the vLLM builtin fallback are separate adapters.

For H>0, the vLLM MP adapter combines request cache salt and LoRA name/path
into a stable request salt shared by lookup, store and retrieve. Local LoRA
integer IDs do not identify KV across processes. Prompt embeddings bypass
external lookup and store because placeholder token IDs do not identify their
contents. H=0 retains the existing request salt behavior.
