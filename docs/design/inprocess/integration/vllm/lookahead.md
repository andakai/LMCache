# In-process lookahead cache keys

The adapter separates accepted hash context from the scheduled KV range.
`ReqMeta.token_ids` and `slot_mapping` describe KV; `lookahead_token_ids` is a
hash-only suffix. The token database consumes their concatenation, but emits
offsets only for complete KV chunks with a complete following window.

For chunk size C and lookahead H, chunk i hashes tokens `[i*C:(i+1)*C+H]`
and the previous digest. The initial digest includes protocol version, C and H.
H=0 preserves the previous hash chain exactly. The model key namespace includes
model/draft/cache configuration, TP size, KV shape and H. Request key tags include
LoRA and cache salt alongside user-provided tags; multimodal substitutions cover
the suffix as well as the KV token range.

The scheduler bounds the save range by both scheduled accepted tokens and
available hash context, advancing `num_saved_tokens` only to that bound. When
decode saving is disabled, the bound also stops at the prompt end. The last
prompt chunk can therefore become eligible on a later decode step. A load of
previously saved decode KV retains its full hash context even when new decode
saves are disabled. No finish-time save is scheduled from `num_computed_tokens`:
the async-load failure case must not publish uninitialized KV.

Physical slot addresses use the resolved single-group block size, which can
differ from the requested cache block size. LMCache chunk boundaries remain
independent of those allocation units.

The worker appends the suffix only for key generation. Masks and GPU slots remain
KV-sized; retrieve results are sliced back to the KV range before failed-block
reporting. `BlockStored` events carry the following token window separately.

For vLLM packed K/V layouts, the V2 GPU connector views its legacy `[2,L,T,D]`
allocation as `[1,L,T,2*D]` for both transfer directions. The packed kernel uses
one plane, so passing the legacy shape directly would copy only half the KV.
HND transfers also convert the packed content width to a single K/V head width
before calling the in-process kernel. The allocation size is unchanged.

The versioned vLLM wrapper supplies resolved cache groups. Until separate proofs
exist, H>0 rejects multiple groups, non-full attention, PP/CP, layerwise/Blend,
and LMCache P/D. Prompt embeddings bypass external load/save. A terminal sample
that completes a hash without a later forward can miss a save; it must not be
treated as permission to copy a newly sampled token's uncomputed KV.
