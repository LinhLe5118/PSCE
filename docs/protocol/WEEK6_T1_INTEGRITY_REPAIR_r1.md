# T1 v1.1.0 — integrity packaging revision r1

Date: 22 September 2026. Authorized by the project owner during H1 review.
This is a **packaging revision**, not a new statistical protocol or E1 GO.

## Reproduced defect

On clean Git commit `8861511`, the original T1 verifier fails before reading
any data: `Manifest checksum mismatch`. The manifest bytes in the worktree
are identical to the Git blob, so this is not an uncommitted local edit.

| File | Recorded SHA-256 | SHA-256 of committed LF bytes |
|---|---|---|
| Original v1.1.0 manifest | `567a614c32404041fb668833e4718271907fc05970c84bd1e8488d1344b0bb1f` | `1a2c09325df5557f049ed6e00f159fe296ec3ab34e7c7617660433db49231ee1` |
| Statistical JSON | `42894dae68558c5a2ac2c952b434d85ab9ddc5ade9d750dbbbfc29f38062b230` | `316b144afb951d04984c976d9c575756da154b4baa3b0020500684ffe3a8eca6` |
| Statistical Markdown | `e859810be3c7f41fe72b6a6f9ab0865227de6be5987121a194194dc0461cbca4` | `18ce4d8d642ce31f7630d42032aa8a4d0a99a49e30bcef799c2d19a4ec64a14b` |

150 of 152 originally pinned files match their recorded hashes. The JSON
mismatch is exactly reproduced by LF versus CRLF. Simple LF/CRLF/BOM/trailing
newline variants do **not** reconstruct the original manifest or Markdown hash;
their old preimage is unavailable, so the whole defect must not be attributed
to line endings or claimed cryptographically repaired back to its original bytes.

The current statistical JSON and Markdown are instead verified byte-for-byte
against their introduction in commit `38abddf64949660d4a9e331e86e30368fd548815`.
Their content is not changed by this revision. The repaired package explicitly
re-anchors integrity to those published Git contents. It does not establish
validity of the old digest, recover a missing precommit version or backdate a seal.

## Revision procedure

- Preserve the original manifest and sidecar unchanged as historical evidence.
- Publish `WEEK6_T1_FREEZE_v1.1.0-r1.sha256.json` and a new sidecar, hashing
  current repository bytes (LF as prescribed by `.gitattributes`).
- Retain all 152 original source/protocol paths and add this repair record.
- Record both old and current digests, the original source commit, and a list
  of administrative changes: current-pointer metadata and verifier default.
- Point the current T1 entrypoint and verifier to r1. The verifier still hashes
  exact bytes; it must not silently normalize input or ignore mismatches.
- Scientific JSON, Markdown, core algorithms, feature definitions and historical
  results remain byte-identical to the published baseline.

H1 pins the repaired T1 package. Any later formula change still needs a new
scientific protocol version, rather than a packaging revision. B2 must review
this explicit re-anchoring and the H1 package before E1. Commit/archive both
before evaluation access; neither checksum verification grants execution GO.
