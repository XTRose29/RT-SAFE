# Consolidated benchmark fixes

Branch `fix/consolidated-benchmark-fixes` combines the complete histories of
`fix/report-execution-alignment` and `fix/redlight-curb-boundary` on top of
`main` at `4ff27bd`. Runtime-platform fixes already merged into `main` remain
included. Experimental Gemini account-pool changes and recorded model scores
are not part of this fix set.

The changes cover movement/capsule-height alignment, action-camera tracking,
native contact scoring, hazard setup and attribution, parse-error history,
red-light event timing, and the physical curb boundary. See
[the report fixes](REPORT_FIXES.md), [native rechecks](REPORT_PUBLISH_RECHECK.md),
and [curb correction](REDLIGHT_CURB_FIX.md).

The combined artifact protocol is `execution_alignment_v3`; its traffic policy
is `two_rule_traffic_events_v7`. Pinned source hashes match the combined code.
Existing validation documents and artifacts retain the protocol/version they
actually tested. Historical scores and the frozen Gemini experiment remain
unchanged; results generated with this branch need separate protocol labels.

## Combined validation

`python -m pytest -q tests evaluation --disable-warnings --maxfail=10`:
**767 passed, 1 skipped, 498 subtests passed**. The
[test output](../validation/consolidated-fixes/pytest.txt) belongs to this combined
source. An integration regression checks native sweep sampling across a legal
curb approach followed by an illegal entry: no violation at the curb, one event
during the entry MOVE, and no duplicate at the following observation.

Prior native-renderer evidence is preserved with its original scope. A fresh
live-renderer evaluation of the combined protocol has not been performed.

Machine-specific paths in earlier validation exports use `${VALIDATION_HOME}`,
`${WORKSPACE}`, and `${REPORT_EXAMPLES_ROOT}` placeholders. Set these to your
local artifact roots when reproducing a probe. Existing historical hashes
refer to the original raw files; [the normalization inventory](../validation/consolidated-fixes/path-normalization.json)
records both original and portable-export hashes. Numerical measurements and
historical source-code hashes are unchanged.
