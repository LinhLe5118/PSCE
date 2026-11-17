# Research Administration

**Audit update, 21 September 2026:** The live ICLR 2027 author guidelines
still list abstract 18 September and full paper 25 September, 23:59 AoE.
Repository evidence does not establish whether an abstract was submitted.
The current IEEEtran draft is not the required ICLR submission template.
The live guide recommends a reproducibility statement; it does **not** require
a frozen-protocol commit hash in that statement, contrary to the older list
below. The protocol hash remains an internal project requirement. Advisor
permission to open evaluation is no longer required by the project owner;
technical H1/D1/B2 gates remain. See the
[21 September audit](PROJECT_READINESS_AUDIT_2026-09-21.md) and
[official guide](https://iclr.cc/Conferences/2027/AuthorGuidelines).

Every date/requirement below is quoted from `PLAN_Final.md` §1 (sourced there
from `iclr.cc/Conferences/2027/AuthorGuidelines`, cross-checked 14/08/2026 per
that document). Nothing here is invented; items not present anywhere in the
repository are marked `[TBD]` rather than guessed, per mega-task section 20's
explicit instruction not to fabricate a deadline or venue.

## Target venue

| Item | Value |
|---|---|
| Conference | ICLR 2027 |
| Track | Main conference, double-blind review |
| Source | `iclr.cc/Conferences/2027/AuthorGuidelines`, verified 14/08/2026 (`PLAN_Final.md` §1) |

**Note, quoted from `PLAN_Final.md` §1:** "A previous plan document recorded
these milestones 7-9 days earlier while citing the same source link. Every
milestone must be checked by opening the original page, not trusted from an
intermediate document even when it cites a link." That warning applies to
this file too -- re-verify against the live ICLR page before treating any
date below as current.

## Deadlines

| Milestone | Date |
|---|---|
| Abstract submission | 18/09/2026 AOE |
| Full paper submission | 25/09/2026 AOE |
| Reviews released | 05/11/2026 |
| Public discussion window | 05/11/2026 → 18/11/2026 |
| Internal discussion | 19/11/2026 → 16/12/2026 |
| Decision | 16/12/2026 |

## Formatting requirements

- Page limit: 9 pages at submission, 10 pages during the review/rebuttal
  period.
- Fully anonymized submission (double-blind).
- AI use statement required, does not count against the page limit.
- Reproducibility Statement required, must cite the commit hash of the frozen
  protocol.
- Anonymized source code in supplementary material.

## Authorship

- No author may be added after 18/09/2026; author order may still change up
  to 25/09/2026.
- Every author needs an OpenReview profile before 25/09/2026; a profile
  created with a non-institutional email may take up to two weeks to clear
  moderation -- `PLAN_Final.md` Appendix B lists the internal deadline for
  this as **17/08/2026** (already past as of this writing, 2026-08-18;
  verify status).
- If no author has a prior accepted paper at a qualifying venue, the group
  may still submit, but at most one paper per person, and at least one
  author must register as a reviewer or the submission is desk-rejected.
- Withdrawal before 25/09/2026 removes the submission entirely; withdrawal
  after that date leaves it public permanently with author names attached.

**[TBD] Author list and order.** Not recorded anywhere in the repository.
`PLAN_Final.md` §11 names four contributors by role (A/B/C/D) mapped to
workstreams, and `docs/archive/PROJECT_CONTEXT_READ_ONLY.md` §4 separately names four
people by first name/initials against roles that only partly line up with
that A/B/C/D table. Whether these are the same four people, in what order,
and whether the advisor is a co-author, is not decided in any committed
document. **Do not infer an author list or order from either table** --
resolve explicitly with the team and record the decision here.

**[TBD] Advisor / affiliation questions to raise:**
- Whether the advisor stands as a co-author (`PLAN_Final.md` Appendix B lists
  "Chốt việc thầy đứng tên" with an internal deadline of 18/09/2026, not yet
  resolved as of this writing).
- Institutional affiliation string for the anonymized-then-deanonymized
  submission -- not recorded anywhere in the repository.
- Whether `BRIEF_2p.md` (referenced in `PLAN_Final.md` §0 as the 2-page
  advisor summary) exists yet -- it was not found under the repository root
  as of this audit; if the advisor conversation needs it, confirm it exists
  before the conversation, not after.

## Contribution / order discussion

**[TBD].** Not recorded. Raise alongside the authorship question above.

## Thesis (đồ án) deadline, separate from ICLR

| Item | Value |
|---|---|
| Defense window | second half of 12/2026 (`PLAN_Final.md` §1) |
| Governing body | university committee, not ICLR |
| Priority rule | the thesis is mandatory and takes priority over the paper in every decision (`PLAN_Final.md` §1: "Thứ tự ưu tiên này không được đảo trong bất kỳ quyết định nào") |

## Four hard deadlines (`PLAN_Final.md` §9, "Bốn mốc không lùi được")

| Date | What | Failure mode |
|---|---|---|
| 21/08/2026 | Definition freeze | Not enough runway before the pilot gate |
| 18/09/2026 | Author list locked permanently | No one, including the advisor, can be added after |
| 04/11/2026 | All thesis-only extensions complete | Collides with rebuttal, both suffer |
| 18/11/2026 | ICLR discussion window closes | Cannot respond to reviewers further |
