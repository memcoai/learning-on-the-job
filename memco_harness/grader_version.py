"""The version of the grading contract.

Every result line carries this string. Bump it whenever the reviewer prompt, the
breach semantics, or the scoring logic changes, so that results produced under
different rules are never compared silently. `memco-harness report` warns when a
set of runs mixes versions.

History
-------
1   First release: reviewer sees the policy statements plus the task's expected
    obligations, and returns a breach list restricted to `applicable_policies`.
2   Breach notes must state the condition that made the policy apply, in terms
    the assistant could have looked up. The breach list and what counts as a
    breach are unchanged, so the curves are still about the same thing; but the
    prompt that produces them is not the one version 1 ran under, and results
    from either side of this line are not compared without saying so.
3   Policy ids are matched leniently. The reviewer is shown each policy under a
    heading of `[its-id] Its title` and would sometimes answer with the id in
    those brackets; the scope filter compared ids exactly, found no such policy,
    and dropped the breach, recording a reply that had breached nothing. Under
    version 2 that silently deleted 41 breaches from a hundred-task run, most of
    them the control arm's, so both arms scored better than they were and the
    gap between them was understated. Brackets, quotes, case and stray spaces no
    longer distinguish one policy from another; a policy genuinely out of scope
    is still dropped, which is what the filter is for.
"""

GRADER_VERSION = "3"
