# Recovery budget feasibility, before any new training

The immutable v1 progress archive contains nine completed epochs (0 through8).
Their recorded total is 13555.669310 seconds, mean 1506.185479
seconds, range 1372.600509 to 1618.283555.
The failed child process lasted13723.918316 seconds including setup and part
of epoch9. The existing log reports combined train, scheduler, post-processing,
validation and checkpoint time; it does not separately measure train and
validation. No separate timing estimate is claimed. Evidence is retained in
budget_evidence.json and v1 is unchanged.

At the observed combined mean,6h permits about14.34 epochs and24h about
57.36. The exact unchanged Selection implementation stops only when
bad_epochs>20: even a best checkpoint at epoch0 followed by no improvement
requires22 completed epochs (epoch0 plus21 bad epochs), about9.20h.
This corrects the informal21-epoch estimate.300 epochs would require about
125.52h per seed at that rate, excluding final validation and setup.
These estimates come from the old implementation; the projector backward can
change throughput and the estimates do not guarantee convergence.

The proposed24h per-seed limit explicitly changes v1's6h wall-time limit.
It preserves batch size, maximum300 epochs, patience20, optimizer, scheduler
and strict rounded-four-decimal Recall20 checkpoint selection. The unchanged
72h queue limit includes all waits and jobs. Five22-epoch runs would consume
about46.02h at the old mean, but five jobs reaching24h require120h.
The queue can therefore remain incomplete. No truncated or partial seed counts
as a completed baseline and no favorable-seed aggregation is permitted.

Current v3 status: the independently registered same-realized-input audit and
its actual31/31 CPU tests passed. The separate old full-model receipt remains
completed_fail and the frozen v2 remains ineligible. V3 requires all seven
raw/config/source roles in prerequisites.json, its own actual CPU preparation
tests, complete root payload review and explicit deployment/start authorization.
At candidate8094f25 freeze it remained prepared with ready_for_training=false
and root review pending. Root subsequently recorded conditional start approval;
the actual deployment verification and fresh resource gates still precede start.
Neither the same-input one-batch pass nor the revised24h budget guarantees
long-run stability, convergence or completion of all five seeds within72h.
