# What the CTB simulation supports—and what it does not

The present study supports reproducible, data-aware simulation experiments on
the discovered CTB event-log process. It does not claim a validated physical
digital twin of the terminal. The distinction is essential for a defensible
thesis.

## Claims that can be defended now

1. **Control-flow evidence.** Minute-resolution timestamps cannot establish
within-case physical parallelism. The explicit event order is therefore a
data-engineering observation model used to discover a sequential Petri net.
Held-out replay and the simulation-log contract support the statement that the
model represents the recorded event sequence—not every physical micro-action.

2. **Data-aware simulation evidence.** The model learns associations between
approved case/context attributes, resource assignments and recorded timings.
It can test whether a specified change propagates through those learned
relations. It must not be described as establishing a causal physical effect.

3. **Capacity correction.** The automatically discovered RMG overlap of 100
is an empirical timestamp-overlap measure, not a physical crane capacity. The
cap of three is a transparent domain constraint. The thesis should continue to
report it as an assumption and retain the uncapped result as a sensitivity
diagnostic, not as a competing physical estimate.

4. **Scenario execution.** T22 removal and higher demand are executable
interventions in the statistical model. Zero T22 assignments and the changed
arrival distribution verify that they were implemented. They do not validate a
real block closure or a real slot-policy recommendation.

## Limitations that must remain explicit

- **No spatial state:** the model has no container position, block
compatibility, crane trajectory, relocation or travel-time state. Resource
reallocation is therefore statistical, not a physical movement model.
- **No explicit queue state:** model-derived pre-service delay combines several
unobserved mechanisms. It is not a pure queueing-time estimate. A Kingman-type
queue formula should not be added post hoc unless arrival/service stations and
their queue disciplines are observed and modelled.
- **No isolated workload effect:** comparing no-rules with rules+workload is a
configuration comparison, not a causal test of workload features alone. A
rules-only ablation is the required additional experiment before claiming the
incremental effect of workload features.
- **Gate-only language:** the Petri net's silent bypass is a structural
overgeneralisation risk. Empirical routing and the output contract can suppress
it in generated logs, but do not repair the net. A structural repair would
require a pre-specified repair method, re-discovery/re-estimation and held-out
validation; it should not be silently applied after the main analysis.

## Best next improvements

1. Add the rules-only ablation using the same temporal split, seeds and KPI
protocol; report rules-only versus no-rules and rules+workload.
2. Add a separate spatial/queue extension only if container location, crane
assignment/trajectory, travel-time and queue/admission data can be obtained.
Validate that extension against these observables before using it for slot-rate
or closure recommendations.
3. If structural repair is pursued, preregister the gate-only repair rule and
compare original and repaired models on held-out fitness, precision, activity
frequencies and scenario sensitivity.
4. Keep the current scenario results framed as model-sensitivity findings.
Operational policy claims require the extended model and external validation.
