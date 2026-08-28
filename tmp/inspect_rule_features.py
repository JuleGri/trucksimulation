from __future__ import annotations

import pickle
import sys
from collections import Counter, deque
from pathlib import Path


def split_features(model: object) -> list[str]:
    rules = getattr(model, "rules", None)
    if not rules:
        return []
    root = 0 if 0 in rules else min(rules)
    queue = deque([root])
    seen = set()
    features: list[str] = []
    while queue:
        node_id = queue.popleft()
        if node_id in seen:
            continue
        seen.add(node_id)
        node = rules[node_id]
        if "feature" not in node:
            continue
        features.append(str(node["feature"]))
        queue.extend(node.get("children", {}).values())
    return features


path = Path(sys.argv[1])
with path.open("rb") as handle:
    params = pickle.load(handle)

print(f"rules_mode={params.rules_mode}")
print(f"use_workload_features={params.use_workload_features}")
families = {
    "execution_time": params.execution_time_distributions,
    "waiting_time": params.waiting_time_distributions,
    "resource_selection": params.resource_weights,
    "transition_routing": params.transition_weights,
}
for name, models in families.items():
    counts = Counter(
        feature
        for model in models.values()
        for feature in split_features(model)
    )
    print(f"\n{name}: {sum(counts.values())} split nodes")
    for feature, count in sorted(counts.items()):
        print(f"  {feature}: {count}")
