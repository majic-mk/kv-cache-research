"""Synthetic constructed mechanisms only; no LLM, benchmark, GPU, or paid API."""
import json
import numpy as np
from reference import (calibrate_first_chunk, scale_map_oracle, rank_only_oracle,
                       selection_report, echo_chunk_layout, reconstruction_ledger)


def main():
    virtual = np.array([[[.01,.02,.03,.04,.01,.02,.03,.04]]], dtype=np.float32)
    regions = np.array([0]*4 + [1]*4)
    cases = {}
    for name, exact in (("constructed_scale_shift", virtual * [1,1,1,1,10,10,10,10]),
                        ("constructed_rank_failure", virtual[...,::-1])):
        base = calibrate_first_chunk(virtual, exact[...,:4])
        map_oracle = scale_map_oracle(virtual, exact, regions)
        map_oracle[...,:4] = exact[...,:4]
        rank_oracle = rank_only_oracle(base, exact, regions)
        rank_oracle[...,:4] = exact[...,:4]
        cases[name] = {method: selection_report(scores, exact, regions, .5)
                       for method,scores in (("first_chunk",base), ("scale_map_oracle",map_oracle),
                                             ("rank_only_oracle",rank_oracle), ("exact",exact))}
    chunks = echo_chunk_layout(16003,3,8,12,5)
    result = {"evidence_class":"synthetic_cpu_only", "model_quality_claim":False,
              "gpu_used":False, "weights_downloaded":False, "cases":cases,
              "example_first_pass_ledger":reconstruction_ledger(chunks[:1],16003)}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
