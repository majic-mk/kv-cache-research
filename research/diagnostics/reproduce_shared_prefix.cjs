#!/usr/bin/env node
"use strict";
// SYNTHETIC SCHEDULING SIMULATION: no model, CUDA, network, or measured durations.
// Run: node research/diagnostics/reproduce_shared_prefix.cjs > /tmp/result.json
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const stored = new Map();
const printed = [];
const sourcePath = path.join(__dirname, "shared_prefix_original.js");
vm.runInNewContext(fs.readFileSync(sourcePath, "utf8"), {
  store: (key, value) => stored.set(key, value),
  text: (value) => printed.push(value),
}, {filename: sourcePath, timeout: 1000});
// Convert cross-realm objects to ordinary Node objects for strict assertions.
const schedules = JSON.parse(JSON.stringify(stored.get("toy_exhaustive")));
const originalSummary = JSON.parse(JSON.stringify(printed[0]));
const lengths = {A: {load: 4, recompute: 10}, B: {load: 2, recompute: 3}};
const requests = [{id: "b", prefix: "B", deadline: 4},
  {id: "a1", prefix: "A", deadline: 6}, {id: "a2", prefix: "A", deadline: 6}];

// Verify every generated schedule separately from the enumeration procedure.
for (const schedule of schedules) {
  const tasks = ["A", "B", "b", "a1", "a2"];
  assert.deepEqual([...schedule.po, ...schedule.go].sort(), tasks.slice().sort());
  for (const task of tasks) {
    const duration = task === "A" || task === "B" ? lengths[task][schedule.modes[task]] : 1;
    assert.equal(schedule.end[task] - schedule.start[task], duration);
    assert(schedule.start[task] >= 0);
  }
  for (const prefix of ["A", "B"]) {
    assert.equal(schedule.po.includes(prefix), schedule.modes[prefix] === "load");
  }
  for (const request of requests) {
    assert(schedule.go.includes(request.id));
    assert(schedule.start[request.id] >= schedule.end[request.prefix]);
  }
  for (const order of [schedule.po, schedule.go]) {
    for (let i = 1; i < order.length; i++) {
      assert(schedule.start[order[i]] >= schedule.end[order[i - 1]]);
    }
  }
  assert.equal(schedule.ontime, requests.filter(r => schedule.end[r.id] <= r.deadline).length);
  assert.equal(schedule.makespan, Math.max(...Object.values(schedule.end)));
}
assert.equal(schedules.length, 52);
assert.equal(new Set(schedules.map(s => JSON.stringify([s.modes, s.po, s.go]))).size, 52);
const modeCounts = {};
for (const schedule of schedules) {
  const key = `A:${schedule.modes.A},B:${schedule.modes.B}`;
  modeCounts[key] = (modeCounts[key] || 0) + 1;
}
assert.deepEqual(modeCounts, {
  "A:load,B:recompute": 12,
  "A:load,B:load": 12,
  "A:recompute,B:load": 8,
  "A:recompute,B:recompute": 20,
});
assert.equal(Math.max(...schedules.map(s => s.ontime)), 3);
assert.equal(schedules.filter(s => s.ontime === 3).length, 2);
assert.equal(originalSummary.allLoadBest.ontime, 2);
assert.equal(originalSummary.optimum.makespan, 6);

const sharedEdf = schedules.find(s => s.modes.A === "load" && s.modes.B === "load"
  && s.po.join(",") === "B,A" && s.go.join(",") === "b,a1,a2");
assert(sharedEdf);
assert.deepEqual(sharedEdf.end, {B: 2, b: 3, A: 6, a1: 7, a2: 8});
assert.equal(sharedEdf.ontime, 1);

// Strong simple baseline: A cannot meet its deadline by recomputation, even
// before considering contention. Reserve its load and put flexible B on G.
assert(lengths.A.recompute + 1 > 6);
assert(lengths.B.recompute + 1 <= 4);
const reservation = schedules.find(s => s.modes.A === "load" && s.modes.B === "recompute"
  && s.po.join(",") === "A" && s.go.join(",") === "B,b,a1,a2");
assert(reservation);
assert.deepEqual(reservation.end, {A: 4, B: 3, b: 4, a1: 5, a2: 6});
assert.equal(reservation.ontime, originalSummary.optimum.ontime);

console.log(JSON.stringify({
  evidence_type: "SYNTHETIC_EXACT_SCHEDULING_ENUMERATION_NOT_LLM_EXPERIMENT",
  duration_units: "arbitrary units; not milliseconds and not hardware measurements",
  source: "Exact arithmetic computation reproduced with explicit storage/output helpers",
  objective: "maximize the count of request completions <= deadline; tie-break by makespan",
  assumptions: [
    "All three requests and both prefix states are available at time zero",
    "One serial PCIe resource P and one serial GPU resource G can overlap",
    "Each prefix is restored exactly once, either by load on P or recomputation on G",
    "Each restored prefix is shared among its requests without additional copy cost",
    "Each request has one unit-duration suffix on G after its prefix is ready",
    "Tasks are non-preemptive and durations are fixed and known",
    "Memory is sufficient; there are no capacity, eviction, batching or cancellation effects",
    "Time zero is the earliest legal start; deadline equality counts as on time",
  ],
  inputs: {prefix_durations: lengths, requests},
  original_summary: originalSummary,
  valid_resource_order_counts: modeCounts,
  tests: {status: "PASS", schedule_invariants_checked: schedules.length,
    assertions: ["all durations and precedences valid", "resources never overlap themselves",
      "exact mode-count partition and 52 unique resource orders", "two schedules achieve all deadlines",
      "best all-load policy achieves 2/3", "singleflight EDF achieves 1/3",
      "mandatory-resource reservation matches the 3/3 oracle"]},
  baseline_results: {shared_singleflight_edf: sharedEdf, compulsory_load_reservation: reservation},
  conclusion: "The example defeats independent/EDF choices, but a simple shared-aware reservation baseline matches the oracle. It does not establish research novelty or LLM performance benefit.",
  schedules,
}, null, 2));
