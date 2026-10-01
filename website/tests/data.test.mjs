import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
const read = (p) =>
  JSON.parse(readFileSync(new URL("../" + p, import.meta.url), "utf8"));
const data = read("app/data.json");
const mean = (scope, key) =>
  data.models.reduce((sum, m) => sum + m[scope][key], 0) / data.models.length;
test("all eight manuscript models have complete, distinct conditions", () => {
  assert.equal(data.models.length, 8);
  assert.equal(new Set(data.models.map((m) => m.name)).size, 8);
  for (const m of data.models) {
    for (const scope of ["realtime", "static", "average"]) {
      const r = m[scope];
      assert.ok(r.safeSuccess <= r.success);
      assert.ok(r.success <= 100);
      assert.ok(r.collisions >= 0);
      assert.ok(r.spl >= 0 && r.spl <= 1);
      const n = scope === "average" ? 108 : 36;
      for (const key of ["success", "safeSuccess"])
        assert.ok(
          Math.abs((r[key] * n) / 100 - Math.round((r[key] * n) / 100)) < 0.06,
          `${m.name} ${scope} ${key} denominator`,
        );
    }
    assert.equal(m.static.passive, 0);
    assert.ok(
      Math.abs(
        m.realtime.active + m.realtime.passive - m.realtime.collisions,
      ) <= 0.021,
    );
    assert.equal(m.effort.length, 3);
  }
});
test("headline aggregates agree with the model table within reported rounding", () => {
  for (const scope of ["static", "realtime"]) {
    for (const key of ["success", "safeSuccess", "collisions"]) {
      assert.ok(
        Math.abs(mean(scope, key) - data.aggregate[scope][key]) < 0.06,
        `${scope} ${key}`,
      );
    }
  }
  assert.equal(
    (
      data.aggregate.realtime.collisions / data.aggregate.static.collisions
    ).toFixed(1),
    "12.3",
  );
});
test("rank changes use the correct result scopes", () => {
  const best = (scope) =>
    [...data.models].sort(
      (a, b) => a[scope].collisions - b[scope].collisions,
    )[0].name;
  assert.equal(best("static"), "Inkling");
  assert.equal(best("realtime"), "Fable");
  assert.equal(best("average"), "Sonnet");
});
test("public download matches the presentation data exactly", () =>
  assert.deepEqual(data, read("public/data/results.json")));
test("recorded examples retain their exact inference and action distinction", () => {
  const ex = read("app/examples.json");
  const f = ex.find((e) => e.model === "Fable");
  assert.equal(f.inferenceExposure, 5.83);
  assert.equal(f.actionDuration, 1);
  assert.equal(f.action.type, "wait");
  for (const e of ex) {
    assert.deepEqual(e, read(`public/data/examples/case-${e.id}.json`));
    assert.ok(existsSync(new URL("../public/" + e.image, import.meta.url)));
    assert.ok(
      existsSync(new URL("../public/" + e.resultImage, import.meta.url)),
    );
  }
});
test("subtitle timeline stays within the film duration", () => {
  const t = read("video/timeline.json");
  let end = 0;
  for (const s of t.scenes) {
    assert.ok(Math.abs(s.start - end) < 0.001);
    assert.ok(s.voiceDuration + s.voiceOffset < s.duration);
    end += s.duration;
  }
  assert.ok(Math.abs(t.duration - end) < 0.001);
});
