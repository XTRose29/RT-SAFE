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

test("effort comparisons retain hard-route denominators and provider defaults", () => {
  for (const model of data.models) {
    const defaultEffort = model.effort[model.name === "Sol" ? 0 : 1];
    for (const key of ["success", "safeSuccess", "collisions", "latency", "decisions"]) {
      assert.ok(Math.abs(defaultEffort[key] - model.realtime[key]) <= .051, `${model.name}: default ${key}`);
    }
    for (const effort of model.effort) {
      assert.equal(effort.spl, undefined, "unreported SPL must remain absent");
      for (const key of ["success", "safeSuccess"]) {
        const episodes = effort[key] * 36 / 100;
        assert.ok(Math.abs(episodes - Math.round(episodes)) < .02, `${model.name} ${effort.label}: 36-episode denominator`);
      }
      assert.ok(Math.abs(effort.active + effort.passive - effort.collisions) < .11, "independent rounding of collision components");
    }
  }
});

test("radar raw values correspond to the displayed aggregate metrics", () => {
  const behavior = read("app/behavior.json");
  for (const p of behavior.profiles) {
    const model = data.models.find(m => m.name === p.name);
    assert.ok(model);
    for (const [i, key] of ["collisions", "latency", "decisions"].entries()) {
      assert.ok(Math.abs(1 / p.raw[i] - model.average[key]) <= .051);
    }
    assert.ok(p.raw[3] > 0 && p.raw[3] <= 4);
    assert.ok(p.raw[4] >= 0 && p.raw[4] <= 1);
    assert.ok(p.raw[5] >= 0 && p.raw[5] <= 1);
    assert.equal(p.values.length, 6);
  }
});

test("every model has a sourced, continuous default-reasoning video excerpt", () => {
  const examples = read("app/model-examples.json");
  assert.deepEqual(examples, read("public/data/model-examples.json"));
  assert.deepEqual(examples.models.map(m => m.name).sort(), data.models.map(m => m.name).sort());
  const total = counts => Object.values(counts).reduce((a, b) => a + b, 0);
  for (const model of examples.models) {
    assert.equal(model.reasoning, "Provider default");
    assert.equal(model.steps.length, 3);
    assert.equal(model.speed, 2);
    assert.ok(Math.abs(model.videoDuration - (model.sourceDuration / model.speed + 2)) < 1 / 30 + .001);
    assert.equal(model.contacts, model.steps.reduce((n, s) => n + total(s.active) + total(s.passive), 0));
    assert.equal(model.inferenceContacts, model.steps.reduce((n, s) => n + total(s.passive), 0));
    assert.equal(model.hazards, model.steps.reduce((n, s) => n + total(s.hazards), 0));
    for (const [i, step] of model.steps.entries()) {
      assert.match(step.manifestSha256, /^[a-f0-9]{64}$/);
      assert.ok(!step.sourceRelativePath.startsWith("/"));
      assert.ok(step.inputFrames.length > 0 && step.actionFrames.length > 0);
      for (const frame of [...step.inputFrames, ...step.actionFrames]) assert.match(frame.sha256, /^[a-f0-9]{64}$/);
      assert.ok(step.end > step.start && step.inference <= step.end - step.start + .01);
      assert.ok(["move_to", "turn", "turn_around", "wait"].includes(step.action.type));
      if (i) {
        assert.equal(step.sourceStep, model.steps[i - 1].sourceStep + 1);
        assert.ok(Math.abs(step.start - model.steps[i - 1].end) < .1);
      }
    }
    for (const event of model.events) {
      assert.ok(event.count > 0);
      assert.ok(model.steps.some(s => Math.abs(event.at * model.speed - (event.phase === "inference" ? s.start + s.inference : s.end)) < .001));
    }
    for (const asset of [model.video, model.poster]) assert.ok(existsSync(new URL("../public/" + asset, import.meta.url)), asset);
  }
});

test("current demo is a continuous 60-second edit with no RL scene", () => {
  const timeline = read("video/nyc-60s/timeline.json");
  let frame = 0;
  assert.equal(timeline.audio, "none");
  assert.equal(timeline.subtitles, "none");
  for (const scene of timeline.scenes) {
    assert.equal(scene.start * timeline.fps, frame);
    assert.ok(!["learning", "rl", "training"].includes(scene.id));
    assert.equal(scene.ranges.reduce((sum, cut) => sum + cut.outputFrames, 0), scene.duration * timeline.fps);
    for (const cut of scene.ranges) {
      assert.ok(cut.sourceStartFrame >= 0 && cut.sourceEndFrame <= 2550, "Cut reaches the old RL scene");
      assert.ok(cut.sourceEndFrame > cut.sourceStartFrame);
      if (scene.id === "examples") assert.equal(cut.outputFrames, cut.sourceEndFrame - cut.sourceStartFrame, "Recorded playback speed must remain accurate");
    }
    frame += scene.duration * timeline.fps;
  }
  assert.equal(frame, 1800);
  assert.equal(frame / timeline.fps, timeline.duration);
  assert.equal(timeline.duration, 60);
});
