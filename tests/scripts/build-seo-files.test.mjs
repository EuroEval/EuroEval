import assert from "node:assert/strict";
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import { validateLeaderboardCsvs } from "../../src/scripts/build-seo-files.mjs";

async function temporaryCsvDir() {
  const root = await mkdtemp(path.join(os.tmpdir(), "euroeval-seo-"));
  const csvDir = path.join(root, "csv");
  return { root, csvDir };
}

async function removeTemporaryDir(root) {
  await rm(root, { recursive: true, force: true });
}

test("preview builds allow a missing CSV directory", async () => {
  const { root, csvDir } = await temporaryCsvDir();
  try {
    assert.deepEqual(
      await validateLeaderboardCsvs({ csvDir, vercelEnv: "preview" }),
      [],
    );
  } finally {
    await removeTemporaryDir(root);
  }
});

test("preview builds allow an empty CSV directory", async () => {
  const { root, csvDir } = await temporaryCsvDir();
  try {
    await mkdir(csvDir);
    assert.deepEqual(
      await validateLeaderboardCsvs({ csvDir, vercelEnv: "preview" }),
      [],
    );
  } finally {
    await removeTemporaryDir(root);
  }
});

for (const vercelEnv of ["production", undefined]) {
  test(`${vercelEnv ?? "unspecified"} builds reject a missing CSV directory`, async () => {
    const { root, csvDir } = await temporaryCsvDir();
    try {
      await assert.rejects(
        validateLeaderboardCsvs({ csvDir, vercelEnv }),
        /is missing.*Generate leaderboard CSVs/,
      );
    } finally {
      await removeTemporaryDir(root);
    }
  });

  test(`${vercelEnv ?? "unspecified"} builds reject an empty CSV directory`, async () => {
    const { root, csvDir } = await temporaryCsvDir();
    try {
      await mkdir(csvDir);
      await assert.rejects(
        validateLeaderboardCsvs({ csvDir, vercelEnv }),
        /contains no required leaderboard CSV files/,
      );
    } finally {
      await removeTemporaryDir(root);
    }
  });
}

test("builds accept an existing user-facing CSV", async () => {
  const { root, csvDir } = await temporaryCsvDir();
  try {
    await mkdir(csvDir);
    await writeFile(path.join(csvDir, "leaderboard.csv"), "model,score\n");
    assert.deepEqual(
      await validateLeaderboardCsvs({ csvDir, vercelEnv: "production" }),
      ["leaderboard.csv"],
    );
  } finally {
    await removeTemporaryDir(root);
  }
});
