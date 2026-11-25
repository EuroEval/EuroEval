import { access } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const sourceRoots = ["api", "src/frontend/services"].map((directory) =>
  path.resolve(process.cwd(), directory) + path.sep,
);

function isWithinSources(file) {
  const resolved = path.resolve(file);
  return sourceRoots.some((root) => resolved.startsWith(root));
}

export async function resolve(specifier, context, nextResolve) {
  if (!specifier.startsWith("./") && !specifier.startsWith("../")) {
    return nextResolve(specifier, context);
  }
  if (!specifier.endsWith(".js") || !context.parentURL?.startsWith("file:")) {
    return nextResolve(specifier, context);
  }

  const parent = fileURLToPath(context.parentURL);
  if (!parent.endsWith(".ts") || !isWithinSources(parent)) {
    return nextResolve(specifier, context);
  }

  const target = fileURLToPath(new URL(specifier, context.parentURL));
  const typescriptTarget = `${target.slice(0, -3)}.ts`;
  if (!isWithinSources(target) || !isWithinSources(typescriptTarget)) {
    return nextResolve(specifier, context);
  }
  try {
    await access(typescriptTarget);
  } catch {
    return nextResolve(specifier, context);
  }
  return nextResolve(pathToFileURL(typescriptTarget).href, context);
}
