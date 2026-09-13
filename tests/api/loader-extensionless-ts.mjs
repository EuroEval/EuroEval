export async function resolve(specifier, context, nextResolve) {
  const isRelative = specifier.startsWith("./") || specifier.startsWith("../");

  if (isRelative && !specifier.endsWith("/") && !specifier.endsWith(".ts")) {
    try {
      return await nextResolve(specifier + ".ts", context);
    } catch (error) {
      if (error.code !== "ERR_MODULE_NOT_FOUND") throw error;
    }
  }

  return nextResolve(specifier, context);
}
