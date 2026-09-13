export async function resolve(specifier, context, nextResolve) {
  const isRelative = specifier.startsWith("./") || specifier.startsWith("../") || specifier.startsWith("/");
  const hasExtension = /\.[a-zA-Z0-9]+$/.test(specifier.split(/[/?#]/)[0]);

  if (isRelative && !hasExtension && !specifier.endsWith("/")) {
    const extensions = [".ts", ".js", ".mjs", ".cjs", ".json"];
    for (const extension of extensions) {
      try {
        return await nextResolve(specifier + extension, context);
      } catch (error) {
        const recoverable = ["ERR_MODULE_NOT_FOUND", "ERR_UNSUPPORTED_DIR_IMPORT"];
        if (!recoverable.includes(error.code)) throw error;
      }
    }
  }

  return nextResolve(specifier, context);
}
