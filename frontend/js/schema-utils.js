// Resolves Pydantic's JSON Schema `$ref`/`$defs` shape and walks a schema's
// properties in the order the UI should render them, using the `unit`/`step`/
// `fine_step`/`log`/`group`/`order`/`help` hints from `config/hints.py::ui_hint`.
//
// This is the one place that understands the schema's wire shape, so `param-form.js`
// and `stage-chain.js` both build on it instead of each re-deriving $ref resolution.

/**
 * Resolve a schema node that may be a `$ref` into its concrete definition.
 * @param {any} node
 * @param {Record<string, any>} defs
 * @returns {any}
 */
export function resolveRef(node, defs) {
  if (node && typeof node === "object" && "$ref" in node) {
    const name = node.$ref.split("/").pop();
    return resolveRef(defs[name], defs);
  }
  return node;
}

/**
 * Pydantic emits an optional field (`X | None`) as `anyOf: [{...X}, {type: "null"}]`
 * with the real constraints on the non-null branch. Collapse that into one node plus
 * a `nullable` flag, so callers don't each re-derive this.
 * @param {any} node
 * @param {Record<string, any>} defs
 * @returns {{ node: any, nullable: boolean }}
 */
export function unwrapNullable(node, defs) {
  const resolved = resolveRef(node, defs);
  if (Array.isArray(resolved?.anyOf)) {
    const branches = resolved.anyOf.map((b) => resolveRef(b, defs));
    const real = branches.find((b) => b.type !== "null");
    const hasNull = branches.some((b) => b.type === "null");
    if (real && hasNull) {
      return { node: { ...real, ...pickHints(resolved) }, nullable: true };
    }
  }
  return { node: resolved, nullable: false };
}

const HINT_KEYS = ["unit", "step", "fine_step", "log", "group", "order", "help", "title", "default"];

/** Copy the UI-hint keys off a node (they live beside `anyOf`, not inside each branch). */
function pickHints(node) {
  const out = {};
  for (const key of HINT_KEYS) {
    if (key in node) out[key] = node[key];
  }
  return out;
}

/**
 * List an object schema's properties as `{ name, node, nullable, required }`, resolved
 * and unwrapped, sorted by the `order` hint (properties without one keep their
 * schema-declaration order, after the ordered ones).
 * @param {any} objectSchema
 * @param {Record<string, any>} defs
 * @returns {Array<{ name: string, node: any, nullable: boolean, required: boolean }>}
 */
export function listProperties(objectSchema, defs) {
  const resolved = resolveRef(objectSchema, defs);
  const required = new Set(resolved.required ?? []);
  const entries = Object.entries(resolved.properties ?? {});
  const withIndex = entries.map(([name, raw], index) => {
    const { node, nullable } = unwrapNullable(raw, defs);
    return { name, node, nullable, required: required.has(name), declOrder: index };
  });
  withIndex.sort((a, b) => {
    const orderA = a.node.order ?? Number.POSITIVE_INFINITY;
    const orderB = b.node.order ?? Number.POSITIVE_INFINITY;
    if (orderA !== orderB) return orderA - orderB;
    return a.declOrder - b.declOrder;
  });
  return withIndex;
}

/**
 * Group already-listed properties by their `group` hint. Ungrouped fields fall under `null`.
 * Insertion order of groups follows first appearance, which — since `listProperties` already
 * sorted by the `order` hint — matches the order the hints were authored in.
 * @param {Array<{ name: string, node: any }>} properties
 * @returns {Array<[string | null, Array<any>]>}
 */
export function groupProperties(properties) {
  const groups = new Map();
  for (const prop of properties) {
    const key = prop.node.group ?? null;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(prop);
  }
  return [...groups.entries()];
}

/** Humanize a field name into a label when a node has no `title` (defensive; Pydantic always sets one). */
export function labelFor(name, node) {
  if (node.title) return node.title;
  return name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}
