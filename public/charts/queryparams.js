/** Separator between the target element ID and the parameter name in the URL. */
const NS_SEPARATOR = "--";

/**
 * Sync HTMX request parameters to the URL search string after each request.
 *
 * Each parameter is namespaced by the target element's ID to avoid collisions
 * when multiple HTMX components share the same page. For example, a request
 * from a form targeting `#optimization-list-items` with a `search` parameter
 * would be stored as `optimization-list-items--search=foo`.
 *
 * Only runs for user-triggered requests (i.e. where a triggering event exists),
 * ignoring programmatic or polling requests.
 *
 * Parameters are synced as follows:
 * - Keys namespaced to this target that are absent from the request are removed.
 * - Keys present in the request are set (added or updated).
 *
 * The URL is updated via {@link History.replaceState} so no new history entry
 * is created.
 *
 * @param {CustomEvent} e - The `htmx:afterRequest` event.
 * @returns {void}
 */
document.addEventListener("htmx:afterRequest", (e) => {
  if (!e.detail.requestConfig.triggeringEvent) return;
  const target = e.detail.target;
  const params = new URLSearchParams(e.detail.requestConfig.parameters);
  const url = new URL(window.location);
  const prefix = `${target.id}${NS_SEPARATOR}`;

  // Remove keys for this target that are no longer in the request
  [...url.searchParams.keys()]
    .filter((k) => k.startsWith(prefix))
    .forEach((k) => {
      const unprefixed = k.slice(prefix.length);
      if (!params.has(unprefixed)) {
        url.searchParams.delete(k);
      }
    });

  // Set keys that are present
  params.forEach((v, k) => {
    url.searchParams.set(`${prefix}${k}`, v);
  });

  history.replaceState(null, "", url);
});

/** Form submit methods */
const GET = "get";
const POST = "post";
const PUT = "put";
const PATCH = "patch";
const DELETE = "delete";

/**
 * Redirect Enter key presses inside HTMX-managed forms through HTMX instead
 * of the native browser submission.
 *
 * Pressing Enter in a form input triggers a native browser form submission,
 * bypassing HTMX entirely. This means {@link htmx:configRequest} never runs,
 * empty parameters are not stripped, and the full page reloads.
 *
 * If the focused input is not contained by an HTMX-managed form, we return
 * early and let the native submission proceed normally.
 *
 * If it is, we prevent the native submission and dispatch the request through
 * {@link htmx.ajax} directly, reading the method and URL from the form's
 * `hx-[method]` attribute. This avoids any coupling to `hx-trigger` and
 * ensures {@link htmx:configRequest} fires normally.
 *
 * `e.target.closest()` traverses strictly upward through ancestors (self +
 * parents), so it only matches a form that actually contains the focused input.
 * It cannot reach sideways to unrelated forms elsewhere in the DOM.
 *
 * @param {KeyboardEvent} e
 * @returns {void}
 */
document.addEventListener("keydown", (e) => {
  if (e.key !== "Enter") {
    return;
  }

  const form = e.target.closest(
    [
      `form[hx-${GET}]`,
      `form[hx-${POST}]`,
      `form[hx-${PUT}]`,
      `form[hx-${PATCH}]`,
      `form[hx-${DELETE}]`,
    ].join(","),
  );
  if (!form) {
    return;
  }

  e.preventDefault();

  const method = [GET, POST, PUT, PATCH, DELETE].find((m) =>
    form.hasAttribute(`hx-${m}`),
  );
  if (!method) {
    throw new Error(
      "Failed to dispatch HTMX form submission: no hx-[method] attribute found on form",
      { cause: { form } },
    );
  }

  const url = form.getAttribute(`hx-${method}`);
  if (!url) {
    throw new Error(
      `Failed to dispatch HTMX form submission: hx-${method} attribute is empty`,
      { cause: { form, method } },
    );
  }

  const target = form.getAttribute("hx-target");
  if (!target) {
    throw new Error(
      "Failed to dispatch HTMX form submission: hx-target attribute is missing or empty",
      { cause: { form, method, url } },
    );
  }

  const swap = form.getAttribute("hx-swap");
  if (!swap) {
    throw new Error(
      "Failed to dispatch HTMX form submission: hx-swap attribute is missing or empty",
      { cause: { form, method, url, target } },
    );
  }

  console.log("dispatching!");
  htmx.ajax(method.toUpperCase(), url, { target, swap, source: form });
});

/**
 * Strip empty parameters from HTMX requests before they are sent.
 *
 * When a form input is empty, browsers include it as an empty string in the
 * request. This can cause backend deserialization errors — for example, a date
 * field with `since=` fails to parse as a valid date.
 *
 * By deleting empty, null, or undefined parameters from the request config,
 * we ensure the backend only receives parameters that carry meaningful values,
 * and can treat absence as `None`/null rather than having to handle empty
 * strings as a special case.
 *
 * @param {CustomEvent} e - The `htmx:configRequest` event.
 * @returns {void}
 */
document.addEventListener("htmx:configRequest", (e) => {
  const params = e.detail.parameters;

  Object.keys(params).forEach((k) => {
    if (params[k] === "" || params[k] === null || params[k] === undefined) {
      delete params[k];
    }
  });
});
