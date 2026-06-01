const initCharts = (root = document) => {
  const charts = root.querySelectorAll("[data-line-chart]");

  charts.forEach((canvas) => {
    const rawConfig = canvas.getAttribute("data-line-config");
    if (!rawConfig) {
      return;
    }

    let parsedConfig;
    try {
      parsedConfig = JSON.parse(rawConfig);
    } catch (error) {
      console.error("Failed to parse chart config", error);
      return;
    }

    if (typeof parsedConfig !== "object" || parsedConfig === null) {
      console.warn("Invalid chart config", parsedConfig);
      return;
    }

    parsedConfig.type = parsedConfig.type || "line";
    parsedConfig.options = parsedConfig.options || {};

    if (parsedConfig.options.responsive === undefined) {
      parsedConfig.options.responsive = false;
    }
    if (parsedConfig.options.maintainAspectRatio === undefined) {
      parsedConfig.options.maintainAspectRatio = false;
    }
    if (parsedConfig.options.animation === undefined) {
      parsedConfig.options.animation = false;
    }

    new Chart(canvas, parsedConfig);
  });
};

document.addEventListener("DOMContentLoaded", () => initCharts(document));
document.addEventListener("htmx:afterSettle", (e) => {
  const root = e.detail.elt;
  if (root && root.querySelectorAll) {
    initCharts(root);
  }
});
