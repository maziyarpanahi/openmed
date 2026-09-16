(function () {
  "use strict";

  const storageKey = "openmed-theme";
  const modes = ["light", "dark"];
  const media = window.matchMedia("(prefers-color-scheme: dark)");
  const root = document.documentElement;
  const button = document.querySelector("[data-openmed-theme]");

  function storedMode() {
    try {
      const value = localStorage.getItem(storageKey);
      return modes.includes(value) ? value : (media.matches ? "dark" : "light");
    } catch {
      return media.matches ? "dark" : "light";
    }
  }

  function applyMode(mode) {
    root.dataset.theme = mode;
    root.dataset.themeMode = mode;

    if (button) {
      const next = modes[(modes.indexOf(mode) + 1) % modes.length];
      button.textContent = `Theme: ${mode}`;
      button.setAttribute(
        "aria-label",
        `Theme is ${mode}. Switch to ${next} theme.`,
      );
    }
  }

  let mode = storedMode();
  applyMode(mode);

  button?.addEventListener("click", () => {
    mode = modes[(modes.indexOf(mode) + 1) % modes.length];
    try {
      localStorage.setItem(storageKey, mode);
    } catch {
      // Storage can be unavailable in hardened browser contexts.
    }
    applyMode(mode);
  });

})();
