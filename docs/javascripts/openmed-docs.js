(function () {
  "use strict";

  function restoreHreflangLinks() {
    for (const alternate of document.querySelectorAll(
      "link[data-openmed-hreflang-rel]",
    )) {
      alternate.setAttribute(
        "rel",
        alternate.dataset.openmedHreflangRel || "alternate",
      );
      delete alternate.dataset.openmedHreflangRel;
    }
  }

  function initializeSystemTheme() {
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const storageKey = "/docs/.__palette";
    let stored = null;
    try {
      stored = localStorage.getItem(storageKey);
    } catch {
      // Storage can be unavailable in hardened browser contexts.
    }
    if (stored) {
      return;
    }

    const apply = () => {
      document.body?.setAttribute(
        "data-md-color-scheme",
        media.matches ? "slate" : "default",
      );
    };
    apply();
    media.addEventListener("change", apply);
  }

  function initializeScrollableRegions() {
    const update = () => {
      for (const region of document.querySelectorAll(
        ".md-typeset__scrollwrap, .md-typeset pre > code",
      )) {
        const scrollable =
          region.scrollWidth > region.clientWidth + 1
          || region.scrollHeight > region.clientHeight + 1;
        if (scrollable) {
          region.setAttribute("tabindex", "0");
          region.dataset.openmedScrollable = "true";
        } else if (region.dataset.openmedScrollable === "true") {
          region.removeAttribute("tabindex");
          delete region.dataset.openmedScrollable;
        }
      }
    };

    let scheduled = false;
    const scheduleUpdate = () => {
      if (scheduled) return;
      scheduled = true;
      window.requestAnimationFrame(() => {
        scheduled = false;
        update();
      });
    };
    const searchResults = document.querySelector(
      '[data-md-component="search-result"]',
    );
    if (searchResults) {
      // Search snippets arrive after page initialization and change per query.
      // Observe content only: our tabindex updates must not trigger this observer.
      new MutationObserver(scheduleUpdate).observe(searchResults, {
        childList: true,
        subtree: true,
      });
      searchResults.addEventListener("toggle", scheduleUpdate, true);
      searchResults.addEventListener("keydown", (event) => {
        const region = event.target;
        if (
          !(region instanceof HTMLElement)
          || region.dataset.openmedScrollable !== "true"
          || event.altKey || event.ctrlKey || event.metaKey
        ) {
          return;
        }
        // Search's global result shortcuts intercept these keys. Handle them
        // on the focused region, including WebKit's nested-link scroll case.
        const movement = {
          ArrowLeft: [-40, 0],
          ArrowRight: [40, 0],
          ArrowUp: [0, -40],
          ArrowDown: [0, 40],
          PageUp: [0, -region.clientHeight],
          PageDown: [0, region.clientHeight],
          Home: [-region.scrollWidth, -region.scrollHeight],
          End: [region.scrollWidth, region.scrollHeight],
        }[event.key];
        if (!movement) return;
        region.scrollLeft += movement[0];
        region.scrollTop += movement[1];
        event.preventDefault();
        event.stopPropagation();
      });
    }
    document.querySelector("#__search")?.addEventListener("change", scheduleUpdate);
    update();
    document.fonts?.ready.then(scheduleUpdate);
    window.addEventListener("resize", scheduleUpdate, { passive: true });
  }

  function initializeDrawer() {
    const button = document.querySelector("[data-openmed-drawer]");
    const checkbox = document.querySelector("#__drawer");
    const navigation = document.querySelector(".md-sidebar--primary");

    if (!button || !checkbox || button.dataset.initialized === "true") {
      return;
    }

    button.dataset.initialized = "true";
    if (navigation) {
      navigation.id = "openmed-docs-navigation";
      button.setAttribute("aria-controls", navigation.id);
    }

    const updateState = () => {
      button.setAttribute("aria-expanded", String(checkbox.checked));
    };

    button.addEventListener("click", () => {
      checkbox.checked = !checkbox.checked;
      checkbox.dispatchEvent(new Event("change", { bubbles: true }));
      updateState();

      if (checkbox.checked && navigation) {
        requestAnimationFrame(() => {
          navigation.querySelector("a[href]")?.focus();
        });
      }
    });

    checkbox.addEventListener("change", updateState);
    document.addEventListener("keydown", (event) => {
      if (event.key === "Tab" && checkbox.checked && navigation) {
        const navigationItems = [...navigation.querySelectorAll(
          'a[href], button:not([disabled]), input:not([disabled]), [tabindex]:not([tabindex="-1"])',
        )].filter((element) => {
          const style = getComputedStyle(element);
          return (
            style.display !== "none"
            && style.visibility !== "hidden"
            && element.getClientRects().length > 0
          );
        });
        const first = navigationItems[0];
        const last = navigationItems.at(-1);
        const active = document.activeElement;
        if (!first || !last) {
          event.preventDefault();
          button.focus();
        } else if (active === button) {
          event.preventDefault();
          (event.shiftKey ? last : first).focus();
        } else if (event.shiftKey && active === first) {
          event.preventDefault();
          button.focus();
        } else if (!event.shiftKey && active === last) {
          event.preventDefault();
          button.focus();
        } else if (!navigation.contains(active)) {
          event.preventDefault();
          (event.shiftKey ? last : first).focus();
        }
        return;
      }
      if (event.key !== "Escape" || !checkbox.checked) {
        return;
      }
      checkbox.checked = false;
      checkbox.dispatchEvent(new Event("change", { bubbles: true }));
      updateState();
      button.focus();
    });

    updateState();
  }

  function initialize() {
    window.setTimeout(restoreHreflangLinks, 0);
    initializeSystemTheme();
    initializeScrollableRegions();
    initializeDrawer();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initialize, { once: true });
  } else {
    initialize();
  }
})();
