// Initialize theme functionality when DOM is loaded
document.addEventListener("DOMContentLoaded", function () {
  // Year
  document.getElementById("year").textContent = new Date().getFullYear();

  // Theme toggle functionality
  const themeToggle = document.getElementById("themeToggle");
  const themeIcon = document.getElementById("themeIcon");
  const html = document.documentElement;

  // Check for saved theme preference or default to 'light' mode
  const currentTheme = localStorage.getItem("theme") || "light";

  // Apply the saved theme on page load
  if (currentTheme === "dark") {
    html.classList.add("dark");
    updateThemeIcon("dark");
  } else {
    html.classList.remove("dark");
    updateThemeIcon("light");
  }

  // Theme toggle event listener
  themeToggle.addEventListener("click", () => {
    html.classList.toggle("dark");

    if (html.classList.contains("dark")) {
      localStorage.setItem("theme", "dark");
      updateThemeIcon("dark");
    } else {
      localStorage.setItem("theme", "light");
      updateThemeIcon("light");
    }
  });

  // Update theme icon based on current theme
  function updateThemeIcon(theme) {
    if (theme === "dark") {
      themeIcon.innerHTML = `
                <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
            `;
    } else {
      themeIcon.innerHTML = `
                <circle cx="12" cy="12" r="5"></circle>
                <line x1="12" y1="1" x2="12" y2="3"></line>
                <line x1="12" y1="21" x2="12" y2="23"></line>
                <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
                <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
                <line x1="1" y1="12" x2="3" y2="12"></line>
                <line x1="21" y1="12" x2="23" y2="12"></line>
                <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
                <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
            `;
    }
  }

  // Code content for different tabs
  const codeContent = {
    python: `<div class="code-line"><span class="comment"># uv pip install openmed</span></div>
<div class="code-line"><span class="keyword">import</span> <span class="function">openmed</span></div>
<div class="code-line"></div>
<div class="code-line"><span class="comment"># Quick analysis with smart model suggestions</span></div>
<div class="code-line"><span class="variable">text</span> = <span class="string">"Patient received 5mg warfarin for atrial fibrillation."</span></div>
<div class="code-line"><span class="variable">result</span> = <span class="function">openmed.analyze_text</span>(text, model_name=<span class="string">"pharma_detection_superclinical"</span>)</div>
<div class="code-line"></div>
<div class="code-line"><span class="keyword">print</span>(<span class="string">f"Found {len(result.entities)} entities"</span>)</div>
<div class="code-line"><span class="keyword">for</span> entity <span class="keyword">in</span> result.entities:</div>
<div class="code-line">    <span class="keyword">print</span>(<span class="string">f"- {entity.text} ({entity.label}): {entity.confidence:.3f}"</span>)</div>`,
  };

  const codeText = {
    python: `import openmed

# Quick analysis with smart model suggestions
text = "Patient received 5mg warfarin for atrial fibrillation."
result = openmed.analyze_text(text, model_name="pharma_detection_superclinical")

# Or use the actual HuggingFace model ID
# result = openmed.analyze_text(text, model_name="OpenMed/OpenMed-NER-PharmaDetect-SuperClinical-434M")

print(f"Found {len(result.entities)} entities")
for entity in result.entities:
    print(f"- {entity.text} ({entity.label}): {entity.confidence:.3f}")`,
  };

  let currentTab = "python";

  function switchTab(tab) {
    // Update tab buttons (only if not hidden)
    document.querySelectorAll(".code-tab").forEach((btn) => {
      if (btn.style.display !== "none") {
        btn.classList.remove("active");
        btn.classList.add("inactive");
      }
    });
    if (event.target.style.display !== "none") {
      event.target.classList.add("active");
      event.target.classList.remove("inactive");
    }

    // Update code content
    // currentTab = tab;
    document.getElementById("codeContent").innerHTML = codeContent[tab];
  }

  function copyCode() {
    const textToCopy = codeText[currentTab];
    navigator.clipboard.writeText(textToCopy).then(() => {
      console.log("Code copied to clipboard");
    });
  }

  function expandCode() {
    // Placeholder for expand functionality
    alert("Expand modal feature - to be implemented");
  }

  // Initialize code content
  document.getElementById("codeContent").innerHTML = codeContent[currentTab];
});
