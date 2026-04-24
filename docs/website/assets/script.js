// Initialize theme functionality when DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  const yearElement = document.getElementById("year");
  if (yearElement) {
    yearElement.textContent = new Date().getFullYear();
  }

  const themeToggle = document.getElementById("themeToggle");
  const themeIcon = document.getElementById("themeIcon");
  const root = document.documentElement;
  const savedTheme = localStorage.getItem("theme") || "light";

  const updateThemeIcon = (theme) => {
    if (!themeIcon) return;
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
  };

  if (savedTheme === "dark") {
    root.classList.add("dark");
  } else {
    root.classList.remove("dark");
  }
  updateThemeIcon(savedTheme);

  if (themeToggle) {
    themeToggle.addEventListener("click", () => {
      const nextTheme = root.classList.toggle("dark") ? "dark" : "light";
      localStorage.setItem("theme", nextTheme);
      updateThemeIcon(nextTheme);
    });
  }

  // Clean visual display with syntax highlighting
  const codeContent = {
    python: `<pre><code><span class="keyword">from</span> <span class="function">openmed</span> <span class="keyword">import</span> analyze_text

<span class="comment"># One-call clinical NLP from Python</span>
<span class="variable">result</span> = <span class="function">analyze_text</span>(
    <span class="string">"Metastatic breast cancer treated with trastuzumab."</span>,
    model_name=<span class="string">"disease_detection_superclinical"</span>,
)

<span class="keyword">for</span> <span class="variable">entity</span> <span class="keyword">in</span> <span class="variable">result</span>.entities:
    <span class="keyword">print</span>(<span class="variable">entity</span>.label, <span class="variable">entity</span>.text, <span class="variable">entity</span>.confidence)</code></pre>`,

    pii: `<pre><code><span class="keyword">from</span> <span class="function">openmed</span> <span class="keyword">import</span> extract_pii, deidentify

<span class="comment"># Smart merging keeps fragmented PHI spans together</span>
<span class="variable">text</span> = <span class="string">"Patient John Doe, DOB 1990-05-15, SSN 123-45-6789"</span>
<span class="variable">result</span> = <span class="function">extract_pii</span>(
    <span class="variable">text</span>,
    use_smart_merging=<span class="keyword">True</span>,
)

<span class="keyword">for</span> <span class="variable">entity</span> <span class="keyword">in</span> <span class="variable">result</span>.entities:
    <span class="keyword">print</span>(<span class="variable">entity</span>.label, <span class="variable">entity</span>.text, <span class="variable">entity</span>.confidence)

<span class="keyword">print</span>(<span class="function">deidentify</span>(<span class="variable">text</span>, method=<span class="string">"mask"</span>).deidentified_text)</code></pre>`,

    mlx: `<pre><code><span class="keyword">from</span> <span class="function">openmed</span> <span class="keyword">import</span> analyze_text
<span class="keyword">from</span> <span class="function">openmed.core</span> <span class="keyword">import</span> OpenMedConfig
<span class="keyword">from</span> <span class="function">openmed.core.backends</span> <span class="keyword">import</span> get_backend

<span class="comment"># MLX is auto-selected on Apple Silicon, HF elsewhere</span>
<span class="variable">config</span> = <span class="function">OpenMedConfig</span>()
<span class="variable">backend</span> = <span class="function">get_backend</span>(config=<span class="variable">config</span>)

<span class="variable">result</span> = <span class="function">analyze_text</span>(
    <span class="string">"Patient John Doe, DOB 1990-05-15, SSN 123-45-6789"</span>,
    model_name=<span class="string">"pii_detection"</span>,
    config=<span class="variable">config</span>,
)

<span class="keyword">print</span>(<span class="variable">backend</span>.__class__.__name__, <span class="variable">result</span>.entities)</code></pre>`,

    swift: `<pre><code><span class="keyword">import</span> <span class="function">OpenMedKit</span>

<span class="keyword">let</span> <span class="variable">modelDirectory</span> = <span class="keyword">try</span> <span class="keyword">await</span> <span class="function">OpenMedModelStore.downloadMLXModel</span>(
    repoID: <span class="string">"OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1-mlx"</span>
)

<span class="keyword">let</span> <span class="variable">openmed</span> = <span class="keyword">try</span> <span class="function">OpenMed</span>(
    backend: .mlx(modelDirectoryURL: <span class="variable">modelDirectory</span>)
)

<span class="keyword">let</span> <span class="variable">entities</span> = <span class="keyword">try</span> <span class="variable">openmed</span>.<span class="function">extractPII</span>(
    <span class="string">"Patient John Doe, DOB 1990-05-15, SSN 123-45-6789"</span>
)</code></pre>`,
  };

  // Full code for copying (includes installation and comments)
  const codeText = {
    python: `from openmed import analyze_text

# One-call clinical NLP from Python
result = analyze_text(
    "Metastatic breast cancer treated with trastuzumab.",
    model_name="disease_detection_superclinical",
)

for entity in result.entities:
    print(entity.label, entity.text, entity.confidence)`,

    pii: `from openmed import extract_pii, deidentify

# Smart merging keeps fragmented PHI spans together
text = "Patient John Doe, DOB 1990-05-15, SSN 123-45-6789"
result = extract_pii(
    text,
    use_smart_merging=True,
)

for entity in result.entities:
    print(entity.label, entity.text, entity.confidence)

print(deidentify(text, method="mask").deidentified_text)`,

    mlx: `from openmed import analyze_text
from openmed.core import OpenMedConfig
from openmed.core.backends import get_backend

# MLX is auto-selected on Apple Silicon, HF elsewhere
config = OpenMedConfig()
backend = get_backend(config=config)

result = analyze_text(
    "Patient John Doe, DOB 1990-05-15, SSN 123-45-6789",
    model_name="pii_detection",
    config=config,
)

print(backend.__class__.__name__)
print(result.entities)`,

    swift: `import OpenMedKit

let modelDirectory = try await OpenMedModelStore.downloadMLXModel(
    repoID: "OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1-mlx"
)

let openmed = try OpenMed(
    backend: .mlx(modelDirectoryURL: modelDirectory)
)

let entities = try openmed.extractPII(
    "Patient John Doe, DOB 1990-05-15, SSN 123-45-6789"
)

print(entities)`,
  };

  let currentTab = "python";
  const codeContainer = document.getElementById("codeContent");
  const codeTabs = document.querySelectorAll(".code-tab");

  const setCodeContent = (tab) => {
    if (!codeContainer || !codeContent[tab]) {
      return;
    }
    currentTab = tab;
    codeContainer.innerHTML = codeContent[tab];
  };

  window.switchTab = (tab, button) => {
    if (!codeContent[tab]) {
      return;
    }

    codeTabs.forEach((tabButton) => {
      tabButton.classList.remove("active");
    });

    if (button) {
      button.classList.add("active");
    }

    setCodeContent(tab);
  };

  const showCopyFeedback = (button) => {
    if (!button) {
      return;
    }

    if (button.classList.contains("copy-btn")) {
      const originalTitle = button.getAttribute("title");
      if (originalTitle !== null) {
        button.dataset.originalTitle = originalTitle;
      }
      button.setAttribute("title", "Copied!");
      button.classList.add("copied");
      setTimeout(() => {
        button.classList.remove("copied");
        if (button.dataset.originalTitle) {
          button.setAttribute("title", button.dataset.originalTitle);
          delete button.dataset.originalTitle;
        } else {
          button.removeAttribute("title");
        }
      }, 2000);
      return;
    }

    const original = button.innerHTML;
    button.classList.add("copied");
    button.innerHTML = "Copied";
    setTimeout(() => {
      button.classList.remove("copied");
      button.innerHTML = original;
    }, 2000);
  };

  window.copyCode = async (button) => {
    const codeString = codeText[currentTab];
    if (!codeString) {
      console.warn(`No code snippet registered for tab: ${currentTab}`);
      return;
    }

    try {
      await navigator.clipboard.writeText(codeString);
      showCopyFeedback(button);
    } catch (error) {
      console.error("Failed to copy code snippet", error);
      if (button) {
        button.innerHTML = "Copy failed";
      }
    }
  };

  window.copyInstall = async (button) => {
    const codeString = 'uv pip install "openmed[mlx]"';

    try {
      await navigator.clipboard.writeText(codeString);
      showCopyFeedback(button);
    } catch (error) {
      console.error("Failed to copy install snippet", error);
      if (button) {
        button.innerHTML = "Copy failed";
      }
    }
  };

  setCodeContent(currentTab);
});
