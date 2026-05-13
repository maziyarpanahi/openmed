/* HealthAdvocate — Frontend Application */

const HA = {
  /* ── API helpers ── */

  async api(endpoint, data) {
    let response;
    try {
      response = await fetch(`/api/${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });
    } catch (e) {
      throw new Error('We could not reach the server. Please check your internet connection and try again.');
    }
    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: 'Request failed' }));
      throw new Error(err.detail || response.statusText);
    }
    return response.json();
  },

  async apiGet(endpoint) {
    let response;
    try {
      response = await fetch(`/api/${endpoint}`);
    } catch (e) {
      throw new Error('We could not reach the server. Please check your internet connection and try again.');
    }
    if (!response.ok) throw new Error(response.statusText);
    return response.json();
  },

  _VALID_ENTITY_CLASSES: new Set(['disease', 'drug', 'anatomy', 'pii', 'procedure', 'symptom']),

  safeEntityClass(cat) {
    const c = (cat || '').toLowerCase();
    return this._VALID_ENTITY_CLASSES.has(c) ? c : 'entity';
  },

  escapeHtml(str) {
    if (typeof str !== 'string') return '';
    return str.replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
  },

  /* ── View Routing ── */

  showView(name) {
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    const target = document.getElementById(`view-${name}`);
    if (target) target.classList.add('active');

    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    const btn = document.querySelector(`.nav-btn[data-view="${name}"]`);
    if (btn) btn.classList.add('active');

    if (name === 'family') this.loadFamilyProfiles();
    if (name === 'tracks') this.loadTrackDashboard();
    if (name === 'home') {
      this.loadDashStrip();
      this.initScrollReveal();
    }

    window.scrollTo({ top: 0, behavior: 'smooth' });
  },

  /* ── Theme Toggle (default: light) ── */

  initTheme() {
    const saved = localStorage.getItem('ha-theme');
    if (saved === 'dark') document.documentElement.setAttribute('data-theme', 'dark');
    this.updateThemeIcon();
  },

  toggleTheme() {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    if (isDark) {
      document.documentElement.removeAttribute('data-theme');
      localStorage.setItem('ha-theme', 'light');
    } else {
      document.documentElement.setAttribute('data-theme', 'dark');
      localStorage.setItem('ha-theme', 'dark');
    }
    this.updateThemeIcon();
  },

  updateThemeIcon() {
    const icon = document.getElementById('theme-icon');
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    const btn = document.getElementById('btn-theme');
    if (isDark) {
      icon.innerHTML = '<path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z"/>';
      btn.setAttribute('aria-label', 'Switch to light theme');
    } else {
      icon.innerHTML = '<circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>';
      btn.setAttribute('aria-label', 'Switch to dark theme');
    }
  },

  /* ── Scroll Reveal (IntersectionObserver) ── */

  initScrollReveal() {
    if (this._scrollObserver) this._scrollObserver.disconnect();
    this._scrollObserver = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('revealed');
          this._scrollObserver.unobserve(entry.target);
        }
      });
    }, { threshold: 0.08 });

    document.querySelectorAll('.reveal:not(.revealed)').forEach(el => this._scrollObserver.observe(el));
  },

  /* ── Dashboard Strip ── */

  async loadDashStrip() {
    try {
      const [tracks, profiles] = await Promise.all([
        this.apiGet('tracks/dashboard').catch(() => null),
        this.apiGet('family/profiles').catch(() => []),
      ]);
      if (tracks) {
        document.getElementById('dash-active').textContent = tracks.active || 0;
        document.getElementById('dash-resolved').textContent = tracks.resolved || 0;
      }
      if (Array.isArray(profiles)) {
        document.getElementById('dash-family').textContent = profiles.length;
      }
    } catch (err) {
      console.warn('Dashboard strip load failed:', err.message);
    }
  },

  /* ── Loading / Errors ── */

  setLoading(el) {
    el.innerHTML = `<div class="skeleton-wrap">
      <div class="skeleton-line h-xl"></div>
      <div class="skeleton-line w-100"></div>
      <div class="skeleton-line w-80"></div>
      <div class="skeleton-line w-60"></div>
    </div>`;
  },

  _setBtnBusy(btn, busy) {
    if (!btn) return;
    btn.disabled = busy;
    if (busy && !btn.dataset.originalText) {
      btn.dataset.originalText = btn.textContent;
      btn.textContent = 'Analyzing...';
    } else if (!busy && btn.dataset.originalText) {
      btn.textContent = btn.dataset.originalText;
      delete btn.dataset.originalText;
    }
  },

  showError(el, msg) {
    el.innerHTML = `<div class="flag-item flag-danger">${this.escapeHtml(msg)}</div>`;
  },

  showEmpty(el, msg) {
    el.innerHTML = `<div class="empty-state">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
      <p>${this.escapeHtml(msg)}</p>
    </div>`;
  },

  /* ── Symptom Assessment ── */

  async assessSymptoms(event) {
    const btn = event?.currentTarget;
    const symptoms = document.getElementById('symptom-input').value;
    const el = document.getElementById('symptom-results');
    if (!symptoms.trim()) { this.showEmpty(el, 'Describe your symptoms to get started.'); return; }
    this.setLoading(el);
    this._setBtnBusy(btn, true);
    try {
      const data = await this.api('symptoms/assess', { symptoms });
      this.renderSymptoms(data, el);
    } catch (err) {
      this.showError(el, err.message);
    } finally {
      this._setBtnBusy(btn, false);
    }
  },

  renderSymptoms(data, el) {
    const VALID_URGENCY = ['low', 'medium', 'high'];
    const u = VALID_URGENCY.includes(data.urgency) ? data.urgency : 'low';
    let html = `
      <div class="result-section"><h3>Urgency Level</h3>
        <span class="urgency-badge urgency-${u}">${u.toUpperCase()}</span>
      </div>
      <div class="result-section"><h3>Explanation</h3>
        <p class="result-text">${this.escapeHtml(data.explanation)}</p>
      </div>`;

    if (data.conditions?.length) {
      html += `<div class="result-section"><h3>Possible Conditions</h3>`;
      for (const c of data.conditions) {
        html += `<div class="condition-item">
          <span class="condition-name">${this.escapeHtml(c.name)}</span>
          <span class="condition-confidence">(${Math.round(c.confidence * 100)}%)</span>
        </div>`;
      }
      html += `</div>`;
    }
    if (data.possible_conditions?.length) {
      html += `<div class="result-section"><h3>LLM Assessment</h3>`;
      for (const c of data.possible_conditions) {
        html += `<div class="condition-item">
          <span class="condition-name">${this.escapeHtml(c.name || '')}</span>
          <span class="condition-confidence">(${this.escapeHtml(c.likelihood || '')})</span>
        </div>`;
      }
      html += `</div>`;
    }
    if (data.action_items?.length) {
      html += `<div class="result-section"><h3>Action Items</h3><ol>`;
      for (const a of data.action_items) html += `<li class="result-text">${this.escapeHtml(a)}</li>`;
      html += `</ol></div>`;
    }
    if (data.red_flags?.length) {
      html += `<div class="result-section"><h3>Red Flags</h3>`;
      for (const r of data.red_flags) html += `<div class="flag-item flag-danger">${this.escapeHtml(r)}</div>`;
      html += `</div>`;
    }
    if (data.validation) {
      html += `<div class="result-section"><h3>Validation</h3>
        <div class="condition-item"><span class="condition-name">Reliability</span> <span class="condition-confidence">${this.escapeHtml(data.validation.reliability || 'N/A')}</span></div>
        ${data.validation.urgency_disagreement ? '<div class="flag-item flag-danger">Urgency disagreement detected — upgraded to HIGH for safety.</div>' : ''}
      </div>`;
    }
    el.innerHTML = html;
  },

  /* ── Document Decoder ── */

  async decodeDocument(event) {
    const btn = event?.currentTarget;
    const text = document.getElementById('doc-input').value;
    const el = document.getElementById('doc-results');
    if (!text.trim()) { this.showEmpty(el, 'Paste a medical document to decode.'); return; }
    this.setLoading(el);
    this._setBtnBusy(btn, true);
    try {
      const data = await this.api('documents/decode', { text });
      this.renderDocument(data, el);
    } catch (err) { this.showError(el, err.message); } finally { this._setBtnBusy(btn, false); }
  },

  renderDocument(data, el) {
    let html = `<div class="result-section"><h3>Summary</h3><p class="result-text">${this.escapeHtml(data.explanation)}</p></div>`;
    if (data.urgency) {
      html += `<div class="result-section"><h3>Urgency</h3>
        <span class="urgency-badge urgency-${this.escapeHtml(data.urgency)}">${data.urgency.toUpperCase()}</span></div>`;
    }
    if (data.entities?.length) {
      html += `<div class="result-section"><h3>Medical Entities</h3><div class="entity-list">`;
      for (const e of data.entities) html += `<span class="entity-chip ${this.safeEntityClass(e.category)}">${this.escapeHtml(e.text)} <small>(${this.escapeHtml(e.category)})</small></span>`;
      html += `</div></div>`;
    }
    if (data.medical_terms_explained?.length) {
      html += `<div class="result-section"><h3>Terms Explained</h3>`;
      for (const t of data.medical_terms_explained) html += `<div class="condition-item"><strong>${this.escapeHtml(t.term)}</strong>: ${this.escapeHtml(t.explanation)}</div>`;
      html += `</div>`;
    }
    if (data.action_items?.length) {
      html += `<div class="result-section"><h3>Action Items</h3><ol>`;
      for (const a of data.action_items) html += `<li class="result-text">${this.escapeHtml(a)}</li>`;
      html += `</ol></div>`;
    }
    if (data.red_flags?.length) {
      html += `<div class="result-section"><h3>Red Flags</h3>`;
      for (const r of data.red_flags) html += `<div class="flag-item flag-danger">${this.escapeHtml(r)}</div>`;
      html += `</div>`;
    }
    if (data.pii_found?.length) {
      html += `<div class="result-section"><h3>Personal Information Detected</h3>`;
      for (const p of data.pii_found) html += `<div class="flag-item flag-warning">Found ${this.escapeHtml(p.category || 'PII')}: "${this.escapeHtml(p.text)}"</div>`;
      html += `</div>`;
    }
    el.innerHTML = html;
  },

  /* ── Bill Decoder ── */

  async decodeBill(event) {
    const btn = event?.currentTarget;
    const bill_text = document.getElementById('bill-input').value;
    const el = document.getElementById('bill-results');
    if (!bill_text.trim()) { this.showEmpty(el, 'Paste a medical bill to analyze.'); return; }
    this.setLoading(el);
    this._setBtnBusy(btn, true);
    try {
      const data = await this.api('bills/decode', { bill_text });
      this.renderBill(data, el);
    } catch (err) { this.showError(el, err.message); } finally { this._setBtnBusy(btn, false); }
  },

  renderBill(data, el) {
    let html = '';
    if (data.total) html += `<div class="result-section"><h3>Total</h3><div class="bill-total">${this.escapeHtml(data.total)}</div></div>`;
    if (data.urgency) {
      html += `<div class="result-section"><h3>Urgency</h3>
        <span class="urgency-badge urgency-${this.escapeHtml(data.urgency)}">${data.urgency.toUpperCase()}</span></div>`;
    }
    if (data.explanation) html += `<div class="result-section"><h3>Explanation</h3><p class="result-text">${this.escapeHtml(data.explanation)}</p></div>`;
    if (data.action_items?.length) {
      html += `<div class="result-section"><h3>Action Items</h3><ol>`;
      for (const a of data.action_items) html += `<li class="result-text">${this.escapeHtml(a)}</li>`;
      html += `</ol></div>`;
    }
    if (data.suspicious_charges?.length) {
      html += `<div class="result-section"><h3>Suspicious Charges</h3>`;
      for (const c of data.suspicious_charges) html += `<div class="flag-item flag-warning">${this.escapeHtml(c)}</div>`;
      html += `</div>`;
    }
    if (data.red_flags?.length) {
      html += `<div class="result-section"><h3>Red Flags</h3>`;
      for (const r of data.red_flags) html += `<div class="flag-item flag-danger">${this.escapeHtml(r)}</div>`;
      html += `</div>`;
    }
    if (data.billing_rights?.length) {
      html += `<div class="result-section"><h3>Your Billing Rights</h3><ul>`;
      for (const r of data.billing_rights) html += `<li class="result-text">${this.escapeHtml(r)}</li>`;
      html += `</ul></div>`;
    }
    if (!data.explanation && !data.suspicious_charges?.length && !data.red_flags?.length) html = '<p class="result-text">No bill issues detected. Try pasting a more detailed bill.</p>';
    el.innerHTML = html;
  },

  /* ── Insurance Denial Fighter ── */

  async fightDenial(event) {
    const btn = event?.currentTarget;
    const denial_text = document.getElementById('denial-input').value;
    const el = document.getElementById('denial-results');
    if (!denial_text.trim()) { this.showEmpty(el, 'Paste a denial letter to analyze.'); return; }
    const patient_info = document.getElementById('denial-patient-info')?.value || '';
    this.setLoading(el);
    this._setBtnBusy(btn, true);
    try {
      const data = await this.api('insurance/fight', { denial_text, patient_info });
      this.renderDenial(data, el);
    } catch (err) { this.showError(el, err.message); } finally { this._setBtnBusy(btn, false); }
  },

  renderDenial(data, el) {
    let html = '';
    if (data.urgency) {
      html += `<div class="result-section"><h3>Urgency</h3>
        <span class="urgency-badge urgency-${this.escapeHtml(data.urgency)}">${data.urgency.toUpperCase()}</span></div>`;
    }
    if (data.explanation) html += `<div class="result-section"><h3>What This Denial Means</h3><p class="result-text">${this.escapeHtml(data.explanation)}</p></div>`;
    if (data.denial_reason) html += `<div class="result-section"><h3>Denial Reason</h3><p class="result-text">${this.escapeHtml(data.denial_reason)}</p></div>`;
    if (data.red_flags?.length) {
      html += `<div class="result-section"><h3>Red Flags</h3>`;
      for (const r of data.red_flags) html += `<div class="flag-item flag-danger">${this.escapeHtml(r)}</div>`;
      html += `</div>`;
    }
    if (data.appeal_arguments?.length) {
      html += `<div class="result-section"><h3>Arguments for Your Appeal</h3><ol>`;
      for (const a of data.appeal_arguments) html += `<li class="result-text">${this.escapeHtml(a)}</li>`;
      html += `</ol></div>`;
    }
    if (data.action_items?.length) {
      html += `<div class="result-section"><h3>Action Items</h3><ol>`;
      for (const a of data.action_items) html += `<li class="result-text">${this.escapeHtml(a)}</li>`;
      html += `</ol></div>`;
    }
    if (data.draft_appeal) html += `<div class="result-section"><h3>Draft Appeal Letter</h3><div class="appeal-letter">${this.escapeHtml(data.draft_appeal)}</div></div>`;
    if (data.entities_found) {
      html += `<div class="result-section"><h3>Medical Entities Found</h3>`;
      if (data.entities_found.conditions?.length) {
        html += `<div class="condition-item"><strong>Conditions:</strong> ${this.escapeHtml(data.entities_found.conditions.map(c => c.text).join(', '))}</div>`;
      }
      if (data.entities_found.medications?.length) {
        html += `<div class="condition-item"><strong>Medications:</strong> ${this.escapeHtml(data.entities_found.medications.map(m => m.text).join(', '))}</div>`;
      }
      html += `</div>`;
    }
    el.innerHTML = html;
  },

  /* ── Drug Checker ── */

  async checkDrug(event) {
    const btn = event?.currentTarget;
    const drug_name = document.getElementById('drug-input').value;
    const el = document.getElementById('drug-results');
    if (!drug_name.trim()) { this.showEmpty(el, 'Enter a drug name to check.'); return; }
    this.setLoading(el);
    this._setBtnBusy(btn, true);
    try {
      const data = await this.api('drugs/check', { drug_name });
      this.renderDrug(data, el);
    } catch (err) { this.showError(el, err.message); } finally { this._setBtnBusy(btn, false); }
  },

  renderDrug(data, el) {
    let html = `<div class="result-section"><h3>${this.escapeHtml(data.drug)}</h3><div class="drug-class">${this.escapeHtml(data.drug_class)}</div>`;
    if (data.generic_available === true) html += `<div class="drug-generic">${this.escapeHtml(data.generic_name)}</div>`;
    else if (data.generic_available === "Unknown") html += `<p class="result-text">Generic availability unknown. ${this.escapeHtml(data.cost_note || '')}</p>`;
    if (data.alternatives?.length) {
      html += `<h4 style="margin-top:14px;font-size:11px;text-transform:uppercase;letter-spacing:0.06em;color:var(--text-3)">Alternatives</h4><ul class="alt-list">`;
      for (const a of data.alternatives) html += `<li>${this.escapeHtml(a)}</li>`;
      html += `</ul>`;
    }
    if (data.cost_note) html += `<p class="result-text" style="margin-top:14px">${this.escapeHtml(data.cost_note)}</p>`;
    html += `</div>`;
    el.innerHTML = html;
  },

  /* ── Appointment Prep ── */

  async prepareAppointment(event) {
    const btn = event?.currentTarget;
    const symptoms = document.getElementById('appt-symptoms').value;
    const el = document.getElementById('appt-results');
    if (!symptoms.trim()) { this.showEmpty(el, 'Describe your symptoms or reason for the visit.'); return; }
    const concern = document.getElementById('appt-concern').value;
    this.setLoading(el);
    this._setBtnBusy(btn, true);
    try {
      const data = await this.api('appointments/prepare', { symptoms, concern });
      this.renderAppointment(data, el);
    } catch (err) { this.showError(el, err.message); } finally { this._setBtnBusy(btn, false); }
  },

  renderAppointment(data, el) {
    let html = '';
    if (data.talking_points?.length) {
      html += `<div class="result-section"><h3>Talking Points</h3><ol>`;
      for (const tp of data.talking_points) html += `<li class="result-text">${this.escapeHtml(tp)}</li>`;
      html += `</ol></div>`;
    }
    if (data.questions_to_ask?.length) {
      html += `<div class="result-section"><h3>Questions to Ask</h3><ol>`;
      for (const q of data.questions_to_ask) html += `<li class="result-text">${this.escapeHtml(q)}</li>`;
      html += `</ol></div>`;
    }
    if (data.advocacy_script) html += `<div class="result-section"><h3>Your Advocacy Script</h3><div class="script-block">${this.escapeHtml(data.advocacy_script)}</div></div>`;
    el.innerHTML = html;
  },

  /* ── Discharge Translator ── */

  async translateDischarge(event) {
    const btn = event?.currentTarget;
    const text = document.getElementById('discharge-input').value;
    const el = document.getElementById('discharge-results');
    if (!text.trim()) { this.showEmpty(el, 'Paste discharge instructions to translate.'); return; }
    this.setLoading(el);
    this._setBtnBusy(btn, true);
    try {
      const data = await this.api('discharge/translate', { text });
      this.renderDischarge(data, el);
    } catch (err) { this.showError(el, err.message); } finally { this._setBtnBusy(btn, false); }
  },

  renderDischarge(data, el) {
    let html = '';
    if (data.urgency) {
      html += `<div class="result-section"><h3>Urgency</h3>
        <span class="urgency-badge urgency-${this.escapeHtml(data.urgency)}">${data.urgency.toUpperCase()}</span></div>`;
    }
    if (data.explanation) html += `<div class="result-section"><h3>Plain Language Summary</h3><div class="plain-language">${this.escapeHtml(data.explanation)}</div></div>`;
    if (data.medication_instructions?.length) {
      html += `<div class="result-section"><h3>Medication Instructions</h3><ul>`;
      for (const m of data.medication_instructions) html += `<li class="result-text">${this.escapeHtml(m)}</li>`;
      html += `</ul></div>`;
    }
    if (data.warning_signs?.length) {
      html += `<div class="result-section"><h3>Warning Signs — When to Go Back to the ER</h3>`;
      for (const w of data.warning_signs) html += `<div class="flag-item flag-danger">${this.escapeHtml(w)}</div>`;
      html += `</div>`;
    }
    if (data.red_flags?.length) {
      html += `<div class="result-section"><h3>Red Flags</h3>`;
      for (const r of data.red_flags) html += `<div class="flag-item flag-danger">${this.escapeHtml(r)}</div>`;
      html += `</div>`;
    }
    if (data.follow_up_steps?.length) {
      html += `<div class="result-section"><h3>Follow-Up Steps</h3><ol>`;
      for (const f of data.follow_up_steps) html += `<li class="result-text">${this.escapeHtml(f)}</li>`;
      html += `</ol></div>`;
    }
    if (data.action_items?.length) {
      html += `<div class="result-section"><h3>Action Items</h3><ol>`;
      for (const a of data.action_items) html += `<li class="result-text">${this.escapeHtml(a)}</li>`;
      html += `</ol></div>`;
    }
    if (data.medications_detected?.length) {
      html += `<div class="result-section"><h3>Medications Detected</h3><div class="entity-list">`;
      for (const m of data.medications_detected) html += `<span class="entity-chip drug">${this.escapeHtml(m.name)}</span>`;
      html += `</div></div>`;
    }
    el.innerHTML = html;
  },

  /* ── Second Opinion ── */

  async createSecondOpinion(event) {
    const btn = event?.currentTarget;
    const records = document.getElementById('secondop-input').value;
    const el = document.getElementById('secondop-results');
    if (!records.trim()) { this.showEmpty(el, 'Paste medical records to create a brief.'); return; }
    this.setLoading(el);
    this._setBtnBusy(btn, true);
    try {
      const data = await this.api('second-opinion/create', { records });
      this.renderSecondOpinion(data, el);
    } catch (err) { this.showError(el, err.message); } finally { this._setBtnBusy(btn, false); }
  },

  renderSecondOpinion(data, el) {
    let html = '';
    if (data.urgency) {
      html += `<div class="result-section"><h3>Urgency</h3>
        <span class="urgency-badge urgency-${this.escapeHtml(data.urgency)}">${data.urgency.toUpperCase()}</span></div>`;
    }
    if (data.explanation) html += `<div class="result-section"><h3>Summary for Second Opinion</h3><p class="result-text">${this.escapeHtml(data.explanation)}</p></div>`;
    if (data.conditions?.length) {
      html += `<div class="result-section"><h3>Conditions</h3><div class="entity-list">`;
      for (const c of data.conditions) html += `<span class="entity-chip disease">${this.escapeHtml(c.text)}</span>`;
      html += `</div></div>`;
    }
    if (data.medications?.length) {
      html += `<div class="result-section"><h3>Medications</h3><div class="entity-list">`;
      for (const m of data.medications) html += `<span class="entity-chip drug">${this.escapeHtml(m.text)}</span>`;
      html += `</div></div>`;
    }
    if (data.key_questions?.length) {
      html += `<div class="result-section"><h3>Key Questions for the Specialist</h3><ol>`;
      for (const q of data.key_questions) html += `<li class="result-text">${this.escapeHtml(q)}</li>`;
      html += `</ol></div>`;
    }
    if (data.records_to_bring?.length) {
      html += `<div class="result-section"><h3>Records to Bring</h3><ul>`;
      for (const r of data.records_to_bring) html += `<li class="result-text">${this.escapeHtml(r)}</li>`;
      html += `</ul></div>`;
    }
    if (data.treatment_concerns?.length) {
      html += `<div class="result-section"><h3>Treatment Concerns</h3>`;
      for (const c of data.treatment_concerns) html += `<div class="flag-item flag-warning">${this.escapeHtml(c)}</div>`;
      html += `</div>`;
    }
    if (data.action_items?.length) {
      html += `<div class="result-section"><h3>Action Items</h3><ol>`;
      for (const a of data.action_items) html += `<li class="result-text">${this.escapeHtml(a)}</li>`;
      html += `</ol></div>`;
    }
    if (data.deidentified_records) html += `<div class="result-section"><h3>De-identified Records (safe to share)</h3><div class="plain-language">${this.escapeHtml(data.deidentified_records)}</div></div>`;
    el.innerHTML = html;
  },

  /* ── Community Health Scanner ── */

  async scanCommunity(event) {
    const btn = event?.currentTarget;
    const text = document.getElementById('community-input').value;
    const el = document.getElementById('community-results');
    if (!text.trim()) { this.showEmpty(el, 'Paste a health bulletin to scan.'); return; }
    this.setLoading(el);
    this._setBtnBusy(btn, true);
    try {
      const data = await this.api('community/scan', { text });
      this.renderCommunity(data, el);
    } catch (err) { this.showError(el, err.message); } finally { this._setBtnBusy(btn, false); }
  },

  renderCommunity(data, el) {
    let html = '';
    if (data.urgency) {
      html += `<div class="result-section"><h3>Urgency</h3>
        <span class="urgency-badge urgency-${this.escapeHtml(data.urgency)}">${data.urgency.toUpperCase()}</span></div>`;
    }
    if (data.explanation) html += `<div class="result-section"><h3>Summary</h3><p class="result-text">${this.escapeHtml(data.explanation)}</p></div>`;
    if (data.credibility) {
      const credClass = data.credibility === 'low' ? 'flag-danger' : data.credibility === 'high' ? 'flag-info' : 'flag-warning';
      html += `<div class="result-section"><h3>Credibility</h3><div class="flag-item ${credClass}">Credibility: ${data.credibility.toUpperCase()}</div></div>`;
    }
    if (data.scientific_context) html += `<div class="result-section"><h3>Scientific Context</h3><p class="result-text">${this.escapeHtml(data.scientific_context)}</p></div>`;
    if (data.recommended_action) html += `<div class="result-section"><h3>Recommended Action</h3><p class="result-text">${this.escapeHtml(data.recommended_action)}</p></div>`;
    if (data.red_flags?.length) {
      html += `<div class="result-section"><h3>Red Flags</h3>`;
      for (const r of data.red_flags) html += `<div class="flag-item flag-danger">${this.escapeHtml(r)}</div>`;
      html += `</div>`;
    }
    if (data.action_items?.length) {
      html += `<div class="result-section"><h3>Action Items</h3><ol>`;
      for (const a of data.action_items) html += `<li class="result-text">${this.escapeHtml(a)}</li>`;
      html += `</ol></div>`;
    }
    if (data.conditions_detected?.length) {
      html += `<div class="result-section"><h3>Conditions Detected</h3><div class="entity-list">`;
      for (const c of data.conditions_detected) html += `<span class="entity-chip disease">${this.escapeHtml(c.name)}</span>`;
      html += `</div></div>`;
    }
    if (data.treatments_detected?.length) {
      html += `<div class="result-section"><h3>Treatments Detected</h3><div class="entity-list">`;
      for (const t of data.treatments_detected) html += `<span class="entity-chip drug">${this.escapeHtml(t.name)}</span>`;
      html += `</div></div>`;
    }
    if (!html) html = '<p class="result-text">No significant findings detected in this text.</p>';
    el.innerHTML = html;
  },

  /* ── Family Tracker ── */

  async createFamilyProfile() {
    const name = document.getElementById('family-name').value;
    const relationship = document.getElementById('family-relationship').value;
    const el = document.getElementById('family-list');
    if (!name.trim()) return;
    try {
      await this.api('family/profiles', { name, relationship });
      document.getElementById('family-name').value = '';
      this.loadFamilyProfiles();
    } catch (err) { this.showError(el, err.message); }
  },

  async loadFamilyProfiles() {
    const el = document.getElementById('family-list');
    if (!el) return;
    try {
      const profiles = await this.apiGet('family/profiles');
      this.renderFamilyProfiles(profiles, el);
    } catch (err) {
      el.innerHTML = '<p class="result-text" style="color:var(--text-3)">Could not load family profiles.</p>';
    }
  },

  renderFamilyProfiles(profiles, el) {
    if (!profiles.length) {
      this.showEmpty(el, 'No family members added yet. Add someone above.');
      return;
    }
    let html = '';
    for (const p of profiles) {
      const conds = p.conditions.map(c => c.name).join(', ') || 'None';
      const meds = p.medications.map(m => m.name).join(', ') || 'None';
      html += `<div class="profile-card">
        <span class="profile-name">${this.escapeHtml(p.name)}</span>
        <span class="profile-relationship"> (${this.escapeHtml(p.relationship)})</span>
        <div class="profile-detail"><strong>Conditions:</strong> ${this.escapeHtml(conds)}</div>
        <div class="profile-detail"><strong>Medications:</strong> ${this.escapeHtml(meds)}</div>
        <div style="margin-top:10px">
          <input type="text" id="cond-${p.id}" placeholder="Add condition" aria-label="Add condition for ${this.escapeHtml(p.name)}" style="width:45%;font-size:13px">
          <button class="btn-ghost btn-sm" data-action="add-condition" data-profile-id="${p.id}">Add</button>
        </div>
      </div>`;
    }
    el.innerHTML = html;
  },

  async addCondition(profileId) {
    const input = document.getElementById(`cond-${profileId}`);
    if (!input?.value.trim()) return;
    try {
      await this.api(`family/profiles/${profileId}/conditions`, { condition: input.value });
      this.loadFamilyProfiles();
    } catch (err) {
      const el = document.getElementById('family-list');
      if (el) this.showError(el, err.message);
    }
  },

  /* ── Health Tracks ── */

  async createTrack() {
    const concern = document.getElementById('track-concern').value;
    const category = document.getElementById('track-category').value;
    const el = document.getElementById('track-dashboard');
    if (!concern.trim()) return;
    try {
      await this.api('tracks', { concern, category });
      document.getElementById('track-concern').value = '';
      this.loadTrackDashboard();
    } catch (err) { if (el) this.showError(el, err.message); }
  },

  async loadTrackDashboard() {
    const el = document.getElementById('track-dashboard');
    if (!el) return;
    try {
      const data = await this.apiGet('tracks/dashboard');
      this.renderTrackDashboard(data, el);
    } catch {
      el.innerHTML = '<p class="result-text" style="color:var(--text-3)">Could not load tracks.</p>';
    }
  },

  renderTrackDashboard(data, el) {
    let html = `<div class="dashboard-stats">
      <div class="stat-card"><div class="stat-number">${data.active}</div><div class="stat-label">Active</div></div>
      <div class="stat-card"><div class="stat-number">${data.monitoring}</div><div class="stat-label">Monitoring</div></div>
      <div class="stat-card"><div class="stat-number">${data.resolved}</div><div class="stat-label">Resolved</div></div>
    </div>`;

    if (data.tracks?.length) {
      for (const t of data.tracks) {
        const safeStatus = this.escapeHtml(t.status);
        html += `<div class="track-item">
          <div><span class="track-concern">${this.escapeHtml(t.concern)}</span><small style="color:var(--text-3);margin-left:6px">${this.escapeHtml(t.category)}</small></div>
          <div style="display:flex;align-items:center;gap:6px">
            <span class="track-status ${safeStatus}">${safeStatus}</span>
            ${t.status !== 'resolved' ? `<button class="btn-ghost btn-sm" data-action="update-track" data-track-id="${t.id}" data-status="resolved">Resolve</button>` : ''}
            ${t.status === 'active' ? `<button class="btn-ghost btn-sm" data-action="update-track" data-track-id="${t.id}" data-status="monitoring">Monitor</button>` : ''}
          </div>
        </div>`;
      }
    } else {
      this.showEmpty(el, 'No health tracks yet. Start tracking a concern above.');
      return;
    }
    el.innerHTML = html;
  },

  async updateTrackStatus(trackId, status) {
    try {
      await this.api(`tracks/${trackId}`, { status });
      this.loadTrackDashboard();
    } catch (err) {
      const el = document.getElementById('track-dashboard');
      if (el) this.showError(el, err.message);
    }
  },
};

/* ── Initialize ── */

document.addEventListener('DOMContentLoaded', () => {
  HA.initTheme();
  HA.loadDashStrip();

  /* Scroll reveal for home view elements */
  HA.initScrollReveal();

  /* Navigation */
  document.getElementById('main-nav').addEventListener('click', (e) => {
    if (!e.target.classList.contains('nav-btn')) return;
    HA.showView(e.target.dataset.view);
  });

  document.getElementById('btn-home').addEventListener('click', () => HA.showView('home'));
  document.getElementById('btn-theme').addEventListener('click', () => HA.toggleTheme());

  /* Entry card clicks */
  document.querySelectorAll('.entry-card').forEach(card => {
    const go = () => HA.showView(card.dataset.goto);
    card.addEventListener('click', go);
    card.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); } });
  });

  /* Delegated click handler for data-action buttons (family conditions, track status) */
  document.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-action]');
    if (!btn) return;
    const action = btn.dataset.action;
    if (action === 'add-condition') {
      const profileId = btn.dataset.profileId;
      if (profileId) HA.addCondition(profileId);
    } else if (action === 'update-track') {
      const trackId = btn.dataset.trackId;
      const status = btn.dataset.status;
      if (trackId && status) HA.updateTrackStatus(trackId, status);
    }
  });
});
