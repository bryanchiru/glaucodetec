const API   = "http://localhost:8000";
let   token = localStorage.getItem("token") || null;

// ─── Inicialización ───────────────────────────────────────────────────────────
window.onload = () => {
  token ? showApp() : showAuth();

  const dz = document.getElementById("dropZone");
  dz.addEventListener("dragover",  e => { e.preventDefault(); dz.style.borderColor = "#1a56db"; });
  dz.addEventListener("dragleave", ()  => { dz.style.borderColor = ""; });
  dz.addEventListener("drop", e => {
    e.preventDefault();
    dz.style.borderColor = "";
    if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
  });
};

// ─── Tabs ─────────────────────────────────────────────────────────────────────
function showTab(name) {
  const names = ["login", "register", "recover"];
  document.querySelectorAll(".tab").forEach((t, i) => t.classList.toggle("active", names[i] === name));
  document.getElementById("tabLogin").classList.toggle("hidden",    name !== "login");
  document.getElementById("tabRegister").classList.toggle("hidden", name !== "register");
  document.getElementById("tabRecover").classList.toggle("hidden",  name !== "recover");
}

// ─── Vistas ───────────────────────────────────────────────────────────────────
function showAuth() {
  document.getElementById("viewAuth").classList.remove("hidden");
  document.getElementById("viewApp").classList.add("hidden");
  document.getElementById("btnLogout").classList.add("hidden");
  document.getElementById("navUser").classList.add("hidden");
}

function showApp() {
  document.getElementById("viewAuth").classList.add("hidden");
  document.getElementById("viewApp").classList.remove("hidden");
  document.getElementById("btnLogout").classList.remove("hidden");
  fetchMe();
  fetchHistory();
}

// ─── Helpers ──────────────────────────────────────────────────────────────────
async function apiFetch(path, opts = {}) {
  const headers = { ...(opts.headers || {}) };
  if (token) headers["Authorization"] = `Bearer ${token}`;
  const res  = await fetch(API + path, { ...opts, headers });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Error en la petición");
  return data;
}

function setMsg(id, text, isError) {
  const el = document.getElementById(id);
  el.textContent = text;
  el.className   = "msg " + (isError ? "error" : "success");
}

// ─── Login ────────────────────────────────────────────────────────────────────
async function login(e) {
  e.preventDefault();
  try {
    const form = new URLSearchParams();
    form.append("username", document.getElementById("loginUser").value);
    form.append("password", document.getElementById("loginPass").value);
    const data = await apiFetch("/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: form,
    });
    token = data.access_token;
    localStorage.setItem("token", token);
    showApp();
  } catch (err) {
    setMsg("loginMsg", err.message, true);
  }
}

// ─── Registro ─────────────────────────────────────────────────────────────────
async function register(e) {
  e.preventDefault();
  try {
    await apiFetch("/auth/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        username: document.getElementById("regUser").value,
        email:    document.getElementById("regEmail").value,
        password: document.getElementById("regPass").value,
      }),
    });
    setMsg("regMsg", "¡Cuenta creada! Ahora inicia sesión.", false);
    setTimeout(() => showTab("login"), 1500);
  } catch (err) {
    setMsg("regMsg", err.message, true);
  }
}

// ─── Recuperar contraseña ─────────────────────────────────────────────────────
async function recover(e) {
  e.preventDefault();
  try {
    const data = await apiFetch("/auth/password-reset-request", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email: document.getElementById("recEmail").value }),
    });
    setMsg("recMsg", data.message, false);
  } catch (err) {
    setMsg("recMsg", err.message, true);
  }
}

// ─── Logout ───────────────────────────────────────────────────────────────────
function logout() {
  token = null;
  localStorage.removeItem("token");
  document.getElementById("resultCard").classList.add("hidden");
  document.getElementById("preview").classList.add("hidden");
  showAuth();
}

// ─── Me ───────────────────────────────────────────────────────────────────────
async function fetchMe() {
  try {
    const me = await apiFetch("/auth/me");
    const el = document.getElementById("navUser");
    el.textContent = `👤 ${me.username}`;
    el.classList.remove("hidden");
  } catch (_) {}
}

// ─── Predicción ───────────────────────────────────────────────────────────────
function handleFile(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = e => {
    const img = document.getElementById("preview");
    img.src = e.target.result;
    img.classList.remove("hidden");
  };
  reader.readAsDataURL(file);
  uploadAndPredict(file);
}

async function uploadAndPredict(file) {
  document.getElementById("resultCard").classList.add("hidden");
  try {
    const fd = new FormData();
    fd.append("file", file);
    const res  = await fetch(API + "/predict", {
      method:  "POST",
      headers: { "Authorization": `Bearer ${token}` },
      body:    fd,
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Error");
    showResult(data);
    fetchHistory();
  } catch (err) {
    alert("Error al procesar imagen: " + err.message);
  }
}

function showResult(data) {
  const isRG = data.class === "RG";
  document.getElementById("resultIcon").textContent   = isRG ? "🔴" : "🟢";
  document.getElementById("resultTitle").textContent  = isRG ? "Posible Glaucoma" : "Sin Signos de Glaucoma";
  document.getElementById("resultTitle").style.color  = isRG ? "#dc2626" : "#16a34a";
  document.getElementById("resultLabel").textContent  = `${data.label} · Confianza: ${data.confidence}%`;

  document.getElementById("barNRG").style.width  = data.prob_NRG + "%";
  document.getElementById("barRG").style.width   = data.prob_RG  + "%";
  document.getElementById("pctNRG").textContent  = data.prob_NRG + "%";
  document.getElementById("pctRG").textContent   = data.prob_RG  + "%";

  const warn = document.getElementById("resultWarning");
  isRG ? warn.classList.remove("hidden") : warn.classList.add("hidden");
  document.getElementById("resultCard").classList.remove("hidden");
}

// ─── Historial ────────────────────────────────────────────────────────────────
async function fetchHistory() {
  try {
    const preds = await apiFetch("/predictions");
    const list  = document.getElementById("historyList");
    if (!preds.length) {
      list.innerHTML = "<p style='color:#94a3b8;font-size:.9rem'>Sin análisis previos</p>";
      return;
    }
    list.innerHTML = preds.slice(0, 10).map(p => `
      <div class="hist-item">
        <span class="hist-name">${p.filename}</span>
        <span class="hist-badge badge-${p.result.toLowerCase()}">${p.result}</span>
        <span class="hist-conf">${p.confidence}%</span>
        <span class="hist-date">${new Date(p.created_at).toLocaleDateString('es')}</span>
      </div>
    `).join("");
  } catch (_) {}
}
