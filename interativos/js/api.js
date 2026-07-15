// Helpers compartilhados para falar com a API real do servidor (src/api.rs).
// Nada aqui inventa dados — todo valor vem de uma resposta HTTP/WS/SSE real.

const API_BASE = '/api';

export async function apiGet(path) {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error((await safeJson(res))?.error || `${res.status} ${res.statusText}`);
  return res.json();
}

export async function apiPost(path, body) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body ?? {}),
  });
  if (!res.ok) throw new Error((await safeJson(res))?.error || `${res.status} ${res.statusText}`);
  return safeJson(res);
}

async function safeJson(res) {
  try { return await res.json(); } catch { return null; }
}

export function wsUrl(path) {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${proto}//${location.host}${API_BASE}${path}`;
}

/** Atualiza a pill de status no topbar com o estado real do modelo/corpus. */
export async function refreshStatusPill() {
  const pill = document.querySelector('[data-status-pill]');
  if (!pill) return;
  try {
    const status = await apiGet('/model/status');
    if (status.loaded) {
      pill.dataset.state = 'ok';
      const params = status.num_parameters ? `${(status.num_parameters / 1e6).toFixed(2)}M params` : '';
      pill.querySelector('[data-status-text]').textContent = `modelo carregado · ${params}`;
    } else {
      pill.dataset.state = 'off';
      pill.querySelector('[data-status-text]').textContent = 'nenhum modelo carregado';
    }
  } catch {
    pill.dataset.state = 'error';
    pill.querySelector('[data-status-text]').textContent = 'api indisponível';
  }
}

export function formatTime() {
  const d = new Date();
  return d.toTimeString().slice(0, 8);
}

export function showError(container, message) {
  container.textContent = '';
  const div = document.createElement('div');
  div.className = 'error-banner';
  div.textContent = message;
  container.appendChild(div);
}
