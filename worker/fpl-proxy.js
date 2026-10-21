// Tiny read-only CORS proxy for the FPL API, used by docs/team.html (the My team page).
//
// The FPL API doesn't send CORS headers, so a static page on GitHub Pages can't call it directly.
// This Worker forwards only the public, no-login team endpoints and adds the CORS header.
//
// Deploy (free Cloudflare account, about two minutes):
//   1. dash.cloudflare.com → Workers & Pages → Create → Worker → paste this file → Deploy
//   2. Copy the URL it gives you (https://fpl-proxy.<you>.workers.dev) into FPL_PROXY in docs/team.html
// Or with the CLI:  npx wrangler deploy worker/fpl-proxy.js --name fpl-proxy --compatibility-date 2026-10-01
//
// Request shape:  GET https://fpl-proxy.<you>.workers.dev/api/entry/123456/event/7/picks/

const ALLOWED = [
  /^\/api\/entry\/\d{1,9}\/$/,
  /^\/api\/entry\/\d{1,9}\/history\/$/,
  /^\/api\/entry\/\d{1,9}\/transfers\/$/,
  /^\/api\/entry\/\d{1,9}\/event\/\d{1,2}\/picks\/$/,
];
const ORIGINS = ['https://fplvalue.co', 'https://www.fplvalue.co', 'https://arlind55.github.io'];

export default {
  async fetch(request, env, ctx) {
    const origin = request.headers.get('Origin') || '';
    const allowOrigin = ORIGINS.includes(origin) || /^http:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/.test(origin) ? origin : ORIGINS[0];
    const cors = { 'Access-Control-Allow-Origin': allowOrigin, 'Access-Control-Allow-Methods': 'GET, OPTIONS', 'Vary': 'Origin' };
    if (request.method === 'OPTIONS') return new Response(null, { status: 204, headers: cors });
    if (request.method !== 'GET') return new Response('Method not allowed', { status: 405, headers: cors });

    const url = new URL(request.url);
    if (!ALLOWED.some(re => re.test(url.pathname))) return new Response('Not found', { status: 404, headers: cors });

    // Short edge cache: picks for a finished gameweek don't change, and this keeps FPL traffic low.
    const cache = caches.default;
    const key = new Request(url.toString(), { method: 'GET' });
    let res = await cache.match(key);
    if (!res) {
      const up = await fetch('https://fantasy.premierleague.com' + url.pathname, {
        headers: { 'User-Agent': 'Mozilla/5.0 (fplvalue.co team page)', 'Accept': 'application/json' },
      });
      res = new Response(up.body, { status: up.status, headers: { 'Content-Type': 'application/json', 'Cache-Control': 'public, max-age=300' } });
      if (up.ok) ctx.waitUntil(cache.put(key, res.clone()));
    }
    const out = new Response(res.body, res);
    for (const [k, v] of Object.entries(cors)) out.headers.set(k, v);
    return out;
  },
};
