#!/usr/bin/env node
'use strict';

/**
 * «Catch 22» — приглашение-плеер: статический сервер для BotHost.
 *
 * Требования платформы (bothost.ru/docs/web-apps-domains):
 *   • слушать 0.0.0.0, а не 127.0.0.1 — иначе reverse proxy не достучится
 *     до контейнера и домен отдаст 502/504;
 *   • брать порт из process.env.PORT (в панели бота задаётся тот же порт,
 *     по умолчанию 8080), а не хардкодить его;
 *   • отвечать 200 на /health — по этому адресу проверяют, что сервис жив.
 *
 * Зависимостей нет: только стандартная библиотека Node.js (>= 18), поэтому
 * `npm install` на хостинге ничего не скачивает и не может упасть, а образ
 * собирается за секунды.
 *
 * Что умеет:
 *   • отдаёт index.html и assets/** как обычная статика;
 *   • поддержка Range — Safari на iPad перематывает mp3 только с ней;
 *   • ETag / Last-Modified → 304, чтобы повторные заходы не тянули 330 КБ mp3;
 *   • gzip для html/css/js, длинный Cache-Control для assets;
 *   • 404-страница в тему плеера, защита от path traversal и dotfiles (.git).
 */

const http = require('http');
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');

const ROOT = __dirname;
const HOST = process.env.HOST || '0.0.0.0';
const PORT = Number.parseInt(process.env.PORT || '8080', 10) || 8080;
const STARTED_AT = Date.now();

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.webmanifest': 'application/manifest+json; charset=utf-8',
  '.map': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.webp': 'image/webp',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif': 'image/gif',
  '.ico': 'image/x-icon',
  '.mp3': 'audio/mpeg',
  '.m4a': 'audio/mp4',
  '.wav': 'audio/wav',
  '.ogg': 'audio/ogg',
  '.woff2': 'font/woff2',
  '.woff': 'font/woff',
  '.ttf': 'font/ttf',
  '.otf': 'font/otf',
  '.txt': 'text/plain; charset=utf-8',
  '.md': 'text/markdown; charset=utf-8',
};

const DEFAULTS = { '.html': 'index.html', '.css': 'player.css' };

// что имеет смысл жать на лету (mp3/png/webp уже сжаты — их не трогаем)
const COMPRESSIBLE = /^(text\/|application\/(javascript|json|manifest\+json)|image\/svg\+xml)/;
const MIN_GZIP = 1024; // мелочь быстрее отдать как есть
const CACHE_IMMUTABLE = 'public, max-age=31536000, immutable'; // ассеты с ?v=
const CACHE_ASSET = 'public, max-age=86400';                  // прочие ассеты
const CACHE_HTML = 'no-cache';                                // страница всегда свежая

const NOT_FOUND_PAGE = `<!DOCTYPE html>
<html lang="ru"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="theme-color" content="#0b0b0a">
<title>404 — Catch 22</title>
<style>
  html,body{height:100%;margin:0}
  body{background:#0b0b0a;color:#e8e0cf;display:flex;flex-direction:column;
       align-items:center;justify-content:center;gap:14px;min-height:100dvh;
       font:16px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
       text-align:center;padding:24px;box-sizing:border-box}
  b{font-size:clamp(40px,12vw,96px);letter-spacing:.08em}
  a{color:#e8b26a}
  p{margin:0;opacity:.75;max-width:32ch}
</style></head>
<body>
  <b>404</b>
  <p>Такой дорожки на кассете нет. Плёнка заканчивается здесь.</p>
  <p><a href="/">← вернуться к плееру</a></p>
</body></html>`;

/* ─────────────────────────── утилиты ─────────────────────────── */

function log(req, status, bytes, note) {
  const t = new Date().toISOString().slice(11, 19);
  const size = bytes == null ? '' : ` ${bytes >= 1024 ? (bytes / 1024).toFixed(1) + 'k' : bytes + 'b'}`;
  console.log(`[${t}] ${req.method} ${req.url} ${status}${size}${note ? ' ' + note : ''}`);
}

function send(req, res, status, headers, body) {
  let payload = body;
  if (payload == null) payload = '';
  if (!Buffer.isBuffer(payload)) payload = Buffer.from(payload);
  const buf = req.method === 'HEAD' ? null : payload;
  const size = buf ? buf.length : 0;
  res.writeHead(status, Object.assign({ 'Content-Length': size }, headers));
  if (buf) res.end(buf); else res.end();
  log(req, status, buf ? size : 0);
}

function notFound(req, res) {
  send(req, res, 404, {
    'Content-Type': MIME['.html'],
    'Cache-Control': 'no-cache',
  }, NOT_FOUND_PAGE);
}

// путь внутри ROOT или null, если запрос пытается выйти наружу либо
// трогает служебные файлы (.git, .env, node_modules и прочие точки)
function resolvePath(urlPath) {
  let decoded;
  try {
    decoded = decodeURIComponent(urlPath);
  } catch {
    return null;
  }
  if (decoded.indexOf('\0') !== -1) return null;
  const rel = path.normalize(decoded).replace(/^([/\\])+/, '');
  if (rel.split(/[/\\]/).some((part) => part.startsWith('.') && part !== '' && part !== '..')) {
    return null; // dotfiles: .git, .env, .nvmrc — наружу не отдаём
  }
  if (rel.split(/[/\\]/).includes('node_modules')) return null;
  const abs = path.resolve(ROOT, rel);
  if (abs !== ROOT && !abs.startsWith(ROOT + path.sep)) return null;
  return abs;
}

function etagOf(stat) {
  return `"${stat.size.toString(16)}-${Math.floor(stat.mtimeMs).toString(16)}"`;
}

function cacheControlFor(urlPath) {
  if (urlPath.indexOf('?v=') !== -1) return CACHE_IMMUTABLE;
  const ext = path.extname(urlPath).toLowerCase();
  if (ext === '.html' || ext === '') return CACHE_HTML;
  return CACHE_ASSET;
}

// «bytes=0-1023» → {start, end} | 'invalid' | null (заголовка не было)
function parseRange(header, size) {
  if (!header) return null;
  const m = /^bytes=(\d*)-(\d*)$/.exec(String(header).trim());
  if (!m) return 'invalid';
  let start;
  let end;
  if (m[1] === '' && m[2] === '') return 'invalid';
  if (m[1] === '') {                       // суффикс: последние N байт
    const tail = Number.parseInt(m[2], 10);
    if (!Number.isFinite(tail) || tail === 0) return 'invalid';
    start = Math.max(size - tail, 0);
    end = size - 1;
  } else {
    start = Number.parseInt(m[1], 10);
    end = m[2] === '' ? size - 1 : Number.parseInt(m[2], 10);
    if (!Number.isFinite(start) || !Number.isFinite(end)) return 'invalid';
    if (start > end || start >= size) return 'invalid';
    end = Math.min(end, size - 1);
  }
  return { start, end };
}

/* ─────────────────────────── обработчик ─────────────────────────── */

async function handle(req, res) {
  const method = req.method || 'GET';
  if (method !== 'GET' && method !== 'HEAD') {
    return send(req, res, 405, { Allow: 'GET, HEAD', 'Content-Type': 'text/plain; charset=utf-8' }, 'method not allowed\n');
  }

  const rawUrl = req.url || '/';
  const qIndex = rawUrl.indexOf('?');
  const rawPath = qIndex === -1 ? rawUrl : rawUrl.slice(0, qIndex);
  const query = qIndex === -1 ? '' : rawUrl.slice(qIndex); // ?v=… → immutable-кэш

  if (rawPath === '/health' || rawPath === '/healthz') {
    const body = JSON.stringify({
      status: 'ok',
      service: 'catch22-player',
      uptime: Math.round((Date.now() - STARTED_AT) / 1000),
      port: PORT,
    }, null, 2) + '\n';
    return send(req, res, 200, {
      'Content-Type': MIME['.json'],
      'Cache-Control': 'no-store',
    }, body);
  }

  let abs = resolvePath(rawPath === '/' ? '/index.html' : rawPath);
  if (abs == null) return notFound(req, res);

  let stat = null;
  try {
    stat = await fs.promises.stat(abs);
  } catch {
    stat = null;
  }
  if (stat && stat.isDirectory()) {
    abs = path.join(abs, 'index.html');
    try {
      stat = await fs.promises.stat(abs);
    } catch {
      stat = null;
    }
  }
  if (!stat || !stat.isFile()) return notFound(req, res);

  const urlPath = rawPath + query;
  const ext = path.extname(abs).toLowerCase();
  const type = MIME[ext] || (DEFAULTS[ext] ? MIME[DEFAULTS[ext]] : 'application/octet-stream');
  const etag = etagOf(stat);
  const lastModified = new Date(stat.mtimeMs).toUTCString();

  const headers = {
    'Content-Type': type,
    'Cache-Control': cacheControlFor(urlPath),
    ETag: etag,
    'Last-Modified': lastModified,
    'Accept-Ranges': 'bytes',
    'X-Content-Type-Options': 'nosniff',
    Vary: 'Accept-Encoding',
  };

  // условный запрос: содержимое не менялось — тела не шлём
  const inm = req.headers['if-none-match'];
  const ims = req.headers['if-modified-since'];
  if ((inm && inm.split(',').map((s) => s.trim()).indexOf(etag) !== -1) ||
      (!inm && ims && new Date(ims).getTime() >= Math.floor(stat.mtimeMs / 1000) * 1000)) {
    res.writeHead(304, Object.assign({}, headers, { ETag: etag, 'Cache-Control': headers['Cache-Control'] }));
    res.end();
    log(req, 304, 0);
    return;
  }

  const range = parseRange(req.headers.range, stat.size);
  if (range === 'invalid') {
    res.writeHead(416, Object.assign({}, headers, { 'Content-Range': `bytes */${stat.size}` }));
    res.end();
    log(req, 416, 0);
    return;
  }

  // Range: mp3 перематывается в Safari/Chrome только с 206 и Content-Range
  if (range) {
    headers['Content-Range'] = `bytes ${range.start}-${range.end}/${stat.size}`;
    headers['Content-Length'] = range.end - range.start + 1;
    res.writeHead(206, headers);
    if (method === 'HEAD') { res.end(); log(req, 206, 0); return; }
    const stream = fs.createReadStream(abs, { start: range.start, end: range.end });
    stream.on('error', () => res.destroy());
    stream.pipe(res);
    log(req, 206, headers['Content-Length']);
    return;
  }

  const wantsGzip = method === 'GET' &&
    /\bgzip\b/.test(String(req.headers['accept-encoding'] || '')) &&
    COMPRESSIBLE.test(type) &&
    stat.size >= MIN_GZIP;

  if (wantsGzip) {
    headers['Content-Encoding'] = 'gzip';
  } else {
    headers['Content-Length'] = stat.size;
  }
  res.writeHead(200, headers);
  if (method === 'HEAD') { res.end(); log(req, 200, 0); return; }

  const stream = fs.createReadStream(abs);
  stream.on('error', () => res.destroy());
  if (wantsGzip) {
    const gzip = zlib.createGzip({ level: 6 });
    stream.pipe(gzip).pipe(res);
    log(req, 200, null, '(gzip)');
  } else {
    stream.pipe(res);
    log(req, 200, stat.size);
  }
}

/* ─────────────────────────── сервер ─────────────────────────── */

const server = http.createServer((req, res) => {
  res.on('error', () => {}); // оборванный клиент не должен ронять процесс
  handle(req, res).catch((err) => {
    console.error('request failed:', err);
    if (!res.headersSent) {
      send(req, res, 500, { 'Content-Type': 'text/plain; charset=utf-8' }, 'internal error\n');
    } else {
      res.destroy();
    }
  });
});

server.on('clientError', (err, socket) => {
  if (socket.writable) socket.end('HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n');
});

server.keepAliveTimeout = 65 * 1000; // чуть больше трафикового idle-таймаута
server.headersTimeout = 70 * 1000;

server.listen(PORT, HOST, () => {
  console.log(`Catch 22 player: http://${HOST}:${PORT}  (health: /health)`);
});

// BotHost перезапускает контейнеры сигналом: закрываемся аккуратно,
// чтобы не оставить висящих соединений (иначе сокет держится до таймаута).
function shutdown(signal) {
  console.log(`${signal}: shutting down`);
  server.close(() => process.exit(0));
  setTimeout(() => process.exit(0), 3000).unref();
}
process.on('SIGTERM', () => shutdown('SIGTERM'));
process.on('SIGINT', () => shutdown('SIGINT'));
