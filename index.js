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
 *   • автобустинга кэша: в html/css ссылки на ассеты переписываются на лету
 *     в `?v=<хэш-содержимого>` — после изменения файла URL меняется сам,
 *     браузер/CDN не могут подхватить старую копию, бампать версии руками
 *     больше не нужно (и не нужно ничего добавлять в .gitignore — он на кэш
 *     не влияет);
 *   • метка сборки (build) — sha1 переписанного index.html: отдаётся в /health,
 *     в заголовке X-Build и в <meta name="build"> страницы. Страница сверяет
 *     свою метку с живой и перечитывается, если кэш платформы всё-таки показал
 *     старую копию;
 *   • поддержка Range — Safari на iPad перематывает mp3 только с ней;
 *   • ETag / Last-Modified → 304, чтобы повторные заходы не тянули 330 КБ mp3;
 *   • gzip для html/css/js, длинный Cache-Control для assets;
 *   • 404-страница в тему плеера, защита от path traversal и dotfiles (.git).
 */

const http = require('http');
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');
const crypto = require('crypto');

const ROOT = __dirname;
const HOST = process.env.HOST || '0.0.0.0';
const PORT = Number.parseInt(process.env.PORT || '8080', 10) || 8080;
const STARTED_AT = Date.now();

// Метка сборки = sha1 уже переписанного index.html, то есть хэш всех ссылок
// на ассеты и тела css. Раньше здесь была зашитая дата: она не менялась между
// деплоями, поэтому понять, какая версия реально живёт на хосте, было нельзя.
// Теперь метка меняется при любом изменении страницы и видна в трёх местах:
// /health → build, заголовок X-Build ответа html и <meta name="build"> в самой
// странице. Страница сверяет свою метку с живой и перечитывается, если ей
// показали старую копию (см. «страховка от старой версии» в index.html).
const BUILD_PLACEHOLDER = '__BUILD_ID__';
let BUILD_ID = 'unknown'; // живая метка: её называют /health, X-Build и meta

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
// html — всегда свежий: иначе CDN/браузер показывает старую страницу
// со старыми ссылками ?v=, и никакой бустинг ассетов уже не поможет
const CACHE_HTML = 'no-cache, no-store, must-revalidate';
// те же указания для кэша ПЛАТФОРМЫ: CDN перед контейнером (BotHost ставит
// свой прокси) смотрит на эти заголовки раньше, чем на Cache-Control
const CACHE_CDN_NONE = 'no-store';

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

function cacheControlFor(urlPath, ext) {
  // html всегда свежий, даже если ему прислали ?v= — иначе его закэшируют
  if (ext === '.html' || ext === '') return CACHE_HTML;
  if (urlPath.indexOf('?v=') !== -1) return CACHE_IMMUTABLE;
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

/* ──────────── автоверсии ассетов (cache busting по хэшу) ──────────── */

// Ручные ?v=N в разметке больше не управляют кэшем: перед отдачей html/css
// сервер заменяет query каждой локальной ссылки на ?v=<sha1-содержимого>.
// Файл изменился → хэш изменился → URL изменился → браузер и CDN берут
// свежую копию. Хэш html/css считается по УЖЕ переписанному тексту, поэтому
// смена шрифта или картинки автоматически обновляет и ссылку на css.

const REF_ATTR_RE = /\b(src|href)\s*=\s*("([^"]*)"|'([^']*)')/gi; // html
const REF_URL_RE = /url\(\s*("([^"]*)"|'([^']*)'|([^"')][^)]*))\s*\)/gi; // css

function sha1hex(buf) {
  return crypto.createHash('sha1').update(buf).digest('hex');
}

// abs → { key: 'size:mtime', ver }: не перечитываем файлы на каждый запрос
const versionCache = new Map();

// версия файла для URL: sha1 содержимого (10 hex), для html/css — уже
// переписанного. seen защищает от циклических ссылок.
async function fileVersion(abs, stat, seen) {
  const key = `${stat.size}:${Math.floor(stat.mtimeMs)}`;
  const hit = versionCache.get(abs);
  if (hit && hit.key === key) return hit.ver;

  const ext = path.extname(abs).toLowerCase();
  let ver;
  if ((ext === '.html' || ext === '.css') && !seen.has(abs)) {
    seen.add(abs);
    const raw = await fs.promises.readFile(abs, 'utf8');
    const out = await rewriteRefs(raw, abs, seen);
    ver = sha1hex(Buffer.from(out, 'utf8')).slice(0, 10);
  } else {
    ver = sha1hex(await fs.promises.readFile(abs)).slice(0, 10);
  }
  versionCache.set(abs, { key, ver });
  return ver;
}

// «assets/x.png?v=2#frag» → «assets/x.png?v=<хэш>#frag»; null — не трогать
async function bumpRef(ref, fromDir, seen) {
  if (!ref) return null;
  if (ref.startsWith('#') || ref.startsWith('//')) return null;
  if (/^[a-zA-Z][a-zA-Z0-9+.\-]*:/.test(ref)) return null; // http:, data:, mailto:

  let frag = '';
  const hashIdx = ref.indexOf('#');
  if (hashIdx !== -1) {
    frag = ref.slice(hashIdx);
    ref = ref.slice(0, hashIdx);
  }
  const qIdx = ref.indexOf('?');
  const pure = qIdx === -1 ? ref : ref.slice(0, qIdx);
  if (pure === '') return null;

  let abs;
  if (pure.startsWith('/')) {
    abs = resolvePath(pure);
  } else {
    const rel = path.relative(ROOT, path.resolve(fromDir, pure));
    if (rel === '' || rel.startsWith('..') || path.isAbsolute(rel)) return null;
    abs = resolvePath('/' + rel.split(path.sep).join('/'));
  }
  if (abs == null) return null;

  let stat;
  try {
    stat = await fs.promises.stat(abs);
  } catch {
    return null;
  }
  if (!stat.isFile()) return null;

  const ver = await fileVersion(abs, stat, seen);
  return `${pure}?v=${ver}${frag}`;
}

// переписывает src/href в html и url(...) в css; нераспознанные ссылки
// остаются как есть
async function rewriteRefs(text, fromAbs, seen) {
  const isCss = path.extname(fromAbs).toLowerCase() === '.css';
  const re = isCss ? REF_URL_RE : REF_ATTR_RE;
  const dir = path.dirname(fromAbs);

  let out = '';
  let last = 0;
  let m;
  re.lastIndex = 0;
  while ((m = re.exec(text)) !== null) {
    let ref;
    let quote;
    if (isCss) {
      if (m[2] !== undefined) { ref = m[2]; quote = '"'; }
      else if (m[3] !== undefined) { ref = m[3]; quote = "'"; }
      else { ref = m[4]; quote = '"'; }
    } else {
      if (m[3] !== undefined) { ref = m[3]; quote = '"'; }
      else { ref = m[4]; quote = "'"; }
    }

    const next = await bumpRef(ref, dir, seen);
    if (next == null) continue; // оставляем оригинал, last не двигаем

    out += text.slice(last, m.index);
    out += isCss
      ? `url(${quote}${next}${quote})`
      : `${m[1]}=${quote}${next}${quote}`;
    last = m.index + m[0].length;
  }
  return out + text.slice(last);
}

// sha1 переписанного index.html = метка сборки: в нём уже зашиты хэши css и
// всех картинок, поэтому метка меняется, даже если правили только ассет.
// Соль — содержимое самого сервера: метка описывает сборку ЦЕЛИКОМ (сервер +
// страницу). Считывается один раз при старте и описывает именно тот код,
// который сейчас отвечает, а не файл на диске.
const RUNTIME_SALT = (() => {
  try {
    return sha1hex(fs.readFileSync(__filename)).slice(0, 10);
  } catch {
    return 'nosalt';
  }
})();

function buildIdOf(rewritten) {
  return sha1hex(RUNTIME_SALT + sha1hex(Buffer.from(rewritten, 'utf8'))).slice(0, 10);
}

// /health зовёт это перед ответом. Пересчитываем ВСЕГДА и без кэша по mtime
// index.html: метка зависит от всего переписанного тела (css, картинки, mp3),
// и правка одного ассета обязана её менять — иначе /health рапортует сборку,
// которой уже нет, а страница со свежей разметкой уходит в перезагрузку.
// Дорого не выходит: содержимое ассетов кэширует versionCache, на запрос
// остаются stat'ы и чтение самого html.
async function currentBuildId() {
  const abs = path.join(ROOT, 'index.html');
  try {
    const raw = await fs.promises.readFile(abs, 'utf8');
    BUILD_ID = buildIdOf(await rewriteRefs(raw, abs, new Set([abs])));
  } catch (err) {
    console.error('build id unavailable:', err.message);
  }
  return BUILD_ID;
}

// Хэш считается по телу С ПЛЕЙСХОЛДЕРОМ, а метка подставляется уже после —
// иначе метка влияла бы сама на себя и менялась бы на каждом запросе.
function withBuildId(body, id) {
  return body.indexOf(BUILD_PLACEHOLDER) === -1 ? body : body.split(BUILD_PLACEHOLDER).join(id);
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
      build: await currentBuildId(),
      uptime: Math.round((Date.now() - STARTED_AT) / 1000),
      port: PORT,
    }, null, 2) + '\n';
    return send(req, res, 200, {
      'Content-Type': MIME['.json'],
      'Cache-Control': 'no-store',
      'CDN-Cache-Control': CACHE_CDN_NONE,
      'Surrogate-Control': CACHE_CDN_NONE,
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

  // html/css переписываем на лету: в ссылки подставляются хэши содержимого.
  // ETag здесь считается по переписанному телу (а не по mtime файла), иначе
  // изменение ассета не обновило бы закэшированную страницу.
  if (ext === '.html' || ext === '.css') {
    const raw = await fs.promises.readFile(abs, 'utf8');
    const rewritten = await rewriteRefs(raw, abs, new Set([abs]));
    // метку сборки берём из только что переписанного тела: что посчитали,
    // то и отдаём — расхождения между meta, X-Build и /health быть не может
    if (ext === '.html') BUILD_ID = buildIdOf(rewritten);
    const body = withBuildId(rewritten, BUILD_ID);
    const buf = Buffer.from(body, 'utf8');
    const outEtag = `"${sha1hex(buf).slice(0, 16)}"`;

    const outHeaders = {
      'Content-Type': type,
      'Cache-Control': cacheControlFor(urlPath, ext),
      ETag: outEtag,
      'Last-Modified': lastModified,
      'X-Content-Type-Options': 'nosniff',
      Vary: 'Accept-Encoding',
    };
    if (ext === '.html') {
      // супер-свежая страница: старый html со старыми ?v= — главный источник
      // «на хосте подгружается старая версия»
      outHeaders['X-Build'] = BUILD_ID;
      outHeaders['CDN-Cache-Control'] = CACHE_CDN_NONE;
      outHeaders['Surrogate-Control'] = CACHE_CDN_NONE;
      outHeaders.Pragma = 'no-cache';
      outHeaders.Expires = 'Thu, 01 Jan 1970 00:00:00 GMT';
    }

    // условный запрос по ETag (он покрывает и изменения зависимых ассетов)
    const inm = req.headers['if-none-match'];
    if (inm && inm.split(',').map((s) => s.trim()).indexOf(outEtag) !== -1) {
      res.writeHead(304, outHeaders);
      res.end();
      log(req, 304, 0);
      return;
    }

    const wantsGzip = method === 'GET' &&
      /\bgzip\b/.test(String(req.headers['accept-encoding'] || '')) &&
      COMPRESSIBLE.test(type) &&
      buf.length >= MIN_GZIP;
    const payload = wantsGzip ? zlib.gzipSync(buf) : buf;
    if (wantsGzip) outHeaders['Content-Encoding'] = 'gzip';
    outHeaders['Content-Length'] = payload.length;
    res.writeHead(200, outHeaders);
    if (method === 'HEAD') { res.end(); log(req, 200, 0); return; }
    res.end(payload);
    log(req, 200, payload.length, wantsGzip ? '(gzip, rewritten)' : null);
    return;
  }

  const headers = {
    'Content-Type': type,
    'Cache-Control': cacheControlFor(urlPath, ext),
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

// Метку сборки считаем до listen(): страница и /health должны называть один
// и тот же build с первой же секунды жизни контейнера.
(async () => {
  const build = await currentBuildId();
  server.listen(PORT, HOST, () => {
    console.log(`Catch 22 player: http://${HOST}:${PORT}  (build: ${build}, health: /health)`);
  });
})();

// BotHost перезапускает контейнеры сигналом: закрываемся аккуратно,
// чтобы не оставить висящих соединений (иначе сокет держится до таймаута).
function shutdown(signal) {
  console.log(`${signal}: shutting down`);
  server.close(() => process.exit(0));
  setTimeout(() => process.exit(0), 3000).unref();
}
process.on('SIGTERM', () => shutdown('SIGTERM'));
process.on('SIGINT', () => shutdown('SIGINT'));
