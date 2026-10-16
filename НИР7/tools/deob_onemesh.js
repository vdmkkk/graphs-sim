// Decode javascript-obfuscator string-table lookups: node deob.js <file.js> 0x35e 0x17c ...
const fs = require('fs'), vm = require('vm');
const src = fs.readFileSync(process.argv[2], 'utf8');

function slice(startNeedle, endNeedle) {
  const s = src.indexOf(startNeedle);
  const e = src.indexOf(endNeedle, s) + endNeedle.length;
  return src.slice(s, e);
}

const arrFn = slice('function _0x149f()', 'return _0x149f();}');
const decFn = slice('function _0x5e64(', 'return _0x5e647a;}');
const i0 = src.indexOf('(function(_0x35845d');
const re = /\}\(_0x149f,0x[0-9a-f]+\)\);/g;
re.lastIndex = i0;
const m = re.exec(src);
const iife = src.slice(i0, m.index + m[0].length);

const ctx = { parseInt };
vm.createContext(ctx);
vm.runInContext(arrFn + '\n' + decFn + '\n' + iife + '\nthis.d=_0x5e64;', ctx, { timeout: 5000 });
for (const h of process.argv.slice(3)) console.log(h, JSON.stringify(ctx.d(parseInt(h, 16))));
