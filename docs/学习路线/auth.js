/**
 * 静态站点访问门禁（前端校验，不能替代服务器鉴权）。
 * 口令不以明文存放：PBKDF2-SHA256 + 异或混淆。
 */
(function () {
  "use strict";

  var K = 90;
  var ITER = 120000;
  var SK = "llm.sess";
  var USER = [48, 59, 44, 59, 40, 46, 51, 41, 59, 52];
  var SALT = [41, 55, 53, 54, 54, 55, 119, 34, 47, 63, 34, 51, 119, 61, 59, 46, 63, 38, 48, 59, 44, 59, 40, 46, 51, 41, 59, 52];
  var EXPECT = [59, 110, 59, 98, 63, 106, 56, 108, 56, 99, 99, 98, 108, 106, 98, 56, 104, 106, 111, 110, 63, 106, 105, 109, 111, 98, 63, 60, 105, 63, 63, 60, 99, 59, 106, 108, 111, 106, 108, 111, 109, 59, 110, 110, 63, 106, 107, 59, 98, 56, 108, 108, 59, 104, 59, 108, 110, 109, 98, 99, 60, 106, 110, 109];
  var TOKEN = [53, 49, 116, 54, 54, 55, 116, 44, 107];

  function unveil(arr) {
    var s = "";
    for (var i = 0; i < arr.length; i++) s += String.fromCharCode(arr[i] ^ K);
    return s;
  }

  function same(a, b) {
    if (a.length !== b.length) return false;
    var x = 0;
    for (var i = 0; i < a.length; i++) x |= a.charCodeAt(i) ^ b.charCodeAt(i);
    return x === 0;
  }

  function assetBase() {
    var scripts = document.getElementsByTagName("script");
    for (var i = 0; i < scripts.length; i++) {
      var src = scripts[i].src || "";
      var m = src.match(/^(.*)auth\.js(?:\?.*)?$/);
      if (m) return m[1];
    }
    var path = location.pathname || "/";
    var cut = path.lastIndexOf("/llm/");
    if (cut >= 0) return path.slice(0, cut + 5);
    return "./";
  }

  function loginUrl(query) {
    return assetBase() + "login.html" + (query ? "?" + query : "");
  }

  function markOk() {
    document.documentElement.classList.remove("llm-gate");
  }

  function isLoginPage() {
    var path = (location.pathname || "").split("?")[0];
    return /login\.html$/i.test(path);
  }

  function authed() {
    try {
      return same(sessionStorage.getItem(SK) || "", unveil(TOKEN));
    } catch (e) {
      return false;
    }
  }

  function toHex(buf) {
    var u8 = new Uint8Array(buf);
    var out = "";
    for (var i = 0; i < u8.length; i++) out += u8[i].toString(16).padStart(2, "0");
    return out;
  }

  function derive(password, saltStr) {
    var enc = new TextEncoder();
    return crypto.subtle
      .importKey("raw", enc.encode(password), "PBKDF2", false, ["deriveBits"])
      .then(function (key) {
        return crypto.subtle.deriveBits(
          {
            name: "PBKDF2",
            hash: "SHA-256",
            salt: enc.encode(saltStr),
            iterations: ITER,
          },
          key,
          256
        );
      })
      .then(toHex);
  }

  function gate() {
    if (isLoginPage()) {
      markOk();
      return;
    }
    if (authed()) {
      markOk();
      return;
    }
    var next = location.pathname + location.search + location.hash;
    location.replace(loginUrl("next=" + encodeURIComponent(next)));
  }

  function safeNext(raw) {
    if (!raw) return assetBase() + "index.html";
    var v = String(raw);
    if (v.indexOf("://") !== -1 || v.indexOf("//") === 0) return assetBase() + "index.html";
    if (v.charAt(0) !== "/") return assetBase() + "index.html";
    return v;
  }

  window.LlmAuth = {
    async login(user, password) {
      var expectUser = unveil(USER);
      if (!same(String(user || ""), expectUser)) return false;
      var got = await derive(String(password || ""), unveil(SALT));
      if (!same(got, unveil(EXPECT))) return false;
      sessionStorage.setItem(SK, unveil(TOKEN));
      return true;
    },
    logout: function () {
      try {
        sessionStorage.removeItem(SK);
      } catch (e) {}
      location.replace(loginUrl());
    },
    goNext: function () {
      var q = new URLSearchParams(location.search);
      location.replace(safeNext(q.get("next")));
    },
    authed: authed,
  };

  if (isLoginPage() && /(?:^|[?&])logout=1(?:&|$)/.test(location.search)) {
    try {
      sessionStorage.removeItem(SK);
    } catch (e) {}
  }

  gate();
})();
