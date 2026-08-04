// Bridges jsdom's `localStorage` onto the globals test files see.
//
// Confirmed directly: vitest's built-in jsdom environment constructs a real
// JSDOM instance with a working `localStorage` (`globalThis.jsdom.window
// .localStorage` is present and functional), but does not copy it onto the
// `window`/`localStorage` globals it exposes to test files -- a gap in this
// vitest+jsdom version combination, not anything about this app. Node also
// defines its own experimental global `localStorage` accessor (the
// "--localstorage-file was not provided" warning), which would otherwise
// shadow jsdom's with one that throws on use.
//
// App.tsx uses bare `localStorage` for persistence (`voxtrans_server_url`,
// `voxtrans_input_device`), so anything that renders it needs a working one.
const dom = (globalThis as unknown as { jsdom?: { window: Window } }).jsdom;

if (dom) {
  Object.defineProperty(globalThis, "localStorage", {
    value: dom.window.localStorage,
    configurable: true,
  });
  Object.defineProperty(window, "localStorage", {
    value: dom.window.localStorage,
    configurable: true,
  });
}
