self.onmessage = async (event) => {
  try {
    const { instance } = await WebAssembly.instantiate(new Uint8Array(event.data));
    self.postMessage({ ok: instance.exports.add(2, 3) === 5 });
  } catch (error) {
    self.postMessage({ ok: false, error: error.name });
  }
};
