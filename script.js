// ====== CONFIG ======
const MODEL_PATH = "https://raw.githubusercontent.com/saifali28922-hash/upscaletool/main/assets/models/realesrganX4plus_v1.onnx
";
const UPSCALE = 4; // RealESRGAN x4

// ✅ Force CPU (WASM only, no GPU)
async function createSession() {
    if (ort.env && ort.env.wasm) {
        ort.env.wasm.numThreads = 1; // safer for browsers without SAB
        ort.env.wasm.proxy = false;
    }

    const session = await ort.InferenceSession.create(MODEL_PATH, {
        executionProviders: ["wasm"], // ✅ CPU only
        graphOptimizationLevel: "all"
    });

    console.log("✅ Using WASM (CPU only)");
    return session;
}

// Draw uploaded image to canvas
async function loadImageToCanvas(file, canvasId) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => {
            const canvas = document.getElementById(canvasId);
            const ctx = canvas.getContext("2d");
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);
            resolve({ canvas, width: img.width, height: img.height });
        };
        img.onerror = reject;
        img.src = URL.createObjectURL(file);
    });
}

// Convert a canvas sub-rect (tile) to NCHW Float32 tensor [1,3,H,W]
function canvasTileToTensor(srcCtx, sx, sy, sw, sh) {
    const imgData = srcCtx.getImageData(sx, sy, sw, sh);
    const { data } = imgData; // RGBA
    const wh = sw * sh;
    const out = new Float32Array(3 * wh);
    for (let i = 0; i < wh; i++) {
        out[i] = data[i * 4] / 255; // R
        out[i + wh] = data[i * 4 + 1] / 255; // G
        out[i + 2 * wh] = data[i * 4 + 2] / 255; // B
    }
    return new ort.Tensor("float32", out, [1, 3, sh, sw]);
}

// Write tensor back to canvas
function writeTensorToCanvas(tensor, dstCtx, dx, dy) {
    const dims = tensor.dims;
    let w, h, layout;

    if (dims.length === 4) {
        if (dims[1] === 3) {
            layout = "NCHW";
            h = dims[2];
            w = dims[3];
        } else if (dims[3] === 3) {
            layout = "NHWC";
            h = dims[1];
            w = dims[2];
        } else {
            throw new Error("Unknown tensor layout: " + dims);
        }
    } else {
        throw new Error("Unexpected tensor dims: " + dims);
    }

    const img = dstCtx.createImageData(w, h);
    const wh = w * h;
    const d = img.data;

    if (layout === "NCHW") {
        const data = tensor.data;
        const plane = wh;
        for (let i = 0; i < wh; i++) {
            d[i * 4] = Math.max(0, Math.min(255, data[i] * 255));
            d[i * 4 + 1] = Math.max(0, Math.min(255, data[i + plane] * 255));
            d[i * 4 + 2] = Math.max(0, Math.min(255, data[i + 2 * plane] * 255));
            d[i * 4 + 3] = 255;
        }
    } else {
        const data = tensor.data;
        for (let i = 0; i < wh; i++) {
            d[i * 4] = Math.max(0, Math.min(255, data[i * 3] * 255));
            d[i * 4 + 1] = Math.max(0, Math.min(255, data[i * 3 + 1] * 255));
            d[i * 4 + 2] = Math.max(0, Math.min(255, data[i * 3 + 2] * 255));
            d[i * 4 + 3] = 255;
        }
    }
    dstCtx.putImageData(img, dx, dy);
}

// Tiled upscaling with async yielding
async function runTiled(session, srcCanvas, tileSize, overlap) {
    const srcCtx = srcCanvas.getContext("2d");
    const sw = srcCanvas.width,
        sh = srcCanvas.height;

    const outW = sw * UPSCALE,
        outH = sh * UPSCALE;
    const outCanvas = document.getElementById("upscaledCanvas");
    outCanvas.width = outW;
    outCanvas.height = outH;
    const outCtx = outCanvas.getContext("2d");

    const inName = session.inputNames[0];
    const outName = session.outputNames[0];

    const cols = Math.ceil(sw / tileSize);
    const rows = Math.ceil(sh / tileSize);
    const total = cols * rows;

    const status = document.getElementById("status");

    let count = 0;
    for (let ty = 0; ty < sh; ty += tileSize) {
        for (let tx = 0; tx < sw; tx += tileSize) {
            const tw = Math.min(tileSize, sw - tx);
            const th = Math.min(tileSize, sh - ty);

            const inputTensor = canvasTileToTensor(srcCtx, tx, ty, tw, th);
            const feeds = {
                [inName]: inputTensor };

            await new Promise(r => setTimeout(r, 0)); // UI responsive

            let results;
            try {
                results = await session.run(feeds);
            } catch (err) {
                console.error("❌ Inference failed on tile:", tx, ty, err);
                throw err;
            }

            const outTensor = results[outName];
            const dims = outTensor.dims;
            const ow = (dims[1] === 3) ? dims[3] : dims[2];
            const oh = (dims[1] === 3) ? dims[2] : dims[1];

            const tmpCanvas = document.createElement("canvas");
            tmpCanvas.width = ow;
            tmpCanvas.height = oh;
            const tmpCtx = tmpCanvas.getContext("2d");
            writeTensorToCanvas(outTensor, tmpCtx, 0, 0);

            const imgData = tmpCtx.getImageData(0, 0, ow, oh);
            outCtx.putImageData(imgData, tx * UPSCALE, ty * UPSCALE);

            count++;
            if (status) status.textContent = `Processing tile ${count}/${total}`;

            if (count % 2 === 0) await new Promise(r => setTimeout(r, 0));
        }
    }
    if (status) status.textContent = `✅ Done (${total} tiles)`;
}

// ===== UI Handlers =====
const uploadEl = document.getElementById("upload");
const runBtn = document.getElementById("runBtn");
const statusEl = document.getElementById("status");
let sessionPromise = null;
let lastOriginal = null;

uploadEl.addEventListener("change", async(e) => {
    const file = e.target.files[0];
    if (!file) return;
    lastOriginal = await loadImageToCanvas(file, "originalCanvas");
    statusEl.textContent = "Image loaded. Ready to upscale.";
});

runBtn.addEventListener("click", async() => {
    if (!lastOriginal) {
        alert("Please choose an image first.");
        return;
    }

    const tileSize = Math.max(64, Math.min(512, parseInt(document.getElementById("tileSize").value || "128", 10)));
    const overlap = Math.max(0, Math.min(32, parseInt(document.getElementById("overlap").value || "8", 10)));

    if (!sessionPromise) {
        statusEl.textContent = "Loading model…";
        sessionPromise = createSession().catch(err => {
            console.error("❌ Failed to create session:", err);
            statusEl.textContent = "Model load failed.";
            throw err;
        });
    }

    const session = await sessionPromise;
    statusEl.textContent = `Upscaling… (tile ${tileSize}, overlap ${overlap})`;

    try {
        await runTiled(session, lastOriginal.canvas, tileSize, overlap);
    } catch (err) {
        console.error("❌ Upscaling failed:", err);
        statusEl.textContent = "Upscaling failed. See console.";
    }

});



