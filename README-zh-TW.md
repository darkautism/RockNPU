# RockNPU

[English](README.md) | **繁體中文**

**讓 RK3588 的 NPU 幫 llama.cpp / Ollama 跑大語言模型。**
開源、免 RKNN/RKLLM 閉源函式庫、不必改 llama.cpp 或 Ollama 原始碼，也不必換模型格式：直接用你手上的 GGUF 模型。

*English: RockNPU is an open-source RK3588 NPU backend for stock llama.cpp and Ollama. It runs your existing GGUF models, needs no vendor blobs and no frontend patches. Technical documentation: [架構.md](架構.md).*

---

## 能帶來什麼

以 TinyLlama‑1.1B Q4_K_M、Orange Pi 5（RK3588，LPDDR4X）實測：

| 項目 | 只用 CPU（llama.cpp） | **RockNPU** | 參考：閉源 RKLLM 官方數據* |
|---|---:|---:|---:|
| 讀提示詞（prefill，128 tokens） | 73 tok/s | **≈ 570 tok/s** | ≈ 525 tok/s（TTFT 244 ms） |
| 讀提示詞（prefill，512 tokens） | 69 tok/s | **≈ 420 tok/s** | — |
| 生成（decode） | 32–33 tok/s | **≈ 31–33 tok/s** | 24.4 tok/s |
| 輸出品質 | 基準 | 生成階段與 CPU 完全相同；讀提示詞為 W8A8 | W8A8 |

\* RKLLM 無法在測試環境執行，數字取自 [airockchip/rknn-llm 官方 benchmark](https://github.com/airockchip/rknn-llm/blob/main/benchmark.md)（W8A8，最高 CPU/NPU 頻率）。
RockNPU 數字為 NPU 700 MHz、llama-bench、4 個 A76 執行緒。詳細方法與原始數據見 [架構.md](架構.md#效能與量測方法)。

其他模型（tok/s；「只用 CPU」為原版 llama.cpp 預設設定）：

| 模型（Q4_K_M） | 讀提示詞 128：只用 CPU | **RockNPU** | RKLLM* | 生成：只用 CPU | **RockNPU** | RKLLM* |
|---|---:|---:|---:|---:|---:|---:|
| Llama‑3.2‑1B‑Instruct | 71 | **≈ 580** | — | 26 | 23 | — |
| Qwen2.5‑1.5B‑Instruct | 55 | **≈ 350** | ≈ 340 | 22.5 | 19–20 | 16.7 |
| Qwen2.5‑0.5B‑Instruct | 71 | 71（不加速，見常見問題） | — | — | — | 41.6（Qwen2 0.5B） |

白話：**貼長文件、長對話歷史給模型時，等待回應的時間縮短為 1/6–1/8**；生成仍由 CPU 負責（RK3588 的記憶體頻寬決定了生成速度的上限，CPU 的 4-bit 路徑在這一步最快），但比原版 llama.cpp 慢 5–15 %，原因見常見問題「生成比原版 llama.cpp 慢一點？」。

---

## 你需要

- 一塊 RK3588 / RK3588S 開發板（Orange Pi 5 系列、Rock 5 系列……）
- Linux 核心內建 `rocket` NPU 驅動（Linux 6.18 以上，例如 Armbian 的 *current* 核心）。
  確認方式：`ls /dev/accel/accel0` 有東西就對了。
- 約 2 GB 空閒硬碟空間（編譯用）。

## 安裝（3 步）

```sh
# 1. 取得 RockNPU
git clone https://github.com/darkautism/RockNPU.git
cd RockNPU

# 2. 一鍵安裝（自動安裝編譯工具、Rust，並編譯 RockNPU）
#    還沒有 llama.cpp 的話加上 --with-llama，會順便編譯 llama-server
./scripts/install.sh --with-llama

# 3. 載入設定（建議把這行加進 ~/.bashrc）
. ~/.local/share/rocknpu/rocknpu.env
```

第一次使用若提示沒有 `/dev/accel/accel0` 權限，執行 `sudo usermod -aG render $USER` 後重新登入即可。

（選用，強烈建議）讓 NPU 跑在 700 MHz，並套用系統調校：

```sh
sudo ./scripts/rocknpu-tune.sh dvfs      # 編譯並載入 NPU 調頻驅動模組（需要核心 headers）
sudo ./scripts/rocknpu-tune.sh install   # 開機自動套用；要還原：sudo ./scripts/rocknpu-tune.sh restore
```

主線核心的 NPU 驅動沒有調頻功能，NPU 會停在開機時的 200 MHz：讀提示詞約慢 40%。`dvfs` 會直接取得、編譯並載入社群維護的 [rk3588-npu-gpu](https://github.com/sky-rk3588/rk3588-npu-gpu) 調頻模組；RockNPU 不修改也不維護這個 kernel module。它不改電壓、不寫入開機核心，`restore` 後重開機即可回到發行版原本的驅動。RockNPU 的 LLM 路徑在約 700 MHz 已接近飽和，實測升到 1 GHz 沒有可重現的效能收益，因此最佳效能設定以 700 MHz 為準。

## 開始使用

### llama.cpp（網頁聊天介面 / OpenAI 相容 API）

```sh
llama-server -m 你的模型.gguf
```

然後用瀏覽器開啟 `http://開發板IP:8080`。
確認有用到 NPU：`llama-server --list-devices` 會列出 `ROCKNPU0: RockNPU RK3588`。

小技巧：`taskset -c 4-7 llama-server -m 你的模型.gguf -t 4` 把 llama.cpp 固定在 4 顆大核上，讀提示詞再快約 10%。
載入模型後的**第一個**請求會稍慢（NPU 正在把權重轉成 8-bit，TinyLlama 約 1 秒），之後就是全速。

### Ollama

```sh
curl -fsSL https://ollama.com/install.sh | sh     # 安裝 Ollama（已安裝可略過）
sudo systemctl stop ollama                         # 停掉系統服務，改用帶 RockNPU 設定的版本
. ~/.local/share/rocknpu/rocknpu.env
ollama serve
# 另開一個終端機：
ollama run tinyllama:1.1b-chat-v1-q4_K_M
```

模型請選 `Q4_K_M` 之類的 K-quant 版本（Ollama 的預設標籤常是 `Q4_0`，NPU 不加速）。

Ollama 0.34.x 使用與 RockNPU 相同的 llama.cpp 版本（`b10969`），不需要修改 Ollama（實測 0.34.4）。
其他版本請以 `LLAMA_REF=<該版本 llama.cpp 的 tag> ./scripts/install.sh` 重新編譯。

設定檔已替 Ollama 處理好兩件事：只用 4 顆大核（`LLAMA_ARG_THREADS=4`；Ollama 預設用 8 顆，小核會拖慢每一步）以及保持 flash attention 開啟（`OLLAMA_FLASH_ATTENTION=1`）。
Ollama（TinyLlama Q4_K_M，329 tokens 提示詞）實測：讀提示詞 **≈ 300 tok/s**、生成 **≈ 30 tok/s**；原版 Ollama（只用 CPU）為 ≈ 100 / ≈ 23 tok/s。
模型載入後的前一兩個請求會慢一些（NPU 正在準備 8-bit 權重）。

> 提醒：Ollama 本體約 2 GB。系統裝在 eMMC/SD 卡的板子，建議把 Ollama 與模型放在 NVMe/SSD 上。

## 常見問題

**Q：`--list-devices` 沒有 `ROCKNPU0`？**
確認已執行 `. ~/.local/share/rocknpu/rocknpu.env`，且 `/dev/accel/accel0` 存在並有讀寫權限。

**Q：有列出 NPU，但速度跟 CPU 一樣？**
請確認環境變數 `LLAMA_ARG_REPACK=false` 有生效（安裝腳本產生的設定檔已包含）。llama.cpp 預設會把權重「重新排列」成只有 CPU 看得懂的格式，NPU 就拿不到工作。
自行下指令時也可以加上 `--no-repack`。

**Q：讀提示詞的結果跟純 CPU 完全一樣嗎？**
不完全一樣：NPU 以 8-bit 整數（W8A8，與閉源 RKLLM 相同的做法）計算讀提示詞的部分，下一個字與 CPU 相同的比例約 91–94 %；生成階段預設由 CPU 計算，與原版 llama.cpp 相同。
重視精度可設 `export ROCKNPU_PREFILL_HILO=down`（誤差約減半，讀提示詞慢約 23 %）或 `=1`（誤差約為 1/5，慢約一半，仍比 CPU 快 3–6 倍）。

**Q：生成比原版 llama.cpp 慢一點？**
是的，約 5–15 %（模型越大差越多）。原版 llama.cpp 會把權重「重新排列」成 CPU 專用格式（repack），生成時快一些；但 NPU 讀不了重新排列後的權重，所以 RockNPU 需要 `LLAMA_ARG_REPACK=false`。如果你的用法幾乎都是很短的提示詞、很長的輸出，可以不載入 RockNPU 設定（或 `export LLAMA_ARG_REPACK=true`），改回純 CPU。

**Q：想讓 NPU 也負責生成（把 CPU 讓給別的程式）？**
`export ROCKNPU_DECODE=npu`（約 20 tok/s，CPU 幾乎閒置）或 `ROCKNPU_DECODE=hybrid`（CPU 與 NPU 一起算，約 26 tok/s）。預設 `cpu` 最快。

**Q：支援哪些模型？**
GGUF 的 Q4_K / Q6_K 權重（例如常見的 `Q4_K_M`），隱藏層寬度是 256 的倍數的模型：Llama 3.x、TinyLlama、Qwen2.5 1.5B 以上等。NPU 以 W8A8 執行投影層；其餘運算與不支援的格式自動交給 CPU，所以任何 llama.cpp 能跑的模型都能跑，只是加速程度不同。
Qwen2.5‑0.5B（寬度 896）的 GGUF 權重不是 K-quant 格式，會完整由 CPU 執行（結果正確，只是沒有加速）。
記憶體：NPU 需要另外保存一份 8-bit 權重，約為模型參數量（1B 模型約 1 GB）。

**Q：NPU 頻率重要嗎？要超頻到 1 GHz 嗎？**
200 MHz（主線預設）→ 700 MHz 很重要：讀提示詞 326 → 556 tok/s。700 MHz → 1 GHz（需加壓到 850 mV）實測幾乎沒有差異（瓶頸在記憶體與主機端），不需要冒險。細節見 [架構.md](架構.md#npu-頻率)。

## 更多

- 技術架構、所有設定參數、效能量測方法：[架構.md](架構.md)
- 研究紀錄與已驗證/已否決的方向：[docs/research-status.md](docs/research-status.md)、[trialanderror.md](trialanderror.md)

## 致謝與授權

感謝 [oRKLLM/ork-driver](https://github.com/oRKLLM/ork-driver) 對 RK35xx NPU 開創性的逆向工程，RockNPU 以其作為硬體與 regcmd 研究參考。

RockNPU 原創程式碼以 MIT 授權釋出。直接衍生自 ork-driver 的 regcmd 基準保留原 ISC 授權，位於 `crates/rocknpu-regcmd/src/int8/ork_isc.rs`，授權聲明見 `docs/licenses/ork-driver-ISC.txt`。
