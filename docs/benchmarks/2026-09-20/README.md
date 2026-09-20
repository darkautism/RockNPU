# RK3588 LLM：原生 CPU 與常駐 prefill

本輪使用兩台 RK3588：o16 探索程式修改及硬體正確性，o8 驗證原生 ARM CPU 與完整請求。LLM 與 NPU 測試都在 ARM 板子執行；Windows x86 僅編輯程式、整理證據。

## 結果：完整請求穩定快於原生 CPU

正式測量兩輪 `CPU,NPU,NPU,CPU`，每個 process 三次請求；共八個 process、24 次請求，每端 12 個樣本，全部 exit 0。

| 512 prompt + 128 generation | 平均請求時間 |
| --- | ---: |
| 原生 ARM CPU，四個 A76 | 12.854186 秒 |
| NPU prefill + CPU generation | 11.460154 秒 |

以平均時間計算，吞吐提升 **12.164%（1.121642×）**、請求時間減少 **10.845%**。兩輪各提升 **10.900%**、**13.397%**，方向一致。這滿足本輪「穩定超過 CPU 一些即可收尾」的門檻，範圍是上述完整請求與測試條件。

完整證據：[summary](abba/summary.json)、[所有樣本](abba/results.json)、[產物 SHA-256 與環境](abba/metadata.json)、每個 process 的 `.stdout`／`.stderr`／`.meta.json`。遠端原件保留於 o8 `/build/hybrid-abba-0920`；execution task `task_a2f60e181d50f3c4`，379.456 秒內完成。各 process 前後的頻率／governor／溫度都保存在 metadata 中。

## 比較條件

- 同一未修改的 llama.cpp source：`391fac16460f15233a7740550d858ac96df3419d`。o8 正式比較共用 `build-native-0920/bin/llama-bench`，Release、`GGML_NATIVE=ON`、`GGML_BACKEND_DL=ON`。
- 模型：TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf，667,814,880 bytes；SHA-256 `5c66751b61537f9e55177b1b67e06af88e0e2df88f86de4909f5bf87fb1ae583`。
- CPU 使用四個 A76（`taskset -c 4-7`、`-t 4`）、performance governor、flash attention on。額外測過全八核心，速度較慢，因此比較採用較快的四核心設定。
- NPU 固定 700 MHz／既有 800 mV。o8 使用既有 devfreq Rocket 研究模組；o16 使用 `3345d5f`（額外 IOMMU cache）及 A55 IRQ 設定。這不是封裝版 200 MHz stock Rocket 的性能。
- CPU process 清除 `GGML_BACKEND_PATH` 與所有 `ROCKNPU_*`，指定 `-dev none -nopo 1` 並核對 CPU-only loader。載入 ACCEL plugin 時單獨 `-dev none` 不足以隔離 CPU。
- 混合模式為 `ROCKNPU_PREFILL_CACHE=1 ROCKNPU_DECODE=0`：NPU 執行對齊的 prompt matmul，CPU 逐字生成。無 W8 sidecar、無額外 W8A8 量化。權重與啟動值在 prefill 轉為 FP16，NPU partial 為 FP32，K 累加使用 host f64。

`-pg512,128` 的 tok/s 是 **640 個總 token ÷ 整段請求秒數**，包含 prompt 處理及生成；不是 decode tok/s。模型載入及 llama-bench warmup 不在計時內；實際 workload 中發生的 cache 準備仍算入計時。

## 方向判斷與原始資料

| 同機測試 | 原生 CPU t4 | 舊 W8 NPU | 新混合模式初測 |
| --- | ---: | ---: | ---: |
| decode 128 token | 33.371925 | 15.797234 | 未以此宣稱收益 |
| prompt 512 + generation 128 | 50.517048 | 20.836110 | 57.741400 |

以上是三次樣本的初測，不取代交錯驗證。原始命令、loader stderr、JSON 及 execution task id 分別保存在本目錄的 `cpu-native-*.json`、`npu-native-*.json`、`hybrid-native-triage.json`。較弱的 generic CPU 約 20 tok/s，已從達標基準排除。

全八核心 CPU 的 prompt-only 為 62.640916 tok/s，完整請求 42.590829 tok/s；四核心分別為 69.462267 及 50.517048。探索機相同 generic binary 的 prompt-only 路徑依序為：舊 bridge 25.399275、常駐 cache 68.711780、加原生排列 gather 91.367919 tok/s。這個分解只用來確認改動有效，正式結論以 o8 原生 CPU 對照為準。

## 正確性與範圍

- [Release library tests](library-tests.json)：28 passed。
- [FP32 常駐硬體測試](prefill-smoke-v2.json)：12 個正／負／零輸入設定通過；小型與 deep-K dyadic 輸入對照獨立 CPU oracle，常駐結果與原 FP32 pool bit-for-bit 相同；包含錯誤形狀、釋放後拒絕，以及錯誤之後的合法新請求。
- direct-scratch W8 的 13 組 int32 oracle 測試亦通過，該路徑保留但不是本輪完整請求加速的來源。
- [確定性生成結果](quality/results.json)：短、長 prompt，各 CPU／舊 NPU bridge／cached，共六次 process 均 exit 0。每次最多 64 token，temperature 0；較弱 generic CPU binary 用於此路徑回歸，不能用其時間作性能基準。
- [短 prompt](quality/comparison-0.json)：三條路徑輸出完全相同。[長 prompt](quality/comparison-1.json)：舊 bridge 與新 cached 完全相同；CPU 的後續生成文字不同。三者第一句均為「She works at a library in a small town.」，後續一個生成 `text material`，另一個生成 `passage` 等差異。這是**模型生成文字**的差異，不是工具加上的前綴；不宣稱 logits／token 或整體模型品質等價。
- 長 prompt trace 證實每條 NPU 路徑有 151 次 prefill matmul、零次 NPU decode。短 prompt 不符合對齊條件時由 CPU fallback。
- [正式原生環境的額外 dispatch trace](native-route-trace.json)：單次 `pp512+tg128` 與 warmup 合計 302 次 NPU prefill matmul（262 Q4_K、40 Q6_K），零次 W8/W4/native-W8 decode，exit 0。這證實正式混合設定有實際 NPU 運算；此開 trace 的 run 不納入效能數據。

常駐 cache 增加記憶體使用量（接近每個加速權重兩 bytes 加 scratch），首次準備與 batch 大小改變會產生成本。只保留每個不可變權重的一個 exact-M layout，context 銷毀時釋放。短 prompt、其他模型與不同硬體設定不能套用本輪速度結論。

使用方式見 [adapter README](../../../adapters/ggml-rocknpu/README.md)。下一階段尚未驗證的閉源驅動方向見 [猜想文件](../../closed-driver-hypotheses.md)；本輪沒有同條件閉源驅動對照，沒有超過閉源的結論。
