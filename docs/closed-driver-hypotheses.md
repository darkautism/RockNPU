# 閉源 LLM 驅動優化猜想

本文件區分實測證據與尚未驗證的方向。本輪已達到有範圍的 CPU 門檻：TinyLlama `pp512+tg128` 混合推論的兩輪交錯驗證提升 12.16%，所有 24 次請求成功，見[完整證據](benchmarks/2026-09-20/README.md)。後續閉源方向依使用者允許保留為猜想。沒有同模型、同量化、同時脈、同上下文、同計時範圍的直接對照，就不宣稱超過閉源驅動。

## 已有證據的邊界

- 本次原生 ARM Release CPU 達到 33.37 tok/s decode，明顯快於目前 W8 NPU。歷史約 20 tok/s 的 CPU 編譯未開啟 ARM dot-product，不能據此宣稱已超過 CPU。全 8 核心也已檢查，這個模型使用四個 A76 的結果較快。
- Prefill 常駐 FP16 權重配 FP32 partial／f64 累加，以及依原生排列讀取輸出，已在探索機把 pp512 從 25.40 提升到 91.37 tok/s。這是新的直接證據，支持修正「FP16 bridge 沒有優化價值」的過度概括。完整請求另在驗證機通過原生 CPU 對照；生成由 CPU 執行，不能延伸成 NPU decode 勝出。
- 詳細原始歷史在 [trialanderror.md](../trialanderror.md) 與 [重現紀錄](repro.md)。歷史約 24.43 tok/s 的 RKLLM 數字只是文獻目標，並非本次兩機的直接對照。
- W8A8 是 GGUF 權重再次量化、啟動值動態量化的近似路徑。與舊 W8A8 路徑的確定性輸出相同，不等於與 CPU Q4_K_M 的 logits 或模型品質相同。歷史固定 token history 對照已看到 CPU 與 W8A8 在第 14 步有 top-1 分歧。
- 多核 direct-submit、shape-keyed persistent scratch、直接累加 worker 輸出已有完整模型正向證據。這些改動保留原有 W8A8 算術；sidecar v2 必須由實際 GGUF 生成並通過來源檢查。
- 700 MHz／800 mV 是現有實驗設定。封裝版 mainline Rocket 的 200 MHz 數據、只有 devfreq 的研究模組、額外含 IOMMU cache 的研究模組，應分開標示。
- 舊 profiling 的約 40 ms/token output wait 含裝置執行與排程／完成等待，不能直接當成純硬體計算時間。約 0.76 ms input staging、0.87 ms regcmd staging 是 worker 階段加總，也不能無條件相加成可節省的牆鐘時間。

## 猜想與證偽條件

| 方向 | 已知依據 | 尚未證明的猜想 | 何時值得重開實驗 |
| --- | --- | --- | --- |
| 較少的權重流量 | W8 resident 權重約 924 MiB；W4 primitive 可正確且更快，但舊完整模型 W4 沒有穩定收益且生成分歧 | 閉源堆疊可能依靠不同分組／校準或混合精度，在保留品質下減少記憶體流量 | 先用獨立 CPU oracle 與固定 token history 評估量化誤差，再測同一完整模型的品質及吞吐；不要再盲掃 worker 數 |
| 保留 FFN 中間資料 | FFN 是主要投影成本；整數硬體鏈已驗證，但 per-channel scales 與單一 LUT/requant domain 不相容；compact int16 品質失敗 | 合適的縮放域或高精度中間格式能減少 host round-trip | 先證明縮放、溢位、SiLU 與輸出量化的誤差界；純 SDP／CNA 轉換與 PC-chain 需明確 kernel 能力，不能假設 stock UAPI 支援 |
| 減少 CPU 小算子的同步 | 解碼圖約 221 個 CPU/NPU partition；flash attention 有完整模型收益；一般 CPU delegate 原型更慢 | 共用持久 threadpool 或特定小算子融合，可能降低 RMSNorm／RoPE／SWIGLU 的同步成本 | 證明省掉實際工作或 barrier，而非只重貼 backend 標籤；包含 CPU output head 與 attention 的整體測試必須變快 |
| 精確 M=1 output head | W8 head 品質不合格；pad M=4 FP16 head 品質 gate 通過但約慢 6% | 真正不需四倍 padding 的高精度 M=1 可把最後 CPU 大投影移至 NPU | 有新的硬體格式／排程證據後才重開；backend routing A/B 必須使用獨立 context，不能在已建圖 context 中切 env 冒充切換 backend |
| 合併裝置工作與完成事件 | 普通 submit 約數微秒；短工作存在排程與喚醒固定成本，但許多工作佇列調整未改善整體 | 將有資料依賴的運算保留於裝置可減少真正的同步邊界 | 需要完整生命週期與逾時語義、能力偵測及品質證據；不以減少 ioctl 數本身作為成功指標 |
| Prefill 的 K partial 留在裝置 | 本輪常駐權重加連續 gather 有顯著實測收益；目前每個 K partial 仍輸出 FP32，再由 CPU f64 累加 | 閉源可能使用更合適的 tile／累加格式，減少大批次的 FP32 回傳流量 | 先證明硬體能保留所需精度及一致的累加語義；不能直接改用現有 FP16 EW 累加冒充等價優化。CPU 門檻達成後可只記錄，不必本輪實驗 |

## 已有負面證據，不重複盲試

1. 700 MHz 升至 1 GHz、NPU ACLK 250 升至 500 MHz 的歷史測試沒有完整模型收益。除非目前 workload 的瓶頸已有改變，不再以提高時脈作為預設方向。
2. IOMMU domain thrash 確實存在；固定 core、偏向 locality、共享 domain 配多 scheduler entities 都已驗證機制但未得到吞吐收益。消除 attach/detach 不代表值得放棄負載平衡。
3. 把 IRQ 放到 A76、固定 CPU4 submit worker、提高 workqueue 優先級、關閉全部 A76 深睡，都沒有通過可重現的整體收益門檻。
4. Per-weight regcmd cache 曾通過整數 microbench，卻在真實模型首個 decode graph 失敗。需要新的 BO／fence／生命週期證據才能再試。
5. 不因短 prompt token 相同、零飽和或單次峰值，就宣稱量化品質等價或效能達標。

兩台規格近似的板子分工為探索與驗證：不必每個候選雙機完整重跑，只把探索機勝出的設定送去複驗。達到穩定、可重現的 CPU 勝出後，以上未驗證方向可以維持猜想狀態，無須為了收尾而逐一實驗。
