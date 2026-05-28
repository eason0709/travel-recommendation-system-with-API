# Travel Recommendation System

本專案是一個以 Python 實作的旅遊景點推薦與決策支援系統。系統會產生模擬使用者旅遊行為資料，使用 Gaussian Mixture Model (GMM) 進行 soft clustering，並可使用模擬景點資料或 OpenStreetMap 真實 POI 資料作為景點候選來源。

目前專案重點不是建立完整商業級旅遊平台，而是展示：

- 使用者偏好如何被轉換成特徵向量
- GMM soft clustering 如何分析使用者偏好族群
- cluster probability 與 cluster confidence 如何呈現模型不確定性
- personal、cluster、collaborative、hybrid 等推薦分數如何比較
- Streamlit App 如何展示 cluster 解釋、推薦景點與視覺化結果

> 注意：若使用 `fetch_real_attractions_osm.py`，景點名稱、座標與部分 metadata 來自 OpenStreetMap；但 `cost_level`、`photo_value`、`popularity`、`estimated_time` 等推薦特徵仍是依據 OSM tag 與景點類型推估，不代表真實平台評分、票價或人氣。

---

## 專案功能摘要

| 功能 | 說明 |
|---|---|
| 模擬使用者資料 | 產生使用者預算、可用時間、天氣、疲勞、社交情境與景點偏好 |
| 模擬景點資料 | 使用 `generate_attractions.py` 產生模擬景點 |
| 真實 POI 匯入 | 使用 `fetch_real_attractions_osm.py` 從 OpenStreetMap 抓取真實景點名稱、類型、座標與 metadata |
| GMM Soft Clustering | 使用 GMM 將使用者分成 soft cluster，並輸出每群機率 |
| Cluster Confidence | 使用最高 cluster probability 作為模型信心 |
| 動態 Cluster Label | 根據當次 clustering 後的 z_mean 自動產生 cluster label，避免 label 與 heatmap 衝突 |
| Heatmap 視覺化 | 使用紅藍漸層與格內數值顯示各 cluster 特徵平均值 |
| 推薦分數 | 產生 personal、cluster、collaborative、hybrid 四種分數 |
| 方法評估 | 比較 personal-only、cluster-only、collaborative-only、hybrid 與 random baseline |
| App 展示 | 顯示使用者 cluster、confidence、probability、cluster 解釋與 cluster-based 推薦 |
| 測試工具 | `test_cluster_inputs.py` 可測試不同輸入參數下會對應到哪個 cluster |

---

## 專案結構

```text
travel_clustering_project/
├── travel_utils.py
├── generate_data.py
├── cluster_interpretation_soft.py
├── generate_attractions.py
├── fetch_real_attractions_osm.py
├── generate_user_interactions.py
├── recommendation_engine.py
├── evaluate_clusters.py
├── analyze_data.py
├── test_cluster_inputs.py
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/       # 執行後產生，建議不要上傳 GitHub
├── figures/    # 執行後產生，建議不要上傳 GitHub
└── models/     # 執行後產生，建議不要上傳 GitHub
```

---

## 檔案說明

| 檔案 | 功能 |
|---|---|
| `travel_utils.py` | 共用工具檔，管理特徵欄位、相似度、scene name、cluster label、confidence 判讀與 CSV 工具 |
| `generate_data.py` | 產生模擬使用者旅遊行為資料 |
| `cluster_interpretation_soft.py` | 訓練 GMM、產生 cluster probability、confidence、heatmap、cluster label summary 與模型檔 |
| `generate_attractions.py` | 產生模擬景點資料 |
| `fetch_real_attractions_osm.py` | 從 OpenStreetMap Overpass API 抓取真實 POI，轉成系統可用的 `attractions.csv` |
| `generate_user_interactions.py` | 根據使用者與景點相似度產生模擬 rating / clicked 紀錄 |
| `recommendation_engine.py` | 為每位使用者與景點計算 personal、cluster、collaborative、hybrid 分數 |
| `evaluate_clusters.py` | 比較不同推薦方法與 random baseline |
| `analyze_data.py` | 額外資料診斷，產生相關係數圖與特徵分布圖 |
| `test_cluster_inputs.py` | 測試不同手動輸入下的 cluster 結果 |
| `app.py` | Streamlit 網頁展示系統 |

---

## 安裝

建議使用 Python 3.10 以上版本，並建立虛擬環境。

```bash
python -m venv venv
```

Windows：

```bash
venv\Scripts\activate
```

macOS / Linux：

```bash
source venv/bin/activate
```

安裝套件：

```bash
pip install -r requirements.txt
```

目前 `requirements.txt` 包含：

```text
pandas
numpy
matplotlib
scikit-learn
streamlit
joblib
pyarrow
requests
```

---

## 執行流程：使用模擬景點資料

如果只使用模擬景點資料，請依序執行：

```bash
python generate_data.py
python cluster_interpretation_soft.py
python generate_attractions.py
python generate_user_interactions.py
python recommendation_engine.py
python evaluate_clusters.py
streamlit run app.py
```

---

## 執行流程：使用 OpenStreetMap 真實 POI 資料

若要用 OpenStreetMap 取得真實地點名稱、座標與類型，將 `generate_attractions.py` 替換成 `fetch_real_attractions_osm.py`。

建議先測試 100 筆：

```bash
python generate_data.py
python cluster_interpretation_soft.py
python fetch_real_attractions_osm.py --area 臺北市 --max-items 100 --per-group-limit 60
python generate_user_interactions.py
python recommendation_engine.py
python evaluate_clusters.py
streamlit run app.py
```

專題展示建議使用 300 筆：

```bash
python fetch_real_attractions_osm.py --area 臺北市 --max-items 300 --per-group-limit 80
```

如果 Overpass API 發生 504 Gateway Timeout 或 429 Too Many Requests，可以降低 `--max-items` 或 `--per-group-limit`：

```bash
python fetch_real_attractions_osm.py --area 臺北市 --max-items 200 --per-group-limit 40
```

可嘗試的地區名稱：

```bash
python fetch_real_attractions_osm.py --area 臺北市 --max-items 300 --per-group-limit 80
python fetch_real_attractions_osm.py --area 台北市 --max-items 300 --per-group-limit 80
python fetch_real_attractions_osm.py --area "Taipei City" --max-items 300 --per-group-limit 80
```

### OpenStreetMap 資料限制

`fetch_real_attractions_osm.py` 會輸出真實 POI 的：

```text
realistic_scene_name
latitude
longitude
osm_id
osm_category
address
opening_hours
website
phone
wikipedia
wikidata
wheelchair
fee
```

但以下欄位仍是根據景點類型與 tag 推估：

```text
cost_level
indoor_score
photo_value
nature_value
culture_value
food_value
popularity
estimated_time
```

因此報告中應描述為：

```text
系統使用 OpenStreetMap 取得真實景點名稱、類型、座標與部分 metadata；
但推薦特徵仍由類型與 tag heuristic 轉換而來。
```

不要寫成：

```text
系統完全使用真實旅遊平台資料。
```

---

## Step A：產生使用者行為資料

```bash
python generate_data.py
```

輸出：

```text
data/travel_behavior.csv
```

主要欄位：

| 欄位 | 說明 |
|---|---|
| `budget` | 使用者旅遊預算 |
| `available_time` | 可用時間 |
| `weather_badness` | 天氣不佳程度，越高代表越不適合戶外 |
| `social_context` | 社交／同行情境 |
| `fatigue` | 疲勞程度 |
| `spontaneity` | 隨興程度 |
| `selected_indoor_score` | 室內偏好 |
| `selected_cost_level` | 可接受消費等級 |
| `selected_photo_value` | 拍照價值偏好 |
| `selected_nature_value` | 自然景點偏好 |
| `selected_culture_value` | 文化景點偏好 |
| `selected_food_value` | 美食偏好 |
| `selected_popularity` | 熱門程度偏好 |

---

## Step B：GMM Soft Clustering

```bash
python cluster_interpretation_soft.py
```

此步驟會：

1. 讀取 `data/travel_behavior.csv`
2. 使用 `SimpleImputer` 補缺失值
3. 使用 `StandardScaler` 標準化
4. 使用 BIC 在 k=2 到 k=8 中選擇 GMM cluster 數
5. 訓練 `GaussianMixture`
6. 輸出每位使用者對各 cluster 的 probability
7. 產生 cluster confidence
8. 產生 heatmap 與 cluster label summary
9. 儲存模型到 `models/soft_cluster_pipeline.pkl`

輸出：

```text
data/travel_behavior_soft_clustered.csv
data/soft_cluster_feature_mean.csv
data/soft_cluster_interpretation.csv
data/soft_cluster_label_summary.csv
figures/soft_cluster_feature_heatmap.png
figures/soft_cluster_confidence_distribution.png
models/soft_cluster_pipeline.pkl
```

重要欄位：

| 欄位 | 說明 |
|---|---|
| `soft_cluster` | probability 最高的 cluster |
| `soft_cluster_confidence` | 最高 cluster probability |
| `cluster_prob_i` | 屬於第 i 個 cluster 的機率 |

### Heatmap 說明

`figures/soft_cluster_feature_heatmap.png` 使用紅藍漸層：

| 顏色 | 意義 |
|---|---|
| 紅色 | 該 cluster 在此特徵上高於整體平均 |
| 藍色 | 該 cluster 在此特徵上低於整體平均 |
| 白色附近 | 接近整體平均 |

格子內的數字是 `z_mean`，也就是標準化後的 cluster feature mean。

---

## Step C：產生或匯入景點資料

### 方法一：模擬景點資料

```bash
python generate_attractions.py
```

輸出：

```text
data/attractions.csv
```

### 方法二：OpenStreetMap 真實 POI

```bash
python fetch_real_attractions_osm.py --area 臺北市 --max-items 300 --per-group-limit 80
```

同樣輸出：

```text
data/attractions.csv
```

因此後續 pipeline 不需要修改。

---

## Step D：產生模擬互動資料

```bash
python generate_user_interactions.py
```

輸出：

```text
data/user_interactions.csv
```

此步驟根據使用者偏好向量、景點特徵向量、時間限制、天氣因素與 popularity bias 產生模擬互動紀錄。

輸出欄位包含：

```text
user_id
soft_cluster
attraction_id
rating
clicked
```

---

## Step E：產生推薦候選清單

```bash
python recommendation_engine.py
```

輸出：

```text
data/all_recommendation_candidates.csv
data/recommendations.csv
```

推薦分數：

| 分數 | 邏輯 |
|---|---|
| `personal_score` | 使用者個人偏好向量與景點特徵向量的 cosine similarity，並考慮時間與天氣 penalty |
| `cluster_score` | 使用者所屬 cluster 的平均偏好向量與景點特徵向量的 cosine similarity |
| `collaborative_score` | 同 cluster 使用者對該景點的平均 rating |
| `hybrid_score` | personal、cluster、collaborative 的加權混合分數 |
| `recommendation_score` | 目前輸出推薦使用的分數，預設為 hybrid score |

---

## Step F：評估推薦方法

```bash
python evaluate_clusters.py
```

輸出：

```text
data/recommendation_method_comparison.csv
figures/recommendation_method_comparison.png
```

比較方法：

| 方法 | 使用分數 |
|---|---|
| `personal_only` | `personal_score` |
| `cluster_only` | `cluster_score` |
| `collaborative_only` | `collaborative_score` |
| `hybrid` | `hybrid_score` |

評估方式是：對每位使用者根據不同分數取 Top-K 推薦，再計算推薦景點與使用者個人偏好向量的平均相似度，並與 random baseline 比較。

目前測試結果的解讀方式：

```text
personal_only 通常表現最好，因為評估指標本身就是看景點與使用者個人偏好的相似度。
cluster_only 的價值不一定是最高準確率，而是提供族群偏好解釋、冷啟動輔助與模型不確定性展示。
```

---

## Step G：啟動 Streamlit App

```bash
streamlit run app.py
```

目前 App 主要功能：

| 功能 | 說明 |
|---|---|
| 手動輸入使用者條件 | 左側 sidebar 可輸入預算、時間、天氣、疲勞、隨興程度與偏好 |
| 中文欄位說明 | 每個輸入欄位附中文與英文名稱，以及高低值意義 |
| 使用既有使用者 | 可從已產生資料中選擇 user ID |
| Soft Cluster 顯示 | 顯示目前使用者最可能的 cluster |
| Cluster Label | 從 `data/soft_cluster_label_summary.csv` 動態讀取 label |
| Cluster Confidence | 顯示模型對此 cluster 判斷的信心 |
| Soft Cluster Probability | 手動輸入模式下顯示各 cluster probability |
| Low Confidence Warning | confidence 低時提醒不要過度依賴單一 label |
| Cluster 計算流程說明 | 說明 imputer、scaler、GMM、probability、confidence 的流程 |
| Cluster-based 推薦 | 使用使用者所屬 cluster 的平均偏好向量推薦景點 |
| Cluster 解釋 | 顯示該 cluster 高於或低於平均的特徵 |
| 方法比較 | 顯示 `evaluate_clusters.py` 的方法比較結果 |
| 圖表展示 | 顯示 heatmap、confidence distribution、method comparison 等圖 |

### App 目前的推薦模式狀態

目前上傳版本的 `app.py` 主要顯示 **cluster-based recommendation**，也就是使用 `cluster_score` 的邏輯做 App 內即時推薦。

不過離線評估檔案 `evaluate_clusters.py` 仍會比較：

```text
personal_only
cluster_only
collaborative_only
hybrid
```

因此報告中可以這樣說：

```text
系統在離線評估階段比較 personal-only、cluster-only、collaborative-only 與 hybrid 四種推薦方法；
目前 App 介面則主要展示 cluster-based 推薦，以突顯分群解釋、cluster confidence 與使用者偏好族群分析。
```

---

## Cluster Label 命名規則

本專案不再固定寫死 `cluster 0~7` 的中文名稱。原因是 GMM 每次重新訓練後 cluster 編號可能重新排列，如果固定寫死 label，容易出現 label 與 heatmap 衝突。

目前流程是：

```text
GMM 分群
→ 計算每個 cluster 的標準化平均特徵 z_mean
→ 根據 z_mean 自動產生 cluster label
→ 輸出 data/soft_cluster_label_summary.csv
→ App 讀取該檔案顯示 label
```

### Label 門檻

目前命名邏輯採用保守門檻：

| z_mean 範圍 | 解讀 |
|---|---|
| `z_mean >= 0.80` | 明顯高，可作為主要 label |
| `0.50 <= z_mean < 0.80` | 輕度偏高，只能作為輕度偏好 |
| `-0.50 < z_mean < 0.50` | 接近平均，不拿來命名 |
| `z_mean <= -0.80` | 明顯低，可用於低偏好型 label |
| `budget >= 3.00` | 視為極高預算特殊型 |

這樣可以避免例如：

```text
budget = 0.141
```

卻被命名成：

```text
預算偏高型
```

### Label 相關輸出

```text
data/soft_cluster_label_summary.csv
```

包含：

| 欄位 | 說明 |
|---|---|
| `cluster` | cluster 編號 |
| `cluster_label` | 自動推論出的中文名稱 |
| `label_strength` | label 強度，分為 weak / moderate / strong |
| `label_basis` | 命名依據 |
| `description` | cluster 說明 |
| `top_high_features` | 最高的幾個 z_mean 特徵 |
| `top_low_features` | 最低的幾個 z_mean 特徵 |

---

## 測試不同輸入的 Cluster 結果

可使用：

```bash
python test_cluster_inputs.py --preset --method mean --show-prob
```

此工具可測試不同參數下的 cluster 結果。

### 測試模式

| 模式 | 說明 |
|---|---|
| `--method gmm` | 讀取 `models/soft_cluster_pipeline.pkl`，最接近 App 的 GMM 預測 |
| `--method mean` | 不讀 pkl，使用 cluster 平均特徵進行近似測試，較穩定 |
| `--method auto` | 先嘗試 GMM，失敗後自動改用 mean |

若出現 pkl 讀取問題，可使用：

```bash
python test_cluster_inputs.py --preset --method mean --show-prob
```

自訂輸入範例：

```bash
python test_cluster_inputs.py --method mean --budget 5 --available-time 4 --weather-badness 0.2 --social-context 1 --fatigue 1 --spontaneity 4 --indoor 3 --cost 5 --photo 5 --nature 2 --culture 3 --food 4 --popularity 5 --show-prob
```

輸出：

```text
data/cluster_input_test_results.csv
```

---

## Optional：資料分析

```bash
python analyze_data.py
```

輸出：

```text
figures/correlation_matrix.png
figures/{feature}_distribution.png
```

此步驟非主流程必要步驟，用於檢查資料分布與相關係數。

---

## 推薦系統設計解讀

### personal-only 的用途

`personal_only` 直接使用使用者個人偏好向量與景點特徵向量比較，因此通常最適合做高個人化推薦。

### cluster-only 的用途

`cluster_only` 先判斷使用者屬於哪個偏好族群，再用該族群的平均偏好推薦景點。它的推薦準確率不一定高於 personal-only，但能提供：

- 使用者類型解釋
- 群體偏好分析
- 冷啟動輔助
- 模型不確定性呈現
- heatmap 與 cluster probability 的決策支援資訊

因此本專案不主張 cluster-only 一定比 personal-only 更準，而是將 clustering 作為可解釋決策支援的一部分。

---

## GitHub 注意事項

`data/`、`figures/`、`models/` 是執行後產生的資料、圖片與模型檔，建議不要上傳 GitHub。

`.gitignore` 建議包含：

```gitignore
data/
figures/
models/
__pycache__/
*.pyc
venv/
.venv/
.env
.DS_Store
Thumbs.db
```

如果這些資料夾已經被 Git 追蹤，請使用：

```bash
git rm -r --cached data figures models
```

這會從 Git 追蹤中移除，但保留本機檔案。

接著：

```bash
git add .gitignore README.md
git commit -m "Update README and stop tracking generated outputs"
git push
```

---

## 常見問題

### Overpass API 504 / 429

如果執行 `fetch_real_attractions_osm.py` 出現：

```text
504 Gateway Timeout
429 Too Many Requests
```

代表 Overpass API 暫時負載過高或查詢量太大。可降低：

```bash
--max-items
--per-group-limit
```

例如：

```bash
python fetch_real_attractions_osm.py --area 臺北市 --max-items 200 --per-group-limit 40
```

### App 顯示舊 cluster label

請重新執行：

```bash
python cluster_interpretation_soft.py
```

並確認存在：

```text
data/soft_cluster_label_summary.csv
```

App 會讀取此檔案顯示最新 label。

### Heatmap 與 label 不一致

通常是因為重新產生了 heatmap，但沒有同步更新或讀取 `soft_cluster_label_summary.csv`。請重新執行：

```bash
python cluster_interpretation_soft.py
streamlit run app.py
```

### GMM pkl 讀取失敗

先重新產生模型：

```bash
python cluster_interpretation_soft.py
```

若只是要測試 cluster 輸入結果，可以使用 mean mode：

```bash
python test_cluster_inputs.py --preset --method mean --show-prob
```

---

## 專案限制

本專案仍有以下限制：

- 使用者資料是模擬產生，不是真實平台使用者資料
- 使用者互動資料是模擬 rating / clicked，不是真實點擊或收藏紀錄
- OpenStreetMap 不提供完整評分、評論、票價、即時人氣或訂票狀態
- OSM 匯入後的推薦特徵仍需 heuristic 轉換
- App 目前主要展示 cluster-based recommendation，不是完整商業級推薦平台
- 評估方法偏向 similarity-based，因此 personal-only 通常較有利
- 尚未整合交通時間、即時天氣決策、真實使用者帳號與使用者實驗

---

## 結論

本專案展示了一個以 soft clustering 輔助旅遊推薦的決策支援系統。`personal_only` 較適合追求個人化準確度，`cluster_only` 則提供使用者類型解釋、cluster confidence、soft probability 與族群偏好分析。若使用 OpenStreetMap 匯入資料，系統可展示真實地點名稱與座標，但推薦特徵仍需依據類型與 tag 推估。因此，本專案較適合定位為「可解釋推薦系統原型」，而非完整真實旅遊平台。
