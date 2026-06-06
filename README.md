# Travel Recommendation System

本專案是一個以 **GMM soft clustering** 為核心的旅遊景點推薦與決策支援系統示範。系統會根據使用者輸入的旅遊偏好，判斷使用者較接近哪一類旅遊偏好族群，並展示 cluster probability、cluster confidence、cluster heatmap、推薦景點與方法比較結果。

本專案重點不是建立完整商業旅遊平台，而是展示：

```text
使用者偏好 → 模型判斷偏好族群 → 推薦景點 → 顯示模型信心與解釋依據
```

---

## 目前版本重點

### 1. Cluster 已改為偏好特徵導向

最新版 GMM clustering 只使用以下 7 個景點偏好欄位：

```text
selected_indoor_score
selected_cost_level
selected_photo_value
selected_nature_value
selected_culture_value
selected_food_value
selected_popularity
```

這樣做的原因是：cluster 應該描述「使用者偏好哪一類景點」，而不是被預算、天氣、疲勞或可用時間等情境因素主導。

以下情境欄位仍會保留在資料與 App 中，但目前不直接用來訓練 cluster：

```text
budget
available_time
weather_badness
social_context
fatigue
spontaneity
```

---

### 2. App 目前展示 cluster-based recommendation

最新版 App 主要展示：

```text
Cluster-based 推薦景點
```

流程為：

```text
使用者輸入
→ GMM 判斷最接近的 cluster
→ 取該 cluster 的平均偏好
→ 根據 cluster 平均偏好推薦景點
```

請注意：目前 App 不提供 `personal_only`、`cluster_only`、`collaborative_only`、`hybrid` 的 UI 切換。這些方法仍由 `recommendation_engine.py` 與 `evaluate_clusters.py` 在離線階段計算與比較。

---

### 3. Cluster label 改為動態產生

GMM 每次重新訓練後，cluster 編號可能改變。因此本專案不再固定寫死：

```text
Cluster 0 = 某固定類型
Cluster 1 = 某固定類型
```

目前 cluster label 會根據當次訓練後的 cluster feature mean 動態產生，並輸出到：

```text
data/soft_cluster_label_summary.csv
```

判讀 cluster 時，應以實際執行後產生的 heatmap 與 label summary 為準。

---

### 4. 新增 Pipeline Data Audit 視覺化檢查

最新版新增：

```text
visualize_pipeline_data.py
```

此工具用來檢查整個 pipeline 是否成功串接，包含資料獲取、資料處理、模型建立、推薦計算、互動資料檢查、方法比較與結果分析。

執行後會產生：

```text
pipeline_audit/pipeline_data_audit.html
pipeline_audit/figures/*.png
```

它不是主模型的一部分，而是診斷與視覺化檢查工具。建議在跑完整個 pipeline 後執行，用來確認所有主要資料檔案、模型輸出與評估結果都已正確產生。

---

## 專案功能總覽

| 功能 | 目前狀態 | 說明 |
|---|---|---|
| 使用者資料生成 | 已實作 | 產生旅遊偏好與情境欄位 |
| 偏好導向 GMM clustering | 已實作 | 只使用 7 個景點偏好欄位訓練 cluster |
| Cluster probability | 已實作 | 顯示使用者屬於各 cluster 的機率 |
| Cluster confidence | 已實作 | 使用最高 cluster probability 作為模型信心 |
| Low confidence warning | 已實作 | 信心低時提醒不要過度解讀單一 cluster |
| Dynamic cluster label | 已實作 | 根據 cluster feature mean 自動命名 |
| Cluster heatmap | 已實作 | 顯示各 cluster 在偏好特徵上的高低 |
| OpenStreetMap POI 匯入 | 已實作 / 選配 | 可用 OSM 真實 POI 作為景點資料來源 |
| Cluster-based App 推薦 | 已實作 | App 目前展示此模式 |
| Personal / Cluster / Collaborative / Hybrid 離線比較 | 已實作 | 由 `recommendation_engine.py` 與 `evaluate_clusters.py` 完成 |
| Pipeline Data Audit 視覺化檢查 | 已實作 | 由 `visualize_pipeline_data.py` 產生 HTML 報告與多張診斷圖表 |
| App 內切換推薦模式 | 目前未保留 | 舊版 README 相關描述已移除 |
| 固定 Cluster 0~7 名稱表 | 已移除 | 改用動態 label summary |

---

## Project Structure

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
├── visualize_pipeline_data.py
├── test_cluster_inputs.py
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/             # 執行後產生
├── figures/          # 執行後產生
├── models/           # 執行後產生
└── pipeline_audit/   # 執行 visualize_pipeline_data.py 後產生
```

---

## 檔案說明

| 檔案 | 功能 |
|---|---|
| `travel_utils.py` | 共用工具檔，集中管理特徵欄位、cluster label、confidence 判讀與顯示名稱 |
| `generate_data.py` | 產生使用者旅遊偏好與情境資料 |
| `cluster_interpretation_soft.py` | 使用 GMM 進行 preference-based soft clustering，輸出 cluster probability、confidence、label summary 與 heatmap |
| `generate_attractions.py` | 產生模擬景點資料，主要作為備用或測試資料來源 |
| `fetch_real_attractions_osm.py` | 從 OpenStreetMap Overpass API 抓取真實 POI，並轉換成本專案使用的景點資料格式 |
| `generate_user_interactions.py` | 根據使用者偏好與景點特徵產生互動資料，例如 rating / clicked |
| `recommendation_engine.py` | 產生推薦候選清單與 Top-K 推薦結果，包含 personal、cluster、collaborative、hybrid 分數 |
| `evaluate_clusters.py` | 比較不同推薦方法與 random baseline 的差異 |
| `analyze_data.py` | 選配，用於單一使用者資料檔的資料分布與相關係數分析 |
| `visualize_pipeline_data.py` | 選配但建議執行，用於檢查完整 pipeline 的檔案狀態、資料分布、cluster、推薦分數與方法比較，並輸出 HTML 報告與圖表 |
| `test_cluster_inputs.py` | 用多組固定輸入測試 cluster 分類是否合理 |
| `app.py` | Streamlit 網頁展示系統 |

---

## Installation

建議使用 Python 虛擬環境。

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

macOS / Linux:

```bash
source venv/bin/activate
```

安裝套件：

```bash
pip install -r requirements.txt
```

`requirements.txt` 建議包含：

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

## 執行順序

### 使用 OpenStreetMap 真實 POI 資料

建議報告展示使用此流程：

```bash
python generate_data.py
python cluster_interpretation_soft.py
python fetch_real_attractions_osm.py --area 臺北市 --max-items 300
python generate_user_interactions.py
python recommendation_engine.py
python evaluate_clusters.py
python visualize_pipeline_data.py
streamlit run app.py
```

### 使用模擬景點資料

若 Overpass API 暫時無法使用，或只是想快速測試系統流程，可以使用模擬景點資料：

```bash
python generate_data.py
python cluster_interpretation_soft.py
python generate_attractions.py
python generate_user_interactions.py
python recommendation_engine.py
python evaluate_clusters.py
python visualize_pipeline_data.py
streamlit run app.py
```

如果只想啟動 App，前提是 `data/`、`models/`、`figures/` 已經存在：

```bash
streamlit run app.py
```

`pipeline_audit/` 不是 App 必要資料夾，只是執行 `visualize_pipeline_data.py` 後產生的診斷報告。

---

## Main Pipeline

### Step A: Generate User Behavior Data

```bash
python generate_data.py
```

輸出：

```text
data/travel_behavior.csv
```

主要欄位分成兩類。

#### 情境欄位

```text
budget
available_time
weather_badness
social_context
fatigue
spontaneity
```

#### 景點偏好欄位

```text
selected_indoor_score
selected_cost_level
selected_photo_value
selected_nature_value
selected_culture_value
selected_food_value
selected_popularity
```

景點偏好欄位會用於 GMM clustering，情境欄位目前主要用於描述與 App 顯示。

---

### Step B: Soft Clustering and Cluster Interpretation

```bash
python cluster_interpretation_soft.py
```

輸出：

```text
data/travel_behavior_soft_clustered.csv
data/soft_cluster_feature_mean.csv
data/soft_cluster_label_summary.csv
data/soft_cluster_interpretation.csv
models/soft_cluster_pipeline.pkl
figures/soft_cluster_feature_heatmap.png
figures/soft_cluster_confidence_distribution.png
```

重要欄位：

| 欄位 | 說明 |
|---|---|
| `soft_cluster` | 使用者最可能所屬的 cluster |
| `soft_cluster_confidence` | 最高 cluster probability，也就是模型信心 |
| `cluster_prob_i` | 使用者屬於第 i 個 cluster 的機率 |

---

## Heatmap 判讀規則

`soft_cluster_feature_heatmap.png` 顯示每個 cluster 在各偏好特徵上的標準化平均值。

建議用以下 threshold 判讀：

| 標記 | 條件 | 意思 |
|---|---:|---|
| 特高 | `z >= +1.0` | 該 cluster 在此特徵明顯高於平均 |
| 偏高 | `+0.5 <= z < +1.0` | 稍微高於平均 |
| 普通 | `-0.5 < z < +0.5` | 接近平均，不要過度解讀 |
| 偏低 | `-1.0 < z <= -0.5` | 稍微低於平均 |
| 特低 | `z <= -1.0` | 該 cluster 在此特徵明顯低於平均 |

---

### Step C: Prepare Attraction Data

#### Option 1: OpenStreetMap POI Import

```bash
python fetch_real_attractions_osm.py --area 臺北市 --max-items 300
```

此步驟會使用 OpenStreetMap Overpass API 抓取真實 POI，並轉換成本專案的：

```text
data/attractions.csv
```

OSM 可提供：

```text
真實地點名稱
經緯度
地址
網站
營業時間
wikipedia / wikidata
wheelchair 等 metadata
```

但 OSM 通常不提供：

```text
完整評分
真實人氣
票價
排隊時間
訂票狀態
使用者評論
```

因此以下欄位仍屬於根據景點類型與 OSM tags 推估：

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

#### Option 2: Simulated Attraction Data

```bash
python generate_attractions.py
```

此步驟可在沒有 OSM 資料時產生測試用景點資料。

---

### Step D: Generate User Interactions

```bash
python generate_user_interactions.py
```

輸出：

```text
data/user_interactions.csv
```

此步驟會根據使用者偏好與景點特徵產生互動資料，例如：

```text
rating
clicked
```

目前此互動資料主要用於展示 collaborative-only 與 hybrid 方法流程，不應直接解讀為真實平台使用者行為。

---

### Step E: Recommendation Engine

```bash
python recommendation_engine.py
```

輸出：

```text
data/all_recommendation_candidates.csv
data/recommendations.csv
```

分數包含：

| 分數 | 說明 |
|---|---|
| `personal_score` | 使用者個人偏好向量與景點向量的相似度 |
| `cluster_score` | 使用者所屬 cluster 平均偏好向量與景點向量的相似度 |
| `collaborative_score` | 同 cluster 使用者對景點的平均互動評分 |
| `hybrid_score` | personal、cluster、collaborative 的混合分數 |

---

### Step F: Evaluate Recommendation Methods

```bash
python evaluate_clusters.py
```

輸出：

```text
data/recommendation_method_comparison.csv
figures/recommendation_method_comparison.png
```

比較方法：

| 方法 | 說明 |
|---|---|
| `personal_only` | 直接根據使用者個人偏好推薦 |
| `cluster_only` | 根據使用者所屬 cluster 的平均偏好推薦 |
| `collaborative_only` | 根據同 cluster 使用者的互動紀錄推薦 |
| `hybrid` | 混合 personal、cluster、collaborative 分數 |

方法比較的合理解讀：

```text
四種方法通常都高於 random baseline。
personal_only 通常略高。
hybrid 接近 personal_only。
cluster_only 與 collaborative_only 略低，但仍有解釋與參考價值。
```

---

### Step G: Pipeline Data Audit Visualization

```bash
python visualize_pipeline_data.py
```

此步驟會產生完整 pipeline 的視覺化檢查報告，用來確認資料獲取、資料處理、模型建立、推薦計算與評估結果是否都有成功串接。

輸出：

```text
pipeline_audit/pipeline_data_audit.html
pipeline_audit/figures/
```

主要檢查內容：

| 檢查項目 | 說明 |
|---|---|
| Expected file status | 檢查主要 CSV 與模型檔是否存在 |
| DataFrame profiles | 顯示各主要 CSV 的 rows、columns、missing values、numeric summary 與資料預覽 |
| User preference distribution | 檢查使用者偏好特徵分布是否合理 |
| User feature correlation matrix | 檢查偏好欄位與情境欄位之間的相關性 |
| Top attraction types | 檢查 OSM POI 或景點資料的類型分布 |
| Attraction feature distribution | 檢查景點資料經特徵轉換後的分布 |
| User count by soft cluster | 檢查各 cluster 人數是否過度不均 |
| Cluster confidence distribution | 檢查模型對使用者分群的信心分布 |
| Mean cluster probability | 檢查各 cluster probability 的整體分布 |
| Cluster feature mean heatmap | 檢查 cluster label 與偏好特徵是否一致 |
| Interaction rating distribution | 檢查互動 rating 分布是否合理 |
| Recommendation score distribution | 檢查 personal、cluster、collaborative、hybrid 等推薦分數分布 |
| Recommendation method comparison | 檢查四種推薦方法與 random baseline 的比較結果 |
| User preference PCA by soft cluster | 以 PCA 降維方式觀察 cluster 在偏好空間中的分布 |

這些圖表可以作為報告中「資料處理、模型建立、測試檢查與結果分析」的補充證據。

需要注意：這些圖表主要檢查 pipeline 輸出結果；OSM API 實際抓取流程仍由 `fetch_real_attractions_osm.py` 負責。

---

### Step H: Run Streamlit App

```bash
streamlit run app.py
```

App 功能包含：

| 功能 | 說明 |
|---|---|
| 手動輸入使用者條件 | 使用 sidebar 輸入預算、時間、天氣、疲勞與景點偏好 |
| 使用既有使用者 | 從已產生資料中選擇 user ID |
| Soft Cluster | 顯示模型判定的 cluster |
| Cluster 類型 | 顯示根據 cluster feature mean 動態產生的 label |
| Cluster Confidence | 顯示最高 cluster probability |
| Soft Cluster Probability | 顯示使用者屬於各 cluster 的機率 |
| Low Confidence Warning | 當模型信心較低時顯示提醒 |
| 使用者輸入 / 行為特徵 | 顯示目前使用者輸入與偏好特徵 |
| Cluster-based 推薦景點 | 根據使用者所屬 cluster 的平均偏好推薦景點 |
| 推薦結果解釋 | 說明推薦分數與 cluster 平均偏好的關係 |
| Cluster 解釋 | 顯示該 cluster 高於或低於平均的特徵 |
| 分析圖表 | 顯示 heatmap、confidence distribution、method comparison 等 |

---

## Cluster 計算流程

本系統的 cluster 不是人工 if-else 規則，而是由 GMM 根據使用者景點偏好特徵計算機率。

流程如下：

```text
使用者輸入偏好特徵
→ 補缺失值 imputation
→ 標準化 StandardScaler
→ 輸入 Gaussian Mixture Model
→ 計算各 cluster probability
→ 取 probability 最大者作為 soft_cluster
→ 最大 probability 作為 cluster confidence
```

模型實際使用的欄位是：

```text
selected_indoor_score
selected_cost_level
selected_photo_value
selected_nature_value
selected_culture_value
selected_food_value
selected_popularity
```

---

## Collaborative-only 是什麼？

`collaborative_only` 是協同過濾概念的簡化示範。

它不是直接看這個使用者填了什麼偏好，而是看：

```text
跟他同一個 cluster 的其他使用者，對哪些景點互動評分較高
```

再推薦那些景點。

目前本專案尚未串接真實旅遊平台的點擊、收藏或評分資料，因此互動資料主要用於展示推薦方法流程，不應直接解讀為真實平台使用者行為。

---

## Cluster-only 與冷啟動輔助

`cluster_only` 的冷啟動輔助不是指「系統完全沒有資料也能推薦」。

比較精確的意思是：

```text
系統已經有一批訓練資料並建立 cluster；
但新使用者還沒有歷史點擊、評分或收藏紀錄時，
可以根據他填的少量偏好去比對既有 cluster，
再用該 cluster 的平均偏好提供初步推薦與解釋。
```

---

## 測試 Cluster 分類

可使用：

```bash
python test_cluster_inputs.py --preset --method gmm --show-prob
```

或：

```bash
python test_cluster_inputs.py --preset --method mean --show-prob
```

測試重點不是「是否剛好分到 cluster 0 或 cluster 1」，而是「label、probability 與 confidence 是否符合輸入偏好」。

---

## Optional: Analyze Data

```bash
python analyze_data.py
```

此步驟不是主流程必要步驟，用於單獨檢查 `travel_behavior.csv` 的資料分布與相關係數圖。

如果需要檢查完整 pipeline，建議使用：

```bash
python visualize_pipeline_data.py
```

---

## GitHub 注意事項

`data/`、`figures/`、`models/`、`pipeline_audit/` 屬於執行後產生的資料、圖片、模型檔或診斷報告。若不想把大量產物放到 GitHub，建議在 `.gitignore` 中排除：

```gitignore
data/
figures/
models/
pipeline_audit/
__pycache__/
*.pyc
venv/
.venv/
.env
.DS_Store
Thumbs.db
```

如果這些資料夾已經被 Git 追蹤，需要使用以下指令從 Git 追蹤中移除，但保留本機檔案：

```bash
git rm -r --cached data figures models pipeline_audit
```

之後再 commit 並 push。

---

## Notes

本專案目前適合作為：

```text
推薦系統流程展示
GMM soft clustering 解釋
cluster probability / confidence 展示
cluster heatmap 判讀
推薦方法比較
pipeline data audit 視覺化檢查
```

OpenStreetMap 可補充真實 POI 名稱與部分 metadata，但不提供完整旅遊平台所需的評分、人氣、票價、評論與即時狀態。因此，本系統不應直接解讀為真實商業旅遊平台的推薦成效。
