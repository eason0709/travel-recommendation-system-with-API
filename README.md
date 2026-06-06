# Travel Recommendation System

本專案是一個以 **GMM soft clustering** 為核心的旅遊景點推薦與決策支援系統示範。系統會產生模擬使用者旅遊偏好資料、建立使用者偏好 cluster，並在 Streamlit App 中展示：

- 使用者輸入條件
- 使用者最接近的旅遊偏好 cluster
- cluster probability 與 confidence
- cluster-based 推薦景點
- cluster heatmap 與方法比較結果

本專案的重點不是建立完整商業旅遊平台，而是展示：

```text
使用者偏好 → 模型判斷偏好族群 → 推薦景點 → 顯示模型信心與解釋依據
```

---

## 目前版本重點

### 1. Cluster 已改為「偏好特徵」導向

最新版的 cluster 不再使用預算、天氣、疲勞、時間等情境欄位來決定使用者類型。

目前 GMM clustering 只使用以下 7 個景點偏好欄位：

```text
selected_indoor_score
selected_cost_level
selected_photo_value
selected_nature_value
selected_culture_value
selected_food_value
selected_popularity
```

這樣做的原因是：cluster 應該描述「使用者偏好哪一類景點」，而不是被預算、天氣、疲勞等情境因素蓋過。  
例如，一個自然景點偏好很高的使用者，不應該只因為預算或情境欄位而被分到高預算型或其他不相關類型。

情境欄位仍然會保留在資料與 App 中，例如：

```text
budget
available_time
weather_badness
social_context
fatigue
spontaneity
```

但它們目前主要用於展示與描述使用者情境，不用來訓練 cluster。

---

### 2. App 目前展示的是 cluster-based recommendation

最新版 App 的推薦區塊目前展示的是：

```text
Cluster-based 推薦景點
```

也就是：

```text
使用者輸入
→ GMM 判斷最接近的 cluster
→ 取該 cluster 的平均偏好
→ 根據 cluster 平均偏好推薦景點
```

請注意：目前 App 不是讓使用者在 `personal_only`、`cluster_only`、`hybrid` 之間切換。  
這些方法仍然會在離線評估腳本 `evaluate_clusters.py` 中比較，但 App 畫面目前主要展示 cluster-based 推薦與解釋流程。

---

### 3. Cluster label 改為動態產生

舊版 README 曾使用固定 cluster 名稱，例如：

```text
Cluster 0 = 低預算室內文化型
Cluster 1 = 好天氣一般消費型
```

最新版不再使用固定 cluster ID 對應固定名稱。

原因是 GMM 每次重訓後，cluster 編號可能改變。  
因此目前 cluster label 會根據 `soft_cluster_feature_mean.csv` 的標準化平均值動態產生，並輸出到：

```text
data/soft_cluster_label_summary.csv
```

所以 README 中不再列出固定的 `Cluster 0 ~ Cluster 7` 名稱表。  
判讀時應以實際執行後產生的 heatmap 和 `soft_cluster_label_summary.csv` 為準。

---

### 4. 方法比較不是用來宣稱某方法大勝

離線評估會比較：

```text
personal_only
cluster_only
collaborative_only
hybrid
```

目前結果通常顯示四種方法都高於 random baseline，且差異不一定很大。  
`personal_only` 往往略高，這是合理的，因為評估指標主要衡量推薦景點與使用者個人偏好的相似度。

比較正確的解讀是：

```text
personal_only：較適合直接個人化排序
cluster_only：較適合展示族群偏好與模型解釋
collaborative_only：展示「相似使用者也喜歡」的概念
hybrid：混合個人偏好、cluster 偏好與互動資訊
```

---

## 專案功能總覽

| 功能                                                 | 目前狀態   | 說明                                                         |
| ---------------------------------------------------- | ---------- | ------------------------------------------------------------ |
| 使用者資料生成                                       | 已實作     | 產生模擬使用者旅遊偏好與情境欄位                             |
| 偏好導向 GMM clustering                              | 已實作     | 只使用 7 個景點偏好欄位訓練 cluster                          |
| Cluster probability                                  | 已實作     | 顯示使用者屬於每個 cluster 的機率                            |
| Cluster confidence                                   | 已實作     | 使用最高 cluster probability 作為模型信心                    |
| Low confidence warning                               | 已實作     | 信心低時提醒不要過度解讀單一 cluster                         |
| Dynamic cluster label                                | 已實作     | 根據 cluster feature mean 自動命名                           |
| Cluster heatmap                                      | 已實作     | 顯示各 cluster 在各偏好特徵上的高低                          |
| 模擬景點資料                                         | 已實作     | 由 `generate_attractions.py` 產生                            |
| OpenStreetMap POI 匯入                               | 選配       | 可用 OSM 真實 POI 補充景點名稱與 metadata                    |
| Cluster-based App 推薦                               | 已實作     | App 目前展示此模式                                           |
| Personal / Cluster / Collaborative / Hybrid 離線比較 | 已實作     | 由 `recommendation_engine.py` 與 `evaluate_clusters.py` 完成 |
| App 內切換推薦模式                                   | 目前未保留 | 舊版 README 相關描述已移除                                   |
| 固定 Cluster 0~7 名稱表                              | 已移除     | 改用動態 label summary                                       |

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
├── test_cluster_inputs.py
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/       # 執行後產生
├── figures/    # 執行後產生
└── models/     # 執行後產生
```

---

## 檔案說明

| 檔案                             | 功能                                                                                                           |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `travel_utils.py`                | 共用工具檔，集中管理特徵欄位、cluster label、confidence 判讀與顯示名稱                                         |
| `generate_data.py`               | 產生模擬使用者旅遊偏好資料；新版資料生成會讓自然、美食、拍照、文化、室內、熱門等偏好更明確                     |
| `cluster_interpretation_soft.py` | 使用 GMM 進行 preference-based soft clustering，輸出 cluster probability、confidence、label summary 與 heatmap |
| `generate_attractions.py`        | 產生模擬景點資料                                                                                               |
| `fetch_real_attractions_osm.py`  | 從 OpenStreetMap Overpass API 抓取真實 POI，並轉換成本專案使用的景點資料格式                                   |
| `generate_user_interactions.py`  | 根據使用者偏好與景點特徵產生模擬 rating / clicked 互動資料                                                     |
| `recommendation_engine.py`       | 產生推薦候選清單與 Top-K 推薦結果，包含 personal、cluster、collaborative、hybrid 分數                          |
| `evaluate_clusters.py`           | 比較不同推薦方法與 random baseline 的差異                                                                      |
| `analyze_data.py`                | 選配，用於資料分布與相關係數分析                                                                               |
| `test_cluster_inputs.py`         | 用多組固定輸入測試 cluster 分類是否合理                                                                        |
| `app.py`                         | Streamlit 網頁展示系統                                                                                         |

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

其中：

| 套件           | 用途                                  |
| -------------- | ------------------------------------- |
| `scikit-learn` | GMM、standardization、imputation      |
| `streamlit`    | App 展示                              |
| `joblib`       | 儲存與載入模型                        |
| `pyarrow`      | 避免部分 pandas / parquet 讀取問題    |
| `requests`     | OpenStreetMap / Overpass API 資料抓取 |

---

## 執行順序

### 使用模擬景點資料

第一次執行建議依照以下流程：

```bash
python generate_data.py
python cluster_interpretation_soft.py
python generate_attractions.py
python generate_user_interactions.py
python recommendation_engine.py
python evaluate_clusters.py
streamlit run app.py
```

### 使用 OpenStreetMap 真實 POI 資料

如果要用 OSM 資料取代模擬景點資料，可以把 `generate_attractions.py` 換成：

```bash
python fetch_real_attractions_osm.py --area 臺北市 --max-items 300
```

完整流程如下：

```bash
python generate_data.py
python cluster_interpretation_soft.py
python fetch_real_attractions_osm.py --area 臺北市 --max-items 300
python generate_user_interactions.py
python recommendation_engine.py
python evaluate_clusters.py
streamlit run app.py
```

如果只想啟動 App，前提是 `data/`、`models/`、`figures/` 已經存在：

```bash
streamlit run app.py
```

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

主要欄位包含兩類。

#### 1. 情境欄位

```text
budget
available_time
weather_badness
social_context
fatigue
spontaneity
```

這些欄位描述使用者當下情境，但新版不使用它們訓練 cluster。

#### 2. 景點偏好欄位

```text
selected_indoor_score
selected_cost_level
selected_photo_value
selected_nature_value
selected_culture_value
selected_food_value
selected_popularity
```

這些欄位會用來訓練 GMM cluster，也會對應到景點資料中的特徵欄位。

---

### Step B: Soft Clustering and Cluster Interpretation

```bash
python cluster_interpretation_soft.py
```

此步驟會使用 GMM 進行 soft clustering，並輸出每位使用者屬於各 cluster 的機率。

最新版 clustering 使用的特徵只有：

```text
selected_indoor_score
selected_cost_level
selected_photo_value
selected_nature_value
selected_culture_value
selected_food_value
selected_popularity
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

| 欄位                      | 說明                                     |
| ------------------------- | ---------------------------------------- |
| `soft_cluster`            | 使用者最可能所屬的 cluster               |
| `soft_cluster_confidence` | 最高 cluster probability，也就是模型信心 |
| `cluster_prob_i`          | 使用者屬於第 i 個 cluster 的機率         |

---

## Heatmap 判讀規則

`soft_cluster_feature_heatmap.png` 顯示每個 cluster 在各偏好特徵上的標準化平均值。

建議用以下 threshold 判讀：

| 標記 |               條件 | 意思                            |
| ---- | -----------------: | ------------------------------- |
| 特高 |        `z >= +1.0` | 該 cluster 在此特徵明顯高於平均 |
| 偏高 | `+0.5 <= z < +1.0` | 稍微高於平均                    |
| 普通 |  `-0.5 < z < +0.5` | 接近平均，不要過度解讀          |
| 偏低 | `-1.0 < z <= -0.5` | 稍微低於平均                    |
| 特低 |        `z <= -1.0` | 該 cluster 在此特徵明顯低於平均 |

範例：

```text
Cluster A:
nature_value = +1.91 → 特高
indoor_score = -1.42 → 特低
cost_level = -1.31 → 特低

解讀：
此群使用者明顯偏好自然景點，
且不偏好室內與高消費景點。
```

---

### Step C1: Generate Simulated Attraction Data

```bash
python generate_attractions.py
```

輸出：

```text
data/attractions.csv
```

景點欄位包含：

```text
attraction_id
name
realistic_scene_name
attraction_type
cost_level
indoor_score
photo_value
nature_value
culture_value
food_value
popularity
estimated_time
```

`realistic_scene_name` 是展示用的現實場景式名稱，不是從真實旅遊平台爬取而來。

---

### Step C2: Optional OpenStreetMap POI Import

```bash
python fetch_real_attractions_osm.py --area 臺北市 --max-items 300
```

此步驟會使用 OpenStreetMap Overpass API 抓取真實 POI，並轉換成本專案的 `data/attractions.csv` 格式。

OSM 可提供的資料包含：

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

### OSM 資料清洗說明

OSM 的 `tourism=attraction` 標籤很泛用，可能混入不適合推薦的 POI，例如住宿、停車場、辦公室或標記不完整的地點。  
因此新版資料匯入流程會盡量排除：

```text
hotel
hostel
guest_house
motel
apartment
parking
toilets
office
泛用且資訊不足的 attraction
```

也會盡量保留較適合旅遊推薦的類型，例如：

```text
museum
gallery
park
viewpoint
aquarium
zoo
historic
monument
garden
library
theatre
cinema
restaurant
cafe
market
shopping
temple
```

仍需注意：OSM 匯入資料適合用來展示真實 POI 名稱與位置，但不能等同於完整旅遊平台資料。

---

### Step D: Generate User Interactions

```bash
python generate_user_interactions.py
```

輸出：

```text
data/user_interactions.csv
```

此步驟會根據使用者偏好向量與景點特徵相似度，產生模擬互動紀錄，例如：

```text
rating
clicked
```

這些互動資料是模擬產生的，不代表真實平台上的使用者評分或點擊。

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

| 分數                  | 說明                                              |
| --------------------- | ------------------------------------------------- |
| `personal_score`      | 使用者個人偏好向量與景點向量的相似度              |
| `cluster_score`       | 使用者所屬 cluster 平均偏好向量與景點向量的相似度 |
| `collaborative_score` | 同 cluster 使用者對景點的平均模擬 rating          |
| `hybrid_score`        | personal、cluster、collaborative 的混合分數       |

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

| 方法                 | 說明                                       |
| -------------------- | ------------------------------------------ |
| `personal_only`      | 直接根據使用者個人偏好推薦                 |
| `cluster_only`       | 根據使用者所屬 cluster 的平均偏好推薦      |
| `collaborative_only` | 根據同 cluster 使用者的模擬互動紀錄推薦    |
| `hybrid`             | 混合 personal、cluster、collaborative 分數 |

### 方法比較的正確解讀

目前結果通常顯示：

```text
四種方法都高於 random baseline
personal_only 通常略高
hybrid 接近 personal_only
cluster_only 與 collaborative_only 略低，但仍有參考價值
```

因此較合理的解讀是：

```text
personal_only 適合直接個人化排序；
cluster_only 提供族群偏好與解釋；
collaborative_only 展示相似使用者互動參考；
hybrid 嘗試在個人偏好與群體資訊之間平衡。
```

---

### Step G: Run Streamlit App

```bash
streamlit run app.py
```

App 功能包含：

| 功能                     | 說明                                                           |
| ------------------------ | -------------------------------------------------------------- |
| 手動輸入使用者條件       | 使用 sidebar 輸入預算、時間、天氣、疲勞與景點偏好              |
| 使用既有使用者           | 從已產生資料中選擇 user ID                                     |
| Soft Cluster             | 顯示模型判定的 cluster                                         |
| Cluster 類型             | 顯示根據 cluster feature mean 動態產生的 label                 |
| Cluster Confidence       | 顯示最高 cluster probability                                   |
| Soft Cluster Probability | 顯示使用者屬於各 cluster 的機率                                |
| Low Confidence Warning   | 當模型信心較低時顯示提醒                                       |
| Cluster 計算流程說明     | 解釋 imputer、scaler、GMM probability 與 confidence 的計算流程 |
| 使用者輸入 / 行為特徵    | 顯示目前使用者輸入與偏好特徵                                   |
| Cluster-based 推薦景點   | 根據使用者所屬 cluster 的平均偏好推薦景點                      |
| 推薦結果解釋             | 說明推薦分數與 cluster 平均偏好的關係                          |
| Cluster 解釋             | 顯示該 cluster 高於或低於平均的特徵                            |
| 分析圖表                 | 顯示 heatmap、confidence distribution、method comparison 等    |

目前 App 主要展示的是 cluster-based recommendation。  
若要展示 personal-only 或 hybrid 的排序結果，請使用 `recommendation_engine.py` 與 `evaluate_clusters.py` 的離線輸出，或自行擴充 App UI。

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

不使用下列欄位定義 cluster：

```text
budget
available_time
weather_badness
social_context
fatigue
spontaneity
```

當 `cluster confidence` 較低時，表示模型對單一 cluster 的判定不夠明確，使用者可能同時具有多個 cluster 的特徵。  
因此 App 會提醒使用者參考 Soft Cluster Probability，而不是只依賴單一 hard cluster。

---

## Collaborative-only 是什麼？

`collaborative_only` 是協同過濾概念的簡化示範。

它不是直接看這個使用者填了什麼偏好，而是看：

```text
跟他同一個 cluster 的其他使用者，過去對哪些景點評分較高
```

再推薦那些景點。

由於本專案沒有真實平台互動資料，所以 `collaborative_score` 是根據模擬出的 `user_interactions.csv` 計算。  
因此它主要用來展示推薦系統方法，不應被解讀為真實使用者行為結果。

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

也就是：

```text
完全沒有使用者資料 → 無法訓練 cluster
有初始資料，但新使用者沒有歷史紀錄 → cluster-only 可以輔助
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

建議測試的典型輸入包含：

| 測試類型       | 主要特徵                              |
| -------------- | ------------------------------------- |
| 自然景點導向型 | nature 高，indoor / food / culture 低 |
| 室內文化型     | indoor 高，culture 高，nature 低      |
| 美食導向型     | food 高                               |
| 拍照打卡型     | photo 高，popularity 高               |
| 高消費型       | cost 高                               |
| 熱門景點型     | popularity 高                         |
| 低消費輕旅行型 | cost 低，偏好不明顯                   |
| 中性模糊型     | 各項接近中間值，confidence 應較低     |

注意：GMM 的 cluster 編號每次重訓可能不同。  
測試重點不是「是否剛好分到 cluster 0 或 cluster 1」，而是「label 與 probability 是否符合輸入偏好」。

---

## 與舊版 README 的主要差異

本 README 已根據最新修改調整，移除或修正了下列舊版內容：

| 舊版內容                                  | 最新處理                                                     |
| ----------------------------------------- | ------------------------------------------------------------ |
| App 可選 `personal_only` / `cluster_only` | 已移除；目前 App 主要展示 cluster-based 推薦                 |
| 固定列出 `Cluster 0 ~ 7` 名稱             | 已移除；改用 `soft_cluster_label_summary.csv` 動態產生       |
| cluster 使用預算、天氣、疲勞等欄位        | 已修正；新版 cluster 只使用 7 個景點偏好欄位                 |
| 將 OSM 資料描述成完整真實旅遊平台資料     | 已修正；OSM 僅提供真實 POI 與 metadata，評分與人氣等仍為推估 |
| `personal_only` 明顯最好                  | 已修正；目前結果通常只是略高，四種方法差異不大               |
| collaborative-only 像真實使用者評分       | 已修正；目前互動資料為模擬產生                               |

---

## Notes

本專案目前仍以模擬資料為主。  
OpenStreetMap 可補充真實 POI 名稱與部分 metadata，但不提供完整旅遊平台所需的評分、人氣、票價、評論與即時狀態。

因此，本系統適合作為：

```text
推薦系統流程展示
GMM soft clustering 解釋
cluster probability / confidence 展示
cluster heatmap 判讀
推薦方法比較
```

不應直接解讀為真實商業旅遊平台的推薦成效。
