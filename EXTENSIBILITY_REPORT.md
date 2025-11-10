# スワイプLP分析ツール - 拡張性レポート

**作成日**: 2025年10月25日  
**バージョン**: app/main_v2.py, app/capture_lp.py

---

## 最終修正内容

### 1. キャプチャサイズをさらに縮小 ✅

**修正内容:**
```python
# 修正前
img_col, metric_col = st.columns([1, 3])  # キャプチャ:メトリクス = 1:3

# 修正後
img_col, metric_col = st.columns([1, 6])  # キャプチャ:メトリクス = 1:6
```

**効果:**
- キャプチャサイズが約半分に縮小
- メトリクス表示エリアがさらに拡大
- より多くの情報を一画面で確認可能

**修正箇所:**
- `app/main_v2.py` 行621

---

### 2. 10ページ固定を解除 ✅

**問題点:**
- ファネルデータと離脱率計算が `range(1, 11)` で固定
- 16ページのLPでも10ページまでしか表示されない

**修正内容:**
```python
# 実際のページ数を取得
max_page = int(filtered_df['page_num_dom'].max()) if not filtered_df.empty else 10

# ファネルデータ生成
for page_num in range(1, max_page + 1):
    count = filtered_df[filtered_df['max_page_reached'] >= page_num]['session_id'].nunique()
    funnel_data.append({'ページ': f'ページ{page_num}', 'セッション数': count})

# 離脱率計算
for page_num in range(1, max_page + 1):
    reached = filtered_df[filtered_df['max_page_reached'] >= page_num]['session_id'].nunique()
    exited = filtered_df[filtered_df['max_page_reached'] == page_num]['session_id'].nunique()
    exit_rate = (exited / reached * 100) if reached > 0 else 0
    page_exit.append({'ページ番号': page_num, '離脱率': exit_rate})
```

**効果:**
- 各LPのページ数に動的に対応
- 16ページのLPなら16ページすべてが表示
- 5ページのLPなら5ページのみ表示
- フォールバック: データがない場合は10ページとして処理

**修正箇所:**
- `app/main_v2.py` 行465-471（ファネル）
- `app/main_v2.py` 行592-600（離脱率）

---

## 骨組みの完成度と拡張性

### ✅ 完成している基盤

#### 1. データ生成・管理レイヤー
```python
# ダミーデータ生成関数
generate_dummy_data(
    lp_url, 
    start_date, 
    end_date, 
    num_sessions=1000, 
    num_pages=10
)
```

**特徴:**
- パラメータ化されたデータ生成
- 日付範囲、セッション数、ページ数を柔軟に設定
- BigQuery APIに置き換えやすい構造

**拡張性:** ⭐⭐⭐⭐⭐
- 新しいカラム追加が容易
- データソースの切り替えが簡単

---

#### 2. LPキャプチャレイヤー
```python
# LPキャプチャ関数
extract_swipe_lp_images(url)
```

**特徴:**
- lpSettings自動解析
- 動画・画像の自動認識
- ページ数の動的対応
- 存在確認機能

**拡張性:** ⭐⭐⭐⭐⭐
- 新しいメディアタイプ追加が容易
- サムネイル生成機能の追加が可能
- キャッシング戦略の改善が可能

---

#### 3. 可視化レイヤー
```python
# Plotlyを使ったグラフ描画
fig = px.bar(data, x='x', y='y', title='title')
st.plotly_chart(fig, use_container_width=True)
```

**特徴:**
- 統一されたグラフ描画パターン
- インタラクティブなグラフ
- レスポンシブデザイン

**拡張性:** ⭐⭐⭐⭐⭐
- 新しいグラフタイプの追加が容易
- カスタマイズが柔軟

---

#### 4. フィルター・セグメンテーションレイヤー
```python
# 日付フィルター
filtered_df = df[
    (df['event_date'] >= start_date) & 
    (df['event_date'] <= end_date)
]

# デバイスフィルター
if device_filter != "すべて":
    filtered_df = filtered_df[filtered_df['device'] == device_filter]
```

**特徴:**
- 段階的なフィルタリング
- 複数条件の組み合わせ
- パフォーマンス最適化

**拡張性:** ⭐⭐⭐⭐
- 新しいフィルター条件の追加が容易
- 複雑な条件の組み合わせが可能

---

#### 5. タブ構造・UI レイヤー
```python
# タブ構造
selected_analysis = st.selectbox(
    "分析タイプを選択",
    ["概要", "ページ別分析", "ボトルネック分析", ...]
)

if selected_analysis == "概要":
    # 概要タブの処理
elif selected_analysis == "ページ別分析":
    # ページ別分析タブの処理
```

**特徴:**
- 明確な分離
- 独立した処理
- 再利用可能なコンポーネント

**拡張性:** ⭐⭐⭐⭐⭐
- 新しいタブの追加が非常に容易
- 既存タブへの影響なし

---

## 新機能追加の難易度とリスク

### 🟢 低リスク・簡単（1-2時間）

#### 1. 新しいグラフの追加
**例:** 「時間帯別の離脱率」「ページ間遷移マトリックス」

**実装方法:**
```python
# 既存のデータから新しい視点で可視化
hourly_exit = filtered_df.groupby('hour')['exit_rate'].mean()
fig = px.line(hourly_exit, title='時間帯別離脱率')
st.plotly_chart(fig, use_container_width=True)
```

**リスク:** ⭐ (非常に低い)
- 既存データを使用
- 既存のグラフパターンを踏襲
- 他の機能への影響なし

---

#### 2. 新しいメトリクスの追加
**例:** 「平均ページ滞在時間の中央値」「直帰率」「エンゲージメント率」

**実装方法:**
```python
# 新しいメトリクスを計算
median_stay_time = filtered_df['stay_ms'].median()
bounce_rate = (bounced_sessions / total_sessions) * 100

# 表示
st.metric("直帰率", f"{bounce_rate:.1f}%")
```

**リスク:** ⭐ (非常に低い)
- 計算ロジックを追加するだけ
- 既存のメトリクス表示パターンを使用

---

#### 3. 新しいフィルター条件の追加
**例:** 「流入元別」「初回訪問/リピーター別」「コンバージョン達成/未達成別」

**実装方法:**
```python
# サイドバーに新しいフィルターを追加
source_filter = st.sidebar.selectbox(
    "流入元",
    ["すべて", "Google", "Facebook", "Twitter", "Direct"]
)

# フィルタリング
if source_filter != "すべて":
    filtered_df = filtered_df[filtered_df['source'] == source_filter]
```

**リスク:** ⭐ (非常に低い)
- 既存のフィルターパターンを踏襲
- データ構造への影響なし

---

#### 4. 新しい表示項目の追加
**例:** 「ページごとのCTA配置情報」「ヒートマップ」「スクリーンショット比較」

**実装方法:**
```python
# 新しいセクションを追加
st.markdown("#### CTA配置情報")
for page_num in range(1, max_page + 1):
    st.write(f"ページ{page_num}: CTAボタン位置 - 80%スクロール地点")
```

**リスク:** ⭐ (非常に低い)
- 独立したセクションとして追加
- 既存の表示ロジックへの影響なし

---

### 🟡 中リスク・中程度（3-5時間）

#### 1. 新しいタブの追加
**例:** 「ユーザーフロー分析」「リアルタイムダッシュボード」

**実装方法:**
```python
# タブリストに追加
analysis_types = [
    "概要", "ページ別分析", ..., "ユーザーフロー分析"  # 新規追加
]

# 処理を追加
elif selected_analysis == "ユーザーフロー分析":
    st.markdown("### ユーザーフロー分析")
    # 新しい分析ロジック
```

**リスク:** ⭐⭐ (低い)
- 既存タブと独立
- データ構造は共通
- テストが必要

---

#### 2. インタラクティブな機能の追加
**例:** 「ページごとの注釈機能」「カスタムセグメント保存」

**実装方法:**
```python
# セッションステートを使用
if 'annotations' not in st.session_state:
    st.session_state.annotations = {}

# 注釈入力
annotation = st.text_input(f"ページ{page_num}の注釈")
if st.button("保存"):
    st.session_state.annotations[page_num] = annotation
```

**リスク:** ⭐⭐ (低い)
- セッションステート管理が必要
- データ永続化を考慮する必要がある

---

#### 3. データエクスポート機能
**例:** 「CSV出力」「PDF レポート生成」「Excel出力」

**実装方法:**
```python
# CSVエクスポート
csv = filtered_df.to_csv(index=False)
st.download_button(
    "CSVダウンロード",
    csv,
    "analysis_report.csv",
    "text/csv"
)
```

**リスク:** ⭐⭐ (低い)
- Streamlitの標準機能を使用
- データ形式の変換が必要

---

### 🔴 高リスク・複雑（1-3日）

#### 1. API統合
**例:** 「BigQuery API」「Gemini 2.5 Pro API」

**実装方法:**
```python
from google.cloud import bigquery

@st.cache_data(ttl=3600)
def fetch_data_from_bigquery(query):
    client = bigquery.Client()
    df = client.query(query).to_dataframe()
    return df

# ダミーデータ生成を置き換え
df = fetch_data_from_bigquery(query)
```

**リスク:** ⭐⭐⭐ (中程度)
- 認証・権限管理が必要
- エラーハンドリングが重要
- コスト管理が必要
- テストが複雑

---

#### 2. データ構造の変更
**例:** 「カラム名の変更」「データ型の変更」「新しいテーブルの追加」

**実装方法:**
```python
# カラム名変更
df = df.rename(columns={'old_name': 'new_name'})

# すべてのグラフ・計算ロジックを更新
# 影響範囲が広い
```

**リスク:** ⭐⭐⭐⭐ (高い)
- 既存のすべてのコードに影響
- 広範囲なテストが必要
- バグが発生しやすい

---

#### 3. パフォーマンス最適化
**例:** 「キャッシング戦略の変更」「並列処理の導入」

**実装方法:**
```python
# キャッシュ戦略の変更
@st.cache_data(ttl=1800, max_entries=100)
def expensive_computation(params):
    # 重い計算
    return result

# 並列処理
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=4) as executor:
    results = executor.map(process_page, pages)
```

**リスク:** ⭐⭐⭐ (中程度)
- パフォーマンスへの影響を測定する必要がある
- デバッグが難しい
- 副作用に注意

---

## 推奨される拡張の優先順位

### フェーズ1: 低リスク・高価値（即座に実装可能）

1. **新しいメトリクスの追加**
   - 直帰率
   - エンゲージメント率
   - ページ滞在時間の中央値

2. **新しいグラフの追加**
   - 時間帯別離脱率
   - 曜日別コンバージョン率
   - ページ間遷移サンキーダイアグラム

3. **フィルター条件の拡張**
   - 流入元別フィルター
   - 初回訪問/リピーター別フィルター
   - コンバージョン達成/未達成別フィルター

---

### フェーズ2: 中リスク・高価値（1週間以内）

1. **データエクスポート機能**
   - CSV出力
   - PDFレポート生成
   - 画像一括ダウンロード

2. **インタラクティブ機能**
   - ページごとの注釈機能
   - カスタムセグメント保存
   - ブックマーク機能

3. **新しいタブの追加**
   - ユーザーフロー分析
   - ヒートマップ分析
   - コホート分析

---

### フェーズ3: 高リスク・高価値（2-4週間）

1. **BigQuery API統合**
   - 実データ取得
   - クエリ最適化
   - コスト管理

2. **Gemini 2.5 Pro API統合**
   - 実AI分析
   - プロンプトエンジニアリング
   - レスポンス処理

3. **パフォーマンス最適化**
   - キャッシング戦略の改善
   - 並列処理の導入
   - データベース最適化

---

## 拡張時のベストプラクティス

### 1. 段階的な実装
```python
# ❌ 悪い例: 一度に大きな変更
def new_feature():
    # 100行の新しいコード
    # データ構造も変更
    # 既存のコードも修正
    pass

# ✅ 良い例: 小さな変更を積み重ねる
def new_metric_calculation():
    # 新しいメトリクスの計算のみ
    return metric

def display_new_metric():
    # 表示ロジックのみ
    st.metric("新しいメトリクス", value)
```

---

### 2. 既存パターンの踏襲
```python
# ✅ 既存のグラフパターンを使用
fig = px.bar(data, x='x', y='y', title='新しいグラフ')
fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
fig.update_layout(height=400)
st.plotly_chart(fig, use_container_width=True, key='plotly_chart_new')
```

---

### 3. テストの実施
```python
# テストデータで確認
test_df = generate_dummy_data(
    lp_url="https://example.com",
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 1, 31),
    num_sessions=100,
    num_pages=5
)

# 新機能をテスト
result = new_feature(test_df)
assert result is not None
assert len(result) > 0
```

---

### 4. ドキュメント化
```python
def new_analysis_function(df: pd.DataFrame, param: str) -> pd.DataFrame:
    """
    新しい分析機能
    
    Args:
        df: 分析対象のDataFrame
        param: パラメータの説明
    
    Returns:
        分析結果のDataFrame
    
    Example:
        >>> result = new_analysis_function(df, "example")
        >>> print(result.head())
    """
    # 実装
    return result
```

---

## まとめ

### 骨組みの完成度: ⭐⭐⭐⭐⭐

現在の実装は、以下の点で非常に優れた基盤となっています:

1. **モジュール化された構造**: 各レイヤーが独立
2. **拡張性の高い設計**: 新機能の追加が容易
3. **統一されたパターン**: コードの一貫性が高い
4. **柔軟なデータ処理**: データソースの切り替えが簡単

### 拡張の容易さ

- **新しいグラフ・メトリクス**: ⭐⭐⭐⭐⭐ (非常に容易)
- **新しいフィルター**: ⭐⭐⭐⭐⭐ (非常に容易)
- **新しいタブ**: ⭐⭐⭐⭐ (容易)
- **API統合**: ⭐⭐⭐ (中程度)
- **データ構造変更**: ⭐⭐ (慎重に対応が必要)

### 推奨される次のステップ

1. **即座に実装可能（今日中）**
   - 新しいメトリクス追加（直帰率、エンゲージメント率）
   - 新しいグラフ追加（時間帯別離脱率）

2. **1週間以内**
   - データエクスポート機能（CSV、PDF）
   - 新しいタブ追加（ユーザーフロー分析）

3. **2-4週間**
   - BigQuery API統合
   - Gemini 2.5 Pro API統合

---

**結論:**

ご認識の通り、**骨組みは完成しており、新しいグラフや表示項目を増やすのはバグの危険性が低く、軽めの修正で済みます**。

既存のパターンを踏襲し、段階的に実装していけば、安全かつ効率的に機能を拡張できます。

