# ページ数表示問題の修正レポート

**実施日**: 2025年10月25日  
**修正バージョン**: app/main_v2.py

---

## 問題の原因

### 1. 10ページまでしか表示されない問題

**根本原因:**
- ダミーデータCSVファイルが10ページ分しか持っていない
- `filtered_df['page_num_dom'].max()` が常に10を返す
- LPキャプチャで16ページ取得しても、ダミーデータの範囲外のページは表示されない

**確認結果:**
```bash
$ python3 -c "import pandas as pd; df = pd.read_csv('dummy_data.csv'); \
  print('最大ページ数:', df['page_num_dom'].max()); \
  print('ユニークページ数:', df['page_num_dom'].nunique())"

最大ページ数: 10
ユニークページ数: 10
```

---

## 実施した修正

### 1. LPキャプチャのページ数を使用

**修正内容:**
```python
# 修正前: ダミーデータのページ数を使用
max_page = int(filtered_df['page_num_dom'].max()) if not filtered_df.empty else 10

# 修正後: LPキャプチャのページ数を使用
swipe_images = extract_swipe_lp_images(current_lp_url)
actual_page_count = len(swipe_images) if swipe_images else 10
```

**効果:**
- LPの実際のページ数（16ページ）を正確に取得
- ダミーデータの制限に依存しない
- 将来的に実データに置き換えても問題なし

---

### 2. ダミーデータにないページを追加

**修正内容:**
```python
# ダミーデータにないページを追加（ダミーデータが10ページまでしかない場合）
for page_num in range(1, actual_page_count + 1):
    if page_num not in page_stats['ページ番号'].values:
        # ダミーデータがないページはデフォルト値で追加
        new_row = pd.DataFrame([{
            'ページ番号': page_num,
            'ビュー数': 0,
            '平均滞在時間(ms)': 0,
            '平均逆行率': 0,
            '平均読込時間(ms)': 0,
            '平均滞在時間(秒)': 0,
            '離脱率': 0
        }])
        page_stats = pd.concat([page_stats, new_row], ignore_index=True)

# ページ番号でソート
page_stats = page_stats.sort_values('ページ番号').reset_index(drop=True)
```

**効果:**
- ページ11-16もexpanderとして表示される
- ダミーデータがないページは0値で表示（実データ導入後は実値に置き換わる）
- ページ順序が正しく保たれる

---

### 3. 複数LP選択時のエラー表示

**修正内容:**
```python
# ページ分析タブの冒頭に追加
if len(selected_lps) > 1:
    st.error("⚠️ ページ分析は1つのLPのみを対象とします。左側のフィルターでLPを1つだけ選択してください。")
    st.info("💡 現在選択されているLP: " + ", ".join(selected_lps))
    st.stop()  # ここで処理を停止
```

**効果:**
- 複数LP選択時に明確なエラーメッセージを表示
- 現在選択されているLPを表示
- 処理を停止して、誤った分析を防止

---

### 4. ページ数の情報表示

**修正内容:**
```python
# LPの実際のページ数を取得
actual_page_count = len(swipe_images) if swipe_images else 10
st.info(f"📊 このLPは {actual_page_count} ページで構成されています")
```

**効果:**
- ユーザーにLPのページ数を明示
- デバッグ情報としても有用

---

## 修正箇所

### app/main_v2.py

**1. 複数LP選択時のエラー表示（行571-575）**
```python
# 複数LP選択時のエラー表示
if len(selected_lps) > 1:
    st.error("⚠️ ページ分析は1つのLPのみを対象とします。左側のフィルターでLPを1つだけ選択してください。")
    st.info("💡 現在選択されているLP: " + ", ".join(selected_lps))
    st.stop()
```

**2. LPキャプチャを先に取得（行597-606）**
```python
# スワイプLPの各ページ画像を取得（ページ数を先に取得）
with st.spinner("📸 スワイプLPの画像を取得中..."):
    swipe_images = extract_swipe_lp_images(current_lp_url)

# LPの実際のページ数を取得
actual_page_count = len(swipe_images) if swipe_images else 10
st.info(f"📊 このLPは {actual_page_count} ページで構成されています")
```

**3. 離脱率計算をLPのページ数に合わせる（行608-617）**
```python
# 離脱率計算（LPの実際のページ数を使用）
page_exit = []
for page_num in range(1, actual_page_count + 1):
    reached = filtered_df[filtered_df['max_page_reached'] >= page_num]['session_id'].nunique()
    exited = filtered_df[filtered_df['max_page_reached'] == page_num]['session_id'].nunique()
    exit_rate = (exited / reached * 100) if reached > 0 else 0
    page_exit.append({'ページ番号': page_num, '離脱率': exit_rate})
```

**4. ダミーデータにないページを追加（行619-635）**
```python
# ダミーデータにないページを追加
for page_num in range(1, actual_page_count + 1):
    if page_num not in page_stats['ページ番号'].values:
        new_row = pd.DataFrame([{...}])
        page_stats = pd.concat([page_stats, new_row], ignore_index=True)

# ページ番号でソート
page_stats = page_stats.sort_values('ページ番号').reset_index(drop=True)
```

---

## 動作確認

### テストケース1: 単一LP選択時

**条件:**
- LP A のみ選択
- LP URL: https://shungene.lm-c.jp/tst08/tst08.html（16ページ）

**期待される動作:**
- ✅ 「このLPは 16 ページで構成されています」と表示
- ✅ ページ1-16のexpanderが表示される
- ✅ ページ1-10: ダミーデータの値が表示
- ✅ ページ11-16: 0値が表示（実データ導入後は実値に置き換わる）
- ✅ すべてのページにキャプチャが表示される

---

### テストケース2: 複数LP選択時

**条件:**
- LP A と LP B を選択

**期待される動作:**
- ✅ エラーメッセージが表示される
- ✅ 「ページ分析は1つのLPのみを対象とします」
- ✅ 「現在選択されているLP: LP A, LP B」
- ✅ ページ分析の処理が停止される

---

## LP URL の仕組み（本番環境）

### 現在の実装（プロトタイプ）

```python
lp_urls = {
    'LP A': 'https://shungene.lm-c.jp/tst08/tst08.html',
    'LP B': 'https://shungene.lm-c.jp/tst08/tst08.html',
    'LP C': 'https://shungene.lm-c.jp/tst08/tst08.html'
}

current_lp_url = lp_urls.get(selected_lps[0], 'https://shungene.lm-c.jp/tst08/tst08.html')
```

**特徴:**
- ハードコードされたURL
- すべてのLPが同じURL（デモ用）

---

### 本番環境での実装（予定）

**BigQueryから取得:**
```python
# BigQueryから各LPのURLを取得
query = """
SELECT DISTINCT
    lp_name,
    lp_url
FROM `project.dataset.lp_master`
WHERE is_active = TRUE
ORDER BY lp_name
"""

lp_urls = {}
for row in client.query(query):
    lp_urls[row['lp_name']] = row['lp_url']

# 選択されたLPのURLを取得
current_lp_url = lp_urls.get(selected_lps[0])
```

**特徴:**
- データベースから動的に取得
- 各LPが異なるURLを持つ
- 新しいLPの追加が容易

---

### キャプチャ表示の仕組み

**ご認識の通り:**

1. **単一LP選択時:**
   - ✅ 選択されたLPのURLを取得
   - ✅ そのURLからLPキャプチャを取得
   - ✅ ページごとにキャプチャを表示

2. **複数LP選択時:**
   - ✅ エラーメッセージを表示
   - ✅ 「LPを1つに絞ってください」
   - ✅ 処理を停止

**理由:**
- ページ分析は特定のLPの詳細分析
- 複数LPを同時に分析すると、どのページがどのLPのものか混乱する
- キャプチャも1つのLPのみを対象とする

---

## 今後の拡張

### 1. ダミーデータの拡張（オプション）

現在のダミーデータを16ページ分に拡張することも可能ですが、以下の理由で不要と判断:

- ✅ 実データ導入時にダミーデータは不要になる
- ✅ 現在の実装で11-16ページも表示される（0値で）
- ✅ キャプチャは正しく16ページすべて表示される

---

### 2. 実データ導入時の対応

**BigQuery API統合後:**
```python
# 実データから各LPのページ数を取得
query = f"""
SELECT 
    page_num_dom,
    COUNT(DISTINCT session_id) as view_count,
    AVG(stay_ms) as avg_stay_time,
    AVG(scroll_pct) as avg_scroll_pct,
    AVG(load_time_ms) as avg_load_time
FROM `project.dataset.page_events`
WHERE lp_url = '{current_lp_url}'
GROUP BY page_num_dom
ORDER BY page_num_dom
"""

page_stats = client.query(query).to_dataframe()
```

**効果:**
- すべてのページに実データが存在
- ダミーデータの制限がなくなる
- 正確な分析が可能

---

## まとめ

### 修正内容

1. ✅ **LPキャプチャのページ数を使用** - 16ページすべてを表示
2. ✅ **ダミーデータにないページを追加** - ページ11-16も表示（0値）
3. ✅ **複数LP選択時のエラー表示** - 明確なエラーメッセージ
4. ✅ **ページ数の情報表示** - 「このLPは 16 ページで構成されています」

### LP URL の仕組み

- ✅ **単一LP選択時**: 選択されたLPのURLからキャプチャを取得・表示
- ✅ **複数LP選択時**: エラーを表示し、LPを1つに絞るよう促す

### 本番環境への移行

- ✅ BigQueryから各LPのURLを動的に取得
- ✅ 実データ導入後、すべてのページに実値が表示される
- ✅ 現在の実装は本番環境にスムーズに移行可能

すべての修正が完了し、16ページすべてが正しく表示されるようになりました。

