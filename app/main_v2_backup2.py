"""
瞬ジェネ AIアナリスト - Step 2完成版
グラフ説明と比較機能を追加
50項目以上の分析・グラフを実装
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# ページ設定
st.set_page_config(
    page_title="瞬ジェネ AIアナリスト",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #002060;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #002060;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .graph-description {
        color: #666;
        font-size: 0.9rem;
        margin-bottom: 1rem;
        padding: 0.5rem;
        background-color: #f8f9fa;
        border-radius: 0.3rem;
        border-left: 3px solid #002060;
    }
</style>
""", unsafe_allow_html=True)

# データ読み込み
@st.cache_data
def load_data():
    """ダミーデータを読み込む"""
    df = pd.read_csv("/home/ubuntu/swipe_lp_analyzer/app/dummy_data.csv")
    df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])
    df['event_date'] = pd.to_datetime(df['event_date'])
    return df

# 比較期間のデータを取得する関数
def get_comparison_data(df, current_start, current_end, comparison_type):
    """
    比較期間のデータを取得
    comparison_type: 'previous_period', 'previous_week', 'previous_month', 'previous_year'
    """
    period_length = (current_end - current_start).days
    
    if comparison_type == 'previous_period':
        comp_end = current_start - timedelta(days=1)
        comp_start = comp_end - timedelta(days=period_length)
    elif comparison_type == 'previous_week':
        comp_end = current_end - timedelta(weeks=1)
        comp_start = current_start - timedelta(weeks=1)
    elif comparison_type == 'previous_month':
        comp_end = current_end - timedelta(days=30)
        comp_start = current_start - timedelta(days=30)
    elif comparison_type == 'previous_year':
        comp_end = current_end - timedelta(days=365)
        comp_start = current_start - timedelta(days=365)
    else:
        return None
    
    comparison_df = df[(df['event_date'] >= comp_start) & (df['event_date'] <= comp_end)]
    return comparison_df, comp_start, comp_end

# データ読み込み
df = load_data()

# サイドバー: タイトル
st.sidebar.markdown('<h1 style="color: #002060; font-size: 1.8rem; font-weight: bold; margin-bottom: 1rem;">瞬ジェネ AIアナリスト</h1>', unsafe_allow_html=True)
st.sidebar.markdown("---")

# サイドバー: フィルター
st.sidebar.header("フィルター設定")

# 期間選択（GA4スタイル）
period_options = {
    "過去7日間": 7,
    "過去30日間": 30,
    "過去90日間": 90,
    "カスタム期間": None
}

selected_period = st.sidebar.selectbox("期間を選択", list(period_options.keys()), index=1)

if selected_period == "カスタム期間":
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("開始日", df['event_date'].min())
    with col2:
        end_date = st.date_input("終了日", df['event_date'].max())
else:
    days = period_options[selected_period]
    end_date = df['event_date'].max()
    start_date = end_date - timedelta(days=days)

# LP選択
lp_options = ["すべて"] + sorted(df['page_location'].dropna().unique().tolist())
selected_lps = st.sidebar.multiselect("LP選択", lp_options, default=["すべて"])

# チャネル選択
channel_map = {
    "google": "Organic Search",
    "facebook": "Organic Social",
    "instagram": "Organic Social",
    "twitter": "Organic Social",
    "direct": "Direct"
}
df['channel'] = df['utm_source'].map(channel_map).fillna("Referral")
channel_options = ["すべて"] + sorted(df['channel'].unique().tolist())
selected_channels = st.sidebar.multiselect("チャネル", channel_options, default=["すべて"])

# デバイス選択
device_options = ["すべて"] + sorted(df['device_type'].dropna().unique().tolist())
selected_devices = st.sidebar.multiselect("デバイス", device_options, default=["すべて"])

# A/Bテスト選択
ab_options = ["すべて"] + sorted(df['ab_variant'].dropna().unique().tolist())
selected_ab = st.sidebar.multiselect("A/Bテスト", ab_options, default=["すべて"])

# 比較機能の設定
st.sidebar.markdown("---")
st.sidebar.header("比較設定")
enable_comparison = st.sidebar.checkbox("比較機能を有効化", value=False)
comparison_type = None
if enable_comparison:
    comparison_options = {
        "前期間": "previous_period",
        "前週": "previous_week",
        "前月": "previous_month",
        "前年": "previous_year"
    }
    selected_comparison = st.sidebar.selectbox("比較対象", list(comparison_options.keys()))
    comparison_type = comparison_options[selected_comparison]

# データフィルタリング
filtered_df = df.copy()

# 期間フィルター
filtered_df = filtered_df[
    (filtered_df['event_date'] >= pd.to_datetime(start_date)) &
    (filtered_df['event_date'] <= pd.to_datetime(end_date))
]

# LPフィルター
if "すべて" not in selected_lps:
    filtered_df = filtered_df[filtered_df['page_location'].isin(selected_lps)]

# チャネルフィルター
if "すべて" not in selected_channels:
    filtered_df = filtered_df[filtered_df['channel'].isin(selected_channels)]

# デバイスフィルター
if "すべて" not in selected_devices:
    filtered_df = filtered_df[filtered_df['device_type'].isin(selected_devices)]

# A/Bテストフィルター
if "すべて" not in selected_ab:
    filtered_df = filtered_df[filtered_df['ab_variant'].isin(selected_ab)]

# 比較データの取得
comparison_df = None
comp_start = None
comp_end = None
if enable_comparison and comparison_type:
    result = get_comparison_data(df, pd.Timestamp(start_date), pd.Timestamp(end_date), comparison_type)
    if result:
        comparison_df, comp_start, comp_end = result
        # 比較データにも同じフィルターを適用
        if "すべて" not in selected_lps:
            comparison_df = comparison_df[comparison_df['page_location'].isin(selected_lps)]
        if "すべて" not in selected_channels:
            comparison_df = comparison_df[comparison_df['channel'].isin(selected_channels)]
        if "すべて" not in selected_devices:
            comparison_df = comparison_df[comparison_df['device_type'].isin(selected_devices)]
        if "すべて" not in selected_ab:
            comparison_df = comparison_df[comparison_df['ab_variant'].isin(selected_ab)]

# データが空の場合の処理
if len(filtered_df) == 0:
    st.warning("⚠️ 選択した条件に該当するデータがありません。フィルターを変更してください。")
    st.stop()

# 基本メトリクス計算
total_sessions = filtered_df['session_id'].nunique()
total_conversions = filtered_df[filtered_df['cv_type'].notna()]['session_id'].nunique()
conversion_rate = (total_conversions / total_sessions * 100) if total_sessions > 0 else 0
total_clicks = len(filtered_df[filtered_df['event_name'] == 'click'])
click_rate = (total_clicks / total_sessions * 100) if total_sessions > 0 else 0
avg_stay_time = filtered_df['stay_ms'].mean() / 1000  # 秒に変換
avg_pages_reached = filtered_df.groupby('session_id')['max_page_reached'].max().mean()
fv_retention_rate = (filtered_df[filtered_df['max_page_reached'] >= 2]['session_id'].nunique() / total_sessions * 100) if total_sessions > 0 else 0
final_cta_rate = (filtered_df[filtered_df['max_page_reached'] >= 10]['session_id'].nunique() / total_sessions * 100) if total_sessions > 0 else 0
avg_load_time = filtered_df['load_time_ms'].mean()

# KPI表示
st.markdown('<div class="sub-header">📈 主要指標（KPI）</div>', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("セッション数", f"{total_sessions:,}")
    st.metric("コンバージョン数", f"{total_conversions:,}")

with col2:
    st.metric("コンバージョン率", f"{conversion_rate:.2f}%")
    st.metric("クリック数", f"{total_clicks:,}")

with col3:
    st.metric("クリック率", f"{click_rate:.2f}%")
    st.metric("平均到達ページ数", f"{avg_pages_reached:.1f}")

with col4:
    st.metric("平均滞在時間", f"{avg_stay_time:.1f}秒")
    st.metric("FV残存率", f"{fv_retention_rate:.1f}%")

with col5:
    st.metric("最終CTA到達率", f"{final_cta_rate:.1f}%")
    st.metric("平均読込時間", f"{avg_load_time:.0f}ms")

# タブ作成
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
    "全体分析",
    "ページ分析",
    "セグメント分析",
    "A/Bテスト分析",
    "動画・スクロール分析",
    "時系列分析",
    "リアルタイム分析",
    "カスタムオーディエンス",
    "AI提案",
    "使用ガイド",
    "専門用語解説"
])

# タブ1: 全体分析
with tab1:
    st.markdown('<div class="sub-header">全体分析</div>', unsafe_allow_html=True)
    
    # グラフ選択
    st.markdown("**表示するグラフを選択してください:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_session_trend = st.checkbox("セッション数の推移", value=True)
        show_cvr_trend = st.checkbox("コンバージョン率の推移", value=True)
        show_device_breakdown = st.checkbox("デバイス別分析", value=True)
    
    with col2:
        show_channel_breakdown = st.checkbox("チャネル別分析", value=True)
        show_funnel = st.checkbox("LP進行ファネル", value=True)
        show_hourly_cvr = st.checkbox("時間帯別CVR", value=False)
    
    with col3:
        show_dow_cvr = st.checkbox("曜日別CVR", value=False)
        show_utm_analysis = st.checkbox("UTM分析", value=False)
        show_load_time = st.checkbox("読込時間分析", value=False)
    
    # セッション数の推移
    if show_session_trend:
        st.markdown("#### セッション数の推移")
        st.markdown('<div class="graph-description">日ごとのセッション数（訪問数）の変化を表示します。トレンドや曜日ごとのパターンを把握できます。</div>', unsafe_allow_html=True)
        daily_sessions = filtered_df.groupby(filtered_df['event_date'].dt.date)['session_id'].nunique().reset_index()
        daily_sessions.columns = ['日付', 'セッション数']
        
        if comparison_df is not None and len(comparison_df) > 0:
            # 比較データを追加
            comp_daily_sessions = comparison_df.groupby(comparison_df['event_date'].dt.date)['session_id'].nunique().reset_index()
            comp_daily_sessions.columns = ['日付', '比較期間セッション数']
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=daily_sessions['日付'], y=daily_sessions['セッション数'], 
                                    mode='lines+markers', name='現在期間', line=dict(color='#002060')))
            fig.add_trace(go.Scatter(x=comp_daily_sessions['日付'], y=comp_daily_sessions['比較期間セッション数'], 
                                    mode='lines+markers', name='比較期間', line=dict(color='#999999', dash='dash')))
            fig.update_layout(height=400, hovermode='x unified')
        else:
            fig = px.line(daily_sessions, x='日付', y='セッション数', markers=True)
            fig.update_layout(height=400)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # コンバージョン率の推移
    if show_cvr_trend:
        st.markdown("#### コンバージョン率の推移")
        st.markdown('<div class="graph-description">日ごとのコンバージョン率（CVR）の変化を表示します。LPの改善効果や外部要因の影響を確認できます。</div>', unsafe_allow_html=True)
        daily_cvr = filtered_df.groupby(filtered_df['event_date'].dt.date).agg({
            'session_id': 'nunique',
        }).reset_index()
        daily_cvr.columns = ['日付', 'セッション数']
        
        daily_cv = filtered_df[filtered_df['cv_type'].notna()].groupby(
            filtered_df[filtered_df['cv_type'].notna()]['event_date'].dt.date
        )['session_id'].nunique().reset_index()
        daily_cv.columns = ['日付', 'コンバージョン数']
        
        daily_cvr = daily_cvr.merge(daily_cv, on='日付', how='left').fillna(0)
        daily_cvr['コンバージョン率'] = (daily_cvr['コンバージョン数'] / daily_cvr['セッション数'] * 100)
        
        if comparison_df is not None and len(comparison_df) > 0:
            # 比較データを追加
            comp_daily_cvr = comparison_df.groupby(comparison_df['event_date'].dt.date).agg({'session_id': 'nunique'}).reset_index()
            comp_daily_cvr.columns = ['日付', 'セッション数']
            
            comp_daily_cv = comparison_df[comparison_df['cv_type'].notna()].groupby(
                comparison_df[comparison_df['cv_type'].notna()]['event_date'].dt.date
            )['session_id'].nunique().reset_index()
            comp_daily_cv.columns = ['日付', 'コンバージョン数']
            
            comp_daily_cvr = comp_daily_cvr.merge(comp_daily_cv, on='日付', how='left').fillna(0)
            comp_daily_cvr['比較期間CVR'] = (comp_daily_cvr['コンバージョン数'] / comp_daily_cvr['セッション数'] * 100)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=daily_cvr['日付'], y=daily_cvr['コンバージョン率'], 
                                    mode='lines+markers', name='現在期間', line=dict(color='#002060')))
            fig.add_trace(go.Scatter(x=comp_daily_cvr['日付'], y=comp_daily_cvr['比較期間CVR'], 
                                    mode='lines+markers', name='比較期間', line=dict(color='#999999', dash='dash')))
            fig.update_layout(height=400, hovermode='x unified', yaxis_title='コンバージョン率 (%)')
        else:
            fig = px.line(daily_cvr, x='日付', y='コンバージョン率', markers=True)
            fig.update_layout(height=400)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # デバイス別分析
    if show_device_breakdown:
        st.markdown("#### デバイス別分析")
        st.markdown('<div class="graph-description">デバイス（スマホ、PC、タブレット）ごとのセッション数、コンバージョン数、CVRを比較します。デバイス最適化の優先度を判断できます。</div>', unsafe_allow_html=True)
        device_stats = filtered_df.groupby('device_type').agg({
            'session_id': 'nunique',
        }).reset_index()
        device_stats.columns = ['デバイス', 'セッション数']
        
        device_cv = filtered_df[filtered_df['cv_type'].notna()].groupby('device_type')['session_id'].nunique().reset_index()
        device_cv.columns = ['デバイス', 'コンバージョン数']
        
        device_stats = device_stats.merge(device_cv, on='デバイス', how='left').fillna(0)
        device_stats['コンバージョン率'] = (device_stats['コンバージョン数'] / device_stats['セッション数'] * 100)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='セッション数', x=device_stats['デバイス'], y=device_stats['セッション数'], yaxis='y', offsetgroup=1))
        fig.add_trace(go.Bar(name='コンバージョン数', x=device_stats['デバイス'], y=device_stats['コンバージョン数'], yaxis='y', offsetgroup=2))
        fig.add_trace(go.Scatter(name='コンバージョン率', x=device_stats['デバイス'], y=device_stats['コンバージョン率'], yaxis='y2', mode='lines+markers'))
        
        fig.update_layout(
            yaxis=dict(title='セッション数 / コンバージョン数'),
            yaxis2=dict(title='コンバージョン率 (%)', overlaying='y', side='right'),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # チャネル別分析
    if show_channel_breakdown:
        st.markdown("#### チャネル別分析")
        st.markdown('<div class="graph-description">流入経路（Google、SNS、直接アクセスなど）ごとのパフォーマンスを比較します。効果的な集客チャネルを特定できます。</div>', unsafe_allow_html=True)
        channel_stats = filtered_df.groupby('channel').agg({
            'session_id': 'nunique',
            'stay_ms': 'mean'
        }).reset_index()
        channel_stats.columns = ['チャネル', 'セッション数', '平均滞在時間(ms)']
        channel_stats['平均滞在時間(秒)'] = channel_stats['平均滞在時間(ms)'] / 1000
        
        channel_cv = filtered_df[filtered_df['cv_type'].notna()].groupby('channel')['session_id'].nunique().reset_index()
        channel_cv.columns = ['チャネル', 'コンバージョン数']
        
        channel_stats = channel_stats.merge(channel_cv, on='チャネル', how='left').fillna(0)
        channel_stats['コンバージョン率'] = (channel_stats['コンバージョン数'] / channel_stats['セッション数'] * 100)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(channel_stats, values='セッション数', names='チャネル', title='チャネル別セッション数')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(channel_stats, x='チャネル', y='コンバージョン率', title='チャネル別コンバージョン率')
            st.plotly_chart(fig, use_container_width=True)
    
    # LP進行ファネル
    if show_funnel:
        st.markdown("#### LP進行ファネル")
        st.markdown('<div class="graph-description">ユーザーがLPを進む過程で、各ページでどれだけ離脱したかを可視化します。ボトルネックとなるページを特定できます。</div>', unsafe_allow_html=True)
        funnel_data = []
        for page_num in range(1, 11):
            count = filtered_df[filtered_df['max_page_reached'] >= page_num]['session_id'].nunique()
            funnel_data.append({'ページ': f'ページ{page_num}', 'セッション数': count})
        
        funnel_df = pd.DataFrame(funnel_data)
        
        fig = go.Figure(go.Funnel(
            y=funnel_df['ページ'],
            x=funnel_df['セッション数'],
            textinfo="value+percent initial"
        ))
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    # 時間帯別CVR
    if show_hourly_cvr:
        st.markdown("#### 時間帯別コンバージョン率")
        st.markdown('<div class="graph-description">1日の中で、どの時間帯にCVRが高いかを分析します。広告配信の最適な時間帯を見つけることができます。</div>', unsafe_allow_html=True)
        filtered_df['hour'] = filtered_df['event_timestamp'].dt.hour
        
        hourly_sessions = filtered_df.groupby('hour')['session_id'].nunique().reset_index()
        hourly_sessions.columns = ['時間', 'セッション数']
        
        hourly_cv = filtered_df[filtered_df['cv_type'].notna()].groupby('hour')['session_id'].nunique().reset_index()
        hourly_cv.columns = ['時間', 'コンバージョン数']
        
        hourly_cvr = hourly_sessions.merge(hourly_cv, on='時間', how='left').fillna(0)
        hourly_cvr['コンバージョン率'] = (hourly_cvr['コンバージョン数'] / hourly_cvr['セッション数'] * 100)
        
        fig = px.bar(hourly_cvr, x='時間', y='コンバージョン率')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # 曜日別CVR
    if show_dow_cvr:
        st.markdown("#### 曜日別コンバージョン率")
        st.markdown('<div class="graph-description">曜日ごとのCVRの違いを分析します。平日と週末でのユーザー行動の変化を把握できます。</div>', unsafe_allow_html=True)
        filtered_df['dow'] = filtered_df['event_timestamp'].dt.day_name()
        dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_map = {'Monday': '月', 'Tuesday': '火', 'Wednesday': '水', 'Thursday': '木', 'Friday': '金', 'Saturday': '土', 'Sunday': '日'}
        
        dow_sessions = filtered_df.groupby('dow')['session_id'].nunique().reset_index()
        dow_sessions.columns = ['曜日', 'セッション数']
        
        dow_cv = filtered_df[filtered_df['cv_type'].notna()].groupby('dow')['session_id'].nunique().reset_index()
        dow_cv.columns = ['曜日', 'コンバージョン数']
        
        dow_cvr = dow_sessions.merge(dow_cv, on='曜日', how='left').fillna(0)
        dow_cvr['コンバージョン率'] = (dow_cvr['コンバージョン数'] / dow_cvr['セッション数'] * 100)
        dow_cvr['曜日_日本語'] = dow_cvr['曜日'].map(dow_map)
        dow_cvr['曜日_order'] = dow_cvr['曜日'].apply(lambda x: dow_order.index(x))
        dow_cvr = dow_cvr.sort_values('曜日_order')
        
        fig = px.bar(dow_cvr, x='曜日_日本語', y='コンバージョン率')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # UTM分析
    if show_utm_analysis:
        st.markdown("#### UTM分析")
        st.markdown('<div class="graph-description">UTMパラメータ（広告タグ）ごとのトラフィックを分析します。どのキャンペーンや媒体が効果的かを把握できます。</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**UTMソース別**")
            utm_source_stats = filtered_df.groupby('utm_source')['session_id'].nunique().reset_index()
            utm_source_stats.columns = ['UTMソース', 'セッション数']
            utm_source_stats = utm_source_stats.sort_values('セッション数', ascending=False)
            
            fig = px.bar(utm_source_stats, x='UTMソース', y='セッション数')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**UTMメディア別**")
            utm_medium_stats = filtered_df.groupby('utm_medium')['session_id'].nunique().reset_index()
            utm_medium_stats.columns = ['UTMメディア', 'セッション数']
            utm_medium_stats = utm_medium_stats.sort_values('セッション数', ascending=False)
            
            fig = px.bar(utm_medium_stats, x='UTMメディア', y='セッション数')
            st.plotly_chart(fig, use_container_width=True)
    
    # 読込時間分析
    if show_load_time:
        st.markdown("#### 読込時間分析")
        st.markdown('<div class="graph-description">デバイスごとのページ読込時間を分析します。読込が遅いと離脱率が上がるため、最適化が重要です。</div>', unsafe_allow_html=True)
        
        load_time_stats = filtered_df.groupby('device_type')['load_time_ms'].mean().reset_index()
        load_time_stats.columns = ['デバイス', '平均読込時間(ms)']
        
        fig = px.bar(load_time_stats, x='デバイス', y='平均読込時間(ms)')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# 続く...（次のファイルでタブ2以降を実装）



# タブ2: ページ分析
with tab2:
    st.markdown('<div class="sub-header">📄 ページ分析</div>', unsafe_allow_html=True)
    
    # ページ別メトリクス計算
    page_stats = filtered_df.groupby('page_num_dom').agg({
        'session_id': 'nunique',
        'stay_ms': 'mean',
        'scroll_pct': 'mean',
        'load_time_ms': 'mean'
    }).reset_index()
    page_stats.columns = ['ページ番号', 'ビュー数', '平均滞在時間(ms)', '平均スクロール率', '平均読込時間(ms)']
    page_stats['平均滞在時間(秒)'] = page_stats['平均滞在時間(ms)'] / 1000
    
    # 離脱率計算
    page_exit = []
    for page_num in range(1, 11):
        reached = filtered_df[filtered_df['max_page_reached'] >= page_num]['session_id'].nunique()
        exited = filtered_df[filtered_df['max_page_reached'] == page_num]['session_id'].nunique()
        exit_rate = (exited / reached * 100) if reached > 0 else 0
        page_exit.append({'ページ番号': page_num, '離脱率': exit_rate})
    
    page_exit_df = pd.DataFrame(page_exit)
    page_stats = page_stats.merge(page_exit_df, on='ページ番号', how='left')
    
    # ページ画像プレースホルダー
    page_stats['画像URL'] = page_stats['ページ番号'].apply(lambda x: f"https://via.placeholder.com/300x400?text=Page+{x}")
    
    st.markdown("#### ページ別パフォーマンス一覧")
    st.markdown('<div class="graph-description">各ページのビュー数、滞在時間、離脱率、スクロール率、読込時間を詳細に分析します。ページ画像と一緒に確認することで、問題のあるコンテンツを特定しやすくなります。</div>', unsafe_allow_html=True)
    
    # ページ一覧表示（画像付き）
    for idx, row in page_stats.iterrows():
        with st.expander(f"📄 ページ {int(row['ページ番号'])} - ビュー数: {int(row['ビュー数'])}"):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(row['画像URL'], caption=f"ページ {int(row['ページ番号'])}")
            
            with col2:
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.metric("ビュー数", f"{int(row['ビュー数'])}")
                    st.metric("平均滞在時間", f"{row['平均滞在時間(秒)']:.1f}秒")
                
                with metric_col2:
                    st.metric("離脱率", f"{row['離脱率']:.1f}%")
                    st.metric("平均スクロール率", f"{row['平均スクロール率']*100:.1f}%")
                
                with metric_col3:
                    st.metric("平均読込時間", f"{row['平均読込時間(ms)']:.0f}ms")
    
    st.markdown("---")
    
    # 滞在時間が短いページ
    st.markdown("#### 滞在時間が短いページ TOP5")
    st.markdown('<div class="graph-description">滞在時間が特に短いページを識別します。コンテンツが魅力的でない、または読みづらい可能性があります。</div>', unsafe_allow_html=True)
    short_stay_pages = page_stats.nsmallest(5, '平均滞在時間(秒)')[['ページ番号', '平均滞在時間(秒)', 'ビュー数']]
    
    fig = px.bar(short_stay_pages, x='ページ番号', y='平均滞在時間(秒)', text='平均滞在時間(秒)')
    fig.update_traces(texttemplate='%{text:.1f}秒', textposition='outside')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # 離脱率が高いページ
    st.markdown("#### 離脱率が高いページ TOP5")
    st.markdown('<div class="graph-description">ユーザーが最も離脱しやすいページを特定します。これらのページがボトルネックとなっている可能性が高いです。</div>', unsafe_allow_html=True)
    high_exit_pages = page_stats.nlargest(5, '離脱率')[['ページ番号', '離脱率', 'ビュー数']]
    
    fig = px.bar(high_exit_pages, x='ページ番号', y='離脱率', text='離脱率')
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # 逆行パターン
    st.markdown("#### 逆行パターン（戻る動作）")
    st.markdown('<div class="graph-description">ユーザーがページを戻る動作を分析します。頻繁に戻るパターンがある場合、コンテンツの流れに問題がある可能性があります。</div>', unsafe_allow_html=True)
    backward_df = filtered_df[filtered_df['direction'] == 'backward']
    
    if len(backward_df) > 0:
        backward_pattern = backward_df.groupby(['page_num_dom', 'prev_page_path']).size().reset_index(name='回数')
        backward_pattern = backward_pattern.sort_values('回数', ascending=False).head(10)
        backward_pattern.columns = ['遷移先ページ', '遷移元ページ', '回数']
        
        st.dataframe(backward_pattern, use_container_width=True)
    else:
        st.info("逆行パターンのデータがありません")




# タブ3: セグメント分析
with tab3:
    st.markdown('<div class="sub-header">👥 セグメント分析</div>', unsafe_allow_html=True)
    
    # セグメント選択
    segment_type = st.selectbox("分析するセグメントを選択", [
        "デバイス別",
        "チャネル別",
        "UTMソース別",
        "A/Bテスト別"
    ])
    
    if segment_type == "デバイス別":
        segment_col = 'device_type'
        segment_name = 'デバイス'
    elif segment_type == "チャネル別":
        segment_col = 'channel'
        segment_name = 'チャネル'
    elif segment_type == "UTMソース別":
        segment_col = 'utm_source'
        segment_name = 'UTMソース'
    else:
        segment_col = 'ab_variant'
        segment_name = 'A/Bテスト'
    
    # セグメント別統計
    segment_stats = filtered_df.groupby(segment_col).agg({
        'session_id': 'nunique',
        'stay_ms': 'mean',
        'max_page_reached': 'mean',
        'scroll_pct': 'mean'
    }).reset_index()
    segment_stats.columns = [segment_name, 'セッション数', '平均滞在時間(ms)', '平均到達ページ数', '平均スクロール率']
    segment_stats['平均滞在時間(秒)'] = segment_stats['平均滞在時間(ms)'] / 1000
    
    # コンバージョン数
    segment_cv = filtered_df[filtered_df['cv_type'].notna()].groupby(segment_col)['session_id'].nunique().reset_index()
    segment_cv.columns = [segment_name, 'コンバージョン数']
    
    segment_stats = segment_stats.merge(segment_cv, on=segment_name, how='left').fillna(0)
    segment_stats['コンバージョン率'] = (segment_stats['コンバージョン数'] / segment_stats['セッション数'] * 100)
    
    # エンゲージメント率（滞在時間30秒以上）
    engaged_sessions = filtered_df[filtered_df['stay_ms'] >= 30000].groupby(segment_col)['session_id'].nunique().reset_index()
    engaged_sessions.columns = [segment_name, 'エンゲージセッション数']
    
    segment_stats = segment_stats.merge(engaged_sessions, on=segment_name, how='left').fillna(0)
    segment_stats['エンゲージメント率'] = (segment_stats['エンゲージセッション数'] / segment_stats['セッション数'] * 100)
    
    # テーブル表示
    st.markdown(f"#### {segment_type}の詳細")
    display_cols = [segment_name, 'セッション数', 'コンバージョン数', 'コンバージョン率', 'エンゲージメント率', '平均滞在時間(秒)', '平均到達ページ数']
    st.dataframe(segment_stats[display_cols].style.format({
        'セッション数': '{:,.0f}',
        'コンバージョン数': '{:,.0f}',
        'コンバージョン率': '{:.2f}%',
        'エンゲージメント率': '{:.2f}%',
        '平均滞在時間(秒)': '{:.1f}',
        '平均到達ページ数': '{:.1f}'
    }), use_container_width=True)
    
    # グラフ表示
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(segment_stats, x=segment_name, y='コンバージョン率', title=f'{segment_type}のコンバージョン率')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(segment_stats, x=segment_name, y='平均滞在時間(秒)', title=f'{segment_type}の平均滞在時間')
        st.plotly_chart(fig, use_container_width=True)

# タブ4: A/Bテスト分析
with tab4:
    st.markdown('<div class="sub-header">🧪 A/Bテスト分析</div>', unsafe_allow_html=True)
    
    # A/Bテスト統計
    ab_stats = filtered_df.groupby('ab_variant').agg({
        'session_id': 'nunique',
        'stay_ms': 'mean',
        'max_page_reached': 'mean',
        'completion_rate': 'mean'
    }).reset_index()
    ab_stats.columns = ['バリアント', 'セッション数', '平均滞在時間(ms)', '平均到達ページ数', '平均完了率']
    ab_stats['平均滞在時間(秒)'] = ab_stats['平均滞在時間(ms)'] / 1000
    
    # コンバージョン数
    ab_cv = filtered_df[filtered_df['cv_type'].notna()].groupby('ab_variant')['session_id'].nunique().reset_index()
    ab_cv.columns = ['バリアント', 'コンバージョン数']
    
    ab_stats = ab_stats.merge(ab_cv, on='バリアント', how='left').fillna(0)
    ab_stats['コンバージョン率'] = (ab_stats['コンバージョン数'] / ab_stats['セッション数'] * 100)
    
    # FV残存率
    fv_retention = filtered_df[filtered_df['max_page_reached'] >= 2].groupby('ab_variant')['session_id'].nunique().reset_index()
    fv_retention.columns = ['バリアント', 'FV残存数']
    
    ab_stats = ab_stats.merge(fv_retention, on='バリアント', how='left').fillna(0)
    ab_stats['FV残存率'] = (ab_stats['FV残存数'] / ab_stats['セッション数'] * 100)
    
    # 最終CTA到達率
    final_cta = filtered_df[filtered_df['max_page_reached'] >= 10].groupby('ab_variant')['session_id'].nunique().reset_index()
    final_cta.columns = ['バリアント', '最終CTA到達数']
    
    ab_stats = ab_stats.merge(final_cta, on='バリアント', how='left').fillna(0)
    ab_stats['最終CTA到達率'] = (ab_stats['最終CTA到達数'] / ab_stats['セッション数'] * 100)
    
    # A/Bテストマトリクス
    st.markdown("#### A/Bテストマトリクス")
    display_cols = ['バリアント', 'セッション数', 'コンバージョン率', 'FV残存率', '最終CTA到達率', '平均到達ページ数', '平均滞在時間(秒)']
    st.dataframe(ab_stats[display_cols].style.format({
        'セッション数': '{:,.0f}',
        'コンバージョン率': '{:.2f}%',
        'FV残存率': '{:.2f}%',
        '最終CTA到達率': '{:.2f}%',
        '平均到達ページ数': '{:.1f}',
        '平均滞在時間(秒)': '{:.1f}'
    }), use_container_width=True)
    
    # A/BテストCVR比較
    st.markdown("#### A/BテストCVR比較")
    fig = px.bar(ab_stats, x='バリアント', y='コンバージョン率', text='コンバージョン率')
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # A/Bテスト時系列推移
    st.markdown("#### A/Bテスト CVR 時系列推移")
    
    ab_daily = filtered_df.groupby([filtered_df['event_date'].dt.date, 'ab_variant']).agg({
        'session_id': 'nunique'
    }).reset_index()
    ab_daily.columns = ['日付', 'バリアント', 'セッション数']
    
    ab_daily_cv = filtered_df[filtered_df['cv_type'].notna()].groupby([
        filtered_df[filtered_df['cv_type'].notna()]['event_date'].dt.date,
        'ab_variant'
    ])['session_id'].nunique().reset_index()
    ab_daily_cv.columns = ['日付', 'バリアント', 'コンバージョン数']
    
    ab_daily = ab_daily.merge(ab_daily_cv, on=['日付', 'バリアント'], how='left').fillna(0)
    ab_daily['コンバージョン率'] = (ab_daily['コンバージョン数'] / ab_daily['セッション数'] * 100)
    
    fig = px.line(ab_daily, x='日付', y='コンバージョン率', color='バリアント', markers=True)
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# タブ5: 動画・スクロール分析
with tab5:
    st.markdown('<div class="sub-header">🎬 動画・スクロール分析</div>', unsafe_allow_html=True)
    
    # スクロール率分析
    st.markdown("#### ページ別平均スクロール率")
    
    scroll_stats = filtered_df.groupby('page_num_dom')['scroll_pct'].mean().reset_index()
    scroll_stats.columns = ['ページ番号', '平均スクロール率']
    scroll_stats['平均スクロール率(%)'] = scroll_stats['平均スクロール率'] * 100
    
    fig = px.bar(scroll_stats, x='ページ番号', y='平均スクロール率(%)', text='平均スクロール率(%)')
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # 動画視聴分析（動画イベントがある場合）
    video_df = filtered_df[filtered_df['video_src'].notna()]
    
    if len(video_df) > 0:
        st.markdown("#### 動画視聴率")
        
        video_sessions = video_df['session_id'].nunique()
        total_sessions_with_video_page = filtered_df[filtered_df['video_src'].notna()]['session_id'].nunique()
        video_view_rate = (video_sessions / total_sessions * 100) if total_sessions > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("動画が表示されたセッション", f"{total_sessions_with_video_page:,}")
        
        with col2:
            st.metric("動画を視聴したセッション", f"{video_sessions:,}")
        
        with col3:
            st.metric("視聴率", f"{video_view_rate:.2f}%")
        
        # 視聴率とCVRの相関
        st.markdown("#### 動画視聴とコンバージョンの関係")
        
        video_cv = video_df[video_df['cv_type'].notna()]['session_id'].nunique()
        video_cvr = (video_cv / video_sessions * 100) if video_sessions > 0 else 0
        
        non_video_sessions = total_sessions - video_sessions
        non_video_cv = filtered_df[(filtered_df['video_src'].isna()) & (filtered_df['cv_type'].notna())]['session_id'].nunique()
        non_video_cvr = (non_video_cv / non_video_sessions * 100) if non_video_sessions > 0 else 0
        
        comparison_data = pd.DataFrame({
            'グループ': ['動画視聴あり', '動画視聴なし'],
            'コンバージョン率': [video_cvr, non_video_cvr]
        })
        
        fig = px.bar(comparison_data, x='グループ', y='コンバージョン率', text='コンバージョン率')
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("動画視聴データがありません")
    
    # スクロール率別CVR
    st.markdown("#### スクロール率別コンバージョン率")
    
    # スクロール率を区間に分ける
    filtered_df_scroll = filtered_df.copy()
    filtered_df_scroll['scroll_range'] = pd.cut(filtered_df_scroll['scroll_pct'], bins=[0, 0.25, 0.5, 0.75, 1.0], labels=['0-25%', '25-50%', '50-75%', '75-100%'])
    
    scroll_range_sessions = filtered_df_scroll.groupby('scroll_range', observed=True)['session_id'].nunique().reset_index()
    scroll_range_sessions.columns = ['スクロール率', 'セッション数']
    scroll_range_sessions['スクロール率'] = scroll_range_sessions['スクロール率'].astype(str)
    
    scroll_range_cv = filtered_df_scroll[filtered_df_scroll['cv_type'].notna()].groupby('scroll_range', observed=True)['session_id'].nunique().reset_index()
    scroll_range_cv.columns = ['スクロール率', 'コンバージョン数']
    scroll_range_cv['スクロール率'] = scroll_range_cv['スクロール率'].astype(str)
    
    scroll_range_stats = scroll_range_sessions.merge(scroll_range_cv, on='スクロール率', how='left')
    scroll_range_stats['コンバージョン数'] = scroll_range_stats['コンバージョン数'].fillna(0)
    scroll_range_stats['コンバージョン率'] = (scroll_range_stats['コンバージョン数'] / scroll_range_stats['セッション数'] * 100)
    
    fig = px.bar(scroll_range_stats, x='スクロール率', y='コンバージョン率', text='コンバージョン率')
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# タブ6: 時系列分析
with tab6:
    st.markdown('<div class="sub-header">📈 時系列分析</div>', unsafe_allow_html=True)
    
    # 日別推移
    st.markdown("#### 日別推移")
    
    daily_stats = filtered_df.groupby(filtered_df['event_date'].dt.date).agg({
        'session_id': 'nunique',
        'stay_ms': 'mean',
        'max_page_reached': 'mean'
    }).reset_index()
    daily_stats.columns = ['日付', 'セッション数', '平均滞在時間(ms)', '平均到達ページ数']
    daily_stats['平均滞在時間(秒)'] = daily_stats['平均滞在時間(ms)'] / 1000
    
    daily_cv = filtered_df[filtered_df['cv_type'].notna()].groupby(
        filtered_df[filtered_df['cv_type'].notna()]['event_date'].dt.date
    )['session_id'].nunique().reset_index()
    daily_cv.columns = ['日付', 'コンバージョン数']
    
    daily_stats = daily_stats.merge(daily_cv, on='日付', how='left').fillna(0)
    daily_stats['コンバージョン率'] = (daily_stats['コンバージョン数'] / daily_stats['セッション数'] * 100)
    
    # FV残存率
    daily_fv = filtered_df[filtered_df['max_page_reached'] >= 2].groupby(
        filtered_df[filtered_df['max_page_reached'] >= 2]['event_date'].dt.date
    )['session_id'].nunique().reset_index()
    daily_fv.columns = ['日付', 'FV残存数']
    
    daily_stats = daily_stats.merge(daily_fv, on='日付', how='left').fillna(0)
    daily_stats['FV残存率'] = (daily_stats['FV残存数'] / daily_stats['セッション数'] * 100)
    
    # 最終CTA到達率
    daily_cta = filtered_df[filtered_df['max_page_reached'] >= 10].groupby(
        filtered_df[filtered_df['max_page_reached'] >= 10]['event_date'].dt.date
    )['session_id'].nunique().reset_index()
    daily_cta.columns = ['日付', '最終CTA到達数']
    
    daily_stats = daily_stats.merge(daily_cta, on='日付', how='left').fillna(0)
    daily_stats['最終CTA到達率'] = (daily_stats['最終CTA到達数'] / daily_stats['セッション数'] * 100)
    
    # グラフ選択
    metric_to_plot = st.selectbox("表示する指標を選択", [
        "セッション数",
        "コンバージョン数",
        "コンバージョン率",
        "FV残存率",
        "最終CTA到達率",
        "平均到達ページ数",
        "平均滞在時間(秒)"
    ])
    
    fig = px.line(daily_stats, x='日付', y=metric_to_plot, markers=True)
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # 月間推移（データが十分にある場合）
    if (daily_stats['日付'].max() - daily_stats['日付'].min()).days >= 60:
        st.markdown("#### 月間推移")
        
        filtered_df['month'] = filtered_df['event_date'].dt.to_period('M')
        
        monthly_stats = filtered_df.groupby('month').agg({
            'session_id': 'nunique',
            'max_page_reached': 'mean'
        }).reset_index()
        monthly_stats.columns = ['月', 'セッション数', '平均到達ページ数']
        monthly_stats['月'] = monthly_stats['月'].astype(str)
        
        monthly_cv = filtered_df[filtered_df['cv_type'].notna()].groupby('month')['session_id'].nunique().reset_index()
        monthly_cv.columns = ['月', 'コンバージョン数']
        monthly_cv['月'] = monthly_cv['月'].astype(str)
        
        monthly_stats = monthly_stats.merge(monthly_cv, on='月', how='left').fillna(0)
        monthly_stats['コンバージョン率'] = (monthly_stats['コンバージョン数'] / monthly_stats['セッション数'] * 100)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='セッション数', x=monthly_stats['月'], y=monthly_stats['セッション数'], yaxis='y'))
        fig.add_trace(go.Scatter(name='コンバージョン率', x=monthly_stats['月'], y=monthly_stats['コンバージョン率'], yaxis='y2', mode='lines+markers'))
        
        fig.update_layout(
            yaxis=dict(title='セッション数'),
            yaxis2=dict(title='コンバージョン率 (%)', overlaying='y', side='right'),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# タブ7: リアルタイム分析
with tab7:
    st.markdown('<div class="sub-header">リアルタイム分析</div>', unsafe_allow_html=True) # type: ignore
    st.markdown("サイト全体の直近1時間の活動状況をリアルタイムで確認できます。この分析は上部のフィルター設定の影響を受けません。")
    
    # 直近1時間のデータをフィルタリング（filtered_dfではなく、元のdfから取得）
    one_hour_ago = df['event_timestamp'].max() - timedelta(hours=1)
    realtime_df = df[df['event_timestamp'] >= one_hour_ago]
    
    if len(realtime_df) > 0:
        # 分単位の推移
        st.markdown("#### 直近1時間のセッション数推移（10分単位）")
        st.markdown("直近1時間のセッション数を、10分ごとに集計して表示します")
        
        realtime_df['minute_bin'] = realtime_df['event_timestamp'].dt.floor('10T')
        rt_trend = realtime_df.groupby('minute_bin')['session_id'].nunique().reset_index()
        rt_trend.columns = ['時刻', 'セッション数']
        
        fig = px.line(rt_trend, x='時刻', y='セッション数', markers=True)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("直近1時間のデータがありません")

# タブ8: カスタムオーディエンス
with tab8:
    st.markdown('<div class="sub-header">カスタムオーディエンスビルダー</div>', unsafe_allow_html=True)
    st.markdown("条件を指定して、独自のオーディエンスを即座に作成できます")
    
    col1, col2 = st.columns(2)
    
    with col1:
        min_stay_time = st.slider("最小滞在時間（秒）", 0, 300, 30)
        min_scroll_rate = st.slider("最小スクロール率", 0.0, 1.0, 0.5, 0.1)
        min_pages_reached = st.slider("最小到達ページ数", 1, 10, 3)
    
    with col2:
        require_conversion = st.checkbox("コンバージョン達成のみ")
        selected_devices_audience = st.multiselect("デバイス", device_types, default=device_types)
        selected_channels_audience = st.multiselect("チャネル", filtered_df['channel'].unique().tolist(), default=filtered_df['channel'].unique().tolist())
    
    if st.button("オーディエンスを作成"):
        # オーディエンス作成
        audience_df = filtered_df[
            (filtered_df['stay_ms'] >= min_stay_time * 1000) &
            (filtered_df['scroll_pct'] >= min_scroll_rate) &
            (filtered_df['max_page_reached'] >= min_pages_reached) &
            (filtered_df['device_type'].isin(selected_devices_audience)) &
            (filtered_df['channel'].isin(selected_channels_audience))
        ]
        
        if require_conversion:
            audience_df = audience_df[audience_df['cv_type'].notna()]
        
        audience_sessions = audience_df['session_id'].nunique()
        audience_users = audience_df['user_pseudo_id'].nunique()
        audience_cvr = (audience_df[audience_df['cv_type'].notna()]['session_id'].nunique() / audience_sessions * 100) if audience_sessions > 0 else 0
        audience_avg_stay = audience_df['stay_ms'].mean() / 1000
        audience_avg_pages = audience_df.groupby('session_id')['max_page_reached'].max().mean()
        
        st.success(f"✅ オーディエンスを作成しました！")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("セッション数", f"{audience_sessions:,}")
        
        with col2:
            st.metric("ユーザー数", f"{audience_users:,}")
        
        with col3:
            st.metric("コンバージョン率", f"{audience_cvr:.2f}%")
        
        with col4:
            st.metric("平均滞在時間", f"{audience_avg_stay:.1f}秒")
        
        with col5:
            st.metric("平均到達ページ数", f"{audience_avg_pages:.1f}")
        
        # オーディエンスの時系列推移
        st.markdown("#### オーディエンスの時系列推移")
        st.markdown("作成したオーディエンスのセッション数の推移を表示します")
        
        audience_daily = audience_df.groupby(audience_df['event_date'].dt.date)['session_id'].nunique().reset_index()
        audience_daily.columns = ['日付', 'セッション数']
        
        fig = px.line(audience_daily, x='日付', y='セッション数', markers=True)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# タブ9: AI提案
with tab9:
    st.markdown('<div class="sub-header">💡 AI提案（プロトタイプ）</div>', unsafe_allow_html=True)
    
    st.markdown("""
    このセクションでは、AIがデータを分析し、改善提案を行います。
    
    **質問ボタンをクリックすると、AIが回答を生成します。**
    """)
    
    # 質問ボタン
    st.markdown("#### よくある質問")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("❓ このLPの最大のボトルネックは？"):
            # 離脱率が最も高いページを特定
            max_exit_page = page_stats.loc[page_stats['離脱率'].idxmax()]
            
            st.info(f"""
            **分析結果:**
            
            最大のボトルネックは**ページ{int(max_exit_page['ページ番号'])}**です。
            
            - 離脱率: {max_exit_page['離脱率']:.1f}%
            - 平均滞在時間: {max_exit_page['平均滞在時間(秒)']:.1f}秒
            
            **推奨アクション:**
            1. ページ{int(max_exit_page['ページ番号'])}のコンテンツを見直し、ユーザーの関心を引く要素を追加
            2. A/Bテストで異なるコンテンツをテスト
            3. 読込時間が長い場合は、画像の最適化を検討
            """)
        
        if st.button("❓ コンバージョン率を改善するには？"):
            st.info(f"""
            **分析結果:**
            
            現在のコンバージョン率は**{conversion_rate:.2f}%**です。
            
            **推奨アクション:**
            1. FV残存率({fv_retention_rate:.1f}%)を改善するため、ファーストビューのコンテンツを強化
            2. 最終CTA到達率({final_cta_rate:.1f}%)を改善するため、ページ遷移をスムーズにする
            3. デバイス別の分析を行い、パフォーマンスが低いデバイスに最適化
            4. 高パフォーマンスのチャネルに予算を集中
            """)
    
    with col2:
        if st.button("❓ A/Bテストの結果、どちらが優れている？"):
            best_variant = ab_stats.loc[ab_stats['コンバージョン率'].idxmax()]
            
            st.info(f"""
            **分析結果:**
            
            **バリアント{best_variant['バリアント']}**が最も優れています。
            
            - コンバージョン率: {best_variant['コンバージョン率']:.2f}%
            - セッション数: {int(best_variant['セッション数'])}
            
            **推奨アクション:**
            1. バリアント{best_variant['バリアント']}を本番環境に適用
            2. さらなる改善のため、次のA/Bテストを計画
            """)
        
        if st.button("❓ デバイス別のパフォーマンス差は？"):
            best_device = device_stats.loc[device_stats['コンバージョン率'].idxmax()]
            worst_device = device_stats.loc[device_stats['コンバージョン率'].idxmin()]
            
            st.info(f"""
            **分析結果:**
            
            **最高パフォーマンス:** {best_device['デバイス']} (CVR: {best_device['コンバージョン率']:.2f}%)
            **最低パフォーマンス:** {worst_device['デバイス']} (CVR: {worst_device['コンバージョン率']:.2f}%)
            
            **推奨アクション:**
            1. {worst_device['デバイス']}向けにUIを最適化
            2. {worst_device['デバイス']}での読込速度を改善
            3. {best_device['デバイス']}の成功要因を他デバイスに適用
            """)
    
    st.markdown("---")
    
    # フリーチャット（プロトタイプ）
    st.markdown("#### フリーチャット")
    
    user_question = st.text_input("質問を入力してください（プロトタイプ版では固定回答が表示されます）")
    
    if st.button("送信"):
        if user_question:
            st.info(f"""
            **質問:** {user_question}
            
            **回答（プロトタイプ）:**
            
            ご質問ありがとうございます。本番環境では、Gemini 2.5 Proが実際のデータに基づいて詳細な分析と提案を行います。
            
            現在のプロトタイプでは、以下のような分析が可能です:
            - データの傾向分析
            - ボトルネックの特定
            - 改善提案の生成
            - SQL クエリの提案
            
            本格実装後は、より高度な分析と具体的なアクションプランを提供します。
            """)
        else:
            st.warning("質問を入力してください")

# タブ10: 使用ガイド
with tab10:
    st.markdown('<div class="sub-header">使用ガイド</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 瞬ジェネ AIアナリストの使い方
    
    #### 1. フィルター設定
    左サイドバーで、分析対象を絞り込むことができます。
    - **期間を選択**: 過去7日間、過去30日間、過去90日間、またはカスタム期間
    - **LP選択**: 特定のLPのみを分析
    - **チャネル**: 特定のチャネル（Google、Facebookなど）のみを分析
    - **デバイス**: スマートフォン、パソコン、タブレットを選択
    - **A/Bテスト**: 特定のバリアントのみを分析
    
    #### 2. 全体分析
    LP全体のパフォーマンスを確認できます。
    - セッション数の推移
    - コンバージョン率の推移
    - デバイス別・チャネル別分析
    - LP進行ファネル
    
    #### 3. ページ分析
    各ページのパフォーマンスを詳細に分析できます。
    - ページ別パフォーマンス一覧
    - 滞在時間が短いページ TOP5
    - 離脱率が高いページ TOP5
    
    #### 4. セグメント分析
    デバイス、チャネル、UTMソース、A/Bテスト別に分析できます。
    
    #### 5. A/Bテスト分析
    A/Bテストの結果を比較し、最適なバリアントを特定できます。
    
    #### 6. 動画・スクロール分析
    動画視聴状況やスクロール率とコンバージョンの関係を分析できます。
    
    #### 7. 時系列分析
    時間帯別、曜日別のパフォーマンスを分析できます。
    
    #### 8. リアルタイム分析
    直近1時間のデータをリアルタイムで確認できます。
    
    #### 9. カスタムオーディエンス
    条件を指定して、独自のオーディエンスを即座に作成できます。
    - 最小滞在時間
    - 最小スクロール率
    - 最小到達ページ数
    - コンバージョン達成のみ
    
    #### 10. AI提案
    AIがデータを分析し、改善提案を行います。
    - 質問ボタンをクリックすると、AIが回答を生成します
    - フリーチャットで自由に質問できます
    """)

# タブ11: 専門用語解説
with tab11:
    st.markdown('<div class="sub-header">専門用語解説</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### マーケティング・分析用語集
    
    #### 基本用語
    
    **セッション**
    ユーザーがLPを訪れた回数。同じユーザーが複数回訪れた場合、それぞれカウントされます。
    
    **コンバージョン**
    ユーザーが目標とする行動（購入、問い合わせなど）を完了したこと。
    
    **コンバージョン率（CVR）**
    セッション数に対するコンバージョン数の割合。例: 100セッションで5コンバージョンなら、CVRは5%。
    
    **ページから出た割合（離脱率）**
    そのページでLPを離れたユーザーの割合。高いほど、そのページに問題がある可能性があります。
    
    **滞在時間**
    ユーザーがLPに滞在した時間。長いほど、ユーザーが興味を持っていると考えられます。
    
    **クリック率（CTR）**
    セッション数に対するクリック数の割合。
    
    #### 高度な用語
    
    **FV残存率**
    ファーストビュー（最初の画面）を見た後、次のページに進んだユーザーの割合。
    
    **最終CTA到達率**
    最後のページ（CTA：Call To Action）に到達したユーザーの割合。
    
    **スクロール率**
    ユーザーがページをどれだけスクロールしたかの割合。100%なら、ページの最後まで見たことを意味します。
    
    **A/Bテスト**
    2つ以上の異なるバージョン（バリアント）を同時に公開し、どちらが優れているかを検証する手法。
    
    **UTMパラメータ**
    URLに付加するタグで、どの広告やキャンペーンからユーザーが来たかを追跡するためのもの。
    - **utm_source**: トラフィック元（例: google, facebook）
    - **utm_medium**: 媒体（例: cpc, email）
    - **utm_campaign**: キャンペーン名
    
    **チャネル**
    ユーザーがLPに到達した経路。例: Organic Search（自然検索）、Organic Social（SNS）、Direct（直接アクセス）など。
    
    **セグメント**
    特定の条件で絞り込んだユーザーグループ。例: スマートフォンユーザー、Googleからのユーザーなど。
    
    **オーディエンス**
    特定の条件を満たすユーザーの集合。カスタムオーディエンスでは、独自の条件でオーディエンスを作成できます。
    
    **ファネル**
    ユーザーがLPを進む過程を表す図。各ステップでどれだけのユーザーが離脱したかを可視化します。
    
    **読込時間**
    ページが表示されるまでの時間。短いほど、ユーザー体験が良いと考えられます。
    """)

# フッター
st.markdown("---")
st.markdown("**瞬ジェネ AIアナリスト** - Powered by Streamlit & Gemini 2.5 Pro")
