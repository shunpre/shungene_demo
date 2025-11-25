"""
瞬ジェネ AIアナライザー - Step 2完成版
グラフ説明と比較機能を追加
"""
import sys
import os

# --- どんな環境でもインポートを成功させるための「魔法の呪文」 ---
# このファイルの絶対パスを取得
file_path = os.path.abspath(__file__)
# このファイルの親ディレクトリ（'app'フォルダ）のパスを取得
dir_path = os.path.dirname(file_path)
# 'app'フォルダの親ディレクトリ（プロジェクトルート）のパスを取得
project_root = os.path.dirname(dir_path)
# Pythonがファイルを検索する場所のリストに、プロジェクトルートを追加
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import time

from app.generate_dummy_data import generate_dummy_data
from app.capture_lp import extract_lp_text_content
import app.ai_analysis as ai_analysis

import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

# --- Streamlitバージョン互換性のためのプロキシクラス ---
class QueryParamsProxy:
    """
    Streamlitのバージョンによって query_params の仕様が異なるため、
    それを吸収するためのプロキシクラス。
    st.query_params (新しいバージョン) があればそれを使い、
    なければ st.experimental_get_query_params / st.experimental_set_query_params (古いバージョン) を使う。
    """
    def __init__(self):
        if hasattr(st, "query_params"):
            self._use_new_api = True
            self._params = st.query_params
        else:
            self._use_new_api = False

    def __getitem__(self, key):
        if self._use_new_api:
            return self._params[key]
        else:
            params = st.experimental_get_query_params()
            if key in params:
                return params[key][0] # 古いAPIはリストを返すため先頭を取得
            raise KeyError(key)

    def get(self, key, default=None):
        if self._use_new_api:
            return self._params.get(key, default)
        else:
            params = st.experimental_get_query_params()
            if key in params:
                return params[key][0]
            return default

    def __setitem__(self, key, value):
        if self._use_new_api:
            self._params[key] = value
        else:
            # 既存のパラメータを取得して更新
            params = st.experimental_get_query_params()
            params[key] = [str(value)] # リストとして保存
            st.experimental_set_query_params(**params)

    def __contains__(self, key):
        if self._use_new_api:
            return key in self._params
        else:
            return key in st.experimental_get_query_params()

# プロキシのインスタンスを作成
query_params_proxy = QueryParamsProxy()

# ページ設定
st.set_page_config(
    page_title="瞬ジェネ AIアナライザー",
    page_icon="https://shungene.lm-c.jp/favicon02.png",
    layout="wide",
    # PCでは常に展開された状態にするため "expanded" に固定
    initial_sidebar_state="expanded",
)

# 上部のオレンジ色のバーを非表示にするためのカスタムCSS
hide_decoration_bar_style = '''
    <style>
        div[data-testid="stDecoration"] {
            display: none;
        }
    </style>
'''
st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)

# ブラウザがスクロールする先の「基点」を設置
st.markdown('<a id="top-anchor"></a>', unsafe_allow_html=True)

# --- Authentication ---
config_path = os.path.join(project_root, 'config.yaml')
with open(config_path) as file:
    config = yaml.load(file, Loader=SafeLoader)

# Fix: Do not pass preauthorized to Authenticate constructor
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# Login Widget
# fields引数でラベルを日本語化
authenticator.login(
    location='main',
    fields={'Form name': 'ログイン', 'Username': 'ユーザー名', 'Password': 'パスワード', 'Login': 'ログイン'}
)

authentication_status = st.session_state.get('authentication_status')
name = st.session_state.get('name')
username = st.session_state.get('username')

if authentication_status is False:
    st.error('ユーザー名またはパスワードが間違っています')
elif authentication_status is None:
    st.warning('ユーザー名とパスワードを入力してください')

if not authentication_status:
    st.stop()

# Logout button in sidebar
authenticator.logout(location='sidebar', key='logout_button')
# ボタンのラベル変更は標準機能では難しいため、メッセージのみ日本語化
st.sidebar.write(f'ようこそ *{name}* さん')

# --- ページ遷移関数 ---
def navigate_to(page_name):
    """指定されたページに遷移する"""
    query_params_proxy["page"] = page_name

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
    /* メインコンテンツの上部余白を調整してサイドバーと高さを合わせる */
    .main > div:first-child {
        padding-top: 1.8rem !important;
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
    /* 通常のボタン (secondary) */
    .stButton>button[kind="secondary"] {
        background-color: #f0f2f6;
        color: #333;
        border: 1px solid #f0f2f6;
    }
    /* サイドバーのボタンを左揃えにする */
    div[data-testid="stSidebarUserContent"] .stButton>button {
        text-align: left !important;
        justify-content: flex-start !important;
    }
    /* サイドバーの通常ボタンのホバー時 */
    div[data-testid="stSidebarUserContent"] .stButton>button[kind="secondary"]:hover {
        background-color: #e6f0ff !important;
        color: #333 !important;
        border: 1px solid #002060 !important;
    }
    
    /* プライマリボタン（データ生成、ダウンロードなど）を赤色にする */
    .stButton>button[kind="primary"] {
        background-color: #ff4b4b !important;
        color: white !important;
        border: 1px solid #ff4b4b !important;
    }
    .stButton>button[kind="primary"]:hover {
        background-color: #ff6b6b !important;
        color: white !important;
        border: 1px solid #ff6b6b !important;
    }
    /* st.info のスタイルを強制的に青系に固定 */
    div[data-testid="stInfo"] {
        background-color: #e6f3ff !important;
        border-color: #1c83e1 !important;
        color: #000 !important;
    }
    /* サイドバーの開閉ボタンのSVGアイコンを非表示にする */
    button[data-testid="stSidebarCollapseButton"] > svg {
        display: none;
    }

    /* サイドバーが開いている時（閉じるボタン）のアイコン */
    body[data-sidebar-state="expanded"] button[data-testid="stSidebarCollapseButton"]::before {
        content: '<';
        font-size: 1.6rem;
        color: #666;
        font-weight: bold;
    }

    /* サイドバーが閉じている時（開くボタン）のアイコン */
    body[data-sidebar-state="collapsed"] button[data-testid="stSidebarCollapseButton"]::before {
        content: '>';
        font-size: 1.6rem;
        color: #666;
        font-weight: bold;
    }
    /* タイトル横のリンクアイコンを非表示にする */
    h1 a, h2 a, h3 a, h4 a, h5 a, h6 a {
        /* リンクアイコンを非表示にするためのより強力なセレクタ */
        visibility: hidden;
        display: none !important;
    }
    /* サイドバーリンクの基本スタイル (非選択時 = secondary) */
    a.sidebar-link {
        display: block; /* ボタンのように振る舞わせる */
        width: 100%;
        padding: 0.5rem 0.75rem; /* st.buttonのpaddingに合わせる */
        border-radius: 0.5rem;
        text-decoration: none; /* 下線を消す */
        font-weight: 400; /* 通常の太さ */
        box-sizing: border-box; /* paddingを含めてwidth 100%にする */

        /* secondaryボタンのスタイルを再現 */
        background-color: #f0f2f6;
        color: #333;
        border: 1px solid #f0f2f6;
        transition: all 0.2s;
    }

    /* ホバー時のスタイル (非選択時) */
    a.sidebar-link:hover {
        background-color: #e6f0ff !important;
        color: #333 !important;
        border: 1px solid #002060 !important;
        text-decoration: none; /* ホバー時も下線なし */
    }

    /* 選択中のリンクのスタイル (primary) */
    a.sidebar-link.active {
        /* primaryボタンのスタイルを再現 */
        background-color: #002060 !important;
        color: white !important;
        border: 1px solid #002060 !important;
        font-weight: bold; /* 選択中を分かりやすく */
    }

    /* --- PC表示でのサイドバー幅を調整 --- */
    @media (min-width: 768px) {
        section[data-testid="stSidebar"] {
            width: 350px !important;
        }
    }

    /* スマホでのみ表示される改行タグ */
    .mobile-br {
        display: none;
    }
    @media (max-width: 768px) {
        .mobile-br {
            display: block;
        }
    }

    /* スマホ表示の時だけ、サイドバーを強制的に閉じる */
    @media (max-width: 768px) {
        /* 
         * Streamlitはページ遷移後もサイドバーの表示状態を記憶してしまうため、
         * ページ読み込み時にサイドバーが開いている状態(expanded)の場合に限り、
         * 強制的に閉じる(collapsed)状態のスタイルを適用する。
         * これにより、スマホでは常に閉じた状態から始まるように見え、開閉ボタンは常に表示される。
         */
        body[data-sidebar-state="expanded"] section[data-testid="stSidebar"] > div:first-child {
            transform: translateX(-100%);
            transition: transform 300ms ease-in-out;
        }
    }
</style>
""", unsafe_allow_html=True)

# --- 堅牢化のためのヘルパー関数 ---
def safe_rate(numerator, denominator):
    """ゼロ除算を回避して率を計算する (inf対応)"""
    if isinstance(denominator, pd.Series):
        # 分母が0の場所をnanに置き換えてから計算し、結果のinf/nanを0で埋める
        denominator_safe = denominator.replace(0, np.nan)
        rate = numerator.divide(denominator_safe)
        return rate.replace([np.inf, -np.inf], np.nan).fillna(0)
    # denominatorが単一の数値の場合
    return numerator / denominator if denominator != 0 else 0.0

# 比較期間のデータを取得する関数
def get_comparison_data(df, current_start, current_end, comparison_type):
    """
    比較期間のデータを取得
    comparison_type: 'previous_period', 'previous_week', 'previous_month', 'previous_year'
    """
    period_length_days = (current_end - current_start).days + 1 # 両端含む日数
    
    if comparison_type == 'previous_period':
        comp_end = current_start - timedelta(days=1)
        comp_start = comp_end - timedelta(days=period_length_days - 1)
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

def safe_extract_lp_text_content(extractor_func, url):
    """capture_lpモジュールがなくてもクラッシュしないようにするラッパー"""
    if extractor_func:
        return extractor_func(url)
    else:
        # モジュールがない場合のデフォルトの戻り値
        return {"headlines": [], "body_copy": [], "ctas": []}


st.sidebar.markdown("""
    <div style="
        display: block;
        color: #002060;
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 1rem;
        line-height: 1.3;
        text-decoration: none;">
        瞬ジェネ<br>AIアナライザー
    </div>
    """, unsafe_allow_html=True)
st.sidebar.markdown("---")



# --- AI Model Selection ---
st.sidebar.markdown("##### AIモデル設定")
model_options = {
    "Gemini 3.0 Pro (Preview)": "gemini-3-pro-preview",
    "Gemini 2.5 Pro": "gemini-2.5-pro"
}
selected_model_label = st.sidebar.selectbox(
    "使用するAIモデル",
    list(model_options.keys()),
    index=0, # Default to 3.0 Pro
    key="model_selector"
)
st.session_state.selected_gemini_model = model_options[selected_model_label]

st.sidebar.markdown("---")

# --- Product Analysis & Scenario Customizer ---
st.sidebar.markdown("##### 商材・サービス設定")
product_description = st.sidebar.text_area(
    "商材・サービス概要",
    placeholder="例: 40代女性向けのエイジングケア美容液。定期購入がメイン。",
    key="product_description_input"
)

if st.sidebar.button("AIで商材を分析", key="analyze_product_btn", type="secondary", use_container_width=True):
    if product_description:
        with st.spinner("商材特性を分析中..."):
            analysis_result_json = ai_analysis.analyze_product_characteristics(product_description)
            try:
                # Clean up JSON string if necessary (remove markdown code blocks)
                analysis_result_json = analysis_result_json.replace("```json", "").replace("```", "").strip()
                analysis_result = json.loads(analysis_result_json)
                
                st.session_state.product_analysis = analysis_result
                st.sidebar.success("分析完了！")
                
                # Update session state with AI suggestions
                params = analysis_result.get("scenario_params", {})
                st.session_state.custom_cvr_multiplier = params.get("cvr_multiplier", 1.0)
                st.session_state.custom_stay_time_mu = params.get("stay_time_mu_base", 2.0)
                st.session_state.custom_fv_exit_rate = params.get("fv_exit_rate", 0.4)
                
            except json.JSONDecodeError:
                st.sidebar.error("AI分析結果の解析に失敗しました。")
            except Exception as e:
                st.sidebar.error(f"エラーが発生しました: {e}")
    else:
        st.sidebar.warning("商材概要を入力してください。")

# Display Analysis Result if available
if "product_analysis" in st.session_state:
    result = st.session_state.product_analysis
    st.sidebar.info(f"""
    **分析結果**:
    *   **ターゲット**: {result.get('target_audience', 'N/A')}
    *   **想定CVR**: {result.get('estimated_cvr_range', 'N/A')}
    *   **ボトルネック**: {', '.join(result.get('bottlenecks', []))}
    """)

st.sidebar.markdown("---")

# シナリオ選択
scenario_options = ["不調（離脱率高）", "好調（高エンゲージメント）", "不調（モバイル課題）", "標準（ベースライン）", "カスタム（AI分析反映）"]
selected_scenario = st.sidebar.selectbox("シナリオを選択", scenario_options, index=3, key="scenario_selector_main")

# Custom Scenario Parameters (only visible if Custom is selected)
if selected_scenario == "カスタム（AI分析反映）":
    st.sidebar.markdown("###### カスタムパラメータ")
    custom_cvr_mult = st.sidebar.slider("CVR倍率", 0.5, 2.0, st.session_state.get("custom_cvr_multiplier", 1.0), 0.1)
    custom_stay_mu = st.sidebar.slider("滞在時間係数", 1.0, 4.0, st.session_state.get("custom_stay_time_mu", 2.0), 0.1)
    custom_fv_exit = st.sidebar.slider("FV離脱率", 0.1, 0.9, st.session_state.get("custom_fv_exit_rate", 0.4), 0.05)

target_cvr_input = st.sidebar.number_input(
    "想定CVR (%)",
    min_value=0.1,
    max_value=100.0,
    value=st.session_state.get('target_cvr', 3.0),
    step=0.1,
    format="%.2f"
)

if st.sidebar.button("ダミーデータを生成", key="global_generate_data", type="primary", use_container_width=True):
    with st.spinner(f"「{selected_scenario}」シナリオのデータを生成中..."):
        # 新しいダミーデータ生成関数を呼び出す
        num_days_gen = 30
        
        # Handle Custom Scenario
        if selected_scenario == "カスタム（AI分析反映）":
            # We need to pass these custom parameters to generate_dummy_data
            # Since generate_dummy_data takes a scenario string, we might need to modify it 
            # or pass a config dict. For now, let's use a hack: modify the SCENARIO_CONFIGS in memory 
            # or pass a special argument. 
            # Better approach: Update generate_dummy_data to accept overrides.
            # For this demo, I will modify generate_dummy_data to accept a 'custom_config' argument 
            # but since I can't easily change the signature without breaking imports, 
            # I will add a temporary 'Custom' entry to SCENARIO_CONFIGS in generate_dummy_data.py
            # Wait, I can't modify the imported module's global variable easily from here in a clean way.
            # I will modify generate_dummy_data.py to export SCENARIO_CONFIGS so I can update it here.
            from app.generate_dummy_data import SCENARIO_CONFIGS
            SCENARIO_CONFIGS['カスタム（AI分析反映）'] = {
                'description': 'AI分析に基づくカスタム設定',
                'num_sessions_per_day_range': (300, 500),
                'fv_exit_rate': custom_fv_exit,
                'transition_mean': 0.90,
                'transition_sd': 0.05,
                'bottleneck_pages': {3: 0.3}, # Default bottleneck
                'cta_click_rate_base': 0.10,
                'cvr_multiplier': custom_cvr_mult,
                'stay_time_mu_base': custom_stay_mu,
                'stay_time_sigma': 0.6,
                'backflow_base': 0.05,
                'device_dist': ['mobile', 'desktop'],
                'device_weights': [0.7, 0.3],
                'num_pages_dist': lambda: random.randint(10, 15),
            }
        
        st.session_state.generated_data = generate_dummy_data(
            scenario=selected_scenario,
            num_days=num_days_gen,
            target_cvr=target_cvr_input / 100 # %を小数に変換して渡す
        )
        st.session_state.data_scenario = selected_scenario # 現在のシナリオを保存
        st.session_state.target_cvr = target_cvr_input # 入力されたCVRを保存
    
    # ページリダイレクトを削除し、現在のページを維持する
    pass

    st.rerun()

# JavaScriptでデータ生成ボタンを赤色にする
# Note: CSSで制御するため、ここのJSは削除

st.sidebar.markdown("---")

# --- 堅牢化のためのヘルパー関数 ---
def assign_channel(row):
    """
    utm_sourceとutm_mediumに基づいてチャネルを割り当てる関数。
    YouTube広告やその他の有料広告に対応。
    """
    source = str(row.get('utm_source', '(direct)')).lower()
    medium = str(row.get('utm_medium', '(none)')).lower()
    referrer = str(row.get('page_referrer', ''))

    # 1. Paid Search (有料検索)
    # mediumがcpc, ppc, paidsearchの場合。sourceが検索エンジン系であることを優先。
    if medium in ['cpc', 'ppc', 'paidsearch']:
        return 'Paid Search'

    # 2. Paid Social (有料ソーシャル)
    # mediumがpaid_social, paidsocial, social_adなど。
    if medium in ['paid_social', 'paidsocial', 'social_ad']:
        return 'Paid Social'

    # 3. Paid Video (有料動画)
    if medium in ['paidvideo', 'paid_video']:
        return 'Paid Video'

    # 4. Display (ディスプレイ広告)
    if medium in ['display', 'banner', 'cpm']:
        return 'Display'

    # 5. Organic Search (自然検索)
    if medium == 'organic':
        return 'Organic Search'

    # 6. Organic Social (自然ソーシャル)
    # mediumがsocial、またはsourceが主要SNSの場合
    if medium == 'social' or source in ['facebook', 'instagram', 'twitter', 'x.com', 't.co', 'linkedin', 'tiktok', 'youtube']:
        return 'Organic Social'

    # 7. Direct (直接流入)
    if source == '(direct)' and medium == '(none)':
        return 'Direct'

    # 8. Email
    if medium == 'email':
        return 'Email'

    # 9. Referral (参照)
    if medium == 'referral':
        return 'Referral'

    return 'Other' # どの条件にも当てはまらない場合

# --- 分析対象のDataFrameを決定 ---
# セッションに生成されたデータがあればそれを使用し、なければ元のCSVデータを使用します。
if "generated_data" not in st.session_state or st.session_state.generated_data.empty:
    st.info("""
    **ガイド ：** 左側のサイドバーで想定CVRとシナリオを設定し、「ダミーデータを生成」ボタンを押して、データ分析の学習にご活用ください。
    """)
    st.stop()
else:
    df = st.session_state.generated_data
    df['event_date'] = pd.to_datetime(df['event_date'])
    df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])


# グルーピングされたメニュー項目
menu_groups = {
    "基本分析": ["全体サマリー", "リアルタイムビュー", "時系列分析", "デモグラフィック情報", "アラート"],
    "LP最適化分析": ["ページ分析", "A/Bテスト分析"],
    "詳細分析": ["広告分析", "インタラクション分析", "動画・スクロール分析", "瞬フォーム分析", "AIアナリスト（チャット）"],
    "ヘルプ": ["LPOの基礎知識", "専門用語解説", "FAQ"]
}

# --- 共通の前処理 ---
# channel列を追加
df['channel'] = df.apply(assign_channel, axis=1)
# LPのベースURL列を追加
df['lp_base_url'] = df['page_location'].str.split('#').str[0]
# twitterをXに置換
df['utm_source_display'] = df['utm_source'].replace('twitter', 'X')
# NaN値を '(none)' に置換
df['utm_source_display'].fillna('(direct)', inplace=True)
df['utm_medium'].fillna('(none)', inplace=True)
# source / medium を作成
df['source_medium'] = df['utm_source_display'] + ' / ' + df['utm_medium']
# 論理的に不自然な組み合わせを除外 (例: direct / cpc)
df = df[~((df['utm_source_display'] == '(direct)') & (df['utm_medium'] != '(none)'))]


DEFAULT_PAGE = "全体サマリー"

# URLクエリから表示するページを取得
selected_analysis = query_params_proxy.get("page", DEFAULT_PAGE)

# 他の処理がst.session_stateを参照している場合に備え、同期させる
st.session_state.selected_analysis = selected_analysis

for group_name, items in menu_groups.items():
    st.sidebar.markdown(f"**{group_name}**")
    for item in items: # type: ignore
        # 現在のページ(selected_analysis)とアイテムが一致するか判定
        is_active = (selected_analysis == item)
        
        # リンクがクリックされたかどうかを判定するためのユニークなキー
        button_key = f"nav_button_{item}"

        # 選択中のページはテキストで強調、ボタン自体は全てsecondary（無色）
        # 選択中の場合は "▶ " を付与して紺色（青）かつ太字で強調
        label = f":blue[**▶ {item}**]" if is_active else item
        button_type = "secondary"
        
        if st.sidebar.button(label, key=button_key, use_container_width=True, type=button_type):
            try:
                query_params_proxy["page"] = item
            except AttributeError:
                query_params_proxy["page"] = item
            st.rerun()

    st.sidebar.markdown("---")

# 選択された分析項目に応じて表示を切り替え

if selected_analysis == "全体サマリー":
    st.markdown('<div class="sub-header">全体サマリー</div>', unsafe_allow_html=True)

    # メインエリア: フィルターと比較設定
    st.markdown('<div class="sub-header">フィルター設定</div>', unsafe_allow_html=True)

    # ページ上部にフィルターを配置
    filter_cols_1 = st.columns(4)
    filter_cols_2 = st.columns(4)

    with filter_cols_1[0]:
        # 期間選択（キーを一意にするためにプレフィックスを追加）
        period_options = [
            "今日", "昨日", "過去7日間", "過去14日間", "過去30日間", "今月", "先月", "全期間", "カスタム"
        ]
        selected_period = st.selectbox("期間を選択", period_options, index=2, key="summary_period_selector")

    with filter_cols_1[1]:
        # LP選択
        lp_options = sorted(df['lp_base_url'].dropna().unique().tolist())
        selected_lp_base_url = st.selectbox(
            "LP選択", 
            lp_options, 
            index=0 if lp_options else None, # 選択肢がなければindexもNone
            key="summary_lp", # キーを明示
            disabled=not lp_options # 選択肢がなければ操作不可
        )

    with filter_cols_1[2]:
        device_options = ["すべて"] + sorted(df['device_type'].dropna().unique().tolist())
        selected_device = st.selectbox("デバイス選択", device_options, index=0, key="summary_device_selector")

    with filter_cols_1[3]:
        # 新規/リピート フィルター
        user_type_options = ["すべて", "新規", "リピート"]
        selected_user_type = st.selectbox("新規/リピート", user_type_options, index=0, key="summary_user_type_selector")

    with filter_cols_2[0]:
        # CV/非CV フィルター
        conversion_status_options = ["すべて", "コンバージョン", "非コンバージョン"]
        selected_conversion_status = st.selectbox("CV/非CV", conversion_status_options, index=0, key="summary_conversion_status_selector")

    with filter_cols_2[1]:
        # チャネルフィルターを追加
        channel_options = ["すべて"] + sorted(df['channel'].unique().tolist())
        selected_channel = st.selectbox("チャネル", channel_options, index=0, key="summary_channel_selector")

    with filter_cols_2[2]:
        # チャネルフィルターを「参照元/メディア」に変更
        source_medium_options = ["すべて"] + sorted(df['source_medium'].unique().tolist())
        selected_source_medium = st.selectbox("参照元/メディア", source_medium_options, index=0, key="summary_source_medium_selector") # ラベルは変更済み

    # 期間設定
    today = df['event_date'].max().date()
    
    if selected_period == "今日":
        start_date = today
        end_date = today
    elif selected_period == "昨日":
        start_date = today - timedelta(days=1)
        end_date = today - timedelta(days=1)
    elif selected_period == "過去7日間":
        start_date = today - timedelta(days=6)
        end_date = today
    elif selected_period == "過去14日間":
        start_date = today - timedelta(days=13)
        end_date = today
    elif selected_period == "過去30日間":
        start_date = today - timedelta(days=29)
        end_date = today
    elif selected_period == "今月":
        start_date = today.replace(day=1)
        end_date = today
    elif selected_period == "先月":
        last_month_end = today.replace(day=1) - timedelta(days=1)
        start_date = last_month_end.replace(day=1)
        end_date = last_month_end
    elif selected_period == "全期間":
        start_date = df['event_date'].min().date()
        end_date = df['event_date'].max().date()
    elif selected_period == "カスタム":
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("開始日", df['event_date'].min())
        with col2:
            end_date = st.date_input("終了日", df['event_date'].max())

    # ページ上部にフィルターを配置ここまで
    comparison_type = None # 初期化
    # 期間フィルターのみを適用したDataFrame（テーブル表示用）
    period_filtered_df = df[
        (df['event_date'] >= pd.to_datetime(start_date)) &
        (df['event_date'] <= pd.to_datetime(end_date))
    ]

    # --- 新規/リピート、CV/非CVの列を追加 ---
    df['user_type'] = np.where(df['ga_session_number'] == 1, '新規', 'リピート')
    conversion_session_ids = df[df['cv_type'].notna()]['session_id'].unique()
    df['conversion_status'] = np.where(df['session_id'].isin(conversion_session_ids), 'コンバージョン', '非コンバージョン')

    # KPIカードやグラフ用のデータフィルタリング（期間＋LP）
    filtered_df = df.copy()

    # 期間フィルター
    filtered_df = filtered_df[
        (filtered_df['event_date'] >= pd.to_datetime(start_date)) &
        (filtered_df['event_date'] <= pd.to_datetime(end_date))
    ]

    # LPフィルター
    if selected_lp_base_url:
        filtered_df = filtered_df[filtered_df['lp_base_url'] == selected_lp_base_url]

    # --- クロス分析用フィルター適用 ---
    if selected_device != "すべて":
        filtered_df = filtered_df[filtered_df['device_type'] == selected_device]

    if selected_user_type != "すべて":
        filtered_df = filtered_df[filtered_df['user_type'] == selected_user_type]

    if selected_conversion_status != "すべて":
        filtered_df = filtered_df[filtered_df['conversion_status'] == selected_conversion_status]

    if selected_channel != "すべて":
        filtered_df = filtered_df[filtered_df['channel'] == selected_channel]

    if selected_source_medium != "すべて":
        filtered_df = filtered_df[filtered_df['source_medium'] == selected_source_medium]

    # --- データダウンロード機能 ---
    st.markdown("##### フィルター適用後のデータをダウンロード")
    st.markdown('<div class="graph-description">現在選択されているフィルター条件で絞り込んだ生データをCSVファイルとしてダウンロードできます。日報や週報の作成にご活用ください。</div>', unsafe_allow_html=True)

    # CSVに変換する関数
    @st.cache_data()
    def convert_df_to_csv(df_to_convert):
        # IMPORTANT: utf-8-sig を使用してExcelでの日本語文字化けを防ぐ
        return df_to_convert.to_csv(index=False).encode('utf-8-sig')

    csv_data = convert_df_to_csv(filtered_df)
    
    # ダウンロードボタン
    st.download_button(
       label="CSV形式でダウンロード",
       data=csv_data,
       file_name=f"analysis_data_{pd.to_datetime(start_date).strftime('%Y%m%d')}_{pd.to_datetime(end_date).strftime('%Y%m%d')}.csv",
       mime='text/csv',
       use_container_width=True,
       type="primary"
    )

    # データが空の場合の処理
    if len(filtered_df) == 0:
        st.warning("⚠️ 選択した条件に該当するデータがありません。フィルターを変更してください。")
        st.stop()

    # 基本メトリクス計算
    total_sessions = filtered_df['session_id'].nunique()
    total_conversions = filtered_df[filtered_df['cv_type'].notna()]['session_id'].nunique()
    conversion_rate = safe_rate(total_conversions, total_sessions) * 100
    clicked_sessions = filtered_df[filtered_df['event_name'] == 'click']['session_id'].nunique()
    total_clicks = clicked_sessions
    click_rate = (total_clicks / total_sessions * 100) if total_sessions > 0 else 0
    avg_stay_time = filtered_df['stay_ms'].mean() / 1000  # 秒に変換
    avg_pages_reached = filtered_df.groupby('session_id')['max_page_reached'].max().mean()
    fv_retention_rate = (filtered_df[filtered_df['max_page_reached'] >= 2]['session_id'].nunique() / total_sessions * 100) if total_sessions > 0 else 0
    final_cta_rate = (filtered_df[filtered_df['max_page_reached'] >= 10]['session_id'].nunique() / total_sessions * 100) if total_sessions > 0 else 0
    avg_load_time = filtered_df['load_time_ms'].mean()

    st.markdown('<div class="sub-header">主要指標（KPI）</div>', unsafe_allow_html=True)

    # 主要KPIを算出して表示
    # 比較機能をKPIヘッダーの下に配置
    comp_cols = st.columns([1, 1, 4]) # チェックボックス、選択ボックス、スペーサー
    with comp_cols[0]:
        enable_comparison = st.checkbox("比較機能を有効化", value=False, key="summary_comparison_checkbox")
    with comp_cols[1]:
        if enable_comparison:
            comparison_options = {
                "前期間": "previous_period", "前週": "previous_week",
                "前月": "previous_month", "前年": "previous_year"
            }
            selected_comparison = st.selectbox("比較対象", list(comparison_options.keys()), key="summary_comparison_selector", label_visibility="collapsed")
            comparison_type = comparison_options[selected_comparison]

    # 比較データの取得
    comparison_df = None
    comp_start = None
    comp_end = None
    if enable_comparison and comparison_type:
        result = get_comparison_data(df, pd.Timestamp(start_date), pd.Timestamp(end_date), comparison_type)
        if result is not None:
            comparison_df, comp_start, comp_end = result
            # 比較データにも同じフィルターを適用
            if selected_lp_base_url:
                comparison_df = comparison_df[comparison_df['lp_base_url'] == selected_lp_base_url]
            # --- 比較データにもクロス分析用フィルターを適用 ---
            if selected_device != "すべて":
                comparison_df = comparison_df[comparison_df['device_type'] == selected_device]
            if selected_user_type != "すべて":
                comparison_df = comparison_df[comparison_df['user_type'] == selected_user_type]
            if selected_conversion_status != "すべて":
                comparison_df = comparison_df[comparison_df['conversion_status'] == selected_conversion_status]
            if selected_channel != "すべて":
                comparison_df = comparison_df[comparison_df['channel'] == selected_channel]
            if selected_channel != "すべて":
                comparison_df = comparison_df[comparison_df['source_medium'] == selected_source_medium]

            # 比較データが空の場合は無効化
            if len(comparison_df) == 0:
                comparison_df = None
                st.info(f"比較期間（{comp_start.strftime('%Y-%m-%d')} 〜 {comp_end.strftime('%Y-%m-%d')}）にデータがありません。")


    # 比較データのKPI計算
    comp_kpis = {}
    if comparison_df is not None and len(comparison_df) > 0:
        comp_total_sessions = comparison_df['session_id'].nunique()
        comp_total_conversions = comparison_df[comparison_df['cv_type'].notna()]['session_id'].nunique()
        comp_conversion_rate = (comp_total_conversions / comp_total_sessions * 100) if comp_total_sessions > 0 else 0
        comp_clicked_sessions = comparison_df[comparison_df['event_name'] == 'click']['session_id'].nunique()
        comp_total_clicks = comp_clicked_sessions
        comp_click_rate = (comp_total_clicks / comp_total_sessions * 100) if comp_total_sessions > 0 else 0
        comp_avg_stay_time = comparison_df['stay_ms'].mean() / 1000
        comp_avg_pages_reached = comparison_df.groupby('session_id')['max_page_reached'].max().mean()
        comp_fv_retention_rate = (comparison_df[comparison_df['max_page_reached'] >= 2]['session_id'].nunique() / comp_total_sessions * 100) if comp_total_sessions > 0 else 0
        comp_final_cta_rate = (comparison_df[comparison_df['max_page_reached'] >= 10]['session_id'].nunique() / comp_total_sessions * 100) if comp_total_sessions > 0 else 0
        comp_avg_load_time = comparison_df['load_time_ms'].mean()
        
        comp_kpis = {
            'sessions': comp_total_sessions,
            'conversions': comp_total_conversions,
            'conversion_rate': comp_conversion_rate,
            'clicks': comp_total_clicks,
            'click_rate': comp_click_rate,
            'avg_stay_time': comp_avg_stay_time,
            'avg_pages_reached': comp_avg_pages_reached,
            'fv_retention_rate': comp_fv_retention_rate,
            'final_cta_rate': comp_final_cta_rate,
            'avg_load_time': comp_avg_load_time
        }

    # KPIカード表示
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        # セッション数
        delta_sessions = total_sessions - comp_kpis.get('sessions', 0) if comp_kpis else None
        st.metric("セッション数", f"{total_sessions:,}", delta=f"{delta_sessions:+,}" if delta_sessions is not None else None)
        
        # FV残存率
        delta_fv = fv_retention_rate - comp_kpis.get('fv_retention_rate', 0) if comp_kpis else None
        st.metric("FV残存率", f"{fv_retention_rate:.1f}%", delta=f"{delta_fv:+.1f}%" if delta_fv is not None else None)

    with col2:
        # コンバージョン数
        delta_conversions = total_conversions - comp_kpis.get('conversions', 0) if comp_kpis else None
        st.metric("コンバージョン数", f"{total_conversions:,}", delta=f"{delta_conversions:+,}" if delta_conversions is not None else None)

        # 最終CTA到達率
        delta_cta = final_cta_rate - comp_kpis.get('final_cta_rate', 0) if comp_kpis else None
        st.metric("最終CTA到達率", f"{final_cta_rate:.1f}%", delta=f"{delta_cta:+.1f}%" if delta_cta is not None else None)

    with col3:
        # コンバージョン率
        delta_cvr = conversion_rate - comp_kpis.get('conversion_rate', 0) if comp_kpis else None
        st.metric("コンバージョン率", f"{conversion_rate:.2f}%", delta=f"{delta_cvr:+.2f}%" if delta_cvr is not None else None)

        # 平均到達ページ数
        delta_pages = avg_pages_reached - comp_kpis.get('avg_pages_reached', 0) if comp_kpis else None
        st.metric("平均到達ページ数", f"{avg_pages_reached:.1f}", delta=f"{delta_pages:+.1f}" if delta_pages is not None else None)

    with col4:
        # クリック数
        delta_clicks = total_clicks - comp_kpis.get('clicks', 0) if comp_kpis else None
        st.metric("クリック数", f"{total_clicks:,}", delta=f"{delta_clicks:+,}" if delta_clicks is not None else None)

        # 平均滞在時間
        delta_stay = avg_stay_time - comp_kpis.get('avg_stay_time', 0) if comp_kpis else None
        st.metric("平均滞在時間", f"{avg_stay_time:.1f}秒", delta=f"{delta_stay:+.1f} 秒" if delta_stay is not None else None)

    with col5:
        # クリック率
        delta_click_rate = click_rate - comp_kpis.get('click_rate', 0) if comp_kpis else None
        st.metric("クリック率", f"{click_rate:.2f}%", delta=f"{delta_click_rate:+.2f}%" if delta_click_rate is not None else None)

        # 平均読込時間
        delta_load = avg_load_time - comp_kpis.get('avg_load_time', 0) if comp_kpis else None
        st.metric("平均読込時間", f"{avg_load_time:.0f}ms", delta=f"{delta_load:+.0f} ms" if delta_load is not None else None, delta_color="inverse")

    # KPIスコアカードと日別KPIテーブルの間にスペースを設ける
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # 日別KPIテーブル
    st.markdown("##### 日別KPI詳細")
    st.markdown('<div class="graph-description">選択した期間内の日ごとの主要指標です。</div>', unsafe_allow_html=True)

    # 日別にKPIを計算
    daily_df = filtered_df.groupby(filtered_df['event_date'].dt.date).agg(
        セッション数=('session_id', 'nunique'),
        クリック数=('event_name', lambda x: (x == 'click').sum()), # この計算は後で上書きされるのでそのままでOK
        平均滞在時間=('stay_ms', 'mean')
    ).reset_index()
    daily_df.rename(columns={'event_date': '日付'}, inplace=True)

    # --- 平均到達ページ数の正しい計算ロジック ---
    # 日ごと、セッションごとに最大到達ページ数を取得
    daily_session_max_page = filtered_df.groupby([filtered_df['event_date'].dt.date, 'session_id'])['max_page_reached'].max().reset_index()
    # 日ごとにその平均を計算
    daily_avg_pages = daily_session_max_page.groupby('event_date')['max_page_reached'].mean().reset_index()
    daily_avg_pages.rename(columns={'event_date': '日付', 'max_page_reached': '平均到達ページ'}, inplace=True)

    # 日別コンバージョン数
    daily_cv = filtered_df[filtered_df['cv_type'].notna()].groupby(filtered_df['event_date'].dt.date)['session_id'].nunique().reset_index()
    daily_cv.columns = ['日付', 'CV数']
    daily_df = pd.merge(daily_df, daily_cv, on='日付', how='left').fillna(0)

    # 日別FV残存数
    daily_fv = filtered_df[filtered_df['max_page_reached'] >= 2].groupby(filtered_df['event_date'].dt.date)['session_id'].nunique().reset_index()
    daily_fv.columns = ['日付', 'FV残存数']
    daily_df = pd.merge(daily_df, daily_fv, on='日付', how='left').fillna(0)

    # 日別最終CTA到達数
    daily_final_cta = filtered_df[filtered_df['max_page_reached'] >= 10].groupby(filtered_df['event_date'].dt.date)['session_id'].nunique().reset_index()
    daily_final_cta.columns = ['日付', '最終CTA到達数']
    daily_df = pd.merge(daily_df, daily_final_cta, on='日付', how='left').fillna(0)

    # 率を計算
    daily_df['CVR'] = daily_df.apply(lambda row: safe_rate(row['CV数'], row['セッション数']) * 100, axis=1)
    daily_df['CTR'] = daily_df.apply(lambda row: safe_rate(row['クリック数'], row['セッション数']) * 100, axis=1)
    daily_df['FV残存率'] = daily_df.apply(lambda row: safe_rate(row['FV残存数'], row['セッション数']) * 100, axis=1)
    daily_df['最終CTA到達率'] = daily_df.apply(lambda row: safe_rate(row['最終CTA到達数'], row['セッション数']) * 100, axis=1)
    daily_df['平均滞在時間'] = daily_df['平均滞在時間'] / 1000
    # 正しく計算した平均到達ページ数をマージ
    daily_df = pd.merge(daily_df, daily_avg_pages, on='日付', how='left')

    # 日付を降順にソート
    daily_df = daily_df.sort_values(by='日付', ascending=False)

    # 表示する列を選択
    display_cols_daily = ['日付', 'セッション数', 'CV数', 'CVR', 'クリック数', 'CTR', 'FV残存率', '最終CTA到達率', '平均到達ページ', '平均滞在時間']
    
    # データフレームを表示（7行分の高さに固定）
    st.dataframe(daily_df[display_cols_daily].style.format({
        'CVR': '{:.2f}%', 'CTR': '{:.2f}%', 'FV残存率': '{:.1f}%', '最終CTA到達率': '{:.1f}%',
        '平均到達ページ': '{:.1f}', '平均滞在時間': '{:.1f}秒'
    }), use_container_width=True, height=282, hide_index=True)
    # page_pathごとのKPIを計算（期間フィルターのみ適用したデータを使用）
    path_sessions = period_filtered_df.groupby('page_path')['session_id'].nunique()
    path_users = period_filtered_df.groupby('page_path')['user_pseudo_id'].nunique()
    path_conversions = period_filtered_df[period_filtered_df['cv_type'].notna()].groupby('page_path')['session_id'].nunique()
    path_clicks = period_filtered_df[period_filtered_df['event_name'] == 'click'].groupby('page_path').size()
    kpi_by_path = pd.DataFrame({
        'セッション数': path_sessions,
        'CV数': path_conversions,
        'クリック数': path_clicks,
        '平均滞在時間': period_filtered_df.groupby('page_path')['stay_ms'].mean() / 1000,
        '平均到達ページ': period_filtered_df.groupby('page_path')['max_page_reached'].mean()
    }).fillna(0)

    kpi_by_path['CVR'] = kpi_by_path.apply(lambda row: safe_rate(row['CV数'], row['セッション数']) * 100, axis=1)
    kpi_by_path['CTR'] = kpi_by_path.apply(lambda row: safe_rate(row['クリック数'], row['セッション数']) * 100, axis=1)
    # FV残存率
    fv_sessions = period_filtered_df[period_filtered_df['max_page_reached'] >= 2].groupby('page_path')['session_id'].nunique()
    kpi_by_path['FV残存率'] = (safe_rate(fv_sessions, path_sessions) * 100).fillna(0) # safe_rateがSeriesを返すように
    # 最終CTA到達率
    final_cta_sessions = period_filtered_df[period_filtered_df['max_page_reached'] >= 10].groupby('page_path')['session_id'].nunique()
    kpi_by_path['最終CTA到達率'] = (safe_rate(final_cta_sessions, path_sessions) * 100).fillna(0) # こちらも同様に修正

    kpi_by_path = kpi_by_path.reset_index()
    kpi_by_path.rename(columns={'page_path': 'ページパス'}, inplace=True)

    # 表示する列を定義（変更なし）
    display_cols = [
        'ページパス', 'セッション数', 'CV数', 'CVR', 'クリック数', 'CTR', 
        'FV残存率', '最終CTA到達率', '平均到達ページ', '平均滞在時間'
    ]

    # --- インタラクションKPIの計算ロジック（期間フィルターのみ適用したデータを使用） ---
    # CTAクリック
    cta_clicks = period_filtered_df[
        (period_filtered_df['event_name'] == 'click') & 
        (period_filtered_df['elem_classes'].str.contains('cta|btn-primary', na=False))
    ].groupby('page_path').size()

    # フローティングバナークリック
    floating_clicks = period_filtered_df[
        (period_filtered_df['event_name'] == 'click') & 
        (period_filtered_df['elem_classes'].str.contains('floating', na=False))
    ].groupby('page_path').size()

    # 離脱防止ポップアップクリック
    exit_popup_clicks = period_filtered_df[
        (period_filtered_df['event_name'] == 'click') & 
        (period_filtered_df['elem_classes'].str.contains('exit', na=False))
    ].groupby('page_path').size()

    interaction_kpis = pd.DataFrame({
        'セッション数': path_sessions,
        'ユニークユーザー数': path_users,
        'CTAクリック数': cta_clicks,
        'FBクリック数': floating_clicks,
        '離脱防止POPクリック数': exit_popup_clicks
    }).fillna(0)

    interaction_kpis['CTAクリック率'] = interaction_kpis.apply(lambda row: safe_rate(row['CTAクリック数'], row['セッション数']) * 100, axis=1)
    interaction_kpis['FBクリック率'] = interaction_kpis.apply(lambda row: safe_rate(row['FBクリック数'], row['セッション数']) * 100, axis=1)
    interaction_kpis['離脱防止POPクリック率'] = interaction_kpis.apply(lambda row: safe_rate(row['離脱防止POPクリック数'], row['セッション数']) * 100, axis=1)

    interaction_kpis = interaction_kpis.reset_index().rename(columns={'page_path': 'ページパス'})

    interaction_display_cols = [
        'ページパス', 'ユニークユーザー数', 'CTAクリック数', 'CTAクリック率',
        'FBクリック数', 'FBクリック率',
        '離脱防止POPクリック数', '離脱防止POPクリック率'
    ]
    
    # --- expanderを使って表を表示 ---
    with st.expander("詳細1: ページパス別 主要指標詳細表"):
        st.markdown('<div class="graph-description">LP（ページパス）ごとの主要なKPIを一覧表示します。</div>', unsafe_allow_html=True) # type: ignore
        st.dataframe(kpi_by_path[display_cols].style.format({
            'セッション数': '{:,.0f}',
            'CV数': '{:,.0f}',
            'CVR': '{:.2f}%',
            'クリック数': '{:,.0f}',
            'CTR': '{:.2f}%',
            'FV残存率': '{:.2f}%',
            '最終CTA到達率': '{:.2f}%',
            '平均到達ページ': '{:.2f}',
            '平均滞在時間': '{:.1f}秒'
        }), use_container_width=True, hide_index=True)

    with st.expander("詳細2: ページパス別 インタラクション指標詳細表"):
        st.markdown('<div class="graph-description">LP（ページパス）ごとの主要なインタラクション指標を一覧表示します。</div>', unsafe_allow_html=True) # type: ignore
        st.dataframe(interaction_kpis[interaction_display_cols].style.format({
            'ユニークユーザー数': '{:,.0f}',
            'CTAクリック数': '{:,.0f}',
            'CTAクリック率': '{:.2f}%',
            'FBクリック数': '{:,.0f}',
            'FBクリック率': '{:.2f}%',
            '離脱防止POPクリック数': '{:,.0f}',
            '離脱防止POPクリック率': '{:.2f}%'
        }), use_container_width=True, hide_index=True)

    st.markdown("---")
    
    # グラフ選択
    st.markdown("**表示するグラフを選択してください:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_session_trend = st.checkbox("セッション数の推移", value=False, key="summary_show_session_trend") # デフォルトをFalseに変更
        show_cvr_trend = st.checkbox("コンバージョン率の推移", value=False, key="summary_show_cvr_trend") # デフォルトをFalseに変更
        show_device_breakdown = st.checkbox("デバイス別分析", value=False, key="summary_show_device_breakdown") # デフォルトをFalseに変更
    
    with col2:
        show_channel_breakdown = st.checkbox("チャネル別分析", value=False, key="summary_show_channel_breakdown") # デフォルトをFalseに変更
        show_funnel = st.checkbox("LP進行ファネル", value=False, key="summary_show_funnel") # デフォルトをFalseに変更
        show_hourly_cvr = st.checkbox("時間帯別CVR", value=False, key="summary_show_hourly_cvr")
    
    with col3:
        show_dow_cvr = st.checkbox("曜日別CVR", value=False, key="summary_show_dow_cvr") # type: ignore
        show_utm_analysis = st.checkbox("UTM分析", value=False, key="summary_show_utm_analysis") # デフォルトをFalseに変更
        show_load_time = st.checkbox("読込時間分析", value=False, key="summary_show_load_time") # デフォルトをFalseに変更
    
    # セッション数の推移
    if show_session_trend:
        st.markdown("#### セッション数の推移")
        st.markdown('<div class="graph-description">日ごとのセッション数（訪問数）の変化を表示します。トレンドや曜日ごとのパターンを把握できます。</div>', unsafe_allow_html=True) # type: ignore
        daily_sessions = filtered_df.groupby(filtered_df['event_date'].dt.date)['session_id'].nunique().reset_index()
        daily_sessions.columns = ['日付', 'セッション数']
        
        if comparison_df is not None and len(comparison_df) > 0:
            # 比較データを追加
            comp_daily_sessions = comparison_df.groupby(comparison_df['event_date'].dt.date)['session_id'].nunique().reset_index()
            comp_daily_sessions.columns = ['日付', '比較期間セッション数']

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=daily_sessions['日付'], y=daily_sessions['セッション数'], 
                                    mode='lines+markers', name='現在期間', line=dict(color='#002060'),
                                    hovertemplate='日付: %{x}<br>セッション数: %{y:,}<extra></extra>'))
            fig.add_trace(go.Scatter(x=comp_daily_sessions['日付'], y=comp_daily_sessions['比較期間セッション数'], 
                                    mode='lines+markers', name='比較期間', line=dict(color='#999999', dash='dash'),
                                    hovertemplate='日付: %{x}<br>比較期間セッション数: %{y:,}<extra></extra>'))
            fig.update_layout(height=400, hovermode='x unified')
            fig.update_layout(dragmode=False)
        else:
            fig = px.line(daily_sessions, x='日付', y='セッション数', markers=True)
            fig.update_layout(height=400, dragmode=False)
        
        st.plotly_chart(fig, use_container_width=True, key='plotly_chart_1') # This already has use_container_width=True
    
    # コンバージョン率の推移
    if show_cvr_trend:
        st.markdown("#### コンバージョン率の推移")
        st.markdown('<div class="graph-description">日ごとのコンバージョン率（CVR）の変化を表示します。LPの改善効果や外部要因の影響を確認できます。</div>', unsafe_allow_html=True) # type: ignore
        daily_cvr = filtered_df.groupby(filtered_df['event_date'].dt.date).agg({
            'session_id': 'nunique',
        }).reset_index()
        daily_cvr.columns = ['日付', 'セッション数']
        
        daily_cv = filtered_df[filtered_df['cv_type'].notna()].groupby(
            filtered_df[filtered_df['cv_type'].notna()]['event_date'].dt.date
        )['session_id'].nunique().reset_index()
        daily_cv.columns = ['日付', 'コンバージョン数']
        
        daily_cvr = daily_cvr.merge(daily_cv, on='日付', how='left').fillna(0) # type: ignore
        daily_cvr['コンバージョン率'] = daily_cvr.apply(lambda row: safe_rate(row['コンバージョン数'], row['セッション数']) * 100, axis=1)
        
        if comparison_df is not None and len(comparison_df) > 0:
            # 比較データを追加
            comp_daily_cvr = comparison_df.groupby(comparison_df['event_date'].dt.date).agg({'session_id': 'nunique'}).reset_index()
            comp_daily_cvr.columns = ['日付', 'セッション数']
            
            comp_daily_cv = comparison_df[comparison_df['cv_type'].notna()].groupby(
                comparison_df[comparison_df['cv_type'].notna()]['event_date'].dt.date
            )['session_id'].nunique().reset_index()
            comp_daily_cv.columns = ['日付', 'コンバージョン数']
            
            comp_daily_cvr = comp_daily_cvr.merge(comp_daily_cv, on='日付', how='left').fillna(0) # type: ignore
            comp_daily_cvr['比較期間CVR'] = comp_daily_cvr.apply(lambda row: safe_rate(row['コンバージョン数'], row['セッション数']) * 100, axis=1)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=daily_cvr['日付'], y=daily_cvr['コンバージョン率'],
                                    mode='lines+markers', name='現在期間', line=dict(color='#002060'),
                                    hovertemplate='日付: %{x}<br>コンバージョン率: %{y:.2f}%<extra></extra>'))
            fig.add_trace(go.Scatter(x=comp_daily_cvr['日付'], y=comp_daily_cvr['比較期間CVR'], 
                                    mode='lines+markers', name='比較期間', line=dict(color='#999999', dash='dash'),
                                    hovertemplate='日付: %{x}<br>比較期間CVR: %{y:.2f}%<extra></extra>'))
            fig.update_layout(height=400, hovermode='x unified', yaxis_title='コンバージョン率 (%)')
            fig.update_layout(dragmode=False)
        else:
            fig = px.line(daily_cvr, x='日付', y='コンバージョン率', markers=True)
            fig.update_layout(height=400, dragmode=False)
        
        st.plotly_chart(fig, use_container_width=True, key='plotly_chart_2') # This already has use_container_width=True
    
    # デバイス別分析
    if show_device_breakdown:
        st.markdown("#### デバイス別分析")
        st.markdown('<div class="graph-description">デバイス（スマホ、PC、タブレット）ごとのセッション数、コンバージョン数、CVRを比較します。デバイス最適化の優先度を判断できます。</div>', unsafe_allow_html=True) # type: ignore
        device_stats = filtered_df.groupby('device_type').agg({
            'session_id': 'nunique',
        }).reset_index()
        device_stats.columns = ['デバイス', 'セッション数']
        
        device_cv = filtered_df[filtered_df['cv_type'].notna()].groupby('device_type')['session_id'].nunique().reset_index()
        device_cv.columns = ['デバイス', 'コンバージョン数']
        
        device_stats = device_stats.merge(device_cv, on='デバイス', how='left').fillna(0)
        device_stats['コンバージョン率'] = device_stats.apply(lambda row: safe_rate(row['コンバージョン数'], row['セッション数']) * 100, axis=1)
        
        fig = go.Figure()
        # 主軸（左Y軸）にセッション数とコンバージョン数の棒グラフを追加
        fig.add_trace(go.Bar(name='セッション数', x=device_stats['デバイス'], y=device_stats['セッション数'], yaxis='y', offsetgroup=1,
                             hovertemplate='デバイス: %{x}<br>セッション数: %{y:,}<extra></extra>'))
        fig.add_trace(go.Bar(name='コンバージョン数', x=device_stats['デバイス'], y=device_stats['コンバージョン数'], yaxis='y', offsetgroup=2,
                             hovertemplate='デバイス: %{x}<br>コンバージョン数: %{y:,}<extra></extra>'))
        # 第二軸（右Y軸）にコンバージョン率の折れ線グラフを追加
        fig.add_trace(go.Scatter(name='コンバージョン率', x=device_stats['デバイス'], y=device_stats['コンバージョン率'], yaxis='y2', mode='lines+markers',
                                 hovertemplate='デバイス: %{x}<br>コンバージョン率: %{y:.2f}%<extra></extra>'))
        
        fig.update_layout(
            yaxis=dict(title='セッション数 / コンバージョン数'),
            yaxis2=dict(title='コンバージョン率 (%)', overlaying='y', side='right', showgrid=False),
            height=400,
            dragmode=False, # type: ignore
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig, use_container_width=True, key='plotly_chart_device_combined')
    
    # チャネル別分析
    if show_channel_breakdown:
        st.markdown("#### チャネル別分析")
        st.markdown('<div class="graph-description">流入経路（Google、SNS、直接アクセスなど）ごとのパフォーマンスを比較します。効果的な集客チャネルを特定できます。</div>', unsafe_allow_html=True) # type: ignore
        channel_stats = filtered_df.groupby('channel').agg({
            'session_id': 'nunique',
            'stay_ms': 'mean'
        }).reset_index()
        channel_stats.columns = ['チャネル', 'セッション数', '平均滞在時間(ms)']
        channel_stats['平均滞在時間(秒)'] = channel_stats['平均滞在時間(ms)'] / 1000
        
        channel_cv = filtered_df[filtered_df['cv_type'].notna()].groupby('channel')['session_id'].nunique().reset_index()
        channel_cv.columns = ['チャネル', 'コンバージョン数']
        
        channel_stats = channel_stats.merge(channel_cv, on='チャネル', how='left').fillna(0)
        channel_stats['コンバージョン率'] = channel_stats.apply(lambda row: safe_rate(row['コンバージョン数'], row['セッション数']) * 100, axis=1)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(channel_stats, values='セッション数', names='チャネル', title='チャネル別セッション数')
            fig.update_traces(hovertemplate='チャネル: %{label}<br>セッション数: %{value:,} (%{percent})<extra></extra>')
            fig.update_layout(
                dragmode=False,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig, use_container_width=True, key='plotly_chart_4')
        
        with col2:
            fig = px.bar(channel_stats, x='チャネル', y='コンバージョン率', title='チャネル別コンバージョン率')
            fig.update_traces(hovertemplate='チャネル: %{x}<br>コンバージョン率: %{y:.2f}%<extra></extra>')
            fig.update_layout(
                dragmode=False,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig, use_container_width=True, key='plotly_chart_5')

    # LP進行ファネルと滞在時間別ファネル
    if show_funnel:
        st.markdown("#### LP進行状況とページ内滞在時間")

        # LPのページ数をデータから動的に取得
        actual_page_count = int(filtered_df['page_num_dom'].max()) if not filtered_df['page_num_dom'].dropna().empty else 10

        col1, col2 = st.columns(2)

        with col1:
            funnel_data = []
            for page_num in range(1, actual_page_count + 1):
                count = filtered_df[filtered_df['max_page_reached'] >= page_num]['session_id'].nunique()
                funnel_data.append({'ページ': f'ページ{page_num}', 'セッション数': count})
            
            funnel_df = pd.DataFrame(funnel_data)
            
            fig_funnel = go.Figure(go.Funnel(
                y=funnel_df['ページ'],
                x=funnel_df['セッション数'],
                textinfo="value+percent initial",
                hovertemplate='ページ: %{y}<br>セッション数: %{x:,}<extra></extra>'
            )) # type: ignore
            fig_funnel.update_layout(height=600, dragmode=False)
            st.markdown("**LP進行ファネル**")
            st.markdown('<div class="graph-description">各ページに到達したセッション数と、次のページへの遷移率です。急激に減少している箇所が大きな離脱ポイントです。</div>', unsafe_allow_html=True) # type: ignore
            st.plotly_chart(fig_funnel, use_container_width=True, key='plotly_chart_funnel_revived')

        with col2:
            # 滞在時間セグメントを定義
            stay_segments_for_calc = [
                ('0-10秒', 0, 10000),
                ('10-30秒', 10000, 30000),
                ('30-60秒', 30000, 60000),
                ('1-3分', 60000, 180000),
                ('3分以上', 180000, float('inf'))
            ]
            
            # ページごとの滞在時間別セッション数を計算
            page_stay_data = []
            for page_num in range(1, actual_page_count + 1):
                # そのページに到達したセッションIDを取得
                reached_session_ids = set(filtered_df[filtered_df['max_page_reached'] >= page_num]['session_id'].unique())
                total_reached = len(reached_session_ids)
                
                # そのページでの滞在時間イベントを持つセッションを取得
                page_specific_stay = filtered_df[
                    (filtered_df['page_num_dom'] == page_num) & 
                    (filtered_df['session_id'].isin(reached_session_ids)) &
                    (filtered_df['stay_ms'].notna()) # NaN値を除外
                ]
                
                row = {'ページ': f'ページ{page_num}', 'ページ番号': page_num}
                
                # そのページで滞在時間イベントがあったセッションの総数
                total_sessions_with_stay = page_specific_stay['session_id'].nunique()

                for label, min_ms, max_ms in stay_segments_for_calc:
                    segment_sessions_count = page_specific_stay[
                        (page_specific_stay['stay_ms'] >= min_ms) & 
                        (page_specific_stay['stay_ms'] < max_ms)
                    ]['session_id'].nunique()
                    
                    # 滞在時間イベントがあったセッション内での割合を計算
                    row[label] = (segment_sessions_count / total_sessions_with_stay * 100) if total_sessions_with_stay > 0 else 0
                
                page_stay_data.append(row)

            page_stay_df = pd.DataFrame(page_stay_data).sort_values('ページ番号', ascending=False)

            # 積み上げ棒グラフでファネルを表現
            fig_stay_pct = go.Figure()            
            # YlGnBuスケールから濃い青系の5色を選択
            colors = px.colors.sequential.YlGnBu[2:7]
            colors[-1] = '#08306b' # 一番濃い色を濃紺に設定
            
            for i, (label, _, _) in enumerate(stay_segments_for_calc):
                fig_stay_pct.add_trace(go.Bar(
                    y=page_stay_df['ページ'],
                    x=page_stay_df[label],
                    name=label,
                    orientation='h', # type: ignore
                    hovertemplate='ページ: %{y}<br>割合: %{x:.2f}%<extra></extra>',
                    marker_color=colors[i]
                ))

            fig_stay_pct.update_layout(barmode='stack', height=600,
                              xaxis_title='セッションの割合 (%)', yaxis_title='ページ', dragmode=False,
                              xaxis_ticksuffix='%', legend=dict(traceorder='normal', orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5))
            st.markdown("**ページ内滞在時間の分布**")
            st.markdown('<div class="graph-description">各ページに到達し、滞在時間が計測されたセッションの行動内訳です。横軸は割合（%）を表します。</div>', unsafe_allow_html=True) # type: ignore
            st.plotly_chart(fig_stay_pct, use_container_width=True, key='plotly_chart_stay_percentage')
    
    # 時間帯別CVR
    if show_hourly_cvr:
        st.markdown("#### 時間帯別コンバージョン率")
        st.markdown('<div class="graph-description">1日の中で、どの時間帯にCVRが高いかを分析します。広告配信の最適な時間帯を見つけることができます。</div>', unsafe_allow_html=True) # type: ignore
        filtered_df['hour'] = filtered_df['event_timestamp'].dt.hour
        
        hourly_sessions = filtered_df.groupby('hour')['session_id'].nunique().reset_index()
        hourly_sessions.columns = ['時間', 'セッション数']
        
        hourly_cv = filtered_df[filtered_df['cv_type'].notna()].groupby('hour')['session_id'].nunique().reset_index()
        hourly_cv.columns = ['時間', 'コンバージョン数']
        
        hourly_cvr = hourly_sessions.merge(hourly_cv, on='時間', how='left').fillna(0)
        hourly_cvr['コンバージョン率'] = hourly_cvr.apply(lambda row: safe_rate(row['コンバージョン数'], row['セッション数']) * 100, axis=1)
        
        fig = px.bar(hourly_cvr, x='時間', y='コンバージョン率')
        fig.update_traces(hovertemplate='時間: %{x}時台<br>コンバージョン率: %{y:.2f}%<extra></extra>')
        fig.update_layout(height=400, xaxis_title='時間帯', dragmode=False, legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5))
        st.plotly_chart(fig, use_container_width=True, key='plotly_chart_7')
    
    # 曜日別CVR
    if show_dow_cvr:
        st.markdown("#### 曜日別コンバージョン率")
        st.markdown('<div class="graph-description">曜日ごとのCVRの違いを分析します。平日と週末でのユーザー行動の変化を把握できます。</div>', unsafe_allow_html=True) # type: ignore
        filtered_df['dow'] = filtered_df['event_timestamp'].dt.day_name()
        dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_map = {'Monday': '月', 'Tuesday': '火', 'Wednesday': '水', 'Thursday': '木', 'Friday': '金', 'Saturday': '土', 'Sunday': '日'}
        
        dow_sessions = filtered_df.groupby('dow')['session_id'].nunique().reset_index()
        dow_sessions.columns = ['曜日', 'セッション数']
        
        dow_cv = filtered_df[filtered_df['cv_type'].notna()].groupby('dow')['session_id'].nunique().reset_index()
        dow_cv.columns = ['曜日', 'コンバージョン数']
        
        dow_cvr = dow_sessions.merge(dow_cv, on='曜日', how='left').fillna(0)
        dow_cvr['コンバージョン率'] = dow_cvr.apply(lambda row: safe_rate(row['コンバージョン数'], row['セッション数']) * 100, axis=1)
        dow_cvr['曜日_日本語'] = dow_cvr['曜日'].map(dow_map)
        dow_cvr['曜日_order'] = dow_cvr['曜日'].apply(lambda x: dow_order.index(x))
        dow_cvr = dow_cvr.sort_values('曜日_order')
        
        fig = px.bar(dow_cvr, x='曜日_日本語', y='コンバージョン率')
        fig.update_traces(hovertemplate='曜日: %{x}<br>コンバージョン率: %{y:.2f}%<extra></extra>')
        fig.update_layout(height=400, xaxis_title='曜日', dragmode=False, legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5))
        st.plotly_chart(fig, use_container_width=True, key='plotly_chart_8')
    
    # UTM分析
    if show_utm_analysis:
        st.markdown("#### UTM分析")
        st.markdown('<div class="graph-description">UTMパラメータ（広告タグ）ごとのトラフィックを分析します。どのキャンペーンや媒体が効果的かを把握できます。</div>', unsafe_allow_html=True) # type: ignore
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**UTMソース別**")
            utm_source_stats = filtered_df.groupby('utm_source')['session_id'].nunique().reset_index()
            utm_source_stats.columns = ['UTMソース', 'セッション数']
            utm_source_stats = utm_source_stats.sort_values('セッション数', ascending=False)
            
            fig = px.bar(utm_source_stats, x='UTMソース', y='セッション数')
            fig.update_layout(dragmode=False, legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5))
            fig.update_traces(hovertemplate='UTMソース: %{x}<br>セッション数: %{y:,}<extra></extra>')
            st.plotly_chart(fig, use_container_width=True, key='plotly_chart_9') # type: ignore
        
        with col2:
            st.markdown("**UTMメディア別**")
            utm_medium_stats = filtered_df.groupby('utm_medium')['session_id'].nunique().reset_index()
            utm_medium_stats.columns = ['UTMメディア', 'セッション数']
            utm_medium_stats = utm_medium_stats.sort_values('セッション数', ascending=False)

            fig = px.bar(utm_medium_stats, x='UTMメディア', y='セッション数')
            fig.update_layout(dragmode=False, legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5))
            fig.update_traces(hovertemplate='UTMメディア: %{x}<br>セッション数: %{y:,}<extra></extra>')
            st.plotly_chart(fig, use_container_width=True, key='plotly_chart_10') # type: ignore
    
    # 読込時間分析
    if show_load_time:
        st.markdown("#### 読込時間分析")
        st.markdown('<div class="graph-description">デバイスごとのページ読込時間を分析します。読込が遅いと離脱率が上がるため、最適化が重要です。</div>', unsafe_allow_html=True) # type: ignore
        
        load_time_stats = filtered_df.groupby('device_type')['load_time_ms'].mean().reset_index()
        load_time_stats.columns = ['デバイス', '平均読込時間(ms)']
        
        fig = px.bar(load_time_stats, x='デバイス', y='平均読込時間(ms)')
        fig.update_traces(hovertemplate='デバイス: %{x}<br>平均読込時間: %{y:.0f}ms<extra></extra>')
        fig.update_layout(height=400, yaxis_title='平均読込時間 (ms)', dragmode=False, legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5))
        st.plotly_chart(fig, use_container_width=True, key='plotly_chart_11')

    st.markdown("---")

    # --- AI分析と考察 ---
    st.markdown("### AIによる分析と考察")
    st.markdown('<div class="graph-description">LP全体の主要指標に基づき、AIが現状の評価と次のアクションを提案します。</div>', unsafe_allow_html=True)
    
    # AI分析の表示状態を管理
    if 'summary_ai_open' not in st.session_state:
        st.session_state.summary_ai_open = False

    if st.button("AI分析を実行", key="summary_ai_btn", type="primary", use_container_width=True):
        st.session_state.summary_ai_open = True

    if st.session_state.summary_ai_open:
        with st.container():
            with st.spinner("AIが全体データを分析中..."):
                kpi_data = {
                    "sessions": total_sessions,
                    "conversions": total_conversions,
                    "conversion_rate": conversion_rate,
                    "avg_stay_time": avg_stay_time,
                    "fv_retention_rate": fv_retention_rate,
                    "final_cta_rate": final_cta_rate
                }
                analysis_result = ai_analysis.analyze_overall_performance(kpi_data, comp_kpis if enable_comparison else None)
                st.markdown(analysis_result)

            if st.button("AI分析を閉じる", key="summary_ai_close"):
                st.session_state.summary_ai_open = False

    # --- よくある質問 ---
    st.markdown("#### このページの分析について質問する")
    if 'summary_faq_toggle' not in st.session_state:
        st.session_state.summary_faq_toggle = {1: False, 2: False, 3: False, 4: False}

    faq_cols = st.columns(2)
    with faq_cols[0]:
        if st.button("このLPの強みと弱みは？", key="faq_summary_1", use_container_width=True):
            st.session_state.summary_faq_toggle[1] = not st.session_state.summary_faq_toggle[1]
            st.session_state.summary_faq_toggle[2], st.session_state.summary_faq_toggle[3], st.session_state.summary_faq_toggle[4] = False, False, False
        if st.session_state.summary_faq_toggle[1]:
            st.info(f"**強み**は、平均滞在時間が{avg_stay_time:.1f}秒と比較的長く、コンテンツに興味を持ったユーザーは読み進めている点です。\n\n**弱み**は、FV残存率が{fv_retention_rate:.1f}%と低く、多くのユーザーが最初のページで離脱している点です。")
        
        if st.button("パフォーマンスが悪い原因を特定するには？", key="faq_summary_3", use_container_width=True):
            st.session_state.summary_faq_toggle[3] = not st.session_state.summary_faq_toggle[3]
            st.session_state.summary_faq_toggle[1], st.session_state.summary_faq_toggle[2], st.session_state.summary_faq_toggle[4] = False, False, False
        if st.session_state.summary_faq_toggle[3]:
            st.info("まず「ページ分析」で離脱率が高いボトルネックページを特定します。次に「セグメント分析」で、特定のデバイス（例：スマホ）やチャネル（例：SNS経由）のパフォーマンスが特に悪いかを確認することで、原因を絞り込めます。")
    with faq_cols[1]:
        if st.button("最も優先して改善すべき指標は？", key="faq_summary_2", use_container_width=True):
            st.session_state.summary_faq_toggle[2] = not st.session_state.summary_faq_toggle[2]
            st.session_state.summary_faq_toggle[1], st.session_state.summary_faq_toggle[3], st.session_state.summary_faq_toggle[4] = False, False, False
        if st.session_state.summary_faq_toggle[2]:
            st.info(f"**FV残存率（現在{fv_retention_rate:.1f}%）**です。多くのユーザーがLPの入口で離脱しているため、ここを改善することが最もインパクトが大きいです。")
        
        if st.button("次にどの分析を見るべき？", key="faq_summary_4", use_container_width=True):
            st.session_state.summary_faq_toggle[4] = not st.session_state.summary_faq_toggle[4]
            st.session_state.summary_faq_toggle[1], st.session_state.summary_faq_toggle[2], st.session_state.summary_faq_toggle[3] = False, False, False
        if st.session_state.summary_faq_toggle[4]:
            st.info("「**ページ分析**」がおすすめです。ユーザーがどのページで最も離脱しているか（ボトルネック）を特定し、具体的な改善箇所を見つけましょう。")


# 続く...（次のファイルでタブ2以降を実装）



# タブ2: ページ分析
elif selected_analysis == "ページ分析":
    st.markdown('<div class="sub-header">ページ分析</div>', unsafe_allow_html=True)

    # メインエリア: フィルターと比較設定
    st.markdown('<div class="sub-header">フィルター設定</div>', unsafe_allow_html=True)

    filter_cols_1 = st.columns(4)
    filter_cols_2 = st.columns(4)

    with filter_cols_1[0]:
        # 期間選択
        period_options = [
            "今日", "昨日", "過去7日間", "過去14日間", "過去30日間", "今月", "先月", "全期間", "カスタム"
        ]
        selected_period = st.selectbox("期間を選択", period_options, index=2, key="page_analysis_period")

    with filter_cols_1[1]:
        # LP選択
        lp_options = sorted(df['lp_base_url'].dropna().unique().tolist())
        selected_lp_base_url = st.selectbox(
            "LP選択", 
            lp_options, 
            index=0 if lp_options else None,
            key="page_analysis_lp",
            disabled=not lp_options
        )

    with filter_cols_1[2]:
        device_options = ["すべて"] + sorted(df['device_type'].dropna().unique().tolist())
        selected_device = st.selectbox("デバイス選択", device_options, index=0, key="page_analysis_device")

    with filter_cols_1[3]:
        # 新規/リピート フィルター
        user_type_options = ["すべて", "新規", "リピート"]
        selected_user_type = st.selectbox("新規/リピート", user_type_options, index=0, key="page_analysis_user_type")

    with filter_cols_2[0]:
        # CV/非CV フィルター
        conversion_status_options = ["すべて", "コンバージョン", "非コンバージョン"]
        selected_conversion_status = st.selectbox("CV/非CV", conversion_status_options, index=0, key="page_analysis_conversion_status")
    
    with filter_cols_2[1]:
        # チャネルフィルターを追加
        channel_options = ["すべて"] + sorted(df['channel'].unique().tolist())
        selected_channel = st.selectbox("チャネル", channel_options, index=0, key="page_analysis_channel")

    with filter_cols_2[2]:
        # チャネルフィルターを「参照元/メディア」に変更
        source_medium_options = ["すべて"] + sorted(df['source_medium'].unique().tolist())
        selected_source_medium = st.selectbox("参照元/メディア", source_medium_options, index=0, key="page_analysis_source_medium") # ラベルは変更済み
    
    enable_comparison = False
    # 期間設定
    today = df['event_date'].max().date()
    
    if selected_period == "今日":
        start_date = today
        end_date = today
    elif selected_period == "昨日":
        start_date = today - timedelta(days=1)
        end_date = today - timedelta(days=1)
    elif selected_period == "過去7日間":
        start_date = today - timedelta(days=6)
        end_date = today
    elif selected_period == "過去14日間":
        start_date = today - timedelta(days=13)
        end_date = today
    elif selected_period == "過去30日間":
        start_date = today - timedelta(days=29)
        end_date = today
    elif selected_period == "今月":
        start_date = today.replace(day=1)
        end_date = today
    elif selected_period == "先月":
        last_month_end = today.replace(day=1) - timedelta(days=1)
        start_date = last_month_end.replace(day=1)
        end_date = last_month_end
    elif selected_period == "全期間":
        start_date = df['event_date'].min().date()
        end_date = df['event_date'].max().date()
    elif selected_period == "カスタム":
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("開始日", df['event_date'].min(), key="page_analysis_start_date")
        with col2:
            end_date = st.date_input("終了日", df['event_date'].max(), key="page_analysis_end_date")

    # --- 新規/リピート、CV/非CVの列を追加 ---
    df['user_type'] = np.where(df['ga_session_number'] == 1, '新規', 'リピート')
    conversion_session_ids = df[df['cv_type'].notna()]['session_id'].unique()
    df['conversion_status'] = np.where(df['session_id'].isin(conversion_session_ids), 'コンバージョン', '非コンバージョン')

    # データフィルタリング
    filtered_df = df.copy()

    # 期間フィルター
    filtered_df = filtered_df[
        (filtered_df['event_date'] >= pd.to_datetime(start_date)) &
        (filtered_df['event_date'] <= pd.to_datetime(end_date))
    ]

    # LPフィルター
    if selected_lp_base_url:
        filtered_df = filtered_df[filtered_df['lp_base_url'] == selected_lp_base_url]

    # --- クロス分析用フィルター適用 ---
    if selected_device != "すべて":
        filtered_df = filtered_df[filtered_df['device_type'] == selected_device]

    if selected_user_type != "すべて":
        filtered_df = filtered_df[filtered_df['user_type'] == selected_user_type]

    if selected_conversion_status != "すべて":
        filtered_df = filtered_df[filtered_df['conversion_status'] == selected_conversion_status]

    if selected_channel != "すべて":
        filtered_df = filtered_df[filtered_df['channel'] == selected_channel]

    if selected_source_medium != "すべて":
        filtered_df = filtered_df[filtered_df['source_medium'] == selected_source_medium]

    # 比較機能は無効化
    comparison_df = None

    # データが空の場合の処理
    if len(filtered_df) == 0:
        st.warning("⚠️ 選択した条件に該当するデータがありません。フィルターを変更してください。")
        st.stop()

    # ページ分析は単一のLP選択時のみ実行
    if selected_lp_base_url:
        pass # 選択されたLPのURL表示は削除
    else:
        st.warning("ページ分析を行うには、上のフィルターで分析したいLPを選択してください。")
        st.stop()
    
    # --- BigQueryデータシミュレーション ---
    # --- コンテンツ情報取得ロジック ---
    # ご指定のURLリストをここに定義します。
    lp_content_urls = {
        1: "https://shungene.lm-c.jp/tst08/01.mp4", # P1
        2: "https://shungene.lm-c.jp/tst08/01.jpg",
        3: "https://shungene.lm-c.jp/tst08/02.jpg",
        4: "https://shungene.lm-c.jp/tst08/03.jpg",
        5: "https://shungene.lm-c.jp/tst08/04.jpg",
        6: "https://shungene.lm-c.jp/tst08/05.jpg",
        7: "https://shungene.lm-c.jp/tst08/06.jpg",
        8: "https://shungene.lm-c.jp/tst08/06.mp4",
        9: "https://shungene.lm-c.jp/tst08/07.jpg", # P9
        10: "https://shungene.lm-c.jp/tst08/08.jpg",
        11: "https://shungene.lm-c.jp/tst08/09.jpg",
        12: "https://shungene.lm-c.jp/tst08/10.jpg",
        13: "https://shungene.lm-c.jp/tst08/11.jpg",
        14: "https://shungene.lm-c.jp/tst08/12.jpg",
        15: "https://shungene.lm-c.jp/tst08/13.jpg", # P15
        16: "https://shungene.lm-c.jp/tst08/14.jpg",
        17: "https://shungene.lm-c.jp/tst08/15.jpg",
        18: "https://shungene.lm-c.jp/tst08/16.jpg",
    }

    # BigQueryからコンテンツ情報を取得する関数をシミュレート
    def get_lp_content_info(lp_url, page_num):
        """
        指定されたページ番号に基づいて、コンテンツのタイプとソースを返します。
        現在はハードコードされたURLリストを使用します。
        """
        url = lp_content_urls.get(page_num) # type: ignore
        if url:
            if url.endswith(('.mp4', '.webm', '.mov')):
                return {'page_number': page_num, 'content_type': 'video', 'content_source': url}
            else:
                return {'page_number': page_num, 'content_type': 'image', 'content_source': url}
        # リストにないページ番号の場合はデフォルトのプレースホルダーを返す
        return {'page_number': page_num, 'content_type': 'image', 'content_source': f"https://via.placeholder.com/150x250.png?text=Page{page_num}"}

    # テーブル表示用のプレースホルダー画像
    VIDEO_PLACEHOLDER_IMAGE = "https://via.placeholder.com/150x250.png?text=動画コンテンツ"
    HTML_PLACEHOLDER_IMAGE = "https://via.placeholder.com/150x250.png?text=HTMLコンテンツ"

    # --- BigQueryデータシミュレーションここまで ---


    # ページ別メトリクス計算
    page_stats = filtered_df.groupby('page_num_dom').agg({
        'session_id': 'nunique'
    }).reset_index()
    page_stats.rename(columns={'page_num_dom': 'ページ番号', 'session_id': 'ビュー数'}, inplace=True)

    # 逆行回数を計算
    # セッションごと、ページごとに逆行イベントをカウント
    backflow_df = filtered_df[filtered_df['direction'] == 'backward'].copy()
    if not backflow_df.empty:
        # ページごとの逆行イベントが発生したセッションのユニーク数をカウント
        backflow_counts = backflow_df.groupby('page_num_dom')['session_id'].nunique().reset_index()
        backflow_counts.rename(columns={'page_num_dom': 'ページ番号', 'session_id': '逆行セッション数'}, inplace=True)
        
        # page_statsにマージ
        page_stats = pd.merge(page_stats, backflow_counts, on='ページ番号', how='left').fillna(0)
        page_stats['逆行率'] = page_stats.apply(lambda row: safe_rate(row['逆行セッション数'], row['ビュー数']) * 100, axis=1)
    else:
        page_stats['逆行率'] = 0
    
    # LPの実際のページ数を取得（画像取得が成功した場合はそれを使用、失敗した場合は推測値）
    # フィルターをかける前の元のデータから最大ページ数を取得することで、フィルターによってページ数が1になる問題を回避
    unfiltered_lp_df = df[df['lp_base_url'] == selected_lp_base_url]
    actual_page_count = int(unfiltered_lp_df['page_num_dom'].max()) if not unfiltered_lp_df.empty and not unfiltered_lp_df['page_num_dom'].isnull().all() else 1

    
    # 離脱率計算（LPの実際のページ数を使用）
    page_exit = []
    for page_num in range(1, actual_page_count + 1):
        reached = filtered_df[filtered_df['max_page_reached'] >= page_num]['session_id'].nunique()
        exited = filtered_df[filtered_df['max_page_reached'] == page_num]['session_id'].nunique()
        exit_rate = (exited / reached * 100) if reached > 0 else 0
        page_exit.append({'ページ番号': page_num, '離脱率': exit_rate})
    
    page_exit_df = pd.DataFrame(page_exit)
    page_stats = page_stats.merge(page_exit_df, on='ページ番号', how='left')

    # 平均滞在時間(秒)を計算して列を追加
    stay_time_df = filtered_df.groupby('page_num_dom')['stay_ms'].mean().reset_index() # type: ignore
    stay_time_df.rename(columns={'page_num_dom': 'ページ番号', 'stay_ms': '平均滞在時間(秒)'}, inplace=True)
    stay_time_df['平均滞在時間(秒)'] /= 1000
    page_stats = page_stats.merge(stay_time_df, on='ページ番号', how='left')
    
    # ダミーデータにないページを追加（ダミーデータが10ページまでしかない場合）
    for page_num in range(1, actual_page_count + 1):
        if page_num not in page_stats['ページ番号'].values:
            # ダミーデータがないページはランダムなダミー値で追加
            # ページが進むほどビュー数が減少するパターン
            new_row = pd.DataFrame([{
                'ページ番号': page_num,
                'ビュー数': 0,
                '平均逆行回数': 0,
                '平均滞在時間(秒)': 0,
                '離脱率': 0  # 離脱率は別途計算
            }])
            page_stats = pd.concat([page_stats, new_row], ignore_index=True)
    
    # ページ番号でソート
    page_stats = page_stats.sort_values('ページ番号').reset_index(drop=True)
    
    # 包括的なページメトリクステーブル
    st.markdown("#### ページごとのパフォーマンス詳細")
    st.markdown('<div class="graph-description">各ページのプレビューと主要指標を一覧で確認できます。</div>', unsafe_allow_html=True)

    # 表示件数選択プルダウン
    actual_page_count = int(filtered_df['page_num_dom'].max()) if not filtered_df.empty and not filtered_df['page_num_dom'].isnull().all() else 0
    st.info(f"📊 このLPは {actual_page_count} ページで構成されています")

    _, pulldown_col = st.columns([5, 1])
    with pulldown_col:
        num_to_display_str = st.selectbox(
            "表示件数",
            ["すべて"] + list(range(5, min(51, actual_page_count + 1), 5)),
            index=0,
            label_visibility="collapsed" # ラベルを非表示にしてコンパクトに
        )

    # 表示するページ数を決定
    if num_to_display_str == "すべて":
        num_to_display = actual_page_count
    else:
        num_to_display = int(num_to_display_str)

    # 実際のページ数と表示件数のうち、小さい方（min）でループする
    for page_num in range(1, min(num_to_display, actual_page_count) + 1):
        page_events = filtered_df[filtered_df['page_num_dom'] == page_num]

    # 18ページ分のカードを表示
    for page_num in range(1, num_to_display + 1):
        with st.container():
            col1, col2 = st.columns([1, 6], gap="large") # キャプチャ用に1、データ用に6の比率。間にスペースを追加

            with col1:
                st.markdown(f"**ページ {page_num}**")
                # コンテンツ情報を取得
                content_info = get_lp_content_info(selected_lp_base_url, page_num)
                content_type = content_info.get('content_type', 'image')
                content_source = content_info.get('content_source')

                # プレビューを表示
                if content_source:
                    if content_type == 'video':
                        st.video(content_source)
                    else:
                        st.image(content_source)

            with col2:
                # このコンテナにクラス名を付けてCSSでターゲットできるようにする
                st.markdown('<div class="page-analysis-metrics-container">', unsafe_allow_html=True)

                # メトリクスを4x2のグリッドで表示
                metric_cols_1 = st.columns(4)
                metric_cols_2 = st.columns(4)

                # --- このループ内で各ページの指標を計算 ---
                page_events = filtered_df[filtered_df['page_num_dom'] == page_num]
                page_data = page_stats[page_stats['ページ番号'] == page_num]

                views = int(page_data['ビュー数'].iloc[0]) if not page_data.empty and 'ビュー数' in page_data.columns else 0
                exit_rate = page_data['離脱率'].iloc[0] if not page_data.empty and '離脱率' in page_data.columns else 0
                stay_time = page_data['平均滞在時間(秒)'].iloc[0] if not page_data.empty and '平均滞在時間(秒)' in page_data.columns else 0
                backflow_rate = page_data['逆行率'].iloc[0] if not page_data.empty and '逆行率' in page_data.columns else 0
                # --- ここまで ---
                
                # 1ページ目の逆行率は表示しない
                if page_num == 1:
                    backflow_rate = None
                    
                # 最終ページの離脱率は表示しない
                if page_num == actual_page_count:
                    exit_rate = None
                    
                # このページに到達したユニークなセッション数を計算
                page_sessions = page_events['session_id'].nunique()

                cta_clicks = page_events[(page_events['event_name'] == 'click') & (page_events['elem_classes'].str.contains('cta|btn-primary', na=False))].shape[0]
                cta_clicked_sessions = page_events[(page_events['event_name'] == 'click') & (page_events['elem_classes'].str.contains('cta|btn-primary', na=False))]['session_id'].nunique()
                cta_click_rate = safe_rate(cta_clicked_sessions, page_sessions) * 100

                fb_clicked_sessions = page_events[(page_events['event_name'] == 'click') & (page_events['elem_classes'].str.contains('floating', na=False))]['session_id'].nunique()
                fb_click_rate = safe_rate(fb_clicked_sessions, page_sessions) * 100

                exit_pop_clicked_sessions = page_events[(page_events['event_name'] == 'click') & (page_events['elem_classes'].str.contains('exit', na=False))]['session_id'].nunique()
                exit_pop_click_rate = safe_rate(exit_pop_clicked_sessions, page_sessions) * 100

                load_time = page_events['load_time_ms'].mean() if not page_events.empty else 0
                
                # メトリクスを配置
                metric_cols_1[0].metric("ビュー数", f"{views:,}")
                metric_cols_1[1].metric("離脱率", f"{exit_rate:.1f}%" if exit_rate is not None else "---")
                metric_cols_1[2].metric("平均滞在時間", f"{stay_time:.1f}秒")
                metric_cols_1[3].metric("逆行率", f"{backflow_rate:.1f}%" if backflow_rate is not None else "---")
                metric_cols_2[0].metric("CTAクリック率", f"{cta_click_rate:.1f}%")
                metric_cols_2[1].metric("FBクリック率", f"{fb_click_rate:.1f}%")
                metric_cols_2[2].metric("離脱POPクリック率", f"{exit_pop_click_rate:.1f}%")
                metric_cols_2[3].metric("読み込み時間", f"{load_time:.0f}ms")

                st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---") # 各ページ間に区切り線を追加
    
    st.markdown("---")
    
    # 離脱率と滞在時間の散布図
    st.markdown('### 離脱率 vs 滞在時間 ポジショニングマップ')
    st.markdown('<div class="graph-description">各ページの離脱率（横軸）と平均滞在時間（縦軸）を散布図に表示します。右下の「要注意ゾーン」（高離脱率・低滞在時間）にあるページは、最優先で改善が必要なボトルネックです。</div>', unsafe_allow_html=True)

    if len(page_stats) > 1:
        # ポジショニングマップ用に最終ページを除外したデータを作成
        plot_data = page_stats[page_stats['ページ番号'] != actual_page_count].copy()

        # 平均値を計算
        avg_exit_rate = plot_data['離脱率'].mean()
        # 滞在時間はfiltered_dfから直接計算
        avg_stay_time = filtered_df['stay_ms'].mean() / 1000

        # 散布図を作成
        fig_scatter = px.scatter(
            plot_data,
            x='離脱率',
            y='平均滞在時間(秒)',
            text='ページ番号',
            size='ビュー数',
            color_discrete_sequence=px.colors.qualitative.Plotly,
            hover_name='ページ番号',
            hover_data={'ページ番号': False, 'ビュー数': ':,', '離脱率': ':.1f', '平均滞在時間(秒)': ':.1f'}
        )

        # 平均線を追加
        fig_scatter.add_vline(x=avg_exit_rate, line_dash="dash", line_color="gray", annotation_text=f"平均離脱率: {avg_exit_rate:.1f}%")
        fig_scatter.add_hline(y=avg_stay_time, line_dash="dash", line_color="gray", annotation_text=f"全ページ平均滞在時間: {avg_stay_time:.1f}秒")

        # ゾーンの背景色と注釈を追加
        fig_scatter.add_shape(type="rect", xref="paper", yref="paper", x0=0.5, y0=0, x1=1, y1=0.5, fillcolor="rgba(255, 0, 0, 0.1)", layer="below", line_width=0)
        fig_scatter.add_annotation(xref="paper", yref="paper", x=0.75, y=0.25, text="<b>要注意ゾーン</b><br>高離脱率<br class='mobile-br'>低滞在時間", showarrow=False, font=dict(color="red", size=14), align="center", xanchor="center", yanchor="middle")

        fig_scatter.add_shape(type="rect", xref="paper", yref="paper", x0=0.5, y0=0.5, x1=1, y1=1, fillcolor="rgba(255, 165, 0, 0.1)", layer="below", line_width=0)
        fig_scatter.add_annotation(xref="paper", yref="paper", x=0.75, y=0.75, text="<b>改善候補</b><br>高離脱率<br class='mobile-br'>高滞在時間", showarrow=False, font=dict(color="orange", size=14), align="center", xanchor="center", yanchor="middle")

        fig_scatter.add_shape(type="rect", xref="paper", yref="paper", x0=0, y0=0, x1=0.5, y1=0.5, fillcolor="rgba(255, 255, 0, 0.1)", layer="below", line_width=0)
        fig_scatter.add_annotation(xref="paper", yref="paper", x=0.25, y=0.25, text="<b>機会損失</b><br>低離脱率<br class='mobile-br'>低滞在時間", showarrow=False, font=dict(color="goldenrod", size=14), align="center", xanchor="center", yanchor="middle")

        fig_scatter.add_shape(type="rect", xref="paper", yref="paper", x0=0, y0=0.5, x1=0.5, y1=1, fillcolor="rgba(0, 128, 0, 0.1)", layer="below", line_width=0)
        fig_scatter.add_annotation(xref="paper", yref="paper", x=0.25, y=0.75, text="<b>良好ゾーン</b><br>低離脱率<br class='mobile-br'>高滞在時間", showarrow=False, font=dict(color="green", size=14), align="center", xanchor="center", yanchor="middle")

        fig_scatter.update_traces(
            textposition='top center',
            marker=dict(sizemin=5),
            textfont_size=12
        )
        fig_scatter.update_layout(
            height=600,
            xaxis_title='離脱率 (%)',
            yaxis_title='平均滞在時間 (秒)',
            showlegend=False,
            dragmode=False
        )
        st.plotly_chart(fig_scatter, use_container_width=True, key='plotly_chart_scatter_exit_stay')
    else:
        st.info("ポジショニングマップを表示するには、2ページ以上のデータが必要です。")
    
    st.markdown("---")
    
    # 滞在時間が短いページ、離脱率が高いページ、逆行パターンを並べて表示
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('##### 滞在時間が短いページ TOP5')
        st.markdown('<div class="graph-description">コンテンツが魅力的でない、または読みづらい可能性があります。</div>', unsafe_allow_html=True)
        
        # ページごとの平均滞在時間を計算
        stay_time_per_page = filtered_df.groupby('page_num_dom')['stay_ms'].mean().reset_index()
        stay_time_per_page.columns = ['ページ番号', '平均滞在時間(秒)']
        stay_time_per_page['平均滞在時間(秒)'] /= 1000
        
        # 上位5件を取得
        short_stay_pages = stay_time_per_page.nsmallest(5, '平均滞在時間(秒)')
        
        if not short_stay_pages.empty:
            display_df = short_stay_pages.copy()
            display_df['ページ番号'] = display_df['ページ番号'].astype(int)
            st.dataframe(display_df.style.format({'平均滞在時間(秒)': '{:.1f}秒'}), use_container_width=True, hide_index=True, height=212) # 高さを固定
        else:
            st.info("データがありません。")

    with col2:
        st.markdown('##### 離脱率が高いページ TOP5')
        st.markdown('<div class="graph-description">ユーザーが最も離脱しやすいボトルネックとなっている可能性が高いページです。</div>', unsafe_allow_html=True)
        high_exit_pages = page_stats.nlargest(5, '離脱率')[['ページ番号', '離脱率']]
        high_exit_pages['ページ番号'] = high_exit_pages['ページ番号'].astype(int)
        st.dataframe(high_exit_pages.style.format({'離脱率': '{:.1f}%'}), use_container_width=True, hide_index=True, height=212) # 高さを固定

    with col3:
        st.markdown('##### 逆行率が高いページ TOP5')
        st.markdown('<div class="graph-description">逆行率が高い場合、コンテンツの流れに問題がある可能性があります。</div>', unsafe_allow_html=True)
        
        # page_statsから逆行率が高いページTOP5を取得
        if '逆行率' in page_stats.columns and not page_stats.empty:
            high_backflow_pages = page_stats.nlargest(5, '逆行率')[['ページ番号', '逆行率']]
            high_backflow_pages['ページ番号'] = high_backflow_pages['ページ番号'].astype(int)
            st.dataframe(high_backflow_pages.style.format({'逆行率': '{:.1f}%'}), use_container_width=True, hide_index=True, height=212)
        else:
            st.info("逆行率のデータがありません。")
    
    st.markdown("---")

    # --- AI分析と考察 ---
    st.markdown("### AIによる分析と考察")
    st.markdown('<div class="graph-description">ページ分析の結果に基づき、AIが現状の評価と改善のための考察を提示します。</div>', unsafe_allow_html=True)

    # AI分析の表示状態を管理
    if 'page_analysis_ai_open' not in st.session_state:
        st.session_state.page_analysis_ai_open = False

    if st.button("AI分析を実行", key="page_analysis_ai_btn", type="primary", use_container_width=True):
        st.session_state.page_analysis_ai_open = True

    if st.session_state.page_analysis_ai_open:
        with st.container():
            with st.spinner("AIがページデータを分析中..."):
                analysis_result = ai_analysis.analyze_page_bottlenecks(page_stats)
                st.markdown(analysis_result)

            if st.button("AI分析を閉じる", key="page_analysis_ai_close"):
                st.session_state.page_analysis_ai_open = False

    # --- よくある質問 ---
    st.markdown("#### このページの分析について質問する")
    if 'page_faq_toggle' not in st.session_state:
        st.session_state.page_faq_toggle = {1: False, 2: False, 3: False, 4: False}

    faq_cols = st.columns(2)
    with faq_cols[0]:
        if st.button("最も改善すべきページはどれ？", key="faq_page_1", use_container_width=True):
            st.session_state.page_faq_toggle[1] = not st.session_state.page_faq_toggle[1]
            st.session_state.page_faq_toggle[2], st.session_state.page_faq_toggle[3], st.session_state.page_faq_toggle[4] = False, False, False
        if st.session_state.page_faq_toggle[1]:
            if not page_stats.empty:
                bottleneck_page = page_stats.loc[page_stats['離脱率'].idxmax()]
                st.info(f"**ページ{int(bottleneck_page['ページ番号'])}** です。離脱率が{bottleneck_page['離脱率']:.1f}%と高く、平均滞在時間が{bottleneck_page['平均滞在時間(秒)']:.1f}秒と短いため、最優先で改善すべきボトルネックです。")
        
        if st.button("滞在時間が短いページの共通点は？", key="faq_page_3", use_container_width=True):
            st.session_state.page_faq_toggle[3] = not st.session_state.page_faq_toggle[3]
            st.session_state.page_faq_toggle[1], st.session_state.page_faq_toggle[2], st.session_state.page_faq_toggle[4] = False, False, False
        if st.session_state.page_faq_toggle[3]:
            st.info("滞在時間が短いページは、ユーザーの期待とコンテンツが一致していない、情報が分かりにくい、または単に興味を引かれていない可能性があります。前のページからの文脈を見直し、コンテンツの魅力を高める必要があります。")
    with faq_cols[1]:
        if st.button("ユーザーが前のページに戻る原因は？", key="faq_page_2", use_container_width=True):
            st.session_state.page_faq_toggle[2] = not st.session_state.page_faq_toggle[2]
            st.session_state.page_faq_toggle[1], st.session_state.page_faq_toggle[3], st.session_state.page_faq_toggle[4] = False, False, False
        if st.session_state.page_faq_toggle[2]:
            st.info("ユーザーが逆行（前のページに戻る）するのは、主に「求めている情報が見つからない」「前のページの情報と比較・再確認したい」という理由が考えられます。逆行が多いページ間のコンテンツの流れを見直し、情報の不足がないか確認することが重要です。")
        
        if st.button("離脱率と滞在時間の関係は？", key="faq_page_4", use_container_width=True):
            st.session_state.page_faq_toggle[4] = not st.session_state.page_faq_toggle[4]
            st.session_state.page_faq_toggle[1], st.session_state.page_faq_toggle[2], st.session_state.page_faq_toggle[3] = False, False, False
        if st.session_state.page_faq_toggle[4]:
            st.info("「離脱率が高く、滞在時間が短い」ページは、コンテンツが全く響いていない重大な問題ページです。逆に「離脱率が高く、滞在時間が長い」ページは、コンテンツは読まれているが次のアクションに繋がっていない「惜しい」ページと言えます。")


# タブ3: セグメント分析
elif selected_analysis == "広告分析":
    st.markdown('<div class="sub-header">広告分析</div>', unsafe_allow_html=True)

    # --- ページ上部の共通フィルター ---
    st.markdown('<div class="sub-header">フィルター設定</div>', unsafe_allow_html=True)
    filter_cols_1 = st.columns(4)
    filter_cols_2 = st.columns(4)

    with filter_cols_1[0]:
        period_options = [
            "今日", "昨日", "過去7日間", "過去14日間", "過去30日間", "今月", "先月", "全期間", "カスタム"
        ]
        selected_period = st.selectbox("期間を選択", period_options, index=2, key="ad_analysis_period")

    with filter_cols_1[1]:
        lp_options = sorted(df['page_location'].dropna().unique().tolist())
        selected_lp = st.selectbox(
            "LP選択", 
            lp_options, 
            index=0 if lp_options else None,
            key="ad_analysis_lp",
            disabled=not lp_options
        )

    with filter_cols_1[2]:
        device_options = ["すべて"] + sorted(df['device_type'].dropna().unique().tolist())
        selected_device = st.selectbox("デバイス選択", device_options, index=0, key="ad_analysis_device")

    with filter_cols_1[3]:
        # 新規/リピート フィルター
        user_type_options = ["すべて", "新規", "リピート"]
        selected_user_type = st.selectbox("新規/リピート", user_type_options, index=0, key="ad_analysis_user_type")

    with filter_cols_2[0]:
        # CV/非CV フィルター
        conversion_status_options = ["すべて", "コンバージョン", "非コンバージョン"]
        selected_conversion_status = st.selectbox("CV/非CV", conversion_status_options, index=0, key="ad_analysis_conversion_status")

    with filter_cols_2[1]:
        # 広告関連のチャネルのみを抽出
        ad_channels = ['Paid Search', 'Paid Social', 'Paid Video', 'Display', 'Other']
        channel_options = ["すべて"] + [ch for ch in df['channel'].unique() if ch in ad_channels]
        selected_channel = st.selectbox("チャネル", channel_options, index=0, key="ad_analysis_channel")

    with filter_cols_2[2]:
        source_medium_options = ["すべて"] + sorted(df['source_medium'].unique().tolist())
        selected_source_medium = st.selectbox("参照元/メディア", source_medium_options, index=0, key="ad_analysis_source_medium")

    # 期間設定
    today = df['event_date'].max().date()
    if selected_period == "今日":
        start_date, end_date = today, today
    elif selected_period == "昨日":
        start_date, end_date = today - timedelta(days=1), today - timedelta(days=1)
    elif selected_period == "過去7日間":
        start_date, end_date = today - timedelta(days=6), today
    elif selected_period == "過去14日間":
        start_date, end_date = today - timedelta(days=13), today
    elif selected_period == "過去30日間":
        start_date, end_date = today - timedelta(days=29), today
    elif selected_period == "今月":
        start_date, end_date = today.replace(day=1), today
    elif selected_period == "先月":
        last_month_end = today.replace(day=1) - timedelta(days=1)
        start_date, end_date = last_month_end.replace(day=1), last_month_end
    elif selected_period == "全期間":
        start_date, end_date = df['event_date'].min().date(), df['event_date'].max().date()
    elif selected_period == "カスタム":
        c1, c2 = st.columns(2)
        with c1:
            start_date = st.date_input("開始日", df['event_date'].min(), key="ad_analysis_start")
        with c2:
            end_date = st.date_input("終了日", df['event_date'].max(), key="ad_analysis_end")

    # --- 新規/リピート、CV/非CVの列を追加 ---
    df['user_type'] = np.where(df['ga_session_number'] == 1, '新規', 'リピート')
    conversion_session_ids = df[df['cv_type'].notna()]['session_id'].unique()
    df['conversion_status'] = np.where(df['session_id'].isin(conversion_session_ids), 'コンバージョン', '非コンバージョン')

    # --- データフィルタリング ---
    filtered_df = df[
        (df['event_date'] >= pd.to_datetime(start_date)) &
        (df['event_date'] <= pd.to_datetime(end_date))
    ]
    if selected_lp:
        filtered_df = filtered_df[filtered_df['page_location'] == selected_lp]
    if selected_device != "すべて":
        filtered_df = filtered_df[filtered_df['device_type'] == selected_device]
    if selected_user_type != "すべて":
        filtered_df = filtered_df[filtered_df['user_type'] == selected_user_type]
    if selected_conversion_status != "すべて":
        filtered_df = filtered_df[filtered_df['conversion_status'] == selected_conversion_status]
    if selected_channel != "すべて":
        filtered_df = filtered_df[filtered_df['channel'] == selected_channel]
    if selected_source_medium != "すべて":
        filtered_df = filtered_df[filtered_df['source_medium'] == selected_source_medium]

    # --- 分析対象の選択 ---
    st.markdown('<div class="sub-header">分析軸の選択</div>', unsafe_allow_html=True)
    st.markdown('<div class="graph-description">「キャンペーン別」または「広告コンテンツ別」にパフォーマンスを分析します。</div>', unsafe_allow_html=True)

    analysis_target = st.radio(
        "分析の切り口を選択してください",
        ('キャンペーン別', '広告コンテンツ別'),
        horizontal=True,
        key="ad_analysis_target"
    )

    st.markdown("---") # type: ignore

    # --- 分析テーブル表示 ---
    if analysis_target == 'キャンペーン別':
        st.markdown("#### キャンペーン別 パフォーマンス")
        segment_col = 'utm_campaign'
        segment_name = 'キャンペーン'
        display_df = filtered_df.dropna(subset=['utm_campaign'])
    else:
        st.markdown("#### 広告コンテンツ別 パフォーマンス")
        segment_col = 'utm_content'
        segment_name = '広告コンテンツ'
        display_df = filtered_df.dropna(subset=['utm_content'])

    # データが空の場合の処理
    if display_df.empty or display_df[segment_col].nunique() == 0:
        st.info("選択された条件に該当する広告データがありません。")
        st.stop()

    # セグメント別統計を計算
    segment_stats = display_df.groupby(segment_col).agg(
        セッション数=('session_id', 'nunique'),
        # クリック数はユニークセッション数でカウント
        クリック数=('session_id', lambda x: display_df.loc[x.index][display_df.loc[x.index]['event_name'] == 'click']['session_id'].nunique()),
        平均滞在時間=('stay_ms', 'mean'),
        平均到達ページ=('max_page_reached', 'mean')
    ).reset_index()
    segment_stats.rename(columns={segment_col: segment_name}, inplace=True)
    
    # コンバージョン数
    segment_cv = display_df[display_df['cv_type'].notna()].groupby(segment_col)['session_id'].nunique().reset_index()
    segment_cv.rename(columns={segment_col: segment_name, 'session_id': 'CV数'}, inplace=True)
    segment_stats = pd.merge(segment_stats, segment_cv, on=segment_name, how='left').fillna(0)

    # FV残存数
    segment_fv = display_df[display_df['max_page_reached'] >= 2].groupby(segment_col)['session_id'].nunique().reset_index()
    segment_fv.rename(columns={segment_col: segment_name, 'session_id': 'FV残存数'}, inplace=True)
    segment_stats = pd.merge(segment_stats, segment_fv, on=segment_name, how='left').fillna(0)

    # 最終CTA到達数
    segment_final_cta = display_df[display_df['max_page_reached'] >= 10].groupby(segment_col)['session_id'].nunique().reset_index()
    segment_final_cta.rename(columns={segment_col: segment_name, 'session_id': '最終CTA到達数'}, inplace=True)
    segment_stats = pd.merge(segment_stats, segment_final_cta, on=segment_name, how='left').fillna(0)

    # エンゲージメント率（滞在時間30秒以上）
    engaged_sessions = display_df[display_df['stay_ms'] >= 30000].groupby(segment_col)['session_id'].nunique().reset_index()
    engaged_sessions.rename(columns={segment_col: segment_name, 'session_id': 'エンゲージセッション数'}, inplace=True)
    segment_stats = pd.merge(segment_stats, engaged_sessions, on=segment_name, how='left').fillna(0)

    # 率の計算
    segment_stats['CVR'] = segment_stats.apply(lambda row: safe_rate(row['CV数'], row['セッション数']) * 100, axis=1)
    segment_stats['CTR'] = segment_stats.apply(lambda row: safe_rate(row['クリック数'], row['セッション数']) * 100, axis=1)
    segment_stats['FV残存率'] = segment_stats.apply(lambda row: safe_rate(row['FV残存数'], row['セッション数']) * 100, axis=1)
    segment_stats['最終CTA到達率'] = segment_stats.apply(lambda row: safe_rate(row['最終CTA到達数'], row['セッション数']) * 100, axis=1)
    segment_stats['エンゲージメント率'] = segment_stats.apply(lambda row: safe_rate(row['エンゲージセッション数'], row['セッション数']) * 100, axis=1)
    segment_stats['平均滞在時間'] = segment_stats['平均滞在時間'] / 1000

    # テーブル表示
    display_cols = [
        segment_name, 'セッション数', 'CV数', 'CVR', 'クリック数', 'CTR', 
        'FV残存率', '最終CTA到達率', '平均到達ページ', '平均滞在時間', 'エンゲージメント率'
    ]
    st.dataframe(segment_stats[display_cols].style.format({
        'セッション数': '{:,.0f}', 'CV数': '{:,.0f}', 'CVR': '{:.2f}%',
        'クリック数': '{:,.0f}', 'CTR': '{:.2f}%', 'FV残存率': '{:.2f}%',
        '最終CTA到達率': '{:.2f}%', '平均到達ページ': '{:.1f}',
        '平均滞在時間': '{:.1f}秒', 'エンゲージメント率': '{:.2f}%',
    }), use_container_width=True, hide_index=True)
    
    # --- 指標選択 ---
    st.markdown("##### グラフに表示する指標を選択")
    all_metrics = [
        'セッション数', 'CV数', 'CVR', 'クリック数', 'CTR', 
        'FV残存率', '最終CTA到達率', '平均到達ページ', '平均滞在時間', 'エンゲージメント率'
    ]
    selected_metrics = st.multiselect(
        "最大2つまで選択できます",
        all_metrics,
        default=['CVR', 'セッション数'],
        max_selections=2,
        label_visibility="collapsed"
    )

    # グラフ表示
    if selected_metrics:
        graph_cols = st.columns(len(selected_metrics))
        for i, metric in enumerate(selected_metrics):
            with graph_cols[i]:
                # 単位を決定
                unit = ''
                if '%' in metric or '率' in metric:
                    unit = '%'
                elif '時間' in metric:
                    unit = '秒'
                
                # グラフを作成
                fig = px.line(
                    segment_stats, 
                    x=segment_name, 
                    y=metric, 
                    title=f'{segment_name}別{metric}',
                    markers=True
                )
                fig.update_layout(dragmode=False, yaxis_title=metric)
                fig.update_traces(hovertemplate=f'%{{x}}<br>{metric}: %{{y:,.2f}}{unit}<extra></extra>')
                st.plotly_chart(fig, use_container_width=True, key=f'ad_analysis_chart_{i}')
    else:
        st.info("グラフを表示するには、上のプルダウンから少なくとも1つの指標を選択してください。")

    st.markdown("---")

    # --- AI分析と考察 ---
    st.markdown("### AIによる分析と考察")
    st.markdown('<div class="graph-description">広告のキャンペーンやコンテンツ別のパフォーマンスに基づき、AIが現状の評価と改善のための考察を提示します。</div>', unsafe_allow_html=True)

    # AI分析の表示状態を管理
    if 'ad_analysis_ai_open' not in st.session_state:
        st.session_state.ad_analysis_ai_open = False

    if st.button("AI分析を実行", key="ad_analysis_ai_btn", type="primary", use_container_width=True):
        st.session_state.ad_analysis_ai_open = True

    if st.session_state.ad_analysis_ai_open:
        with st.container():
            with st.spinner("AIがセグメントデータを分析中..."):
                # AI分析を実行
                ai_response = ai_analysis.analyze_ad_performance_expert(segment_stats, analysis_target)
                st.markdown(ai_response)
            
            if st.button("AI分析を閉じる", key="ad_analysis_ai_close"):
                st.session_state.ad_analysis_ai_open = False

    # --- よくある質問 ---
    st.markdown("#### このページの分析について質問する")
    if 'segment_faq_toggle' not in st.session_state:
        st.session_state.segment_faq_toggle = {1: False, 2: False, 3: False, 4: False}

    if 'ad_faq_toggle' not in st.session_state:
        st.session_state.ad_faq_toggle = {1: False, 2: False, 3: False, 4: False}

    faq_cols = st.columns(2)
    with faq_cols[0]:
        if st.button(f"パフォーマンスが最も良い{segment_name}は？", key="faq_segment_1", use_container_width=True):
            st.session_state.segment_faq_toggle[1] = not st.session_state.segment_faq_toggle[1]
            st.session_state.segment_faq_toggle[2], st.session_state.segment_faq_toggle[3], st.session_state.segment_faq_toggle[4] = False, False, False # type: ignore
        if st.session_state.segment_faq_toggle[1] and not segment_stats.empty:
            if not segment_stats.empty and 'CVR' in segment_stats.columns:
                best_segment = segment_stats.loc[segment_stats['CVR'].idxmax()]
                st.info(f"**{best_segment[segment_name]}** です。コンバージョン率が **{best_segment['CVR']:.2f}%** と最も高いパフォーマンスを示しています。")
        
        if st.button(f"パフォーマンスが良いセグメントに集中すべき？", key="faq_segment_3", use_container_width=True):
            st.session_state.segment_faq_toggle[3] = not st.session_state.segment_faq_toggle[3] # type: ignore
            st.session_state.segment_faq_toggle[1], st.session_state.segment_faq_toggle[2], st.session_state.segment_faq_toggle[4] = False, False, False
        if st.session_state.segment_faq_toggle.get(3, False):
            st.info("はい、短期的には最も効果的なアプローチです。パフォーマンスが良いセグメント（例：特定の広告チャネルやデバイス）への予算配分を増やすことで、全体のコンバージョン数を効率的に伸ばすことができます。")
    with faq_cols[1]:
        if st.button(f"パフォーマンスが最も悪い{segment_name}の原因は？", key="faq_segment_2", use_container_width=True):
            st.session_state.segment_faq_toggle[2] = not st.session_state.segment_faq_toggle[2]
            st.session_state.segment_faq_toggle[1], st.session_state.segment_faq_toggle[3], st.session_state.segment_faq_toggle[4] = False, False, False # type: ignore
        if st.session_state.segment_faq_toggle[2] and not segment_stats.empty:
            if not segment_stats.empty and 'CVR' in segment_stats.columns:
                worst_segment = segment_stats.loc[segment_stats['CVR'].idxmin()]
                st.info(f"**{worst_segment[segment_name]}** のパフォーマンスが低い原因として、{analysis_target}が「デバイス別」なら「表示崩れや操作性の問題」、{analysis_target}が「チャネル別」なら「広告ターゲティングとLP内容のミスマッチ」などが考えられます。")
        
        if st.button(f"セグメント毎にLPを変えるべき？", key="faq_segment_4", use_container_width=True):
            st.session_state.segment_faq_toggle[4] = not st.session_state.segment_faq_toggle[4]
            st.session_state.segment_faq_toggle[1], st.session_state.segment_faq_toggle[2], st.session_state.segment_faq_toggle[3] = False, False, False
        if st.session_state.segment_faq_toggle.get(4, False):
            st.info("はい、中長期的には非常に有効な施策です。例えば、PCユーザーには詳細な情報を、スマホユーザーには要点を絞ったコンテンツを見せるなど、セグメントに合わせてLPをパーソナライズすることで、CVRの大幅な向上が期待できます。")

# タブ4: A/Bテスト分析
elif selected_analysis == "A/Bテスト分析":

    st.markdown('<div class="sub-header">A/Bテスト分析</div>', unsafe_allow_html=True)
    # メインエリア: フィルターと比較設定
    st.markdown('<div class="sub-header">フィルター設定</div>', unsafe_allow_html=True)

    filter_cols_1 = st.columns(4)
    filter_cols_2 = st.columns(4)

    with filter_cols_1[0]:
        # 期間選択
        period_options = [
            "今日", "昨日", "過去7日間", "過去14日間", "過去30日間", "今月", "先月", "全期間", "カスタム"
        ]
        selected_period = st.selectbox("期間を選択", period_options, index=2, key="ab_test_period")

    with filter_cols_1[1]:
        # LP選択
        lp_options = sorted(df['page_location'].dropna().unique().tolist())
        selected_lp = st.selectbox(
            "LP選択", 
            lp_options, 
            index=0 if lp_options else None,
            key="ab_test_lp",
            disabled=not lp_options
        )

    with filter_cols_1[2]:
        device_options = ["すべて"] + sorted(df['device_type'].dropna().unique().tolist())
        selected_device = st.selectbox("デバイス選択", device_options, index=0, key="ab_test_device")

    with filter_cols_1[3]:
        # 新規/リピート フィルター
        user_type_options = ["すべて", "新規", "リピート"]
        selected_user_type = st.selectbox("新規/リピート", user_type_options, index=0, key="ab_test_user_type")

    with filter_cols_2[0]:
        # CV/非CV フィルター
        conversion_status_options = ["すべて", "コンバージョン", "非コンバージョン"]
        selected_conversion_status = st.selectbox("CV/非CV", conversion_status_options, index=0, key="ab_test_conversion_status")
    
    with filter_cols_2[1]:
        # チャネルフィルターを追加
        channel_options = ["すべて"] + sorted(df['channel'].unique().tolist())
        selected_channel = st.selectbox("チャネル", channel_options, index=0, key="ab_test_channel")

    with filter_cols_2[2]:
        # チャネルフィルターを「参照元/メディア」に変更
        source_medium_options = ["すべて"] + sorted(df['source_medium'].unique().tolist())
        selected_source_medium = st.selectbox("参照元/メディア", source_medium_options, index=0, key="ab_test_source_medium") # ラベルは変更済み
    
    enable_comparison = False
    # 期間設定
    today = df['event_date'].max().date()
    
    if selected_period == "今日":
        start_date = today
        end_date = today
    elif selected_period == "昨日":
        start_date = today - timedelta(days=1)
        end_date = today - timedelta(days=1)
    elif selected_period == "過去7日間":
        start_date = today - timedelta(days=6)
        end_date = today
    elif selected_period == "過去14日間":
        start_date = today - timedelta(days=13)
        end_date = today
    elif selected_period == "過去30日間":
        start_date = today - timedelta(days=29)
        end_date = today
    elif selected_period == "今月":
        start_date = today.replace(day=1)
        end_date = today
    elif selected_period == "先月":
        last_month_end = today.replace(day=1) - timedelta(days=1)
        start_date = last_month_end.replace(day=1)
        end_date = last_month_end
    elif selected_period == "全期間":
        start_date = df['event_date'].min().date()
        end_date = df['event_date'].max().date()
    elif selected_period == "カスタム":
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("開始日", df['event_date'].min(), key="ab_test_start_date")
        with col2:
            end_date = st.date_input("終了日", df['event_date'].max(), key="ab_test_end_date")

    # --- 新規/リピート、CV/非CVの列を追加 ---
    df['user_type'] = np.where(df['ga_session_number'] == 1, '新規', 'リピート')
    conversion_session_ids = df[df['cv_type'].notna()]['session_id'].unique()
    df['conversion_status'] = np.where(df['session_id'].isin(conversion_session_ids), 'コンバージョン', '非コンバージョン')

    # データフィルタリング
    filtered_df = df.copy()

    # 期間フィルター
    filtered_df = filtered_df[
        (filtered_df['event_date'] >= pd.to_datetime(start_date)) &
        (filtered_df['event_date'] <= pd.to_datetime(end_date))
    ]

    # LPフィルター
    if selected_lp:
        filtered_df = filtered_df[filtered_df['page_location'] == selected_lp]

    # --- クロス分析用フィルター適用 ---
    if selected_device != "すべて":
        filtered_df = filtered_df[filtered_df['device_type'] == selected_device]

    if selected_user_type != "すべて":
        filtered_df = filtered_df[filtered_df['user_type'] == selected_user_type]

    if selected_conversion_status != "すべて":
        filtered_df = filtered_df[filtered_df['conversion_status'] == selected_conversion_status]

    if selected_channel != "すべて":
        filtered_df = filtered_df[filtered_df['channel'] == selected_channel]

    if selected_source_medium != "すべて":
        filtered_df = filtered_df[filtered_df['source_medium'] == selected_source_medium]

    # 比較機能は無効化
    comparison_df = None

    # データが空の場合の処理
    if filtered_df.empty:
        st.warning("⚠️ 選択した条件に該当するデータがありません。フィルターを変更してください。")
        st.stop()

    # A/Bテスト種別のマッピング
    test_type_map = {
        'hero_image': 'FVテスト',
        'cta_button': 'CTAテスト',
        'headline': 'ヘッドラインテスト',
        'layout': 'レイアウトテスト',
        'copy': 'コピーテスト',
        'form': 'フォームテスト',
        'video': '動画テスト'
    }
    if 'ab_test_target' in filtered_df.columns:
        filtered_df['ab_test_target'] = filtered_df['ab_test_target'].map(test_type_map).fillna('-')
    else:
        filtered_df['ab_test_target'] = '-'

    # p_valueカラムが存在するかどうかでaggの内容を分岐
    agg_dict = {
        'session_id': 'nunique',
        'stay_ms': 'mean',
        'max_page_reached': 'mean',
        'completion_rate': 'mean'
    }

    if 'p_value' in filtered_df.columns:
        agg_dict['p_value'] = 'first'
        ab_stats = filtered_df.groupby(['ab_test_target', 'ab_variant']).agg(agg_dict).reset_index()
        ab_stats.columns = ['テスト種別', 'バリアント', 'セッション数', '平均滞在時間(ms)', '平均到達ページ数', '平均完了率', 'p値']
    else:
        ab_stats = filtered_df.groupby(['ab_test_target', 'ab_variant']).agg(agg_dict).reset_index()
        ab_stats.columns = ['テスト種別', 'バリアント', 'セッション数', '平均滞在時間(ms)', '平均到達ページ数', '平均完了率']
        # p_valueカラムが存在しない場合は、1.0で初期化
        ab_stats['p値'] = 1.0
    

    ab_stats['平均滞在時間(秒)'] = ab_stats['平均滞在時間(ms)'] / 1000
    ab_stats['p値'] = ab_stats['p値'].fillna(1.0) # p値がない場合は1.0で埋める
    
    # コンバージョン数（テスト種別とバリアントでグループ化）
    ab_cv = filtered_df[filtered_df['cv_type'].notna()].groupby(['ab_test_target', 'ab_variant'])['session_id'].nunique().reset_index()
    ab_cv.columns = ['テスト種別', 'バリアント', 'コンバージョン数']
    
    ab_stats = ab_stats.merge(ab_cv, on=['テスト種別', 'バリアント'], how='left').fillna({'コンバージョン数': 0})
    ab_stats['コンバージョン率'] = ab_stats.apply(lambda row: safe_rate(row['コンバージョン数'], row['セッション数']) * 100, axis=1)
    
    # FV残存率（テスト種別とバリアントでグループ化）
    fv_retention = filtered_df[filtered_df['max_page_reached'] >= 2].groupby(['ab_test_target', 'ab_variant'])['session_id'].nunique().reset_index()
    fv_retention.columns = ['テスト種別', 'バリアント', 'FV残存数']
    
    ab_stats = ab_stats.merge(fv_retention, on=['テスト種別', 'バリアント'], how='left').fillna({'FV残存数': 0})
    ab_stats['FV残存率'] = ab_stats.apply(lambda row: safe_rate(row['FV残存数'], row['セッション数']) * 100, axis=1)
    
    # 最終CTA到達率（テスト種別とバリアントでグループ化）
    final_cta = filtered_df[filtered_df['max_page_reached'] >= 10].groupby(['ab_test_target', 'ab_variant'])['session_id'].nunique().reset_index()
    final_cta.columns = ['テスト種別', 'バリアント', '最終CTA到達数']
    
    ab_stats = ab_stats.merge(final_cta, on=['テスト種別', 'バリアント'], how='left').fillna({'最終CTA到達数': 0})
    ab_stats['最終CTA到達率'] = ab_stats.apply(lambda row: safe_rate(row['最終CTA到達数'], row['セッション数']) * 100, axis=1)
    
    # テスト種別が'-'の行（テスト対象外のデータ）を除外
    ab_stats = ab_stats[ab_stats['テスト種別'] != '-'].reset_index(drop=True)

    # --- A/Bテスト統計計算 ---
    ab_stats['CVR差分(pt)'] = np.nan # デフォルトをNaNに

    # テスト種別ごとにCVR差分を計算
    for test_type in ab_stats['テスト種別'].unique():
        test_df = ab_stats[ab_stats['テスト種別'] == test_type]
        # バリアントAを基準とする
        if 'A' in test_df['バリアント'].values:
            baseline_cvr = test_df[test_df['バリアント'] == 'A']['コンバージョン率'].iloc[0]
            
            # バリアントAの差分は0
            a_index = test_df[test_df['バリアント'] == 'A'].index
            ab_stats.loc[a_index, 'CVR差分(pt)'] = 0.0

            # 他のバリアントの差分を計算
            other_variants_index = test_df[test_df['バリアント'] != 'A'].index
            ab_stats.loc[other_variants_index, 'CVR差分(pt)'] = test_df.loc[other_variants_index, 'コンバージョン率'] - baseline_cvr

    # p値から有意差と有意性を計算
    ab_stats['有意差'] = ab_stats['p値'].apply(lambda x: '★★★' if x < 0.01 else ('★★' if x < 0.05 else ('★' if x < 0.1 else '-')))
    ab_stats['有意性'] = 1 - ab_stats['p値']  # バブルチャート用


    # A/Bテスト比較
    st.markdown("#### A/Bテスト比較")
    st.markdown('<div class="graph-description">各バリアント（AとB）の主要な指標を比較し、どちらが優れているかを評価します。</div>', unsafe_allow_html=True)
    display_cols = ['セッション数', 'コンバージョン率', 'CVR差分(pt)', '有意差', 'p値', 'FV残存率', '最終CTA到達率', '平均到達ページ数', '平均滞在時間(秒)'] # type: ignore
    
    # 'control' バリアントを除外して表示用のDataFrameを作成
    ab_stats_for_display = ab_stats[ab_stats['バリアント'] != 'control'].copy()
    
    # バリアントBの行をハイライトする関数
    def highlight_variant_b(row):
        return ['background-color: #fffbe6'] * len(row) if row.name[1] == 'B' else [''] * len(row)

    # マルチインデックスを設定して表示
    display_df = ab_stats_for_display.set_index(['テスト種別', 'バリアント'])
    if not display_df.empty:
        st.dataframe(display_df[display_cols].style.format({
            'セッション数': '{:,.0f}',
            'コンバージョン率': '{:.2f}%',
            'CVR差分(pt)': lambda x: f'{x:+.2f}pt' if pd.notna(x) and x != 0 else '---',
            'p値': '{:.4f}',
            'FV残存率': '{:.2f}%',
            '最終CTA到達率': '{:.2f}%',
            '平均到達ページ数': '{:.1f}',
            '平均滞在時間(秒)': '{:.1f}'
        }).apply(highlight_variant_b, axis=1), use_container_width=True)
    else:
        st.info("表示するA/Bテストのデータがありません。")
    
    # CVR向上率×有意性のバブルチャート
    # 'control' バリアントを除外し、バリアント'B'のみを対象とする
    ab_bubble = ab_stats[ab_stats['バリアント'] == 'B'].copy()

    # チャートタイトルと説明 (プルダウンは削除)
    st.markdown("#### CVR向上率×有意性バブルチャート")
    st.markdown('<div class="graph-description">CVR差分（X軸）と有意性（Y軸）を可視化。バブルサイズはサンプルサイズを表します。右上（高CVR差分×高有意性）が最も優れたバリアントです。</div>', unsafe_allow_html=True)

    if not ab_bubble.empty:
        fig = px.scatter(ab_bubble, 
                        x='CVR差分(pt)',
                        y='有意性',
                        size='セッション数',
                        text='バリアント', # バブルに表示するテキスト
                        color='テスト種別', # テスト種別で色分け
                        custom_data=['テスト種別', 'コンバージョン率', 'p値', '有意差'] # hovertemplateで使用するデータを渡す
                        )
        
        # 有意水準の参考線を追加
        fig.add_hline(y=0.95, line_dash="dash", line_color="green", annotation_text="p<0.05 (★★)", annotation_position="bottom right")
        fig.add_hline(y=0.99, line_dash="dash", line_color="red", annotation_text="p<0.01 (★★★)", annotation_position="bottom right") # type: ignore
        fig.add_vline(x=0, line_dash="dash", line_color="gray")

        # X軸の範囲を調整して、常にx=0の線が見えるようにする
        x_min = ab_bubble['CVR差分(pt)'].min()
        x_max = ab_bubble['CVR差分(pt)'].max()
        range_x = [None, None]
        if x_max < 0:
            range_x[1] = 0.5 # 少し余白を持たせる
        if x_min > 0:
            range_x[0] = -0.5 # 少し余白を持たせる
        # マウスオーバー時の表示内容とスタイルをカスタマイズ
        fig.update_traces(
            textposition='top center',
            hovertemplate=(
                "<b>%{text}</b><br>" +
                "テスト種別: %{customdata[0]}<br>" +
                "CVR差分(pt): %{x:+.2f}pt<br>" +
                "有意性: %{y:.2f}<br>" +
                "コンバージョン率: %{customdata[1]:.2f}%<br>" +
                "p値: %{customdata[2]:.4f}<extra></extra>"
            )
        )
        fig.update_layout(height=500,
                         hoverlabel=dict(bordercolor='#002060'), # ホバーの枠線色
                         xaxis_title='CVR差分 (pt)', dragmode=False,
                         xaxis_range=range_x,
                         yaxis_title='有意性 (1 - p値)',
                         legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5))
        
        # 背景色と注釈を追加
        fig.add_shape(type="rect", xref="paper", yref="paper", x0=0.5, y0=0.5, x1=1, y1=1, fillcolor="rgba(0, 128, 0, 0.1)", layer="below", line_width=0)
        fig.add_annotation(xref="paper", yref="paper", x=0.75, y=0.75, text="<b>最良ゾーン</b><br>CVR向上<br class='mobile-br'>有意差あり", showarrow=False, font=dict(color="green", size=14), align="center", xanchor="center", yanchor="middle")

        fig.add_shape(type="rect", xref="paper", yref="paper", x0=0, y0=0.5, x1=0.5, y1=1, fillcolor="rgba(255, 0, 0, 0.1)", layer="below", line_width=0)
        fig.add_annotation(xref="paper", yref="paper", x=0.25, y=0.75, text="<b>悪化ゾーン</b><br>CVR悪化<br class='mobile-br'>有意差あり", showarrow=False, font=dict(color="red", size=14), align="center", xanchor="center", yanchor="middle")

        fig.add_shape(type="rect", xref="paper", yref="paper", x0=0.5, y0=0, x1=1, y1=0.5, fillcolor="rgba(255, 165, 0, 0.1)", layer="below", line_width=0)
        fig.add_annotation(xref="paper", yref="paper", x=0.75, y=0.25, text="<b>有望ゾーン</b><br>CVR向上<br class='mobile-br'>有意差なし", showarrow=False, font=dict(color="orange", size=14), align="center", xanchor="center", yanchor="middle")

        fig.add_shape(type="rect", xref="paper", yref="paper", x0=0, y0=0, x1=0.5, y1=0.5, fillcolor="rgba(128, 128, 128, 0.1)", layer="below", line_width=0)
        fig.add_annotation(xref="paper", yref="paper", x=0.25, y=0.25, text="<b>判断保留ゾーン</b><br>CVR悪化<br class='mobile-br'>有意差なし", showarrow=False, font=dict(color="grey", size=14), align="center", xanchor="center", yanchor="middle")

        st.plotly_chart(fig, use_container_width=True, key='plotly_chart_ab_bubble')
    else:
        st.info("バブルチャートを表示するためのバリアント「B」のデータがありません。")
        
    # A/Bテスト時系列推移
    st.markdown("#### A/Bテスト CVR 時系列推移")
    st.markdown('<div class="graph-description">各A/Bテストのバリアントごとの日次のコンバージョン率（CVR）の推移を可視化します。</div>', unsafe_allow_html=True)

    # A/Bテスト関連のデータをフィルタリング
    ab_daily_df = filtered_df[(filtered_df['ab_test_target'] != '-') & (filtered_df['ab_variant'] != 'control')].copy()
    ab_daily_df['event_date'] = pd.to_datetime(ab_daily_df['event_date']).dt.date

    # 日別、テスト種別、バリアント別のセッション数を計算
    daily_sessions = ab_daily_df.groupby(['event_date', 'ab_test_target', 'ab_variant'])['session_id'].nunique().reset_index()
    daily_sessions.rename(columns={'session_id': 'sessions'}, inplace=True)

    # 日別、テスト種別、バリアント別のコンバージョン数を計算
    daily_conversions = ab_daily_df[ab_daily_df['cv_type'].notna()].groupby(['event_date', 'ab_test_target', 'ab_variant'])['session_id'].nunique().reset_index()
    daily_conversions.rename(columns={'session_id': 'conversions'}, inplace=True)

    # データをマージ
    cvr_data = pd.merge(daily_sessions, daily_conversions, on=['event_date', 'ab_test_target', 'ab_variant'], how='left').fillna(0)
    cvr_data['cvr'] = cvr_data.apply(lambda row: safe_rate(row['conversions'], row['sessions']) * 100, axis=1)

    # テスト種別選択
    test_types = cvr_data['ab_test_target'].unique().tolist()
    
    _, col2 = st.columns([5, 1])
    with col2:
        selected_test_type = st.selectbox('テスト種別を選択', test_types, key="ab_test_cvr_ts_select")
    
    # グラフの作成
    fig_cvr_timeseries = go.Figure()

    # カラーパレット
    color_map = {
        'control': 'grey',
        'A': 'blue',
        'B': 'red'
    }

    filtered_cvr_data = cvr_data[cvr_data['ab_test_target'] == selected_test_type]

    for test_type, group in filtered_cvr_data.groupby('ab_test_target'):
        for variant in group['ab_variant'].unique():
            variant_data = group[group['ab_variant'] == variant]
            fig_cvr_timeseries.add_trace(go.Scatter(
                x=variant_data['event_date'],
                y=variant_data['cvr'],
                mode='lines+markers',
                name=f"{test_type} - {variant}",
                line=dict(color=color_map.get(variant, 'black'))
            ))

    fig_cvr_timeseries.update_layout(
        xaxis_title="日付",
        yaxis_title="コンバージョン率 (%)",
        yaxis_ticksuffix="%",
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        hovermode="x unified",
        height=500,
        dragmode=False
    )
    st.plotly_chart(fig_cvr_timeseries, use_container_width=True)

    st.markdown("---")

    # --- AI分析と考察 ---
    st.markdown("### AIによる分析と考察")
    st.markdown('<div class="graph-description">A/Bテストの結果に基づき、AIが統計的な評価と次のアクションを提案します。</div>', unsafe_allow_html=True)

    # AI分析の表示状態を管理
    if 'ab_test_ai_open' not in st.session_state:
        st.session_state.ab_test_ai_open = False

    if st.button("AI分析を実行", key="ab_test_ai_btn", type="primary", use_container_width=True):
        st.session_state.ab_test_ai_open = True

    if st.session_state.ab_test_ai_open:
        with st.container():
            with st.spinner("AIがA/Bテスト結果を分析中..."):
                if not ab_stats.empty and len(ab_stats) >= 2:
                    # AI分析を実行
                    ai_response = ai_analysis.analyze_ab_test_expert(ab_stats)
                    st.markdown(ai_response)
                else:
                    st.warning("比較するバリアントが2つ未満のため、詳細な分析は実行できません。")
            if st.button("AI分析を閉じる", key="ab_test_ai_close"):
                st.session_state.ab_test_ai_open = False

    # --- よくある質問 ---
    st.markdown("#### このページの分析について質問する")
    if 'ab_test_faq_toggle' not in st.session_state:
        st.session_state.ab_test_faq_toggle = {1: False, 2: False, 3: False, 4: False}

    faq_cols = st.columns(2)
    with faq_cols[0]:
        if st.button("どのバリアントが最も良かったですか？", key="faq_ab_1", use_container_width=True):
            st.session_state.ab_test_faq_toggle[1] = not st.session_state.ab_test_faq_toggle[1]
            st.session_state.ab_test_faq_toggle[2], st.session_state.ab_test_faq_toggle[3], st.session_state.ab_test_faq_toggle[4] = False, False, False
        if st.session_state.ab_test_faq_toggle[1]:
            if not ab_stats.empty:
                winner = ab_stats.sort_values('コンバージョン率', ascending=False).iloc[0]
                st.info(f"**「{winner['バリアント']}」** がCVR {winner['コンバージョン率']:.2f}%で最も良い結果でした。")
        
        if st.button("p値とは何ですか？", key="faq_ab_3", use_container_width=True):
            st.session_state.ab_test_faq_toggle[3] = not st.session_state.ab_test_faq_toggle[3]
            st.session_state.ab_test_faq_toggle[1], st.session_state.ab_test_faq_toggle[2], st.session_state.ab_test_faq_toggle[4] = False, False, False
        if st.session_state.ab_test_faq_toggle[3]:
            st.info("p値は「観測された差が偶然である確率」を示します。一般的にp値が0.05（5%）未満の場合、「統計的に有意な差がある」と判断し、その結果は信頼できると考えます。")
    with faq_cols[1]:
        if st.button("このテスト結果は信頼できますか？", key="faq_ab_2", use_container_width=True):
            st.session_state.ab_test_faq_toggle[2] = not st.session_state.ab_test_faq_toggle[2]
            st.session_state.ab_test_faq_toggle[1], st.session_state.ab_test_faq_toggle[3], st.session_state.ab_test_faq_toggle[4] = False, False, False
        if st.session_state.ab_test_faq_toggle[2]:
            if not ab_stats.empty:
                winner = ab_stats.sort_values('コンバージョン率', ascending=False).iloc[0]
                if winner['p値'] < 0.05:
                    st.info(f"はい、信頼できる可能性が高いです。勝者バリアントのp値は{winner['p値']:.4f}であり、統計的有意差の基準である0.05を下回っています。")
                else:
                    st.warning(f"まだ信頼できるとは言えません。p値が{winner['p値']:.4f}と0.05を上回っているため、この差が偶然である可能性を否定できません。もう少しテスト期間を延長してサンプルサイズを増やすことを推奨します。")
        
        if st.button("次のA/Bテストは何をすべき？", key="faq_ab_4", use_container_width=True):
            st.session_state.ab_test_faq_toggle[4] = not st.session_state.ab_test_faq_toggle[4]
            st.session_state.ab_test_faq_toggle[1], st.session_state.ab_test_faq_toggle[2], st.session_state.ab_test_faq_toggle[3] = False, False, False
        if st.session_state.ab_test_faq_toggle[4]:
            if not ab_stats.empty:
                winner = ab_stats.sort_values('コンバージョン率', ascending=False).iloc[0]
                st.info(f"今回の勝者「{winner['バリアント']}」の要素をベースに、さらに改善できる点をテストしましょう。例えば、CTAボタンの文言を変える、フォームの項目を減らす、などの新しい仮説でテストを計画するのが良いでしょう。")

# タブ5: インタラクション分析
elif selected_analysis == "インタラクション分析":

    st.markdown('<div class="sub-header">インタラクション分析</div>', unsafe_allow_html=True)

    # メインエリア: フィルターと比較設定
    st.markdown('<div class="sub-header">フィルター設定</div>', unsafe_allow_html=True) # type: ignore

    filter_cols_1 = st.columns(4)
    filter_cols_2 = st.columns(4)

    with filter_cols_1[0]:
        # 期間選択
        period_options = [
            "今日", "昨日", "過去7日間", "過去14日間", "過去30日間", "今月", "先月", "全期間", "カスタム"
        ]
        selected_period = st.selectbox("期間を選択", period_options, index=2, key="interaction_period")

    with filter_cols_1[1]:
        # LP選択
        lp_options = sorted(df['page_location'].dropna().unique().tolist())
        selected_lp = st.selectbox(
            "LP選択", 
            lp_options, 
            index=0 if lp_options else None,
            key="interaction_lp",
            disabled=not lp_options
        )

    with filter_cols_1[2]:
        device_options = ["すべて"] + sorted(df['device_type'].dropna().unique().tolist())
        selected_device = st.selectbox("デバイス選択", device_options, index=0, key="interaction_device")

    with filter_cols_1[3]:
        # 新規/リピート フィルター
        user_type_options = ["すべて", "新規", "リピート"]
        selected_user_type = st.selectbox("新規/リピート", user_type_options, index=0, key="interaction_user_type")

    with filter_cols_2[0]:
        # CV/非CV フィルター
        conversion_status_options = ["すべて", "コンバージョン", "非コンバージョン"]
        selected_conversion_status = st.selectbox("CV/非CV", conversion_status_options, index=0, key="interaction_conversion_status")
    
    with filter_cols_2[1]:
        # チャネルフィルターを追加
        channel_options = ["すべて"] + sorted(df['channel'].unique().tolist())
        selected_channel = st.selectbox("チャネル", channel_options, index=0, key="interaction_channel")

    with filter_cols_2[2]:
        # チャネルフィルターを「参照元/メディア」に変更
        source_medium_options = ["すべて"] + sorted(df['source_medium'].unique().tolist())
        selected_source_medium = st.selectbox("参照元/メディア", source_medium_options, index=0, key="interaction_source_medium") # ラベルは変更済み
    
    enable_comparison = False
    # 期間設定
    today = df['event_date'].max().date()
    
    if selected_period == "今日":
        start_date = today
        end_date = today
    elif selected_period == "昨日":
        start_date = today - timedelta(days=1)
        end_date = today - timedelta(days=1)
    elif selected_period == "過去7日間":
        start_date = today - timedelta(days=6)
        end_date = today
    elif selected_period == "過去14日間":
        start_date = today - timedelta(days=13)
        end_date = today
    elif selected_period == "過去30日間":
        start_date = today - timedelta(days=29)
        end_date = today
    elif selected_period == "今月":
        start_date = today.replace(day=1)
        end_date = today
    elif selected_period == "先月":
        last_month_end = today.replace(day=1) - timedelta(days=1)
        start_date = last_month_end.replace(day=1)
        end_date = last_month_end
    elif selected_period == "全期間":
        start_date = df['event_date'].min().date()
        end_date = df['event_date'].max().date()
    elif selected_period == "カスタム":
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("開始日", df['event_date'].min(), key="interaction_start_date")
        with col2:
            end_date = st.date_input("終了日", df['event_date'].max(), key="interaction_end_date")

    # --- 新規/リピート、CV/非CVの列を追加 ---
    df['user_type'] = np.where(df['ga_session_number'] == 1, '新規', 'リピート')
    conversion_session_ids = df[df['cv_type'].notna()]['session_id'].unique()
    df['conversion_status'] = np.where(df['session_id'].isin(conversion_session_ids), 'コンバージョン', '非コンバージョン')

    # データフィルタリング
    filtered_df = df.copy()

    # 期間フィルター
    filtered_df = filtered_df[
        (filtered_df['event_date'] >= pd.to_datetime(start_date)) &
        (filtered_df['event_date'] <= pd.to_datetime(end_date))
    ]

    # LPフィルター
    if selected_lp:
        filtered_df = filtered_df[filtered_df['page_location'] == selected_lp]

    # --- クロス分析用フィルター適用 ---
    if selected_device != "すべて":
        filtered_df = filtered_df[filtered_df['device_type'] == selected_device]

    if selected_user_type != "すべて":
        filtered_df = filtered_df[filtered_df['user_type'] == selected_user_type]

    if selected_conversion_status != "すべて":
        filtered_df = filtered_df[filtered_df['conversion_status'] == selected_conversion_status]

    if selected_channel != "すべて":
        filtered_df = filtered_df[filtered_df['channel'] == selected_channel]

    if selected_source_medium != "すべて":
        filtered_df = filtered_df[filtered_df['source_medium'] == selected_source_medium]

    # 比較機能は無効化
    comparison_df = None

    # データが空の場合の処理
    if len(filtered_df) == 0:
        st.warning("⚠️ 選択した条件に該当するデータがありません。フィルターを変更してください。")
        st.stop()

    # このページで必要なKPIを計算
    total_sessions = filtered_df['session_id'].nunique()

    # --- インタラクション要素一覧 ---
    st.markdown("#### インタラクション要素一覧")
    st.markdown('<div class="graph-description">LP内の各インタラクション要素について、要素が表示されたセッション数、クリック数、およびクリック率を表示します。</div>', unsafe_allow_html=True)

    interactions_for_list = {
        'CTAボタンクリック': {'condition': (filtered_df['event_name'] == 'click') & (filtered_df['elem_classes'].str.contains('cta|btn-primary', na=False)), 'page_num': 9},
        'フローティングバナークリック': {'condition': (filtered_df['event_name'] == 'click') & (filtered_df['elem_classes'].str.contains('floating', na=False)), 'page_num': 1}, # 全ページで表示されると仮定
        '離脱防止ポップアップクリック': {'condition': (filtered_df['event_name'] == 'click') & (filtered_df['elem_classes'].str.contains('exit', na=False)), 'page_num': 1}, # 全ページで表示されると仮定
        '動画視聴完了': {'condition': filtered_df['event_name'] == 'video_completion', 'page_num': 1} # 動画はP1にあると仮定
    }

    interaction_list_data = []
    for name, details in interactions_for_list.items():
        condition = details['condition']
        page_num = details['page_num']

        # クリック数または視聴開始数（イベントの行数）
        action_count = condition.sum()
        
        # その行動を取ったユニークなセッション数
        action_session_count = filtered_df[condition]['session_id'].nunique()

        # 表示セッション数（要素が存在するページに到達したセッション数）
        # page_numが1の場合は全セッションが表示機会を持つとみなす
        if page_num == 1:
            impression_sessions = total_sessions
        else:
            impression_sessions = filtered_df[filtered_df['max_page_reached'] >= page_num]['session_id'].nunique()

        # クリック率（行動セッション数 / 表示セッション数）
        rate = safe_rate(action_session_count, impression_sessions) * 100

        interaction_list_data.append({
            'インタラクション要素': name,
            '表示セッション数': impression_sessions,
            'クリック数または視聴完了数': action_count,
            'クリック率または視聴完了率': rate
        })

    interaction_list_df = pd.DataFrame(interaction_list_data)

    st.dataframe(interaction_list_df.style.format({
        '表示セッション数': '{:,.0f}',
        'クリック数または視聴完了数': '{:,.0f}',
        'クリック率または視聴完了率': '{:.2f}%'
    }), use_container_width=True, hide_index=True)

    st.markdown("---")


    # --- CV貢献度分析ロジック ---
    interactions = {
        'CTAボタンクリック': (filtered_df['event_name'] == 'click') & (filtered_df['elem_classes'].str.contains('cta|btn-primary', na=False)),
        'フローティングバナークリック': (filtered_df['event_name'] == 'click') & (filtered_df['elem_classes'].str.contains('floating', na=False)),
        '離脱防止ポップアップクリック': (filtered_df['event_name'] == 'click') & (filtered_df['elem_classes'].str.contains('exit', na=False)),
        '動画視聴完了': filtered_df['event_name'] == 'video_completion'
    }

    contribution_data = []

    for name, condition in interactions.items():
        # インタラクションを行ったセッションID
        interacted_session_ids = filtered_df[condition]['session_id'].unique()
        
        # 全セッションID
        all_session_ids = filtered_df['session_id'].unique()

        # インタラクションを行わなかったセッションID
        non_interacted_session_ids = np.setdiff1d(all_session_ids, interacted_session_ids)

        # CVしたセッションID
        cv_session_ids = filtered_df[filtered_df['cv_type'].notna()]['session_id'].unique()

        # グループA: インタラクション有り
        interacted_cv_sessions = np.intersect1d(interacted_session_ids, cv_session_ids)
        cvr_with_interaction = safe_rate(len(interacted_cv_sessions), len(interacted_session_ids)) * 100

        # グループB: インタラクション無し
        non_interacted_cv_sessions = np.intersect1d(non_interacted_session_ids, cv_session_ids)
        cvr_without_interaction = safe_rate(len(non_interacted_cv_sessions), len(non_interacted_session_ids)) * 100

        # CVRリフト率
        cvr_lift = safe_rate(cvr_with_interaction - cvr_without_interaction, cvr_without_interaction) * 100

        contribution_data.append({
            'インタラクション要素': name,
            'インタラクション有りCVR (%)': cvr_with_interaction,
            'インタラクション無しCVR (%)': cvr_without_interaction,
            'CVRリフト率 (%)': cvr_lift
        })

    contribution_df = pd.DataFrame(contribution_data)

    # CV貢献度テーブル
    st.markdown("#### インタラクション別 CV貢献度")
    st.markdown('<div class="graph-description">各行動（インタラクション）の「有り/無し」でユーザーを分け、それぞれのコンバージョン率（CVR）を比較します。「CVRリフト率」が高いほど、その行動がCVに強く貢献していることを示します。</div>', unsafe_allow_html=True)

    st.dataframe(contribution_df.style.format({
        'インタラクション有りCVR (%)': '{:.2f}%',
        'インタラクション無しCVR (%)': '{:.2f}%',
        'CVRリフト率 (%)': '{:+.1f}%'
    }), use_container_width=True, hide_index=True)

    # CV貢献度 比較グラフ
    st.markdown("#### CV貢献度 比較グラフ")
    st.markdown('<div class="graph-description">各インタラクションの「有り/無し」でのCVRを棒グラフで比較します。差が大きいほど、その行動がCVに与える影響が大きいことを意味します。</div>', unsafe_allow_html=True)

    # グラフ用にデータを整形
    plot_df = contribution_df.melt(
        id_vars=['インタラクション要素'],
        value_vars=['インタラクション有りCVR (%)', 'インタラクション無しCVR (%)'],
        var_name='グループ',
        value_name='CVR'
    )

    fig = px.bar(
        plot_df,
        x='インタラクション要素',
        y='CVR',
        color='グループ',
        barmode='group',
        text='CVR'
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(
        height=500,
        yaxis_title='コンバージョン率 (%)',
        xaxis_title='インタラクション要素',
        legend_title='グループ',
        dragmode=False,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )
    fig.update_traces(hovertemplate='%{x}<br>%{fullData.name}: %{y:.2f}%<extra></extra>')
    st.plotly_chart(fig, use_container_width=True, key='plotly_chart_interaction_contribution')

    st.markdown("---")

    # --- AI分析と考察 ---
    st.markdown("### AIによる分析と考察")
    st.markdown('<div class="graph-description">各インタラクション要素のパフォーマンスに基づき、AIがユーザーの関心と行動を分析します。</div>', unsafe_allow_html=True)

    # AI分析の表示状態を管理
    if 'interaction_ai_open' not in st.session_state:
        st.session_state.interaction_ai_open = False

    if st.button("AI分析を実行", key="interaction_ai_btn", type="primary", use_container_width=True):
        st.session_state.interaction_ai_open = True

    if st.session_state.interaction_ai_open:
        with st.container():
            with st.spinner("AIがインタラクションデータを分析中..."):
                # AI分析を実行
                ai_response = ai_analysis.analyze_interaction_expert(contribution_df)
                st.markdown(ai_response)
            if st.button("AI分析を閉じる", key="interaction_ai_close"):
                st.session_state.interaction_ai_open = False

    # --- よくある質問 ---
    st.markdown("#### このページの分析について質問する")
    if 'interaction_faq_toggle' not in st.session_state:
        st.session_state.interaction_faq_toggle = {1: False, 2: False, 3: False, 4: False}

    faq_cols = st.columns(2)
    with faq_cols[0]:
        if st.button("最もクリックされている要素は？", key="faq_interaction_1", use_container_width=True):
            st.session_state.interaction_faq_toggle[1] = not st.session_state.interaction_faq_toggle[1]
            st.session_state.interaction_faq_toggle[2], st.session_state.interaction_faq_toggle[3], st.session_state.interaction_faq_toggle[4] = False, False, False
        if st.session_state.interaction_faq_toggle[1] and not contribution_df.empty:
            best_lift_element = contribution_df.loc[contribution_df['CVRリフト率 (%)'].idxmax()]
            st.info(f"コンバージョンへの貢献度が最も高いのは「**{best_lift_element['インタラクション要素']}**」です。この行動を取ったユーザーのCVRは、取らなかったユーザーに比べて **{best_lift_element['CVRリフト率 (%)']:.1f}%** 高くなっています。")

        
        if st.button("クリック率が低い要素はどうすれば？", key="faq_interaction_3", use_container_width=True):
            st.session_state.interaction_faq_toggle[3] = not st.session_state.interaction_faq_toggle[3]
            st.session_state.interaction_faq_toggle[1], st.session_state.interaction_faq_toggle[2], st.session_state.interaction_faq_toggle[4] = False, False, False
        if st.session_state.interaction_faq_toggle[3]:
            st.info("クリック率が低い要素は、まずデザイン（色、サイズ、配置）を見直して視認性を高めましょう。それでも改善しない場合は、要素の文言（コピー）がユーザーにとって魅力的か、メリットが伝わるかを再検討する必要があります。")
    with faq_cols[1]:
        if st.button("CTAボタンのCTRを上げるには？", key="faq_interaction_2", use_container_width=True):
            st.session_state.interaction_faq_toggle[2] = not st.session_state.interaction_faq_toggle[2]
            st.session_state.interaction_faq_toggle[1], st.session_state.interaction_faq_toggle[3], st.session_state.interaction_faq_toggle[4] = False, False, False
        if st.session_state.interaction_faq_toggle[2]:
            st.info("CTAボタンのCTRを上げるには、1) ボタンの色を背景色と対照的な目立つ色にする、2) 「資料請求」→「無料で資料をもらう」のように具体的なアクションやメリットを文言に入れる、3) ボタンのサイズを大きくする、などのA/Bテストが有効です。")
        
        if st.button("デバイスによってクリック率は変わる？", key="faq_interaction_4", use_container_width=True):
            st.session_state.interaction_faq_toggle[4] = not st.session_state.interaction_faq_toggle[4]
            st.session_state.interaction_faq_toggle[1], st.session_state.interaction_faq_toggle[2], st.session_state.interaction_faq_toggle[3] = False, False, False
        if st.session_state.interaction_faq_toggle[4]:
            st.info("はい、大きく変わることがあります。例えば、PCではクリックしやすくても、スマホではボタンが小さすぎて押しにくい、といった問題が考えられます。「セグメント分析」でデバイス別のパフォーマンスを確認し、最適化することが重要です。")

# タブ6: 動画・スクロール分析
elif selected_analysis == "動画・スクロール分析":

    st.markdown('<div class="sub-header">動画・スクロール分析</div>', unsafe_allow_html=True)
    # メインエリア: フィルターと比較設定
    st.markdown('<div class="sub-header">フィルター設定</div>', unsafe_allow_html=True)

    filter_cols_1 = st.columns(4)
    filter_cols_2 = st.columns(4)

    with filter_cols_1[0]:
        # 期間選択
        period_options = [
            "今日", "昨日", "過去7日間", "過去14日間", "過去30日間", "今月", "先月", "全期間", "カスタム"
        ]
        selected_period = st.selectbox("期間を選択", period_options, index=2, key="video_scroll_period")

    with filter_cols_1[1]:
        # LP選択
        lp_options = sorted(df['page_location'].dropna().unique().tolist())
        selected_lp = st.selectbox(
            "LP選択", 
            lp_options, 
            index=0 if lp_options else None,
            key="video_scroll_lp",
            disabled=not lp_options
        )

    with filter_cols_1[2]:
        device_options = ["すべて"] + sorted(df['device_type'].dropna().unique().tolist())
        selected_device = st.selectbox("デバイス選択", device_options, index=0, key="video_scroll_device")

    with filter_cols_1[3]:
        # 新規/リピート フィルター
        user_type_options = ["すべて", "新規", "リピート"]
        selected_user_type = st.selectbox("新規/リピート", user_type_options, index=0, key="video_scroll_user_type")

    with filter_cols_2[0]:
        # CV/非CV フィルター
        conversion_status_options = ["すべて", "コンバージョン", "非コンバージョン"]
        selected_conversion_status = st.selectbox("CV/非CV", conversion_status_options, index=0, key="video_scroll_conversion_status")
    
    with filter_cols_2[1]:
        # チャネルフィルターを追加
        channel_options = ["すべて"] + sorted(df['channel'].unique().tolist())
        selected_channel = st.selectbox("チャネル", channel_options, index=0, key="video_scroll_channel")

    with filter_cols_2[2]:
        # チャネルフィルターを「参照元/メディア」に変更
        source_medium_options = ["すべて"] + sorted(df['source_medium'].unique().tolist())
        selected_source_medium = st.selectbox("参照元/メディア", source_medium_options, index=0, key="video_scroll_source_medium") # ラベルは変更済み
    
    enable_comparison = False
    # 期間設定
    today = df['event_date'].max().date()
    
    if selected_period == "今日":
        start_date = today
        end_date = today
    elif selected_period == "昨日":
        start_date = today - timedelta(days=1)
        end_date = today - timedelta(days=1)
    elif selected_period == "過去7日間":
        start_date = today - timedelta(days=6)
        end_date = today
    elif selected_period == "過去14日間":
        start_date = today - timedelta(days=13)
        end_date = today
    elif selected_period == "過去30日間":
        start_date = today - timedelta(days=29)
        end_date = today
    elif selected_period == "今月":
        start_date = today.replace(day=1)
        end_date = today
    elif selected_period == "先月":
        last_month_end = today.replace(day=1) - timedelta(days=1)
        start_date = last_month_end.replace(day=1)
        end_date = last_month_end
    elif selected_period == "全期間":
        start_date = df['event_date'].min().date()
        end_date = df['event_date'].max().date()
    elif selected_period == "カスタム":
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("開始日", df['event_date'].min(), key="video_scroll_start_date")
        with col2:
            end_date = st.date_input("終了日", df['event_date'].max(), key="video_scroll_end_date")

    # --- 新規/リピート、CV/非CVの列を追加 ---
    df['user_type'] = np.where(df['ga_session_number'] == 1, '新規', 'リピート')
    conversion_session_ids = df[df['cv_type'].notna()]['session_id'].unique()
    df['conversion_status'] = np.where(df['session_id'].isin(conversion_session_ids), 'コンバージョン', '非コンバージョン')

    # データフィルタリング
    filtered_df = df.copy()

    # 期間フィルター
    filtered_df = filtered_df[
        (filtered_df['event_date'] >= pd.to_datetime(start_date)) &
        (filtered_df['event_date'] <= pd.to_datetime(end_date))
    ]

    # LPフィルター
    if selected_lp:
        filtered_df = filtered_df[filtered_df['page_location'] == selected_lp]

    # --- クロス分析用フィルター適用 ---
    if selected_device != "すべて":
        filtered_df = filtered_df[filtered_df['device_type'] == selected_device]

    if selected_user_type != "すべて":
        filtered_df = filtered_df[filtered_df['user_type'] == selected_user_type]

    if selected_conversion_status != "すべて":
        filtered_df = filtered_df[filtered_df['conversion_status'] == selected_conversion_status]

    if selected_channel != "すべて":
        filtered_df = filtered_df[filtered_df['channel'] == selected_channel]

    if selected_source_medium != "すべて":
        filtered_df = filtered_df[filtered_df['source_medium'] == selected_source_medium]

    # 比較機能は無効化
    comparison_df = None

    # データが空の場合の処理
    if len(filtered_df) == 0:
        st.warning("⚠️ 選択した条件に該当するデータがありません。フィルターを変更してください。")
        st.stop()

    # このページで必要なKPIを計算
    total_sessions = filtered_df['session_id'].nunique()

    # --- 動画視聴ファネル ---
    st.markdown("#### 動画視聴ファネル")
    st.markdown('<div class="graph-description">動画の再生開始から視聴完了までのユーザーの残存率を可視化します。どの段階で離脱が多いかを把握できます。</div>', unsafe_allow_html=True)

    # GTMで計測されると想定されるイベント名
    video_events = {
        '再生開始': 'video_play',
        '25%視聴': 'video_progress_25',
        '50%視聴': 'video_progress_50',
        '75%視聴': 'video_progress_75',
        '視聴完了': 'video_completion'
    }

    funnel_data = []
    # 再生開始したユニークセッション数を基準とする
    start_sessions_count = filtered_df[filtered_df['event_name'] == video_events['再生開始']]['session_id'].nunique()

    if start_sessions_count > 0:
        for stage, event_name in video_events.items():
            # そのイベントを発生させたユニークセッション数をカウント
            sessions_count = filtered_df[filtered_df['event_name'] == event_name]['session_id'].nunique()
            funnel_data.append({'段階': stage, 'セッション数': sessions_count})

        video_funnel_df = pd.DataFrame(funnel_data)

        fig_video_funnel = go.Figure(go.Funnel(
            y=video_funnel_df['段階'],
            x=video_funnel_df['セッション数'],
            textinfo="value+percent initial",
            hovertemplate='段階: %{y}<br>セッション数: %{x:,}<extra></extra>'
        ))
        fig_video_funnel.update_layout(height=500, dragmode=False)
        st.plotly_chart(fig_video_funnel, use_container_width=True, key='plotly_chart_video_funnel')
    else:
        st.info("選択された期間に動画の再生データがありません。")

    st.markdown("---")

    # 逆行率分析
    st.markdown("ページ別平均逆行率")
    st.markdown('<div class="graph-description">各ページでユーザーがどれだけ逆方向にスクロールしたかを表示します。逆行率が高いページは、ユーザーが迷っているまたは情報を再確認している可能性があります。</div>', unsafe_allow_html=True) # type: ignore
    scroll_stats = filtered_df.groupby('page_num_dom')['scroll_pct'].mean().reset_index()
    scroll_stats.columns = ['ページ番号', '平均逆行率']
    scroll_stats['平均逆行率(%)'] = scroll_stats['平均逆行率'] * 100
    
    fig = px.bar(scroll_stats, x='ページ番号', y='平均逆行率(%)', text='平均逆行率(%)')
    fig.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='outside',
        hovertemplate='ページ: %{x}<br>逆行率: %{y:.1f}%<extra></extra>'
    )
    fig.update_layout(height=400, showlegend=False, xaxis_title='ページ番号', yaxis_title='平均逆行率 (%)', dragmode=False)
    st.plotly_chart(fig, use_container_width=True, key='plotly_chart_18') # This already has use_container_width=True
    
    # 動画視聴分析（動画イベントがある場合）
    video_df = filtered_df[filtered_df['video_src'].notna()]
    
    if len(video_df) > 0:
        st.markdown("#### 動画視聴率")
        
        video_sessions = video_df['session_id'].nunique()
        total_sessions_with_video_page = filtered_df[filtered_df['video_src'].notna()]['session_id'].nunique()
        video_view_rate = safe_rate(video_sessions, total_sessions) * 100
        
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
        video_cvr = safe_rate(video_cv, video_sessions) * 100
        
        non_video_sessions = total_sessions - video_sessions
        non_video_cv = filtered_df[(filtered_df['video_src'].isna()) & (filtered_df['cv_type'].notna())]['session_id'].nunique()
        non_video_cvr = safe_rate(non_video_cv, non_video_sessions) * 100
        
        comparison_data = pd.DataFrame({
            'グループ': ['動画視聴あり', '動画視聴なし'],
            'コンバージョン率': [video_cvr, non_video_cvr]
        })
        
        fig = px.bar(comparison_data, x='グループ', y='コンバージョン率', text='コンバージョン率')
        fig.update_traces(
            texttemplate='%{text:.2f}%',
            textposition='outside',
            hovertemplate='%{x}<br>CVR: %{y:.2f}%<extra></extra>'
        )
        fig.update_layout(height=400, showlegend=False, yaxis_title='コンバージョン率 (%)', dragmode=False)
        st.plotly_chart(fig, use_container_width=True, key='plotly_chart_19') # This already has use_container_width=True
    
    # 逆行率別CVR
    st.markdown("逆行率別コンバージョン率")
    st.markdown('<div class="graph-description">逆行率の範囲ごとにコンバージョン率を表示します。逆行率が高いほどコンバージョン率が低い傾向があるかを確認できます。</div>', unsafe_allow_html=True) # type: ignore
    
    # 逆行率を区間に分ける
    filtered_df_scroll = filtered_df.copy()
    filtered_df_scroll['scroll_range'] = pd.cut(filtered_df_scroll['scroll_pct'], bins=[0, 0.25, 0.5, 0.75, 1.0], labels=['0-25%', '25-50%', '50-75%', '75-100%'])
    
    scroll_range_sessions = filtered_df_scroll.groupby('scroll_range', observed=True)['session_id'].nunique().reset_index()
    scroll_range_sessions.columns = ['逆行率', 'セッション数']
    scroll_range_sessions['逆行率'] = scroll_range_sessions['逆行率'].astype(str)
    
    scroll_range_cv = filtered_df_scroll[filtered_df_scroll['cv_type'].notna()].groupby('scroll_range', observed=True)['session_id'].nunique().reset_index()
    scroll_range_cv.columns = ['逆行率', 'コンバージョン数']
    scroll_range_cv['逆行率'] = scroll_range_cv['逆行率'].astype(str)
    
    scroll_range_stats = scroll_range_sessions.merge(scroll_range_cv, on='逆行率', how='left')
    scroll_range_stats['コンバージョン数'] = scroll_range_stats['コンバージョン数'].fillna(0)
    scroll_range_stats['コンバージョン率'] = scroll_range_stats.apply(lambda row: safe_rate(row['コンバージョン数'], row['セッション数']) * 100, axis=1)
    
    fig = px.bar(scroll_range_stats, x='逆行率', y='コンバージョン率', text='コンバージョン率')
    fig.update_traces(
        texttemplate='%{text:.2f}%',
        textposition='outside',
        hovertemplate='逆行率: %{x}<br>CVR: %{y:.2f}%<extra></extra>'
    )
    fig.update_layout(height=400, showlegend=False, xaxis_title='逆行率', yaxis_title='コンバージョン率 (%)', dragmode=False)
    st.plotly_chart(fig, use_container_width=True, key='plotly_chart_20') # This already has use_container_width=True

    st.markdown("---")

    # --- AI分析と考察 ---
    st.markdown("### AIによる分析と考察")
    st.markdown('<div class="graph-description">動画視聴やスクロール行動とコンバージョンの関係性を分析し、エンゲージメント向上のヒントを提示します。</div>', unsafe_allow_html=True)

    # AI分析の表示状態を管理
    if 'video_scroll_ai_open' not in st.session_state:
        st.session_state.video_scroll_ai_open = False

    if st.button("AI分析を実行", key="video_scroll_ai_btn", type="primary", use_container_width=True):
        st.session_state.video_scroll_ai_open = True

    if st.session_state.video_scroll_ai_open:
        with st.container():
            with st.spinner("AIがエンゲージメントデータを分析中..."):
                # AI分析を実行
                # video_dfなどはフィルタリング済みデータから再取得する必要があるが、
                # ここでは簡略化のため、必要な統計情報を渡す形にする
                video_stats = None
                if len(filtered_df[filtered_df['video_src'].notna()]) > 0:
                    video_df = filtered_df[filtered_df['video_src'].notna()]
                    video_sessions = video_df['session_id'].nunique()
                    video_cv = video_df[video_df['cv_type'].notna()]['session_id'].nunique()
                    video_cvr = safe_rate(video_cv, video_sessions) * 100
                    
                    non_video_sessions = total_sessions - video_sessions
                    non_video_cv = filtered_df[(filtered_df['video_src'].isna()) & (filtered_df['cv_type'].notna())]['session_id'].nunique()
                    non_video_cvr = safe_rate(non_video_cv, non_video_sessions) * 100
                    
                    video_stats = {
                        'video_cvr': video_cvr,
                        'non_video_cvr': non_video_cvr,
                        'video_sessions': video_sessions
                    }
                
                ai_response = ai_analysis.analyze_video_scroll_expert(video_stats, scroll_stats)
                st.markdown(ai_response)

            if st.button("AI分析を閉じる", key="video_scroll_ai_close"):
                st.session_state.video_scroll_ai_open = False

    # --- よくある質問 ---
    st.markdown("#### このページの分析について質問する")
    if 'video_faq_toggle' not in st.session_state:
        st.session_state.video_faq_toggle = {1: False, 2: False, 3: False, 4: False}

    faq_cols = st.columns(2)
    with faq_cols[0]:
        if st.button("動画はコンバージョンに貢献していますか？", key="faq_video_1", use_container_width=True):
            st.session_state.video_faq_toggle[1] = not st.session_state.video_faq_toggle[1]
            st.session_state.video_faq_toggle[2], st.session_state.video_faq_toggle[3], st.session_state.video_faq_toggle[4] = False, False, False
        if st.session_state.video_faq_toggle[1]:
            if len(video_df) > 0:
                st.info(f"はい、貢献している可能性が高いです。動画視聴ユーザーのCVRは{video_cvr:.2f}%で、非視聴ユーザーの{non_video_cvr:.2f}%より高いです。")
            else:
                st.info("このLPには動画データがありません。")
        
        if st.button("逆行率が高いページは何が問題？", key="faq_video_3", use_container_width=True):
            st.session_state.video_faq_toggle[3] = not st.session_state.video_faq_toggle[3]
            st.session_state.video_faq_toggle[1], st.session_state.video_faq_toggle[2], st.session_state.video_faq_toggle[4] = False, False, False
        if st.session_state.video_faq_toggle[3]:
            st.info("逆行率が高いのは、ユーザーが「情報不足で前のページに戻って確認している」または「ページの構成が分かりにくく迷っている」兆候です。ページ間の情報の流れを見直し、ナビゲーションを分かりやすくする必要があります。")
    with faq_cols[1]:
        if st.button("動画のどこを改善すれば良いですか？", key="faq_video_2", use_container_width=True):
            st.session_state.video_faq_toggle[2] = not st.session_state.video_faq_toggle[2]
            st.session_state.video_faq_toggle[1], st.session_state.video_faq_toggle[3], st.session_state.video_faq_toggle[4] = False, False, False
        if st.session_state.video_faq_toggle[2]:
            st.info("動画の視聴維持率データを分析することが重要です。多くのユーザーが離脱する箇所を特定し、その部分のコンテンツ（メッセージ、テンポ、ビジュアル）を改善しましょう。特に最初の5秒でユーザーの心を掴むことが重要です。")
        
        if st.button("スクロールされないページはどうすれば？", key="faq_video_4", use_container_width=True):
            st.session_state.video_faq_toggle[4] = not st.session_state.video_faq_toggle[4]
            st.session_state.video_faq_toggle[1], st.session_state.video_faq_toggle[2], st.session_state.video_faq_toggle[3] = False, False, False
        if st.session_state.video_faq_toggle[4]:
            st.info("スクロールされないのは、ファーストビュー（FV）に魅力がない証拠です。ユーザーが「続きを読む価値がある」と感じるような、強力なキャッチコピー、魅力的な画像、権威付け（実績や推薦文など）をFVに配置することが効果的です。")



# タブ6: 時系列分析
elif selected_analysis == "時系列分析":
    st.markdown('<div class="sub-header">時系列分析</div>', unsafe_allow_html=True)

    # メインエリア: フィルターと比較設定
    st.markdown('<div class="sub-header">フィルター設定</div>', unsafe_allow_html=True)

    filter_cols_1 = st.columns(4)
    filter_cols_2 = st.columns(4)

    with filter_cols_1[0]:
        # 期間選択
        period_options = [
            "今日", "昨日", "過去7日間", "過去14日間", "過去30日間", "今月", "先月", "全期間", "カスタム"
        ]
        selected_period = st.selectbox("期間を選択", period_options, index=2, key="timeseries_period")

    with filter_cols_1[1]:
        # LP選択
        lp_options = sorted(df['page_location'].dropna().unique().tolist())
        selected_lp = st.selectbox(
            "LP選択", 
            lp_options, 
            index=0 if lp_options else None,
            key="timeseries_lp",
            disabled=not lp_options
        )

    with filter_cols_1[2]:
        device_options = ["すべて"] + sorted(df['device_type'].dropna().unique().tolist())
        selected_device = st.selectbox("デバイス選択", device_options, index=0, key="timeseries_device")

    with filter_cols_1[3]:
        # 新規/リピート フィルター
        user_type_options = ["すべて", "新規", "リピート"]
        selected_user_type = st.selectbox("新規/リピート", user_type_options, index=0, key="timeseries_user_type")

    with filter_cols_2[0]:
        # CV/非CV フィルター
        conversion_status_options = ["すべて", "コンバージョン", "非コンバージョン"]
        selected_conversion_status = st.selectbox("CV/非CV", conversion_status_options, index=0, key="timeseries_conversion_status")
    
    with filter_cols_2[1]:
        # チャネルフィルターを追加
        channel_options = ["すべて"] + sorted(df['channel'].unique().tolist())
        selected_channel = st.selectbox("チャネル", channel_options, index=0, key="timeseries_channel")

    with filter_cols_2[2]:
        # チャネルフィルターを「参照元/メディア」に変更
        source_medium_options = ["すべて"] + sorted(df['source_medium'].unique().tolist())
        selected_source_medium = st.selectbox("参照元/メディア", source_medium_options, index=0, key="timeseries_source_medium") # ラベルは変更済み
    
    enable_comparison = False
    # 期間設定
    today = df['event_date'].max().date()
    
    if selected_period == "今日":
        start_date = today
        end_date = today
    elif selected_period == "昨日":
        start_date = today - timedelta(days=1)
        end_date = today - timedelta(days=1)
    elif selected_period == "過去7日間":
        start_date = today - timedelta(days=6)
        end_date = today
    elif selected_period == "過去14日間":
        start_date = today - timedelta(days=13)
        end_date = today
    elif selected_period == "過去30日間":
        start_date = today - timedelta(days=29)
        end_date = today
    elif selected_period == "今月":
        start_date = today.replace(day=1)
        end_date = today
    elif selected_period == "先月":
        last_month_end = today.replace(day=1) - timedelta(days=1)
        start_date = last_month_end.replace(day=1)
        end_date = last_month_end
    elif selected_period == "全期間":
        start_date = df['event_date'].min().date()
        end_date = df['event_date'].max().date()
    elif selected_period == "カスタム":
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("開始日", df['event_date'].min(), key="timeseries_start_date")
        with col2:
            end_date = st.date_input("終了日", df['event_date'].max(), key="timeseries_end_date")

    # --- 新規/リピート、CV/非CVの列を追加 ---
    df['user_type'] = np.where(df['ga_session_number'] == 1, '新規', 'リピート')
    conversion_session_ids = df[df['cv_type'].notna()]['session_id'].unique()
    df['conversion_status'] = np.where(df['session_id'].isin(conversion_session_ids), 'コンバージョン', '非コンバージョン')

    # データフィルタリング
    filtered_df = df.copy()

    # 期間フィルター
    filtered_df = filtered_df[
        (filtered_df['event_date'] >= pd.to_datetime(start_date)) &
        (filtered_df['event_date'] <= pd.to_datetime(end_date))
    ]

    # LPフィルター
    if selected_lp:
        filtered_df = filtered_df[filtered_df['page_location'] == selected_lp]

    # --- クロス分析用フィルター適用 ---
    if selected_device != "すべて":
        filtered_df = filtered_df[filtered_df['device_type'] == selected_device]

    if selected_user_type != "すべて":
        filtered_df = filtered_df[filtered_df['user_type'] == selected_user_type]

    if selected_conversion_status != "すべて":
        filtered_df = filtered_df[filtered_df['conversion_status'] == selected_conversion_status]

    if selected_channel != "すべて":
        filtered_df = filtered_df[filtered_df['channel'] == selected_channel]

    if selected_source_medium != "すべて":
        filtered_df = filtered_df[filtered_df['source_medium'] == selected_source_medium]

    # 比較機能は無効化
    comparison_df = None

    # データが空の場合の処理
    if len(filtered_df) == 0:
        st.warning("⚠️ 選択した条件に該当するデータがありません。フィルターを変更してください。")
        st.stop()

    # 日別推移
    st.markdown("#### 日別推移") # type: ignore

    # テスト種別でフィルタリングするためのプルダウンメニュー
    # filtered_dfにab_test_target列がない場合があるため、ここでマッピングを適用
    if 'ab_test_target' not in filtered_df.columns:
        filtered_df['ab_test_target'] = df['ab_test_target'].map(test_type_map).fillna('-')

    # daily_statsの計算
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
    
    daily_stats = daily_stats.merge(daily_cv, on='日付', how='left').fillna(0) # type: ignore
    daily_stats['コンバージョン率'] = daily_stats.apply(lambda row: safe_rate(row['コンバージョン数'], row['セッション数']) * 100, axis=1) # type: ignore
    
    # FV残存率
    daily_fv = filtered_df[filtered_df['max_page_reached'] >= 2].groupby(
        filtered_df[filtered_df['max_page_reached'] >= 2]['event_date'].dt.date
    )['session_id'].nunique().reset_index()
    daily_fv.columns = ['日付', 'FV残存数']
    
    daily_stats = daily_stats.merge(daily_fv, on='日付', how='left').fillna(0) # type: ignore
    daily_stats['FV残存率'] = daily_stats.apply(lambda row: safe_rate(row['FV残存数'], row['セッション数']) * 100, axis=1)
    
    # 最終CTA到達率
    daily_cta = filtered_df[filtered_df['max_page_reached'] >= 10].groupby(
        filtered_df[filtered_df['max_page_reached'] >= 10]['event_date'].dt.date
    )['session_id'].nunique().reset_index()
    daily_cta.columns = ['日付', '最終CTA到達数']
    
    daily_stats = daily_stats.merge(daily_cta, on='日付', how='left').fillna(0)
    daily_stats['最終CTA到達率'] = daily_stats.apply(lambda row: safe_rate(row['最終CTA到達数'], row['セッション数']) * 100, axis=1)

    # グラフ選択
    metric_to_plot = st.selectbox("表示する指標を選択", [
        "セッション数", "コンバージョン数", "コンバージョン率", "FV残存率",
        "最終CTA到達率", "平均到達ページ数", "平均滞在時間(秒)"
    ], key="timeseries_metric_select")
    
    fig = px.line(daily_stats, x='日付', y=metric_to_plot, markers=True)
    # Add appropriate hovertemplate based on metric type
    if '率' in metric_to_plot or 'CVR' in metric_to_plot or 'CTR' in metric_to_plot:
        fig.update_traces(hovertemplate='日付: %{x}<br>' + metric_to_plot + ': %{y:.2f}<extra></extra>')
    elif 'ページ' in metric_to_plot:
        fig.update_traces(hovertemplate='日付: %{x}<br>' + metric_to_plot + ': %{y:.2f}<extra></extra>')
    elif '時間' in metric_to_plot:
        fig.update_traces(hovertemplate='日付: %{x}<br>' + metric_to_plot + ': %{y:.2f}<extra></extra>')
    else:
        fig.update_traces(hovertemplate='日付: %{x}<br>' + metric_to_plot + ': %{y:,}<extra></extra>')
    fig.update_layout(height=400, yaxis_title=metric_to_plot, dragmode=False)
    st.plotly_chart(fig, use_container_width=True, key='plotly_chart_21')
    
    # 月間推移（データが十分にある場合）
    if len(daily_stats) > 0 and (pd.to_datetime(daily_stats['日付'].max()) - pd.to_datetime(daily_stats['日付'].min())).days >= 60:
        st.markdown("#### 月間推移")
        
        filtered_df_monthly = filtered_df.copy()
        filtered_df_monthly['月'] = filtered_df_monthly['event_date'].dt.to_period('M').astype(str)
        
        monthly_stats = filtered_df_monthly.groupby('月').agg({
            'session_id': 'nunique',
            'max_page_reached': 'mean'
        }).reset_index()
        monthly_stats.columns = ['月', 'セッション数', '平均到達ページ数']
        
        monthly_cv = filtered_df_monthly[filtered_df_monthly['cv_type'].notna()].groupby('月')['session_id'].nunique().reset_index()
        monthly_cv.columns = ['月', 'コンバージョン数']
        
        monthly_stats = monthly_stats.merge(monthly_cv, on='月', how='left').fillna(0) # type: ignore
        monthly_stats['コンバージョン率'] = monthly_stats.apply(lambda row: safe_rate(row['コンバージョン数'], row['セッション数']) * 100, axis=1)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='セッション数', x=monthly_stats['月'], y=monthly_stats['セッション数'], yaxis='y'))
        fig.add_trace(go.Scatter(name='コンバージョン率', x=monthly_stats['月'], y=monthly_stats['コンバージョン率'], yaxis='y2', mode='lines+markers'))
        
        fig.update_layout(
            yaxis=dict(title='セッション数'),
            yaxis2=dict(title='コンバージョン率 (%)', overlaying='y', side='right'),
            height=400,
            dragmode=False
        )
        st.plotly_chart(fig, use_container_width=True, key='plotly_chart_22')

    st.markdown("---")

    # 曜日・時間帯別 CVRヒートマップ
    st.markdown("#### 曜日・時間帯別 CVRヒートマップ")
    st.markdown('<div class="graph-description">曜日と時間帯をクロス集計し、コンバージョン率（CVR）をヒートマップで表示します。色が濃い部分がCVRの高い曜日と時間帯です。</div>', unsafe_allow_html=True)

    # 曜日と時間の列を追加
    heatmap_df = filtered_df.copy()
    heatmap_df['hour'] = heatmap_df['event_timestamp'].dt.hour
    heatmap_df['dow_name'] = heatmap_df['event_timestamp'].dt.day_name()

    # 時間と曜日でグループ化してセッション数とCV数を計算
    heatmap_sessions = heatmap_df.groupby(['hour', 'dow_name'])['session_id'].nunique().reset_index(name='セッション数')
    heatmap_cv = heatmap_df[heatmap_df['cv_type'].notna()].groupby(['hour', 'dow_name'])['session_id'].nunique().reset_index(name='コンバージョン数')

    # データをマージしてCVRを計算
    heatmap_stats = pd.merge(heatmap_sessions, heatmap_cv, on=['hour', 'dow_name'], how='left').fillna(0)
    heatmap_stats['コンバージョン率'] = heatmap_stats.apply(lambda row: safe_rate(row['コンバージョン数'], row['セッション数']) * 100, axis=1)

    # 曜日の順序を定義
    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_map_jp = {'Monday': '月', 'Tuesday': '火', 'Wednesday': '水', 'Thursday': '木', 'Friday': '金', 'Saturday': '土', 'Sunday': '日'}
    heatmap_stats['dow_name'] = pd.Categorical(heatmap_stats['dow_name'], categories=dow_order, ordered=True)
    heatmap_stats = heatmap_stats.sort_values(['dow_name', 'hour'])

    # ピボットテーブルを作成
    heatmap_pivot = heatmap_stats.pivot_table(index='dow_name', columns='hour', values='コンバージョン率')
    heatmap_pivot = heatmap_pivot.reindex(dow_order) # 曜日の順序を保証
    heatmap_pivot.index = heatmap_pivot.index.map(dow_map_jp) # 曜日を日本語に変換

    # ヒートマップを描画
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_pivot.values,
        x=[f"{h}時" for h in heatmap_pivot.columns],
        y=heatmap_pivot.index,
        colorscale='Blues',
        hovertemplate='曜日: %{y}<br>時間帯: %{x}<br>CVR: %{z:.2f}%<extra></extra>'
    ))
    fig_heatmap.update_layout(title='曜日・時間帯別 CVR', height=500, dragmode=False)
    st.plotly_chart(fig_heatmap, use_container_width=True, key='plotly_chart_heatmap_cvr')

    st.markdown("---")

    # --- AI分析と考察 ---
    st.markdown("### AIによる分析と考察")
    st.markdown('<div class="graph-description">時系列データからパフォーマンスの波を読み解き、広告配信やプロモーションの最適化タイミングを提案します。</div>', unsafe_allow_html=True)
    
    # AI分析の表示状態を管理
    if 'timeseries_ai_open' not in st.session_state:
        st.session_state.timeseries_ai_open = False

    if st.button("AI分析を実行", key="timeseries_ai_btn", type="primary", use_container_width=True):
        st.session_state.timeseries_ai_open = True

    if st.session_state.timeseries_ai_open:
        with st.container():
            with st.spinner("AIが時系列データを分析中..."):
                # AI分析を実行
                # heatmap_statsを渡す
                ai_response = ai_analysis.analyze_timeseries_expert(heatmap_stats)
                st.markdown(ai_response)
            if st.button("AI分析を閉じる", key="timeseries_ai_close"):
                st.session_state.timeseries_ai_open = False

    # --- よくある質問 ---
    st.markdown("#### このページの分析について質問する")
    if 'time_faq_toggle' not in st.session_state:
        st.session_state.time_faq_toggle = {1: False, 2: False, 3: False, 4: False}

    faq_cols = st.columns(2)
    with faq_cols[0]:
        if st.button("CVRが最も高い時間帯はいつ？", key="faq_time_1", use_container_width=True):
            st.session_state.time_faq_toggle[1] = not st.session_state.time_faq_toggle[1]
            st.session_state.time_faq_toggle[2], st.session_state.time_faq_toggle[3], st.session_state.time_faq_toggle[4] = False, False, False
        if st.session_state.time_faq_toggle[1]:
            if not heatmap_stats.empty:
                golden_time = heatmap_stats.loc[heatmap_stats['コンバージョン率'].idxmax()]
                st.info(f"**{dow_map_jp[golden_time['dow_name']]}曜日の{int(golden_time['hour'])}時台**です。この時間帯のCVRは{golden_time['コンバージョン率']:.2f}%と最も高くなっています。")
        
        if st.button("週末と平日でパフォーマンスは違う？", key="faq_time_3", use_container_width=True):
            st.session_state.time_faq_toggle[3] = not st.session_state.time_faq_toggle[3]
            st.session_state.time_faq_toggle[1], st.session_state.time_faq_toggle[2], st.session_state.time_faq_toggle[4] = False, False, False
        if st.session_state.time_faq_toggle[3]:
            st.info("ヒートマップを確認することで、週末と平日のパフォーマンスの違いを視覚的に把握できます。一般的にBtoB商材は平日に、BtoC商材は週末や夜間にパフォーマンスが高まる傾向があります。")
    with faq_cols[1]:
        if st.button("ゴールデンタイムをどう活用すれば良い？", key="faq_time_2", use_container_width=True):
            st.session_state.time_faq_toggle[2] = not st.session_state.time_faq_toggle[2]
            st.session_state.time_faq_toggle[1], st.session_state.time_faq_toggle[3], st.session_state.time_faq_toggle[4] = False, False, False
        if st.session_state.time_faq_toggle[2]:
            st.info("CVRが高い「ゴールデンタイム」には、リスティング広告の入札単価を強化したり、SNS広告の配信を集中させることが有効です。また、メルマガ配信やSNS投稿もこの時間帯を狙うと効果的です。")
        
        if st.button("CVRが低い時間帯はどうすべき？", key="faq_time_4", use_container_width=True):
            st.session_state.time_faq_toggle[4] = not st.session_state.time_faq_toggle[4]
            st.session_state.time_faq_toggle[1], st.session_state.time_faq_toggle[2], st.session_state.time_faq_toggle[3] = False, False, False
        if st.session_state.time_faq_toggle[4]:
            st.info("CVRが著しく低い時間帯は、広告の配信を停止または抑制することで、無駄な広告費を削減し、全体の広告費用対効果（ROAS）を改善できます。")

# タブ7: リアルタイム分析
elif selected_analysis == "リアルタイムビュー":

    st.markdown('<div class="sub-header">リアルタイムビュー (Live)</div>', unsafe_allow_html=True)
    st.markdown('<div class="graph-description">現在サイトに訪問しているユーザーの活動をリアルタイムでモニタリングします。<br>※デモモード: 過去のデータをリアルタイム風に再生します。</div>', unsafe_allow_html=True)
    
    # ストリーミング制御
    if 'streaming_active' not in st.session_state:
        st.session_state.streaming_active = False
    
    col_control, col_status = st.columns([1, 4])
    with col_control:
        if not st.session_state.streaming_active:
            if st.button("モニタリング開始", type="primary", key="start_stream"):
                st.session_state.streaming_active = True
                st.rerun()
        else:
            if st.button("停止", type="secondary", key="stop_stream"):
                st.session_state.streaming_active = False
                st.rerun()
    
    with col_status:
        if st.session_state.streaming_active:
            st.success("● Live Monitoring Active")
        else:
            st.info("モニタリング停止中")

    # プレースホルダーの作成
    kpi_placeholder = st.empty()
    chart_placeholder = st.empty()
    log_placeholder = st.empty()

    if st.session_state.streaming_active:
        # ストリーミングループ
        # デモ用に直近のデータをベースに少しずつ追加していく
        base_time = datetime.now()
        
        # 初期データ（過去1時間分）
        current_df = df[df['event_timestamp'] >= (base_time - timedelta(hours=1))].copy()
        
        # ループ実行（最大100回または停止されるまで）
        for i in range(100):
            if not st.session_state.streaming_active:
                break
            
            # 擬似的な新着データ生成
            # ランダムに1〜5件のイベントを追加
            new_events_count = random.randint(1, 5)
            new_data = []
            for _ in range(new_events_count):
                # ランダムなイベントを生成
                evt_type = random.choice(['page_view', 'scroll', 'click', 'conversion'])
                if evt_type == 'conversion':
                    if random.random() > 0.1: # CVはレアにする
                        evt_type = 'page_view'
                
                new_event = {
                    'event_timestamp': base_time + timedelta(seconds=i*2), # 時間を進める
                    'session_id': f"live_user_{random.randint(1000, 1050)}",
                    'event_name': evt_type,
                    'page_location': 'https://shungene.lm-c.jp/tst08/tst08.html',
                    'stay_ms': random.randint(1000, 5000),
                    'max_page_reached': random.randint(1, 10),
                    'load_time_ms': random.randint(100, 500),
                    'cv_type': 'primary' if evt_type == 'conversion' else None
                }
                new_data.append(new_event)
            
            new_df = pd.DataFrame(new_data)
            new_df['event_timestamp'] = pd.to_datetime(new_df['event_timestamp'])
            
            # データフレームに追加
            current_df = pd.concat([current_df, new_df], ignore_index=True)
            
            # 直近1時間に絞る
            display_df = current_df[current_df['event_timestamp'] >= (current_df['event_timestamp'].max() - timedelta(hours=1))]
            
            # KPI再計算
            rt_sessions = display_df['session_id'].nunique()
            rt_active_users = display_df[display_df['event_timestamp'] >= (display_df['event_timestamp'].max() - timedelta(minutes=5))]['session_id'].nunique()
            rt_cvs = display_df[display_df['event_name'] == 'conversion'].shape[0]
            rt_avg_stay = display_df['stay_ms'].mean() / 1000 if len(display_df) > 0 else 0
            
            # KPI更新
            with kpi_placeholder.container():
                cols = st.columns(4)
                cols[0].metric("現在のアクティブユーザー", f"{rt_active_users}人", delta=random.choice([-1, 0, 1, 2]))
                cols[1].metric("直近1時間のセッション", f"{rt_sessions}人")
                cols[2].metric("直近1時間のCV数", f"{rt_cvs}件", delta_color="inverse")
                cols[3].metric("平均滞在時間", f"{rt_avg_stay:.1f}秒")

            # グラフ更新
            with chart_placeholder.container():
                display_df['minute_bin'] = display_df['event_timestamp'].dt.floor('1T') # 1分単位
                trend = display_df.groupby('minute_bin')['session_id'].count().reset_index() # イベント数
                trend.columns = ['時刻', 'アクティビティ']
                
                fig = px.bar(trend, x='時刻', y='アクティビティ', title="リアルタイム・アクティビティ推移")
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True, key=f"rt_chart_{i}")
            
            # ログ更新
            with log_placeholder.container():
                st.markdown("##### 最新のアクティビティ")
                for _, row in new_df.iterrows():
                    icon = "🟢" if row['event_name'] == 'page_view' else "🔵" if row['event_name'] == 'scroll' else "👆" if row['event_name'] == 'click' else "🎉"
                    st.text(f"{row['event_timestamp'].strftime('%H:%M:%S')} {icon} {row['session_id']} performed {row['event_name']}")
            
            time.sleep(1.5) # 更新間隔
    else:
        # 停止中の表示
        st.info("「モニタリング開始」ボタンを押すと、リアルタイムデモが始まります。")
        
        # 静的データの表示（プレビュー）
        one_hour_ago = df['event_timestamp'].max() - timedelta(hours=1)
        static_df = df[df['event_timestamp'] >= one_hour_ago]
        
        cols = st.columns(4)
        cols[0].metric("現在のアクティブユーザー", "-")
        cols[1].metric("直近1時間のセッション", f"{static_df['session_id'].nunique()}")
        cols[2].metric("直近1時間のCV数", f"{static_df[static_df['cv_type'].notna()]['session_id'].nunique()}")
        cols[3].metric("平均滞在時間", f"{static_df['stay_ms'].mean()/1000:.1f}秒")


# タブ8: カスタムオーディエンス
elif selected_analysis == "デモグラフィック情報":

    st.markdown('<div class="sub-header">デモグラフィック情報</div>', unsafe_allow_html=True)
    # メインエリア: フィルターと比較設定
    st.markdown('<div class="sub-header">フィルター設定</div>', unsafe_allow_html=True) # type: ignore

    # --- フィルター設定 ---
    filter_cols_1 = st.columns(4)
    filter_cols_2 = st.columns(4)

    with filter_cols_1[0]:
        period_options = {"過去7日間": 7, "過去30日間": 30, "過去90日間": 90, "カスタム期間": None}
        selected_period = st.selectbox("期間を選択", list(period_options.keys()), index=1, key="demographic_period")

    with filter_cols_1[1]:
        lp_options = sorted(df['page_location'].dropna().unique().tolist())
        selected_lp = st.selectbox(
            "LP選択", 
            lp_options, 
            index=0 if lp_options else None,
            key="demographic_lp",
            disabled=not lp_options
        )

    with filter_cols_1[2]:
        device_options = ["すべて"] + sorted(df['device_type'].dropna().unique().tolist())
        selected_device = st.selectbox("デバイス選択", device_options, index=0, key="demographic_device")

    with filter_cols_1[3]:
        # 新規/リピート フィルター
        user_type_options = ["すべて", "新規", "リピート"]
        selected_user_type = st.selectbox("新規/リピート", user_type_options, index=0, key="demographic_user_type")

    with filter_cols_2[0]:
        # CV/非CV フィルター
        conversion_status_options = ["すべて", "コンバージョン", "非コンバージョン"]
        selected_conversion_status = st.selectbox("CV/非CV", conversion_status_options, index=0, key="demographic_conversion_status")

    with filter_cols_2[1]:
        # チャネルフィルターを追加
        channel_options = ["すべて"] + sorted(df['channel'].unique().tolist())
        selected_channel = st.selectbox("チャネル", channel_options, index=0, key="demographic_channel")

    with filter_cols_2[2]:
        # チャネルフィルターを「参照元/メディア」に変更
        source_medium_options = ["すべて"] + sorted(df['source_medium'].unique().tolist())
        selected_source_medium = st.selectbox("参照元/メディア", source_medium_options, index=0, key="demographic_source_medium") # ラベルは変更済み

    enable_comparison = False

    # カスタム期間の場合
    if selected_period == "カスタム期間":
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("開始日", df['event_date'].min(), key="demographic_start_date")
        with col2:
            end_date = st.date_input("終了日", df['event_date'].max(), key="demographic_end_date")
    else:
        days = period_options[selected_period]
        end_date = df['event_date'].max()
        start_date = end_date - timedelta(days=days)

    st.markdown("---")

    # データフィルタリング
    filtered_df = df.copy()

    # 期間フィルター
    filtered_df = filtered_df[
        (filtered_df['event_date'] >= pd.to_datetime(start_date)) &
        (filtered_df['event_date'] <= pd.to_datetime(end_date))
    ]

    # LPフィルター
    if selected_lp and selected_lp != "すべて":
        filtered_df = filtered_df[filtered_df['page_location'] == selected_lp]
    
    # --- クロス分析用フィルター適用 ---
    if selected_device != "すべて":
        filtered_df = filtered_df[filtered_df['device_type'] == selected_device]

    if selected_user_type != "すべて":
        filtered_df = filtered_df[filtered_df['user_type'] == selected_user_type]

    if selected_conversion_status != "すべて":
        filtered_df = filtered_df[filtered_df['conversion_status'] == selected_conversion_status]

    if selected_channel != "すべて":
        filtered_df = filtered_df[filtered_df['channel'] == selected_channel]

    if selected_source_medium != "すべて":
        filtered_df = filtered_df[filtered_df['source_medium'] == selected_source_medium]

    # 比較機能は無効化
    comparison_df = None

    # データが空の場合の処理
    if len(filtered_df) == 0:
        st.warning("⚠️ 選択した条件に該当するデータがありません。フィルターを変更してください。")
        st.stop()

    # このページで必要なKPIを計算
    total_sessions = filtered_df['session_id'].nunique()

    st.markdown("ユーザーの属性情報（年齢、性別、地域、デバイス）を分析します。")

    # 年齢層別分析
    with st.expander("年齢層別分析", expanded=False):
        st.markdown('<div class="graph-description">年齢層ごとのセッション数、コンバージョン率、平均滞在時間を表示します。</div>', unsafe_allow_html=True)
        # 年齢層のダミーデータを 'age_group' 列として追加（BigQueryに実データがあればこの処理は不要）
        if 'age_group' not in filtered_df.columns:
            age_bins = [18, 25, 35, 45, 55, 65, 100]
            age_labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
            # 'age' 列がない場合はダミーの年齢を生成
            if 'age' not in filtered_df.columns:
                filtered_df['age'] = np.random.randint(18, 80, size=len(filtered_df))
            filtered_df['age_group'] = pd.cut(filtered_df['age'], bins=age_bins, labels=age_labels, right=False)

        # 年齢層別に集計
        age_sessions = filtered_df.groupby('age_group')['session_id'].nunique()
        age_cv = filtered_df[filtered_df['cv_type'].notna()].groupby('age_group')['session_id'].nunique()
        age_stay = filtered_df.groupby('age_group')['stay_ms'].mean() / 1000

        age_demo_df = pd.DataFrame({ # type: ignore
            'セッション数': age_sessions,
            'CV数': age_cv,
            '平均滞在時間 (秒)': age_stay
        }).fillna(0).reset_index().rename(columns={'age_group': '年齢層'}) # type: ignore
        age_demo_df['CVR (%)'] = age_demo_df.apply(lambda row: safe_rate(row['CV数'], row['セッション数']) * 100, axis=1)

        st.dataframe(age_demo_df.style.format({
            'セッション数': '{:,.0f}',
            'CV数': '{:,.0f}',
            'CVR (%)': '{:.1f}%',
            '平均滞在時間 (秒)': '{:.1f}'
        }), use_container_width=True, hide_index=True)

        fig = px.bar(age_demo_df, x='年齢層', y='CVR (%)', text='CVR (%)')
        fig.update_traces(
            texttemplate='%{text:.1f}%',
            textposition='outside',
            hovertemplate='%{x}<br>CVR: %{y:.1f}%<extra></extra>'
        )
        fig.update_layout(height=400, showlegend=False, xaxis_title='年齢層', yaxis_title='CVR (%)', dragmode=False)
        st.plotly_chart(fig, use_container_width=True, key='plotly_chart_age_cvr')

    with st.expander("性別分析", expanded=False):
        st.markdown('<div class="graph-description">性別ごとのセッション数、コンバージョン率、平均滞在時間を表示します。</div>', unsafe_allow_html=True)
        # 性別のダミーデータを 'gender' 列として追加（BigQueryに実データがあればこの処理は不要）
        if 'gender' not in filtered_df.columns:
            filtered_df['gender'] = np.random.choice(['男性', '女性', 'その他/未回答'], size=len(filtered_df), p=[0.52, 0.45, 0.03])

        # 性別で集計
        gender_sessions = filtered_df.groupby('gender')['session_id'].nunique()
        gender_cv = filtered_df[filtered_df['cv_type'].notna()].groupby('gender')['session_id'].nunique()
        gender_stay = filtered_df.groupby('gender')['stay_ms'].mean() / 1000

        gender_demo_df = pd.DataFrame({ # type: ignore
            'セッション数': gender_sessions,
            'CV数': gender_cv,
            '平均滞在時間 (秒)': gender_stay
        }).fillna(0).reset_index().rename(columns={'gender': '性別'}) # type: ignore
        gender_demo_df['CVR (%)'] = gender_demo_df.apply(lambda row: safe_rate(row['CV数'], row['セッション数']) * 100, axis=1)

        st.dataframe(gender_demo_df.style.format({
            'セッション数': '{:,.0f}',
            'CV数': '{:,.0f}',
            'CVR (%)': '{:.1f}%',
            '平均滞在時間 (秒)': '{:.1f}'
        }), use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(gender_demo_df, values='セッション数', names='性別', title='性別割合')
            fig.update_layout(height=400, dragmode=False)
            st.plotly_chart(fig, use_container_width=True, key='plotly_chart_gender_pie')
        with col2:
            fig = px.bar(gender_demo_df, x='性別', y='CVR (%)', text='CVR (%)')
            fig.update_traces(
                texttemplate='%{text:.1f}%',
                textposition='outside',
                hovertemplate='%{x}<br>CVR: %{y:.1f}%<extra></extra>'
            )
            fig.update_layout(height=400, showlegend=False, xaxis_title='性別', yaxis_title='CVR (%)', dragmode=False)
            st.plotly_chart(fig, use_container_width=True, key='plotly_chart_gender_cvr')
    
    # 地域別分析
    with st.expander("地域別分析", expanded=False):
        st.markdown('<div class="graph-description">都道府県ごとのセッション数、コンバージョン率を表示します。</div>', unsafe_allow_html=True)
        
        # 地域別ダミーデータ（サマリー表用） - これはBigQueryに地域データがない場合の代替として残します
        # BigQueryに地域データがある場合は、以下も動的生成に切り替えます
        region_demo_data = {
            '地域': ['東京都', '大阪府', '神奈川県', '愛知県', '福岡県', '北海道', 'その他'],
            'セッション数': [int(total_sessions * 0.25), int(total_sessions * 0.15), int(total_sessions * 0.10), int(total_sessions * 0.08), int(total_sessions * 0.07), int(total_sessions * 0.06), int(total_sessions * 0.29)],
            'CVR (%)': [3.8, 3.5, 3.2, 3.1, 3.4, 2.9, 3.0]
        }
        region_demo_df = pd.DataFrame(region_demo_data)
        st.dataframe(region_demo_df.style.format({
            'セッション数': '{:,.0f}',
            'CVR (%)': '{:.1f}'
        }), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("##### 都道府県別 CVRマップ")

        # GeoJSONデータを読み込む
        try:
            geojson_url = "https://raw.githubusercontent.com/dataofjapan/land/master/japan.geojson"
            import json
            import requests
            
            @st.cache_data
            def get_geojson():
                res = requests.get(geojson_url)
                return res.json()

            japan_geojson = get_geojson()

            # GeoJSONの各featureに、キーとして使える都道府県名（'東京', '大阪'など）を追加
            for feature in japan_geojson["features"]:
                pref_name_full = feature["properties"]["nam_ja"]
                # '北海道'はそのまま、他は'都','府','県'を削除
                feature["properties"]["pref_key"] = pref_name_full if pref_name_full == '北海道' else pref_name_full[:-1]

            # 表示用データフレームにもキー列を追加
            region_demo_df_for_map = region_demo_df.copy()
            region_demo_df_for_map['pref_key'] = region_demo_df_for_map['地域'].apply(
                lambda x: x if x == '北海道' else x[:-1] if x not in ['その他'] else 'その他'
            )

            # 地図用のデータフレームを作成
            map_df = pd.DataFrame({
                'pref_key': [f["properties"]["pref_key"] for f in japan_geojson["features"]]
            })

            # CVRデータをマージ
            map_df = map_df.merge(region_demo_df_for_map[['pref_key', 'CVR (%)']].rename(columns={'CVR (%)': 'コンバージョン率'}), on='pref_key', how='left')

            # 表にない都道府県は 'その他' のCVRで埋める
            other_cvr = region_demo_df[region_demo_df['地域'] == 'その他']['CVR (%)'].iloc[0]
            map_df['コンバージョン率'] = map_df['コンバージョン率'].fillna(other_cvr)

            # 地図を作成
            fig_map = px.choropleth_mapbox(
                map_df,
                geojson=japan_geojson,
                locations='pref_key', # locationsをキー列に変更
                featureidkey="properties.pref_key", # featureidkeyをキー列に変更
                color='コンバージョン率',
                color_continuous_scale="Blues",
                range_color=(map_df['コンバージョン率'].min(), map_df['コンバージョン率'].max()),
                mapbox_style="carto-positron",
                zoom=4.5,
                center={"lat": 36.2048, "lon": 138.2529},
                opacity=0.7,
                labels={'コンバージョン率': 'CVR (%)'},
                hover_name='pref_key' # ホバー時に都道府県名を表示
            )
            fig_map.update_layout(
                margin={"r":0,"t":0,"l":0,"b":0},
                height=600,
                coloraxis_colorbar=dict(
                    title="CVR (%)",
                    tickvals=[map_df['コンバージョン率'].min(), map_df['コンバージョン率'].max()],
                    ticktext=[f"{map_df['コンバージョン率'].min():.1f}%", f"{map_df['コンバージョン率'].max():.1f}%"]
                )
            )
            
            st.plotly_chart(fig_map, use_container_width=True)

        except Exception as e:
            st.error(f"地図の描画に失敗しました: {e}")
    
    # デバイス別分析
    with st.expander("デバイス別分析", expanded=False):
        st.markdown('<div class="graph-description">デバイスごとのセッション数、コンバージョン率、平均滞在時間を表示します。</div>', unsafe_allow_html=True)
        # デバイス別に集計
        device_sessions = filtered_df.groupby('device_type')['session_id'].nunique()
        device_cv = filtered_df[filtered_df['cv_type'].notna()].groupby('device_type')['session_id'].nunique()
        device_stay = filtered_df.groupby('device_type')['stay_ms'].mean() / 1000

        device_demo_df = pd.DataFrame({ # type: ignore
            'セッション数': device_sessions,
            'CV数': device_cv,
            '平均滞在時間 (秒)': device_stay,
        }).fillna(0).reset_index().rename(columns={'device_type': 'デバイス'}) # type: ignore
        device_demo_df['CVR (%)'] = device_demo_df.apply(lambda row: safe_rate(row['CV数'], row['セッション数']) * 100, axis=1)

        st.dataframe(device_demo_df.style.format({
            'セッション数': '{:,.0f}',
            'CV数': '{:,.0f}',
            'CVR (%)': '{:.1f}%',
            '平均滞在時間 (秒)': '{:.1f}'
        }), use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(device_demo_df, values='セッション数', names='デバイス', title='デバイス別セッション数')
            fig.update_layout(height=400, dragmode=False)
            st.plotly_chart(fig, use_container_width=True, key='plotly_chart_device_pie')
        with col2:
            fig = px.bar(device_demo_df, x='デバイス', y='CVR (%)', text='CVR (%)')
            fig.update_traces(
                texttemplate='%{text:.1f}%',
                textposition='outside',
                hovertemplate='%{x}<br>CVR: %{y:.1f}%<extra></extra>'
            )
            fig.update_layout(height=400, showlegend=False, xaxis_title='デバイス', yaxis_title='CVR (%)', dragmode=False)
            st.plotly_chart(fig, use_container_width=True, key='plotly_chart_device_cvr')

    st.markdown("---")

    # --- AI分析と考察 ---
    st.markdown("### AIによる分析と考察")
    st.markdown('<div class="graph-description">ユーザー属性（デモグラフィック）ごとの行動の違いを分析し、ターゲットユーザーの解像度を高めます。</div>', unsafe_allow_html=True)
    
    # AI分析の表示状態を管理
    if 'demographic_ai_open' not in st.session_state:
        st.session_state.demographic_ai_open = False

    if st.button("AI分析を実行", key="demographic_ai_btn", type="primary", use_container_width=True):
        st.session_state.demographic_ai_open = True

    if st.session_state.demographic_ai_open:
        with st.container():
            with st.spinner("AIがデモグラフィックデータを分析中..."):
                # AI分析を実行
                # age_demo_dfを渡す
                ai_response = ai_analysis.analyze_demographics_expert(age_demo_df)
                st.markdown(ai_response)
            if st.button("AI分析を閉じる", key="demographic_ai_close"):
                st.session_state.demographic_ai_open = False

    # --- よくある質問 ---
    st.markdown("#### このページの分析について質問する")
    if 'demographic_faq_toggle' not in st.session_state:
        st.session_state.demographic_faq_toggle = {1: False, 2: False, 3: False, 4: False}

    faq_cols = st.columns(2)
    with faq_cols[0]:
        if st.button("最もCVRが高い年齢層は？", key="faq_demo_1", use_container_width=True):
            st.session_state.demographic_faq_toggle[1] = not st.session_state.demographic_faq_toggle[1]
            st.session_state.demographic_faq_toggle[2], st.session_state.demographic_faq_toggle[3], st.session_state.demographic_faq_toggle[4] = False, False, False
        if st.session_state.demographic_faq_toggle[1]:
            if not age_demo_df.empty:
                best_age_group = age_demo_df.loc[age_demo_df['CVR (%)'].idxmax()]
                st.info(f"**{best_age_group['年齢層']}** です。この年齢層のCVRは{best_age_group['CVR (%)']:.1f}%と最も高くなっています。")
        
        if st.button("特定の地域だけCVRが高い理由は？", key="faq_demo_3", use_container_width=True):
            st.session_state.demographic_faq_toggle[3] = not st.session_state.demographic_faq_toggle[3]
            st.session_state.demographic_faq_toggle[1], st.session_state.demographic_faq_toggle[2], st.session_state.demographic_faq_toggle[4] = False, False, False
        if st.session_state.demographic_faq_toggle[3]:
            st.info("地域によってCVRに差が出るのは、地域限定のキャンペーン、競合の状況、地域特有のニーズ、または広告の地域ターゲティング設定などが原因として考えられます。")
    with faq_cols[1]:
        if st.button("この分析結果をどう広告に活かす？", key="faq_demo_2", use_container_width=True):
            st.session_state.demographic_faq_toggle[2] = not st.session_state.demographic_faq_toggle[2]
            st.session_state.demographic_faq_toggle[1], st.session_state.demographic_faq_toggle[3], st.session_state.demographic_faq_toggle[4] = False, False, False
        if st.session_state.demographic_faq_toggle[2]:
            if not age_demo_df.empty:
                best_age_group = age_demo_df.loc[age_demo_df['CVR (%)'].idxmax()]
                st.info(f"CVRが高い **{best_age_group['年齢層']}** や特定の性別・地域に広告のターゲティングを絞り込む、または予算を重点的に配分することで、広告の費用対効果を高めることができます。")
        
        if st.button("男女でLPの訴求を変えるべき？", key="faq_demo_4", use_container_width=True):
            st.session_state.demographic_faq_toggle[4] = not st.session_state.demographic_faq_toggle[4]
            st.session_state.demographic_faq_toggle[1], st.session_state.demographic_faq_toggle[2], st.session_state.demographic_faq_toggle[3] = False, False, False
        if st.session_state.demographic_faq_toggle[4]:
            st.info("もし男女でCVRやサイト内行動に大きな差が見られる場合は、訴求メッセージやデザインを男女別に最適化（パーソナライズ）することが有効です。例えば、男性には機能性を、女性には共感を呼ぶストーリーを訴求するなどの方法が考えられます。")




# タブ9: AI提案
elif selected_analysis == "AIによる分析・考察":
    st.markdown('<div class="sub-header">AI による分析・考察</div>', unsafe_allow_html=True)

    # メインエリア: フィルターと比較設定
    st.markdown('<div class="sub-header">フィルター設定</div>', unsafe_allow_html=True)
    filter_cols_1 = st.columns(4)
    filter_cols_2 = st.columns(4)

    with filter_cols_1[0]:
        # 期間選択
        period_options = [
            "今日", "昨日", "過去7日間", "過去14日間", "過去30日間", "今月", "先月", "全期間", "カスタム"
        ]
        selected_period = st.selectbox("期間を選択", period_options, index=2, key="ai_analysis_period")

    with filter_cols_1[1]:
        # LP選択
        lp_options = sorted(df['lp_base_url'].dropna().unique().tolist())
        selected_lp_base_url = st.selectbox(
            "LP選択", 
            lp_options, 
            index=0 if lp_options else None,
            key="ai_analysis_lp",
            disabled=not lp_options
        )

    with filter_cols_1[2]:
        device_options = ["すべて"] + sorted(df['device_type'].dropna().unique().tolist())
        selected_device = st.selectbox("デバイス選択", device_options, index=0, key="ai_analysis_device")

    with filter_cols_1[3]:
        # 新規/リピート フィルター
        user_type_options = ["すべて", "新規", "リピート"]
        selected_user_type = st.selectbox("新規/リピート", user_type_options, index=0, key="ai_analysis_user_type")

    with filter_cols_2[0]:
        # CV/非CV フィルター
        conversion_status_options = ["すべて", "コンバージョン", "非コンバージョン"]
        selected_conversion_status = st.selectbox("CV/非CV", conversion_status_options, index=0, key="ai_analysis_conversion_status")

    with filter_cols_2[1]:
        # チャネルフィルターを追加
        channel_options = ["すべて"] + sorted(df['channel'].unique().tolist())
        selected_channel = st.selectbox("チャネル", channel_options, index=0, key="ai_analysis_channel")

    with filter_cols_2[2]:
        # チャネルフィルターを「参照元/メディア」に変更
        source_medium_options = ["すべて"] + sorted(df['source_medium'].unique().tolist())
        selected_source_medium = st.selectbox("参照元/メディア", source_medium_options, index=0, key="ai_analysis_source_medium") # ラベルは変更済み

    # 期間設定
    today = df['event_date'].max().date()
    
    if selected_period == "今日":
        start_date = today
        end_date = today
    elif selected_period == "昨日":
        start_date = today - timedelta(days=1)
        end_date = today - timedelta(days=1)
    elif selected_period == "過去7日間":
        start_date = today - timedelta(days=6)
        end_date = today
    elif selected_period == "過去14日間":
        start_date = today - timedelta(days=13)
        end_date = today
    elif selected_period == "過去30日間":
        start_date = today - timedelta(days=29)
        end_date = today
    elif selected_period == "今月":
        start_date = today.replace(day=1)
        end_date = today
    elif selected_period == "先月":
        last_month_end = today.replace(day=1) - timedelta(days=1)
        start_date = last_month_end.replace(day=1)
        end_date = last_month_end
    elif selected_period == "全期間":
        start_date = df['event_date'].min().date()
        end_date = df['event_date'].max().date()
    elif selected_period == "カスタム":
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("開始日", df['event_date'].min(), key="ai_analysis_start_date")
        with col2:
            end_date = st.date_input("終了日", df['event_date'].max(), key="ai_analysis_end_date")

    # --- 新規/リピート、CV/非CVの列を追加 ---
    df['user_type'] = np.where(df['ga_session_number'] == 1, '新規', 'リピート')
    conversion_session_ids = df[df['cv_type'].notna()]['session_id'].unique()
    df['conversion_status'] = np.where(df['session_id'].isin(conversion_session_ids), 'コンバージョン', '非コンバージョン')

    comparison_type = None # 初期化
    # データフィルタリング
    filtered_df = df.copy()

    # 期間フィルター
    filtered_df = filtered_df[
        (filtered_df['event_date'] >= pd.to_datetime(start_date)) &
        (filtered_df['event_date'] <= pd.to_datetime(end_date))
    ]

    # LPフィルター
    if selected_lp_base_url:
        filtered_df = filtered_df[filtered_df['lp_base_url'] == selected_lp_base_url]

    # --- クロス分析用フィルター適用 ---
    if selected_device != "すべて":
        filtered_df = filtered_df[filtered_df['device_type'] == selected_device]

    if selected_user_type != "すべて":
        filtered_df = filtered_df[filtered_df['user_type'] == selected_user_type]

    if selected_conversion_status != "すべて":
        filtered_df = filtered_df[filtered_df['conversion_status'] == selected_conversion_status]

    if selected_channel != "すべて":
        filtered_df = filtered_df[filtered_df['channel'] == selected_channel]

    if selected_source_medium != "すべて":
        filtered_df = filtered_df[filtered_df['source_medium'] == selected_source_medium]

    # is_conversion列を作成
    filtered_df['is_conversion'] = filtered_df['cv_type'].notna().astype(int)

    # データが空の場合の処理
    if len(filtered_df) == 0:
        st.warning("⚠️ 選択した条件に該当するデータがありません。フィルターを変更してください。")
        st.stop()

    # 基本メトリクス計算
    total_sessions = filtered_df['session_id'].nunique()
    total_conversions = filtered_df[filtered_df['cv_type'].notna()]['session_id'].nunique()
    conversion_rate = safe_rate(total_conversions, total_sessions) * 100 # type: ignore
    # クリックしたユニークなセッション数をカウント
    clicked_sessions = filtered_df[filtered_df['event_name'] == 'click']['session_id'].nunique()
    total_clicks = clicked_sessions
    click_rate = safe_rate(total_clicks, total_sessions) * 100
    avg_stay_time = filtered_df['stay_ms'].mean() / 1000  # 秒に変換
    avg_pages_reached = filtered_df.groupby('session_id')['max_page_reached'].max().mean()
    fv_retention_rate = safe_rate(filtered_df[filtered_df['max_page_reached'] >= 2]['session_id'].nunique(), total_sessions) * 100
    final_cta_rate = safe_rate(filtered_df[filtered_df['max_page_reached'] >= 10]['session_id'].nunique(), total_sessions) * 100
    avg_load_time = filtered_df['load_time_ms'].mean()

    st.markdown('<div class="sub-header">主要指標（KPI）</div>', unsafe_allow_html=True)

    # 比較機能をKPIヘッダーの下に配置
    comp_cols = st.columns([1, 1, 4]) # チェックボックス、選択ボックス、スペーサー
    with comp_cols[0]:
        enable_comparison = st.checkbox("比較機能を有効化", value=False, key="ai_analysis_compare_check")
    with comp_cols[1]:
        if enable_comparison:
            comparison_options = {
                "前期間": "previous_period", "前週": "previous_week",
                "前月": "previous_month", "前年": "previous_year"
            }
            selected_comparison = st.selectbox("比較対象", list(comparison_options.keys()), key="ai_analysis_compare_select", label_visibility="collapsed")
            comparison_type = comparison_options[selected_comparison]

    # 比較データの取得
    comparison_df = None
    comp_start = None
    comp_end = None
    if enable_comparison and comparison_type:
        result = get_comparison_data(df, pd.Timestamp(start_date), pd.Timestamp(end_date), comparison_type)
        if result is not None:
            comparison_df, comp_start, comp_end = result
            # 比較データにも同じフィルターを適用
            if selected_lp_base_url:
                comparison_df = comparison_df[comparison_df['lp_base_url'] == selected_lp_base_url]
            # --- 比較データにもクロス分析用フィルターを適用 ---
            if selected_device != "すべて":
                comparison_df = comparison_df[comparison_df['device_type'] == selected_device]
            if selected_user_type != "すべて":
                comparison_df = comparison_df[comparison_df['user_type'] == selected_user_type]
            if selected_conversion_status != "すべて":
                comparison_df = comparison_df[comparison_df['conversion_status'] == selected_conversion_status]
            if selected_channel != "すべて":
                comparison_df = comparison_df[comparison_df['channel'] == selected_channel]
            if selected_channel != "すべて":
                comparison_df = comparison_df[comparison_df['source_medium'] == selected_source_medium]

            # 比較データが空の場合は無効化
            if len(comparison_df) == 0:
                comparison_df = None
                st.info(f"比較期間（{comp_start.strftime('%Y-%m-%d')} 〜 {comp_end.strftime('%Y-%m-%d')}）にデータがありません。")


    # 比較データのKPI計算
    comp_kpis = {}
    if comparison_df is not None and len(comparison_df) > 0:
        comp_total_sessions = comparison_df['session_id'].nunique()
        comp_total_conversions = comparison_df[comparison_df['cv_type'].notna()]['session_id'].nunique() # type: ignore
        comp_conversion_rate = safe_rate(comp_total_conversions, comp_total_sessions) * 100
        comp_clicked_sessions = comparison_df[comparison_df['event_name'] == 'click']['session_id'].nunique()
        comp_total_clicks = comp_clicked_sessions
        comp_click_rate = safe_rate(comp_total_clicks, comp_total_sessions) * 100
        comp_avg_stay_time = comparison_df['stay_ms'].mean() / 1000
        comp_avg_pages_reached = comparison_df.groupby('session_id')['max_page_reached'].max().mean()
        comp_fv_retention_rate = (comparison_df[comparison_df['max_page_reached'] >= 2]['session_id'].nunique() / comp_total_sessions * 100) if comp_total_sessions > 0 else 0
        comp_final_cta_rate = (comparison_df[comparison_df['max_page_reached'] >= 10]['session_id'].nunique() / comp_total_sessions * 100) if comp_total_sessions > 0 else 0
        comp_avg_load_time = comparison_df['load_time_ms'].mean()
        
        comp_kpis = {
            'sessions': comp_total_sessions,
            'conversions': comp_total_conversions,
            'conversion_rate': comp_conversion_rate,
            'clicks': comp_total_clicks,
            'click_rate': comp_click_rate,
            'avg_stay_time': comp_avg_stay_time,
            'avg_pages_reached': comp_avg_pages_reached,
            'fv_retention_rate': comp_fv_retention_rate,
            'final_cta_rate': comp_final_cta_rate,
            'avg_load_time': comp_avg_load_time
        }

    # KPIカード表示 (他のページからコピー)
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1: # type: ignore
        # セッション数
        delta_sessions = total_sessions - comp_kpis.get('sessions', 0) if comp_kpis else None
        st.metric("セッション数", f"{total_sessions:,}", delta=f"{delta_sessions:+,}" if delta_sessions is not None else None) # type: ignore
        
        # FV残存率
        delta_fv = fv_retention_rate - comp_kpis.get('fv_retention_rate', 0) if comp_kpis else None
        st.metric("FV残存率", f"{fv_retention_rate:.1f}%", delta=f"{delta_fv:+.1f}%" if delta_fv is not None else None)

    with col2:
        # コンバージョン数
        delta_conversions = total_conversions - comp_kpis.get('conversions', 0) if comp_kpis else None
        st.metric("コンバージョン数", f"{total_conversions:,}", delta=f"{delta_conversions:+,}" if delta_conversions is not None else None) # type: ignore

        # 最終CTA到達率
        delta_cta = final_cta_rate - comp_kpis.get('final_cta_rate', 0) if comp_kpis else None
        st.metric("最終CTA到達率", f"{final_cta_rate:.1f}%", delta=f"{delta_cta:+.1f}%" if delta_cta is not None else None)

    with col3:
        # コンバージョン率
        delta_cvr = conversion_rate - comp_kpis.get('conversion_rate', 0) if comp_kpis else None
        st.metric("コンバージョン率", f"{conversion_rate:.2f}%", delta=f"{delta_cvr:+.2f}%" if delta_cvr is not None else None) # type: ignore

        # 平均到達ページ数
        delta_pages = avg_pages_reached - comp_kpis.get('avg_pages_reached', 0) if comp_kpis else None
        st.metric("平均到達ページ数", f"{avg_pages_reached:.1f}", delta=f"{delta_pages:+.1f}" if delta_pages is not None else None)

    with col4:
        # クリック数
        delta_clicks = total_clicks - comp_kpis.get('clicks', 0) if comp_kpis else None
        st.metric("クリック数", f"{total_clicks:,}", delta=f"{delta_clicks:+,}" if delta_clicks is not None else None) # type: ignore

        # 平均滞在時間
        delta_stay = avg_stay_time - comp_kpis.get('avg_stay_time', 0) if comp_kpis else None
        st.metric("平均滞在時間", f"{avg_stay_time:.1f}秒", delta=f"{delta_stay:+.1f} 秒" if delta_stay is not None else None)

    with col5:
        # クリック率
        delta_click_rate = click_rate - comp_kpis.get('click_rate', 0) if comp_kpis else None
        st.metric("クリック率", f"{click_rate:.2f}%", delta=f"{delta_click_rate:+.2f}%" if delta_click_rate is not None else None) # type: ignore

        # 平均読込時間
        delta_load = avg_load_time - comp_kpis.get('avg_load_time', 0) if comp_kpis else None
        st.metric("平均読込時間", f"{avg_load_time:.0f}ms", delta=f"{delta_load:+.0f} ms" if delta_load is not None else None, delta_color="inverse")

    # --- ユーザー入力フォーム ---
    # AI分析の表示状態を管理するフラグを初期化
    if 'ai_analysis_open' not in st.session_state:
        st.session_state.ai_analysis_open = False

    st.markdown("---")
    st.markdown("### 目標値・現状値の入力")
    st.markdown('<div class="graph-description">AIが選択されたLPの内容とデータを多角的に分析し、現状評価や改善案を提案します。分析精度向上のため、目標値と現状値を入力してください。月間目標は選択期間に応じて日割り計算され、空欄でもAIが推測します。</div>', unsafe_allow_html=True)

    form_cols = st.columns(2)
    with form_cols[0]:
        st.markdown("##### **月間目標値**")
        target_cvr = st.number_input("目標CVR (%)", min_value=0.0, step=0.1, format="%.2f", value=None)
        target_cv = st.number_input("目標CV数", min_value=0, step=1, value=None)
        target_cpa = st.number_input("目標CPA", min_value=0, step=100, value=None)
    
    with form_cols[1]:
        st.markdown("##### **現状値**")
        current_cvr = st.number_input("現状CVR (%)", min_value=0.0, step=0.1, format="%.2f", value=None)
        current_cv = st.number_input("現状CV数", min_value=0, step=1, value=None)
        current_cpa = st.number_input("現状CPA", min_value=0, step=100, value=None)

    st.markdown("---")
    st.markdown("### ターゲット顧客・その他の情報")
    target_customer = st.text_area("ターゲット顧客について教えてください", placeholder="例：30代女性、都内在住、美容への関心が高い、オーガニック製品を好む")
    other_info = st.text_area("その他、分析で特に重視してほしい点などがあればご記入ください", placeholder="例：競合の〇〇と比較してほしい、特定の部分のコピーを重点的に見てほしい")

    if st.button("AI分析を実行", key="ai_analysis_main_btn", type="primary", use_container_width=True): # type: ignore
        st.session_state.ai_analysis_open = True

    if st.session_state.ai_analysis_open:
        with st.container():
            # LPのURLからテキストコンテンツを抽出
            lp_text_content = safe_extract_lp_text_content(extract_lp_text_content, selected_lp_base_url)
            main_headline = lp_text_content['headlines'][0] if lp_text_content['headlines'] else "（ヘッドライン取得不可）"
            # f-string内でエラーを起こさないようにトリプルクォートを別の文字に置換
            main_headline_escaped = main_headline.replace('"""', "'''")
            # ユーザー入力も同様に置換
            target_customer_escaped = target_customer.replace('"""', "'''") # type: ignore

            # AI分析に必要なデータをここで計算
            # ページ別統計
            page_stats = filtered_df.groupby('max_page_reached').agg(
                離脱セッション数=('session_id', 'nunique'),
                平均滞在時間_ms=('stay_ms', 'mean')
            ).reset_index()
            page_stats['離脱率'] = (page_stats['離脱セッション数'] / total_sessions * 100) if total_sessions > 0 else 0
            page_stats.rename(columns={'max_page_reached': 'ページ番号'}, inplace=True)
            max_exit_page = page_stats.loc[page_stats['離脱率'].idxmax()] if not page_stats.empty else {'ページ番号': 'N/A', '離脱率': 0}

            # デバイス別統計（修正）
            device_stats = filtered_df.groupby('device_type').agg(
                セッション数=('session_id', 'nunique'),
                コンバージョン数=('cv_type', lambda x: x.notna().sum())
            ).reset_index().rename(columns={'device_type': 'デバイス'})
            if not device_stats.empty:
                device_stats['コンバージョン率'] = (device_stats['コンバージョン数'] / device_stats['セッション数'] * 100).fillna(0)
                worst_device = device_stats.loc[device_stats['コンバージョン率'].idxmin()]
            else:
                worst_device = {'デバイス': 'N/A', 'コンバージョン率': 0}

            # チャネル別統計（修正）
            channel_stats = filtered_df.groupby('channel').agg(
                セッション数=('session_id', 'nunique'),
                コンバージョン数=('cv_type', lambda x: x.notna().sum())
            ).reset_index().rename(columns={'channel': 'チャネル'})
            if not channel_stats.empty:
                channel_stats['コンバージョン率'] = channel_stats.apply(lambda row: safe_rate(row['コンバージョン数'], row['セッション数']) * 100, axis=1)

            best_channel = channel_stats.loc[channel_stats['コンバージョン率'].idxmax()] if not channel_stats.empty else {'チャネル': 'N/A'}
            worst_channel = channel_stats.loc[channel_stats['コンバージョン率'].idxmin()] if not channel_stats.empty else {'チャネル': 'N/A'}
            
            # AIによる訴求ポイントの推察（簡易版）
            # 本来はLLMで要約するが、ここではキーワードで代用
            body_text = " ".join(lp_text_content['body_copy'])
            keywords = ["限定", "割引", "無料", "簡単", "満足度"]
            found_keywords = [kw for kw in keywords if kw in body_text]
            if found_keywords:
                inferred_appeal_point = f"LPのテキストから「{', '.join(found_keywords)}」というキーワードが検出されました。これらが主要な訴求ポイントと推察されます。"
            else:
                inferred_appeal_point = "LPのテキストから主要な訴求ポイントを自動推察します。（現在はキーワード検出のみ）"
            inferred_appeal_point_escaped = inferred_appeal_point.replace('"""', "'''")
            
            # 分析期間の日数を計算
            analysis_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1
            
            # 月間目標を日割り計算
            daily_target_cv = (target_cv / 30) * analysis_days if target_cv is not None and target_cv > 0 else 0
            daily_target_cvr = target_cvr if target_cvr is not None else 0 # CVRは期間によらないのでそのまま
            daily_target_cpa = target_cpa if target_cpa is not None else 0 # CPAも期間によらないのでそのまま

            # AI分析用のデータを準備
            kpi_data = {
                'current_cvr': current_cvr if current_cvr is not None else 0,
                'current_cv': current_cv if current_cv is not None else 0,
                'current_cpa': current_cpa if current_cpa is not None else 0,
                'fv_retention_rate': fv_retention_rate,
                'final_cta_rate': final_cta_rate,
                'max_exit_page': max_exit_page,
                'worst_device': worst_device,
                'best_channel': best_channel,
                'worst_channel': worst_channel,
                'inferred_appeal_point': inferred_appeal_point,
                'conversion_rate': conversion_rate
            }

            target_info = {
                'target_cvr': target_cvr,
                'target_cv': target_cv,
                'target_cpa': target_cpa,
                'daily_target_cv': daily_target_cv,
                'daily_target_cvr': daily_target_cvr,
                'target_customer': target_customer,
                'other_info': other_info,
                'analysis_days': analysis_days
            }

            with st.spinner("AIが改善提案を作成中..."):
                # AI分析を実行
                ai_response = ai_analysis.analyze_improvement_proposal_expert(lp_text_content, kpi_data, target_info)
                st.markdown(ai_response)

            # 閉じるボタン
            if st.button("AI分析を閉じる", key="ai_analysis_close"):
                st.session_state.ai_analysis_open = False
            st.success("AI分析が完了しました！上記の提案を参考に、LPの改善を進めてください。")
    
    # 既存の質問ボタンは保持
    
    # 質問ボタンにトグル機能を追加
    st.markdown("---")
    st.markdown("### このページの分析について質問する")
    
    # FAQ用のデータ計算を事前に初期化
    page_stats_global = pd.DataFrame(columns=['ページ番号', '離脱セッション数', '平均滞在時間_ms', '離脱率', '平均滞在時間_秒'])
    ab_stats_global = pd.DataFrame(columns=['バリアント', 'セッション数', 'コンバージョン数', 'コンバージョン率']) # type: ignore
    device_stats_global = pd.DataFrame(columns=['デバイス', 'セッション数', 'コンバージョン数', 'コンバージョン率'])

    if not filtered_df.empty and total_sessions > 0:
        # ページ別統計
        page_stats_global = filtered_df.groupby('max_page_reached').agg(
            離脱セッション数=('session_id', 'nunique'),
            平均滞在時間_ms=('stay_ms', 'mean')
        ).reset_index()
        page_stats_global['離脱率'] = (page_stats_global['離脱セッション数'] / total_sessions * 100) if total_sessions > 0 else 0
        page_stats_global['平均滞在時間_秒'] = page_stats_global['平均滞在時間_ms'] / 1000
        page_stats_global.rename(columns={'max_page_reached': 'ページ番号'}, inplace=True) # type: ignore

        # ab_variant列が存在する場合のみ集計
        if 'ab_variant' in filtered_df.columns and filtered_df['ab_variant'].notna().any():
            ab_stats_global = filtered_df.groupby('ab_variant').agg(
                セッション数=('session_id', 'nunique')
            ).reset_index().rename(columns={'ab_variant': 'バリアント'})
            ab_cv_stats = filtered_df[filtered_df['is_conversion'] == 1].groupby('ab_variant')['session_id'].nunique().reset_index(name='コンバージョン数').rename(columns={'ab_variant': 'バリアント'})
            ab_stats_global = pd.merge(ab_stats_global, ab_cv_stats, on='バリアント', how='left').fillna(0)
        else:
            ab_stats_global = pd.DataFrame(columns=['バリアント', 'セッション数', 'コンバージョン数'])
        
        if not ab_stats_global.empty and 'セッション数' in ab_stats_global.columns and ab_stats_global['セッション数'].sum() > 0:
            ab_stats_global['コンバージョン率'] = ab_stats_global.apply(lambda row: safe_rate(row['コンバージョン数'], row['セッション数']) * 100, axis=1)
        
        # デバイス別統計
        device_stats_global = filtered_df.groupby('device_type').agg(
            セッション数=('session_id', 'nunique')
        ).reset_index()
        device_cv_stats = filtered_df[filtered_df['is_conversion'] == 1].groupby('device_type')['session_id'].nunique().reset_index(name='コンバージョン数')
        device_stats_global = pd.merge(device_stats_global, device_cv_stats, on='device_type', how='left').fillna(0).rename(columns={'device_type': 'デバイス'})
        if not device_stats_global.empty and 'セッション数' in device_stats_global.columns and device_stats_global['セッション数'].sum() > 0: # type: ignore
            device_stats_global['コンバージョン率'] = device_stats_global.apply(lambda row: safe_rate(row['コンバージョン数'], row['セッション数']) * 100, axis=1)

    # FAQボタンの表示
    col1, col2 = st.columns(2)
    
    # session_stateの初期化
    if 'ai_faq_toggle' not in st.session_state: # type: ignore
        st.session_state.ai_faq_toggle = {1: False, 2: False, 3: False, 4: False}

    with col1:
        if st.button("このLPの最大のボトルネックは？", key="faq_btn_1", use_container_width=True):
            # ボタンが押されたら状態をトグルし、他を閉じる
            st.session_state.ai_faq_toggle[1] = not st.session_state.ai_faq_toggle[1]
            st.session_state.ai_faq_toggle[2] = False
            st.session_state.ai_faq_toggle[3] = False
            st.session_state.ai_faq_toggle[4] = False
        
        if st.session_state.ai_faq_toggle.get(1, False): # type: ignore
            # 離脱率が最も高いページを特定（データがある場合のみ）
            if not page_stats_global.empty and '離脱率' in page_stats_global.columns and not page_stats_global['離脱率'].empty:
                max_exit_page = page_stats_global.loc[page_stats_global['離脱率'].idxmax()]
                
                st.info(f"""
                **分析結果:**
                
                最大のボトルネックは**ページ{int(max_exit_page['ページ番号'])}**です。
                
                - 離脱率: {max_exit_page['離脱率']:.1f}%
                - 平均滞在時間: {max_exit_page['平均滞在時間_秒']:.1f}秒
                
                **推奨アクション:**
                1. ページ{int(max_exit_page['ページ番号'])}のコンテンツを見直し、ユーザーの関心を引く要素を追加
                2. A/Bテストで異なるコンテンツをテスト
                3. 読込時間が長い場合は、画像の最適化を検討
                """)
            else:
                st.warning("分析データがありません。")
        
        if st.button("コンバージョン率を改善するには？", key="faq_btn_2", use_container_width=True):
            st.session_state.ai_faq_toggle[2] = not st.session_state.ai_faq_toggle[2]
            st.session_state.ai_faq_toggle[1] = False
            st.session_state.ai_faq_toggle[3] = False
            st.session_state.ai_faq_toggle[4] = False
        
        if st.session_state.ai_faq_toggle.get(2, False): # type: ignore
            st.info(f"""
            **分析結果:**
            
            現在のコンバージョン率は**{conversion_rate:.2f}%**です。
            
            **推奨アクション:**
            1. FV残存率({fv_retention_rate:.1f}%)を改善するため、ファーストビューのコンテンツを強化
            2. 最終CTA到達率({final_cta_rate:.1f}%)を改善するため、ページ遷移をスムーズにする
            3. デバイス別の分析を行い、パフォーマンスが低いデバイスに最適化
            4. 高パフォーマンスのチャネルに予算を集中
            """)
    
    with col2: # type: ignore
        if st.button("A/Bテストの結果、どちらが優れている？", key="faq_btn_3", use_container_width=True):
            st.session_state.ai_faq_toggle[3] = not st.session_state.ai_faq_toggle[3]
            st.session_state.ai_faq_toggle[1] = False
            st.session_state.ai_faq_toggle[2] = False
            st.session_state.ai_faq_toggle[4] = False

        if st.session_state.ai_faq_toggle.get(3, False): # type: ignore
            if not ab_stats_global.empty and 'コンバージョン率' in ab_stats_global.columns and not ab_stats_global['コンバージョン率'].empty:
                best_variant = ab_stats_global.loc[ab_stats_global['コンバージョン率'].idxmax()]
                st.info(f"""
            **分析結果:**
            
            **バリアント{best_variant['バリアント']}**が最も優れています。
            
            - コンバージョン率: {best_variant['コンバージョン率']:.2f}%
            - セッション数: {int(best_variant['セッション数'])}
            
            **推奨アクション:**
            1. バリアント{best_variant['バリアント']}を本番環境に適用
            2. さらなる改善のため、次のA/Bテストを計画
            """)
            else:
                st.warning("A/Bテストの分析データがありません。")
        
        if st.button("デバイス別のパフォーマンス差は？", key="faq_btn_4", use_container_width=True):
            st.session_state.ai_faq_toggle[4] = not st.session_state.ai_faq_toggle[4]
            st.session_state.ai_faq_toggle[1] = False
            st.session_state.ai_faq_toggle[2] = False
            st.session_state.ai_faq_toggle[3] = False

        if st.session_state.ai_faq_toggle.get(4, False): # type: ignore
            if not device_stats_global.empty and 'コンバージョン率' in device_stats_global.columns and not device_stats_global['コンバージョン率'].empty:
                best_device = device_stats_global.loc[device_stats_global['コンバージョン率'].idxmax()]
                worst_device = device_stats_global.loc[device_stats_global['コンバージョン率'].idxmin()]
                st.info(f"""
            **分析結果:**
            
            **最高パフォーマンス:** {best_device['デバイス']} (CVR: {best_device['コンバージョン率']:.2f}%)
            **最低パフォーマンス:** {worst_device['デバイス']} (CVR: {worst_device['コンバージョン率']:.2f}%)
            
            **推奨アクション:**
            1. {worst_device['デバイス']}向けにUIを最適化
            2. {worst_device['デバイス']}での読込速度を改善
            3. {best_device['デバイス']}の成功要因を他デバイスに適用
            """)
            else:
                st.warning("分析データがありません。")

# タブ11: 専門用語解説
elif selected_analysis == "専門用語解説":
    st.markdown('<div class="sub-header">専門用語解説</div>', unsafe_allow_html=True)

    st.markdown("LP分析で使用される主要な用語を詳しく解説します。")
    
    # カテゴリー別に表示
    with st.expander("基本指標（KPI）", expanded=False):
        st.markdown("""
        **セッション（Session）**
        ユーザーがウェブサイトを訪れた1回の訪問。同じユーザーが複数回訪れた場合、それぞれ別のセッションとしてカウントされます。通常、30分間操作がないとセッションが終了します。
        
        **ユニークユーザー（Unique User）**
        特定の期間内にサイトを訪れたユニークな個人の数。CookieやデバイスIDで識別されます。
        
        **ページビュー（Page View / PV）**
        ページが表示された回数。同じページを何度も見た場合、その分だけカウントされます。
        
        **直帰率（Bounce Rate）**
        1ページだけを見てサイトを離れたセッションの割合。高いほど、ユーザーがすぐに離脱していることを意味します。
        
        **離脱率（Exit Rate）**
        特定のページでサイトを離れたセッションの割合。そのページがユーザージャーニーの最後になった割合を示します。
        
        **滞在時間（Session Duration）**
        ユーザーがサイトに滞在した時間。長いほどエンゲージメントが高いと考えられますが、コンテンツがわかりにくい可能性もあります。
        """)
    
    with st.expander("コンバージョン関連"):
        st.markdown("""
        **コンバージョン（Conversion / CV）**
        ユーザーが目標とする行動（購入、問い合わせ、会員登録など）を完了したこと。LPの最終目標です。
        
        **コンバージョン率（Conversion Rate / CVR）**
        セッション数に対するコンバージョン数の割合。  
        計算式: CVR = (コンバージョン数 ÷ セッション数) × 100  
        例: 1,000セッションで50コンバージョンならCVR = 5%
        
        **マイクロコンバージョン（Micro Conversion）**
        最終目標に至る前の中間目標。例: 資料ダウンロード、動画視聴、ページスクロールなど。
        
        **CPA（Cost Per Acquisition）**
        1件のコンバージョンを獲得するためにかかったコスト。  
        計算式: CPA = 広告費 ÷ コンバージョン数  
        侎いほど効率的です。
        
        **ROAS（Return On Ad Spend）**
        広告費用対効果。広告費1円あたりの売上。  
        計算式: ROAS = 売上 ÷ 広告費 × 100  
        例: 広告費10万円で売上50万円ならROAS = 500%
        """)
    
    with st.expander("LP特有の指標"):
        st.markdown("""
        **ファーストビュー（First View / FV）**
        ページを開いたときに最初に表示される画面範囲。スクロールしないで見える部分。LPで最も重要な要素で、ファーストビューで興味を引けないと即離脱されます。
        
        **FV残存率（FV Retention Rate）**
        ファーストビューを見た後、次のセクションに進んだユーザーの割合。高いほどFVが効果的です。業界平均は60-80%程度。
        
        **スクロール率（Scroll Depth）**
        ユーザーがページをどれだけスクロールしたかの割合。25%、50%、75%、100%で測定されることが多いです。100%はページの最後まで到達したことを意味します。
        
        **CTA（Call To Action）**
        ユーザーに具体的な行動を促すボタンやリンク。「今すぐ購入」「無料で試す」「資料をダウンロード」など。LPの最重要要素です。
        
        **最終CTA到達率**
        LPの最後に配置されたCTA（コンバージョンボタン）に到達したユーザーの割合。高いほどLP全体のコンテンツが効果的です。
        
        **ファネル（Funnel）**
        ユーザーがLPを進む過程を段階的に表した図。各ステップでどれだけのユーザーが離脱したかを可視化し、ボトルネック（問題箇所）を特定します。
        """)
    
    with st.expander("A/Bテスト・最適化"):
        st.markdown("""
        **A/Bテスト（A/B Testing）**
        2つ以上の異なるバージョン（バリアント）を同時に公開し、どちらが優れているかをデータで検証する手法。例: ヘッダー画像をAパターンとBパターンで比較。
        
        **バリアント（Variant）**
        A/Bテストで比較する各バージョン。Aパターン（オリジナル）、Bパターン（変更版）など。
        
        **統計的有意差（Statistical Significance）**
        A/Bテストの結果が偶然ではなく、本当に差があることを示す指標。通常95%以上の信頼水準で判断します。
        
        **多変量テスト（Multivariate Testing / MVT）**
        複数の要素を同時にテストする手法。例: ヘッダー画像、CTAボタンの色、コピーを同時に変えて最適な組み合わせを見つけます。
        
        **LPO（Landing Page Optimization）**
        LPのコンバージョン率を高めるための最適化施策。A/Bテスト、ヒートマップ分析、ユーザーテストなどを組み合わせて実施します。
        """)
    
    with st.expander("トラフィック・チャネル"):
        st.markdown("""
        **UTMパラメータ（UTM Parameters）**
        URLに付加するタグで、どの広告やキャンペーンからユーザーが来たかを追跡するためのもの。
        - **utm_source**: トラフィック元（例: google, facebook, newsletter）
        - **utm_medium**: 媒体（例: cpc, email, social）
        - **utm_campaign**: キャンペーン名（例: summer_sale_2024）
        - **utm_content**: 広告コンテンツ（例: banner_a, text_link）
        - **utm_term**: 検索キーワード（例: running+shoes）
        
        **チャネル（Channel）**
        ユーザーがLPに到達した経路のカテゴリー。
        - **Organic Search**: 自然検索（Google、Yahooなど）
        - **Paid Search**: 有料検索広告（リスティング広告）
        - **Organic Social**: SNSからの自然流入
        - **Paid Social**: SNS広告
        - **Direct**: 直接アクセス（ブックマーク、URL直打ち）
        - **Referral**: 他サイトからのリンク
        - **Email**: メールからの流入
        
        **リファラー（Referrer）**
        ユーザーが直前に訪れていたページのURL。どこから流入してきたかを特定できます。
        
        **ランディングページ（Landing Page / LP）**
        ユーザーが最初に着地したページ。広告や検索結果から誘導するために特別に設計されたページ。
        """)
    
    with st.expander("セグメント・オーディエンス"):
        st.markdown("""
        **セグメント（Segment）**
        特定の条件で絞り込んだユーザーグループ。例:
        - デバイス別: スマートフォン、タブレット、PC
        - チャネル別: Googleからのユーザー、SNSからのユーザー
        - 行動別: 購入済みユーザー、カート放棄ユーザー
        
        **オーディエンス（Audience）**
        特定の条件を満たすユーザーの集合。リターゲティング広告やパーソナライズに使用します。
        
        **カスタムオーディエンス（Custom Audience）**
        独自の条件で作成したオーディエンス。例: 「滞在時間60秒以上、スクロール率75%以上、コンバージョン未完了」のユーザー。
        
        **リターゲティング（Retargeting）**
        一度サイトを訪れたユーザーに対して、再度広告を表示する手法。コンバージョンしなかったユーザーを再誘導します。
        """)
    
    with st.expander("パフォーマンス指標"):
        st.markdown("""
        **読込時間（Page Load Time）**
        ページが完全に表示されるまでの時間。短いほどユーザー体験が良く、CVRも向上します。目標は3秒以内。
        
        **ファーストビュー表示時間（First Contentful Paint / FCP）**
        ページの最初のコンテンツが表示されるまでの時間。ユーザーが「ページが読み込まれている」と感じるまでの時間。
        
        **インタラクティブまでの時間（Time to Interactive / TTI）**
        ページが完全に操作可能になるまでの時間。ボタンやリンクがクリックできるようになるまでの時間。
        
        **クリック率（Click Through Rate / CTR）**
        表示回数（インプレッション）に対するクリック数の割合。  
        計算式: CTR = (クリック数 ÷ 表示回数) × 100  
        広告やCTAボタンの効果を測定します。
        
        **エンゲージメント率（Engagement Rate）**
        ユーザーがサイトでアクティブに行動した割合。クリック、スクロール、動画視聴などを含みます。
        """)
    
    with st.expander("分析ツール・手法"):
        st.markdown("""
        **ヒートマップ（Heatmap）**
        ユーザーのクリックやスクロール、マウスの動きを色で可視化したもの。赤い部分が最も注目されているエリア。
        
        **セッションリプレイ（Session Replay）**
        ユーザーの行動を動画のように再生する機能。ユーザーがどこで迷ったか、どこで離脱したかを詳細に分析できます。
        
        **コホート分析（Cohort Analysis）**
        同じ時期に訪れたユーザーグループ（コホート）の行動を時間経過で追跡する分析手法。リピート率やLTVの分析に使用します。
        
        **アトリビューション（Attribution）**
        コンバージョンに至るまでの複数のタッチポイント（広告、メール、SNSなど）の貢献度を評価する手法。
        - **ラストクリック**: 最後の接触に100%の貢献を割り当て。
        - **ファーストクリック**: 最初の接触に100%の貢献を割り当て。
        - **線形**: 全ての接触に均等に貢献を割り当て。
        """)

# タブ14: FAQ
elif selected_analysis == "FAQ":
    st.markdown('<div class="sub-header">「瞬ジェネ AIアナリスト」に関するFAQ</div>', unsafe_allow_html=True)
    st.markdown('<div class="footer">瞬ジェネ AIアナライザー - Powered by Gemini 3.0Pro</div>', unsafe_allow_html=True)

    st.markdown("#### 【基本的な使い方・機能について】")
    with st.expander("Q1. このアプリで何ができますか？", expanded=False):
        st.markdown("""
        A1. 瞬ジェネで作成したスワイプLPのパフォーマンスを多角的に分析し、改善点を見つけることができます。
        ページごとの離脱率や、どんなユーザーがコンバージョンしているかなどを可視化し、さらにAIがデータに基づいて具体的な改善案を提案しますので、LPのパフォーマンスアップが期待できます。
        """)
    with st.expander("Q2. どのような課題を解決できますか？", expanded=False):
        st.markdown("""
        A2. このツールは、以下のような課題を解決します。
        - GA4などのデータを見ても数字の意味がわからず改善案が浮かばない
        - データを図解化することでLPの状況が直感的にわかるようにしたい
        - A/Bテストの結果について本当に有意性があるのかいつも判断に迷う
        - 関係者へ分析結果を分かりやすく共有したい
        - LP改善を自社で行いたい
        """)
    with st.expander("Q3. 分析を始めるには、まず何をすれば良いですか？", expanded=False):
        st.markdown("""
        A3. まずは「AIによる分析・考察」ページで、分析したいLPを選び、目標値などを入力して「AI分析を実行」をクリックし、現状を把握します。
        その後、画面左のサイドバー（スマホの場合は画面左上の「>」をタップ）から、詳細に分析したいページ（例：「全体サマリー」「ページ分析」など）を選んでください。
        各ページの「このページの分析について質問する」をクリックすると、各ページの重要な指標となる分析結果が表示されます。
        """)

    st.markdown("#### 【AI分析機能について】")
    with st.expander("Q4. AI分析では、どのようなことを教えてくれますか？", expanded=False):
        st.markdown("""
        A4. 表示されているデータから「パフォーマンスが良い点・悪い点」「最も改善すべきボトルネック」「考えられる原因」などを自動で分析します。さらに、それに基づいて「次に何をすべきか」という具体的な改善提案まで提示します。
        """)
    with st.expander("Q5. AIの分析結果や提案は、毎回同じものですか？", expanded=False):
        st.markdown("""
        A5. いいえ、毎回異なります。AIの回答は、フィルターで選択された期間やLPのデータ、そして「AIによる分析・考察」ページでユーザーが入力した目標値などに基づいて、その都度動的に生成されます。そのため、常に現状に即したパーソナルな分析結果が得られます。
        """)
    with st.expander("Q6. AIの分析結果や提案は、絶対的に正しいですか？", expanded=False):
        st.markdown("""
        A6. いいえ、そもそもLP改善で絶対的な正解はありません。あくまで参考の1つとして捉え、最終判断はユーザーご自身で行ってください。ただし、詳細なデータをもとに高度で専門的なアドバイスを受けることができますので、運用のお役に立つことは間違いありません。
        """)

    st.markdown("#### 【データについて】")
    with st.expander("Q7. 分析に使われているデータはどこから来ていますか？", expanded=False):
        st.markdown("""
        A7. BigQueryと連携し、その実データを直接参照して分析する仕組みです。
        スワイプLPを多角的に分析できるよう100以上のデータ計測が可能です。
        """)
    with st.expander("Q8. データはリアルタイムに更新されますか？", expanded=False):
        st.markdown("""
        A8. いいえ。基本的にBigQueryのデータ更新と連動しており、瞬ジェネでは1日1回、早朝に更新される仕様です。
        「リアルタイムビュー」のみ、直近1時間程度の準リアルタイムデータを確認できます。
        """)

    st.markdown("#### 【料金・技術について】")
    with st.expander("Q9. このアプリを利用するのに料金はかかりますか？", expanded=False):
        st.markdown("""
        A9. このアプリ自体の利用は無料です。
        ただし、実データを分析するために利用するGoogle Cloud Platform (GCP) のサービス（BigQueryとGemini API）については、ご利用量に応じて料金が発生します。
        """)
    with st.expander("Q10. BigQueryやGemini APIの料金は、どのくらいかかりますか？", expanded=False):
        st.markdown("""
        A10. 小規模な利用（月間10万アクセス、1日数回の分析）であれば、GCPの無料枠で収まるか、月額数百円〜数千円程度と想定されます。
        GCPの予算アラート機能を設定すれば、予算消化に近づいたらメールで知らせてくれますので、想定外の費用が発生するのを防ぐことができるので安心です。
        """)

    st.markdown("#### 【トラブルシューティング】")
    with st.expander("Q11. データが表示されず、「該当するデータがありません」と表示されます。", expanded=False):
        st.markdown("""
        A11. フィルターでデータを絞り込みすぎている可能性があります。期間を広げたり、「デバイス」や「チャネル」のフィルターを「すべて」に戻したりして、条件をゆるめてみてください。
        """)
    with st.expander("Q12. ページ分析で画像のキャプチャが表示されません。", expanded=False):
        st.markdown("""
        A12. このアプリは、LPのHTML構造を解析して画像を取得しています。特殊な方法で画像を表示しているサイトの場合、うまく取得できないことがあります。その場合は、代替のプレースホルダー画像が表示されます。
        """)
    
# タブ12: アラート
elif selected_analysis == "アラート":
    # --- dfを決定 ---
    st.markdown('<div class="sub-header">アラート</div>', unsafe_allow_html=True)
    st.markdown('<div class="graph-description">主要指標の急な変化や異常を自動で検知し、お知らせします。この分析は、日次の全体パフォーマンスに基づいています。</div>', unsafe_allow_html=True)

    # アラート表示用のフラグ
    has_high_alerts = False
    has_medium_alerts = False

    # BigQueryのv_alertsビューと同様の計算をpandasで実行
    # 1. 日次KPIサマリーを作成 (v_kpi_daily相当)
    daily_kpi = df.groupby(df['event_date'].dt.date).agg(
        sessions=('session_id', 'nunique'), # type: ignore
        # cv_typeがNaNでないセッションのユニーク数をカウント
        conversions=('session_id', lambda x: df.loc[x.index][df.loc[x.index]['cv_type'].notna()]['session_id'].nunique())
    ).reset_index()
    daily_kpi['cvr'] = safe_rate(daily_kpi['conversions'], daily_kpi['sessions'])

    # 2. 移動平均と前日比を計算 (ma相当)
    if len(daily_kpi) > 7:
        daily_kpi = daily_kpi.sort_values('event_date').reset_index(drop=True)
        daily_kpi['sessions_ma7'] = daily_kpi['sessions'].rolling(window=7, min_periods=1).mean().shift(1)
        daily_kpi['cvr_ma7'] = daily_kpi['cvr'].rolling(window=7, min_periods=1).mean().shift(1)
        daily_kpi['sessions_prev'] = daily_kpi['sessions'].shift(1)
        daily_kpi['cvr_prev'] = daily_kpi['cvr'].shift(1)

        # 3. 変化率を計算
        daily_kpi['sessions_dod'] = safe_rate(daily_kpi['sessions'] - daily_kpi['sessions_prev'], daily_kpi['sessions_prev'])
        daily_kpi['cvr_dod'] = safe_rate(daily_kpi['cvr'] - daily_kpi['cvr_prev'], daily_kpi['cvr_prev'])
        daily_kpi['sessions_vs_ma7'] = safe_rate(daily_kpi['sessions'] - daily_kpi['sessions_ma7'], daily_kpi['sessions_ma7'])
        daily_kpi['cvr_vs_ma7'] = safe_rate(daily_kpi['cvr'] - daily_kpi['cvr_ma7'], daily_kpi['cvr_ma7'])

        # 最新日のデータを取得
        latest_alert_data = daily_kpi.iloc[-1]

        # アラートフラグ
        alerts = []

        # --- 重要度：高 ---
        if latest_alert_data['cvr_dod'] < -0.5:
            alerts.append({
                'level': 'high', 'title': 'CVRが急落',
                'description': f"**コンバージョン率が前日比で {abs(latest_alert_data['cvr_dod']):.1%} 大幅に低下しました。**",
                'details': f"前日: {latest_alert_data['cvr_prev']:.2%}, 本日: {latest_alert_data['cvr']:.2%}",
                'action': '時系列分析で確認', 'page': '時系列分析'
            })
        if latest_alert_data['sessions_dod'] < -0.5:
            alerts.append({
                'level': 'high', 'title': 'セッションが急減',
                'description': f"**セッション数が前日比で {abs(latest_alert_data['sessions_dod']):.1%} 大幅に減少しました。**",
                'details': f"前日: {int(latest_alert_data['sessions_prev']):,}, 本日: {int(latest_alert_data['sessions']):,}",
                'action': '全体サマリで確認', 'page': '全体サマリ'
            })

        # --- 重要度：中 ---
        if -0.5 <= latest_alert_data['cvr_dod'] < -0.3:
            alerts.append({
                'level': 'medium', 'title': 'CVRが低下',
                'description': f"**コンバージョン率が前日比で {abs(latest_alert_data['cvr_dod']):.1%} 低下しています。**",
                'details': f"前日: {latest_alert_data['cvr_prev']:.2%}, 本日: {latest_alert_data['cvr']:.2%}",
                'action': '時系列分析で確認', 'page': '時系列分析'
            })
        if -0.5 <= latest_alert_data['sessions_dod'] < -0.3:
            alerts.append({
                'level': 'medium', 'title': 'セッションが減少',
                'description': f"**セッション数が前日比で {abs(latest_alert_data['sessions_dod']):.1%} 減少しています。**",
                'details': f"前日: {int(latest_alert_data['sessions_prev']):,}, 本日: {int(latest_alert_data['sessions']):,}",
                'action': '時系列分析で確認', 'page': '時系列分析'
            })

        # アラートを表示
        high_alerts = [a for a in alerts if a['level'] == 'high']
        medium_alerts = [a for a in alerts if a['level'] == 'medium']

        if high_alerts:
            has_high_alerts = True
            st.markdown("#### 重要度：高")
            for alert in high_alerts:
                with st.container(): # type: ignore
                    col1, col2, col3 = st.columns([1, 4, 1.5])
                    with col1:
                        st.error(alert['title'])
                    with col2:
                        st.markdown(alert['description'])
                        st.markdown(f"<small>{alert['details']}</small>", unsafe_allow_html=True)
                    with col3:
                        st.button(alert['action'], key=f"alert_{alert['title']}", use_container_width=True, on_click=navigate_to, args=(alert['page'],))

        if medium_alerts: # type: ignore
            has_medium_alerts = True
            st.markdown("#### 重要度：中")
            for alert in medium_alerts:
                with st.container(): # type: ignore
                    col1, col2, col3 = st.columns([1, 4, 1.5])
                    with col1:
                        st.warning(alert['title'])
                    with col2:
                        st.markdown(alert['description'])
                        st.markdown(f"<small>{alert['details']}</small>", unsafe_allow_html=True)
                    with col3:
                        st.button(alert['action'], key=f"alert_{alert['title']}", use_container_width=True, on_click=navigate_to, args=(alert['page'],))

    if not has_high_alerts:
        st.info("現在、重要度の高いアラートはありません。")

    # 2つのセクションの間に区切り線を入れる
    if has_high_alerts and has_medium_alerts:
        st.markdown("---")

    if not has_medium_alerts and has_high_alerts: # 高アラートはあるが中アラートはない場合
        st.markdown("---")
        st.info("現在、重要度：中のアラートはありません。")
    else: # type: ignore
        st.info("アラートを生成するための十分なデータがありません（最低8日分のデータが必要です）。")

elif selected_analysis == "瞬フォーム分析":

    st.markdown('<div class="sub-header">瞬フォーム分析</div>', unsafe_allow_html=True)
    st.markdown('<div class="graph-description">このページは現在開発中のデモです。表示されている数値はダミーであり、他の分析ページとは連動していません。</div>', unsafe_allow_html=True)

    # --- フィルター設定 ---
    st.markdown('<div class="sub-header">フィルター設定</div>', unsafe_allow_html=True)
    filter_cols_1 = st.columns(4)
    filter_cols_2 = st.columns(4)

    # フィルターのキーにプレフィックスを追加してユニークにする
    with filter_cols_1[0]:
        period_options = ["今日", "昨日", "過去7日間", "過去14日間", "過去30日間", "今月", "先月", "全期間", "カスタム"]
        selected_period = st.selectbox("期間を選択", period_options, index=2, key="shun_form_period")

    with filter_cols_1[1]:
        lp_options = sorted(df['page_location'].dropna().unique().tolist())
        selected_lp = st.selectbox(
            "LP選択", 
            lp_options, 
            index=0 if lp_options else None, 
            key="shun_form_lp", 
            disabled=not lp_options
        )

    with filter_cols_1[2]:
        device_options = ["すべて"] + sorted(df['device_type'].dropna().unique().tolist())
        selected_device = st.selectbox(
            "デバイス選択", 
            device_options, 
            index=0, 
            key="shun_form_device"
        )

    with filter_cols_1[3]:
        user_type_options = ["すべて", "新規", "リピート"]
        selected_user_type = st.selectbox(
            "新規/リピート", 
            user_type_options, 
            index=0, 
            key="shun_form_user_type"
        )

    with filter_cols_2[0]:
        conversion_status_options = ["すべて", "コンバージョン", "非コンバージョン"]
        selected_conversion_status = st.selectbox("CV/非CV", conversion_status_options, index=0, key="shun_form_conversion_status")

    with filter_cols_2[1]:
        channel_options = ["すべて"] + sorted(df['channel'].unique().tolist())
        selected_channel = st.selectbox(
            "チャネル", 
            channel_options, 
            index=0, 
            key="shun_form_channel"
        )

    with filter_cols_2[2]:
        source_medium_options = ["すべて"] + sorted(df['source_medium'].unique().tolist())
        selected_source_medium = st.selectbox(
            "参照元/メディア", 
            source_medium_options, 
            index=0, 
            key="shun_form_source_medium"
        )

    # 期間設定
    today = df['event_date'].max().date()
    if selected_period == "今日":
        start_date, end_date = today, today
    elif selected_period == "昨日":
        start_date, end_date = today - timedelta(days=1), today - timedelta(days=1)
    elif selected_period == "過去7日間":
        start_date, end_date = today - timedelta(days=6), today
    elif selected_period == "過去14日間":
        start_date, end_date = today - timedelta(days=13), today
    elif selected_period == "過去30日間":
        start_date, end_date = today - timedelta(days=29), today
    elif selected_period == "今月":
        start_date, end_date = today.replace(day=1), today
    elif selected_period == "先月":
        last_month_end = today.replace(day=1) - timedelta(days=1)
        start_date, end_date = last_month_end.replace(day=1), last_month_end
    elif selected_period == "全期間":
        start_date, end_date = df['event_date'].min().date(), df['event_date'].max().date()
    elif selected_period == "カスタム":
        c1, c2 = st.columns(2)
        with c1:
            start_date = st.date_input("開始日", df['event_date'].min().date(), key="shun_form_start")
        with c2:
            end_date = st.date_input("終了日", df['event_date'].max().date(), key="shun_form_end")

    st.markdown("---")

    # --- 新規/リピート、CV/非CVの列を追加 ---
    df['user_type'] = np.where(df['ga_session_number'] == 1, '新規', 'リピート')
    conversion_session_ids = df[df['cv_type'].notna()]['session_id'].unique()
    df['conversion_status'] = np.where(df['session_id'].isin(conversion_session_ids), 'コンバージョン', '非コンバージョン')

    # データフィルタリング
    filtered_df = df[
        (df['event_date'] >= pd.to_datetime(start_date)) &
        (df['event_date'] <= pd.to_datetime(end_date))
    ]
    if selected_lp:
        filtered_df = filtered_df[filtered_df['page_location'] == selected_lp]
    if selected_device != "すべて":
        filtered_df = filtered_df[filtered_df['device_type'] == selected_device]
    if selected_user_type != "すべて":
        filtered_df = filtered_df[filtered_df['user_type'] == selected_user_type]
    if selected_conversion_status != "すべて":
        filtered_df = filtered_df[filtered_df['conversion_status'] == selected_conversion_status]
    if selected_channel != "すべて":
        filtered_df = filtered_df[filtered_df['channel'] == selected_channel]
    if selected_source_medium != "すべて":
        filtered_df = filtered_df[filtered_df['source_medium'] == selected_source_medium]

    # フォーム関連のイベントに絞る
    form_events = ['form_start', 'form_submit', 'form_progress', 'form_field_interaction']
    form_df = filtered_df[filtered_df['event_name'].isin(form_events)]

    if form_df.empty:
        st.warning("選択された条件に該当するフォームのデータがありません。")
        st.stop()

    # --- スコアカード ---
    st.markdown('<div class="sub-header">スコアカード</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)

    # KPI計算
    form_start_sessions = form_df[form_df['event_name'] == 'form_start']['session_id'].nunique()
    form_submit_sessions = form_df[form_df['event_name'] == 'form_submit']['session_id'].nunique()
    form_submission_rate = safe_rate(form_submit_sessions, form_start_sessions) * 100

    # フォーム表示直帰（フォームを開始したが、次のインタラクションがない）
    form_interaction_sessions = form_df[form_df['event_name'] != 'form_start']['session_id'].nunique()
    form_bounce_sessions = form_start_sessions - form_interaction_sessions
    form_bounce_rate = safe_rate(form_bounce_sessions, form_start_sessions) * 100

    # 平均進行ページ数
    if 'form_page_number' in form_df.columns:
        avg_progress_page = form_df.groupby('session_id')['form_page_number'].max().mean()
    else:
        avg_progress_page = 0

    # 平均滞在時間
    if 'form_duration_ms' in form_df.columns:
        avg_form_duration = form_df.groupby('session_id')['form_duration_ms'].sum().mean() / 1000
    else:
        avg_form_duration = 0

    # ページ逆行率
    if 'form_direction' in form_df.columns:
        backflow_sessions = form_df[form_df['form_direction'] == 'backward']['session_id'].nunique()
        backflow_rate = safe_rate(backflow_sessions, form_start_sessions) * 100
    else:
        backflow_rate = 0

    col1.metric("フォーム開始セッション", f"{form_start_sessions:,}")
    col2.metric("フォーム送信セッション", f"{form_submit_sessions:,}")
    col3.metric("フォーム送信率", f"{form_submission_rate:.1f}%")
    col4.metric("フォーム表示直帰率", f"{form_bounce_rate:.1f}%", delta_color="inverse")
    col1.metric("平均進行ページ数", f"{avg_progress_page:.1f}")
    col2.metric("平均滞在時間", f"{avg_form_duration:.1f}秒")
    col3.metric("ページ逆行率", f"{backflow_rate:.1f}%", delta_color="inverse")

    # --- ページごとの分析 ---
    st.markdown('<div class="sub-header">ページごとの分析</div>', unsafe_allow_html=True)
    st.markdown('<div class="graph-description">各ページの滞在時間や逆行率などを確認し、ユーザーがどの質問でつまずいているか（ボトルネック）を特定します。</div>', unsafe_allow_html=True)
    
    # ダミーデータ
    if 'form_page_number' in form_df.columns:
        page_analysis = form_df.groupby('form_page_number').agg(
            セッション数=('session_id', 'nunique'),
            平均滞在時間_ms=('form_duration_ms', 'mean'),
        ).reset_index()
        page_analysis.rename(columns={'form_page_number': 'ページ'}, inplace=True)
        page_analysis['平均滞在時間(秒)'] = page_analysis['平均滞在時間_ms'] / 1000

        # 離脱率
        exit_counts = []
        total_form_sessions = form_df['session_id'].nunique()
        for page_num in page_analysis['ページ']:
            reached_sessions = form_df[form_df['form_page_number'] >= page_num]['session_id'].nunique()
            exited_sessions = form_df[form_df.groupby('session_id')['form_page_number'].transform('max') == page_num]['session_id'].nunique()
            exit_rate = safe_rate(exited_sessions, reached_sessions) * 100
            exit_counts.append({'ページ': page_num, '離脱率(%)': exit_rate})
        
        exit_df = pd.DataFrame(exit_counts)
        page_analysis = pd.merge(page_analysis, exit_df, on='ページ', how='left')

        st.dataframe(page_analysis[['ページ', 'セッション数', '平均滞在時間(秒)', '離脱率(%)']].style.format({'平均滞在時間(秒)': '{:.1f}', '離脱率(%)': '{:.1f}%'}), use_container_width=True, hide_index=True)
    else:
        st.info("フォームのページ別データを表示できません。")

    st.markdown("---")

    # --- AI分析と考察 ---
    st.markdown("### AIによる分析と考察")
    st.markdown('<div class="graph-description">瞬フォームのパフォーマンスデータに基づき、AIが現状の評価と改善のための考察を提示します。</div>', unsafe_allow_html=True)

    if 'shun_form_ai_open' not in st.session_state:
        st.session_state.shun_form_ai_open = False

    if st.button("AI分析を実行", key="shun_form_ai_btn", type="primary", use_container_width=True):
        st.session_state.shun_form_ai_open = True

    if st.session_state.shun_form_ai_open:
        with st.container():
            with st.spinner("AIがフォームデータを分析中..."):
                st.markdown("#### 1. 現状の評価")
                st.info(f"""
                フォーム全体のパフォーマンスを分析した結果、**フォーム送信率（{form_submission_rate:.1f}%）** に改善の余地があることが分かりました。
                特に、**ページ3** での平均滞在時間が短く、ページ逆行率が他のページより高い傾向にあります。このページがユーザーにとってのボトルネックとなっている可能性が高いです。（※この部分は現在デモ用の固定テキストです）
                一方で、離脱防止POPや一時保存からの再開率は高く、フォームを完了したいというユーザーの意欲は高いと推察されます。
                """)

                st.markdown("#### 2. 今後の考察と改善案")
                st.warning("""
                **ページ3の質問内容の見直しが最優先課題です。（※この部分は現在デモ用の固定テキストです）**
                - **考察**: ページ3の質問がユーザーにとって分かりにくい、または答えるのが面倒だと感じさせている可能性があります。
                - **改善案**:
                    1. **質問文の簡略化**: より直感的で分かりやすい言葉に修正します。
                    2. **選択肢の見直し**: 選択肢が多すぎる場合は減らす、またはラジオボタンからプルダウンに変更するなど、UIを改善します。
                    3. **入力補助機能の追加**: 例えば、住所入力であれば郵便番号からの自動入力機能を追加します。
                
                これらの改善案についてA/Bテストを実施し、最も効果的な変更を特定することをお勧めします。
                """)

            if st.button("AI分析を閉じる", key="shun_form_ai_close"):
                st.session_state.shun_form_ai_open = False

    # --- よくある質問 ---
    st.markdown("#### このページの分析について質問する")
    if 'shun_form_faq_toggle' not in st.session_state:
        st.session_state.shun_form_faq_toggle = {1: False, 2: False, 3: False, 4: False}

    faq_cols = st.columns(2)
    with faq_cols[0]:
        if st.button("フォームのどこで離脱が多い？", key="faq_shun_form_1", use_container_width=True):
            st.session_state.shun_form_faq_toggle[1] = not st.session_state.shun_form_faq_toggle[1]
            st.session_state.shun_form_faq_toggle[2], st.session_state.shun_form_faq_toggle[3], st.session_state.shun_form_faq_toggle[4] = False, False, False
        if st.session_state.shun_form_faq_toggle[1]:
            st.info("ページごとの分析表で「ページ逆行率」が高いページや、「平均滞在時間」が極端に短いページが離脱の多いボトルネックです。このダミーデータでは「ページ3」が該当します。")

        if st.button("フォーム送信率を上げるには？", key="faq_shun_form_3", use_container_width=True):
            st.session_state.shun_form_faq_toggle[3] = not st.session_state.shun_form_faq_toggle[3]
            st.session_state.shun_form_faq_toggle[1], st.session_state.shun_form_faq_toggle[2], st.session_state.shun_form_faq_toggle[4] = False, False, False
        if st.session_state.shun_form_faq_toggle[3]:
            st.info("入力項目を減らす、必須項目を分かりやすくする、入力エラーをリアルタイムで表示する、などのEFO（入力フォーム最適化）施策が有効です。また、「一時保存からの再開率」が低い場合は、その機能をより目立たせることも重要です。")

    with faq_cols[1]:
        if st.button("平均進行ページ数が少ない原因は？", key="faq_shun_form_2", use_container_width=True):
            st.session_state.shun_form_faq_toggle[2] = not st.session_state.shun_form_faq_toggle[2]
            st.session_state.shun_form_faq_toggle[1], st.session_state.shun_form_faq_toggle[3], st.session_state.shun_form_faq_toggle[4] = False, False, False
        if st.session_state.shun_form_faq_toggle[2]:
            st.info("フォームの序盤（ページ1や2）でユーザーの興味を引けていない、または質問の意図が伝わっていない可能性があります。フォーム導入前のLPの訴求と、フォーム序盤の質問内容に一貫性があるか確認しましょう。")

        if st.button("離脱防止POPは効果がある？", key="faq_shun_form_4", use_container_width=True):
            st.session_state.shun_form_faq_toggle[4] = not st.session_state.shun_form_faq_toggle[4]
            st.session_state.shun_form_faq_toggle[1], st.session_state.shun_form_faq_toggle[2], st.session_state.shun_form_faq_toggle[3] = False, False, False
        if st.session_state.shun_form_faq_toggle[4]:
            st.info("「離脱防止POPから再開率」の指標で効果を測定できます。この数値が高い（例: 89.0%）場合、POPが表示されることで多くのユーザーがフォーム入力に復帰しており、効果的であると言えます。")


# タブ9: AIアナリスト（チャット）
elif selected_analysis == "AIアナリスト（チャット）":
    st.markdown('<div class="sub-header">AIアナリスト（チャット）</div>', unsafe_allow_html=True)
    st.markdown('<div class="graph-description">AIアナリストと対話しながら、データの深掘りや要因分析を行えます。</div>', unsafe_allow_html=True)

    # チャット履歴の初期化
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "こんにちは！私はAIアナリストです。現在のデータについて何でも聞いてください。\n例：「昨日のCVRが低かった原因は？」「スマホユーザーの傾向は？」"}
        ]

    # チャット履歴の表示
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ユーザー入力
    if prompt := st.chat_input("質問を入力してください..."):
        # ユーザーメッセージを表示
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # AI応答生成
        with st.chat_message("assistant"):
            with st.spinner("データを分析中..."):
                # コンテキストデータの作成（現在のフィルター適用済みデータを使用）
                # データ量が多すぎるとエラーになるため、サマリーを作成
                
                # 1. 基本KPI
                total_sessions = df['session_id'].nunique()
                cv_sessions = df[df['cv_type'].notna()]['session_id'].nunique()
                cvr = (cv_sessions / total_sessions * 100) if total_sessions > 0 else 0
                
                # 2. 日別トレンド（直近7日）
                daily_trend = df.groupby('event_date')['session_id'].nunique().tail(7).to_dict()
                
                # 3. デバイス別
                device_stats = df.groupby('device_type')['session_id'].nunique().to_dict()
                
                data_summary = f"""
                Total Sessions: {total_sessions}
                CV Sessions: {cv_sessions}
                CVR: {cvr:.2f}%
                Recent Daily Sessions: {daily_trend}
                Device Stats: {device_stats}
                Current Scenario: {st.session_state.get('data_scenario', 'Unknown')}
                """
                
                response = ai_analysis.chat_with_data(prompt, data_summary)
                st.markdown(response)
                
        # 履歴に追加
        st.session_state.messages.append({"role": "assistant", "content": response})

# フッター
st.markdown("---")
st.markdown("**瞬ジェネ AIアナライザー** - Powered by Gemini 3.0Pro")