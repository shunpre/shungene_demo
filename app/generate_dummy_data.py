"""
ダミーデータ生成スクリプト
BigQueryのevents_flat_tblテーブル構造に対応したリアルなイベントデータを生成
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import random
from scipy.stats import gamma, lognorm, norm

# --- Scenario Configurations ---
# --- Scenario Configurations ---
SCENARIO_CONFIGS = {
    '不調（離脱率高）': { # Old: Leaky Bucket
        'description': '流入多・離脱多（広告ロス多発）',
        'num_sessions_per_day_range': (800, 1200),
        'fv_exit_rate': 0.70, # FV離脱率 70%
        'transition_mean': 0.60, # ページ遷移率 かなり低い
        'transition_sd': 0.10,
        'bottleneck_pages': {3: 0.6, 5: 0.5}, # 3ページ目と5ページ目で半数が脱落
        'cta_click_rate_base': 0.05, # CTAクリック率 低い
        'cvr_multiplier': 0.8, # 想定CVR x 0.8 (不調)
        'stay_time_mu_base': 2.0, # ln(seconds). e^2.0 ≈ 7.4秒 (短め)
        'stay_time_sigma': 0.8, # ばらつき大
        'backflow_base': 0.02, # 戻る人は少ない
        'device_dist': ['mobile', 'desktop'],
        'device_weights': [0.9, 0.1], # ほぼスマホ
        'num_pages_dist': lambda: random.randint(8, 12), # ページ数変動
    },
    '好調（高エンゲージメント）': { # Old: Niche Fanbase
        'description': '流入少・高CVR（コアファン層）',
        'num_sessions_per_day_range': (50, 100),
        'fv_exit_rate': 0.15, # FV離脱率 15%
        'transition_mean': 0.98, # ほとんど次へ進む
        'transition_sd': 0.01,
        'bottleneck_pages': {}, # ボトルネックなし
        'cta_click_rate_base': 0.25, # CTAクリック率 高い
        'cvr_multiplier': 1.3, # 想定CVR x 1.3 (好調)
        'stay_time_mu_base': 3.5, # ln(seconds). e^3.5 ≈ 33秒 (じっくり読む)
        'stay_time_sigma': 0.4, # ばらつき小
        'backflow_base': 0.15, # 何度も読み返す
        'device_dist': ['mobile', 'desktop', 'tablet'],
        'device_weights': [0.6, 0.3, 0.1],
        'num_pages_dist': lambda: random.randint(15, 20), # 情報量多い
    },
    '不調（モバイル課題）': { # Old: Mobile Struggle
        'description': 'スマホだけ不調（レスポンシブ課題）',
        'num_sessions_per_day_range': (400, 600),
        'fv_exit_rate': 0.40,
        'transition_mean': 0.90,
        'transition_sd': 0.03,
        'bottleneck_pages': {2: 0.4, 6: 0.4}, # 特定ページでスマホユーザーが離脱
        'cta_click_rate_base': 0.10,
        'cvr_multiplier': 0.8, # 想定CVR x 0.8 (不調)
        'stay_time_mu_base': 2.5, # e^2.5 ≈ 12秒
        'stay_time_sigma': 0.6,
        'backflow_base': 0.05,
        'device_dist': ['mobile', 'desktop'],
        'device_weights': [0.7, 0.3],
        'num_pages_dist': lambda: random.randint(10, 14),
        # スマホ特有のデバフ
        'device_coeff': {
            'mobile': {'cvr': 0.2, 'stay': 0.6, 'load': 1.5}, # スマホ: CVR激減, 滞在短い(諦める), 遅い
            'desktop': {'cvr': 1.5, 'stay': 1.2, 'load': 0.8}, # PC: 快適
            'tablet': {'cvr': 1.0, 'stay': 1.0, 'load': 1.0}
        }
    },
    '標準（ベースライン）': { # Old: Standard
        'description': '標準的なパフォーマンス',
        'num_sessions_per_day_range': (300, 500),
        'fv_exit_rate': 0.35,
        'transition_mean': 0.92,
        'transition_sd': 0.03,
        'bottleneck_pages': {4: 0.2}, # 少しだけ離脱ポイントあり
        'cta_click_rate_base': 0.12,
        'cvr_multiplier': 1.0, # 想定CVR x 1.0 (普通)
        'stay_time_mu_base': 3.0, # e^3.0 ≈ 20秒
        'stay_time_sigma': 0.6,
        'backflow_base': 0.05,
        'device_dist': ['mobile', 'desktop', 'tablet'],
        'device_weights': [0.7, 0.25, 0.05],
        'num_pages_dist': lambda: random.randint(10, 16),
    }
}

# 共通設定 (シナリオで上書きされなければこれを使う)
DEFAULT_CONFIG = {
    'device_coeff': {
        'mobile': {'cvr': 0.9, 'stay': 0.9, 'load': 1.1},
        'desktop': {'cvr': 1.1, 'stay': 1.1, 'load': 0.9},
        'tablet': {'cvr': 1.0, 'stay': 1.0, 'load': 1.0}
    },
    'channel_dist': ['Organic Search', 'Paid Search', 'Paid Social', 'Direct', 'Referral', 'Other'],
    'channel_weights': [0.35, 0.25, 0.20, 0.10, 0.05, 0.05],
    'channel_coeff': {
        'Paid Search': {'cvr': 1.2, 'stay': 1.1},
        'Paid Social': {'cvr': 0.8, 'stay': 0.8}, # ソーシャルは流し見が多い
        'Organic Search': {'cvr': 1.1, 'stay': 1.2},
        'Direct': {'cvr': 1.0, 'stay': 1.0},
        'Referral': {'cvr': 1.1, 'stay': 1.1},
        'Other': {'cvr': 0.7, 'stay': 0.8}
    },
    'hour_seasonality': {12: 1.2, 20: 1.3, 21: 1.3, 22: 1.2, 23: 1.1}, # 夜と昼にピーク
    'weekday_seasonality': {'Mon': 1.0, 'Tue': 1.0, 'Wed': 1.0, 'Thu': 1.0, 'Fri': 0.9, 'Sat': 1.1, 'Sun': 1.1}, # 週末強い
    'load_time_k': 2,
    'load_time_theta_ms': 300,
    'load_time_impact_stay_ms': -0.05 / 100, # 100ms遅れるごとに滞在5%減
    'backflow_stay_bonus': 0.3, # 戻ったときは滞在30%増
    'cta_scroll_penalty': 0.2,
    'cta_video_bonus': 0.4,
    'fb_depth_bonus': 0.1,
    'exit_pop_bounce_bonus': 0.2,
    'info_jump_backflow_bonus': 0.6,
}

def generate_dummy_data(scenario: str = '標準（ベースライン）', num_days: int = 30, num_pages: int = 10, target_cvr: float = 0.04, difficulty: str = '初級（穏やかな波）'):
    """
    リアルなスワイプLPイベントデータを生成
    difficulty: '初級（穏やかな波）', '中級（乱高下）', '上級（急降下）'
    """
    print(f"DEBUG: generate_dummy_data called with difficulty={difficulty}")
    # シナリオ設定のロードとマージ
    scenario_config = SCENARIO_CONFIGS.get(scenario, SCENARIO_CONFIGS['標準（ベースライン）'])
    config = DEFAULT_CONFIG.copy()
    config.update(scenario_config)
    
    # CVR設定: 想定CVR x シナリオ倍率
    base_cvr_original = target_cvr * config.get('cvr_multiplier', 1.0)

    # 基準日時
    end_date = datetime.now()
    start_date = end_date - timedelta(days=num_days)
    
    # LP URL
    lp_url_base = "https://shungene.lm-c.jp/tst08/tst08.html"
    
    # イベント名
    event_names = [
        "page_view", "swipe_page", "click", "form_start", "form_submit", 
        "form_progress", "scroll", "video_play", "conversion", "session_start"
    ]
    
    # UTMパラメータ設定
    traffic_sources = {
        "google": {"mediums": ["organic", "cpc"], "referrer": "https://www.google.com/"},
        "yahoo": {"mediums": ["organic", "cpc"], "referrer": "https://www.yahoo.co.jp/"},
        "bing": {"mediums": ["organic", "cpc"], "referrer": "https://www.bing.com/"},
        "facebook": {"mediums": ["social", "paidsocial", "referral"], "referrer": "https://www.facebook.com/"},
        "instagram": {"mediums": ["social", "paidsocial"], "referrer": "https://www.instagram.com/"},
        "twitter": {"mediums": ["social", "paidsocial"], "referrer": "https://t.co/"},
        "youtube": {"mediums": ["paidvideo", "referral"], "referrer": "https://www.youtube.com/"},
        "smartnews": {"mediums": ["display", "referral"], "referrer": "https://www.smartnews.com/"},
        "line": {"mediums": ["social", "paidsocial"], "referrer": "https://line.me/"},
        "direct": {"mediums": ["(none)"], "referrer": None}
    }
    utm_campaigns = ["spring_sale", "summer_campaign", "brand_awareness", None]
    ab_variants = ["A", "B"]
    ab_test_targets = ['hero_image', 'cta_button', 'headline', 'layout', None]
    
    data = []
    
    # ユーザーIDプール
    total_sessions_approx = config['num_sessions_per_day_range'][1] * num_days
    user_id_pool = [f"user_{i:06d}" for i in range(int(total_sessions_approx / 1.5))] # リピート率考慮

    # Crash Day for Advanced Mode (e.g., 7 days ago)
    crash_day_index = num_days - 7

    for i in range(num_days):
        current_date = start_date + timedelta(days=i)
        weekday_name = current_date.strftime('%a')
        
        # Difficulty Logic: Adjust CVR and Volatility
        daily_cvr_multiplier = 1.0
        
        if difficulty == '初級（穏やかな波）':
            # Stable, slight random fluctuation (+- 10%)
            daily_cvr_multiplier = random.uniform(0.9, 1.1)
        
        elif difficulty == '中級（乱高下）':
            # High volatility (+- 40%), maybe some sine wave
            import math
            wave = math.sin(i / 2.0) * 0.2 # Sine wave
            noise = random.uniform(-0.3, 0.3) # Large noise
            daily_cvr_multiplier = 1.0 + wave + noise
            daily_cvr_multiplier = max(0.2, daily_cvr_multiplier) # Prevent negative
            
        elif difficulty == '上級（急降下）':
            # Stable then Crash
            if i >= crash_day_index:
                daily_cvr_multiplier = 0.4 # Drops to 40% performance
            else:
                daily_cvr_multiplier = random.uniform(0.9, 1.1)
        
        # Apply multiplier to base CVR for this day
        base_cvr = base_cvr_original * daily_cvr_multiplier
        
        # セッション数決定
        weekday_factor = config['weekday_seasonality'].get(weekday_name[:3], 1.0)
        num_sessions_today = int(random.uniform(*config['num_sessions_per_day_range']) * weekday_factor)
        
        # Apply difficulty multiplier to sessions too (optional, but realistic)
        # For crash mode, maybe traffic stays same but CVR drops? Or both drop?
        # Let's keep traffic mostly independent for now, but maybe slight drop in crash
        if difficulty == '上級（急降下）' and i >= crash_day_index:
             num_sessions_today = int(num_sessions_today * 0.8) # Traffic also dips slightly

        # ユーザー選択
        daily_user_ids = random.choices(user_id_pool, k=num_sessions_today) # 重複ありで選択（同日リピート）

        for user_pseudo_id in daily_user_ids:
            ga_session_id = random.randint(1000000000, 9999999999)
            ga_session_number = random.randint(1, 10)
            session_id = f"{user_pseudo_id}-{ga_session_id}"
            
            # 時間帯
            hour_of_day = random.choices(list(range(24)), weights=[config['hour_seasonality'].get(h, 1.0) for h in range(24)])[0]
            session_start_time = datetime.combine(current_date, time(hour_of_day, random.randint(0, 59), random.randint(0, 59)))

            # デバイス・チャネル
            device_type = random.choices(config['device_dist'], weights=config['device_weights'])[0]
            channel = random.choices(config['channel_dist'], weights=config['channel_weights'])[0]
            
            # UTMパラメータ決定
            utm_source, utm_medium, page_referrer = "(direct)", "(none)", None
            if channel != 'Direct':
                src_key = random.choice(list(traffic_sources.keys()))
                if src_key != 'direct':
                    utm_source = src_key
                    utm_medium = random.choice(traffic_sources[src_key]['mediums'])
                    page_referrer = traffic_sources[src_key]['referrer']

            utm_campaign = random.choice(utm_campaigns)
            utm_content = f"ad_{random.randint(1, 5)}" if utm_medium in ['cpc', 'paidsocial', 'display'] else None

            # A/Bテスト
            session_variant = random.choice(ab_variants)
            ab_test_target = random.choice(ab_test_targets)
            
            # ページ数決定
            session_num_pages = config['num_pages_dist']()
            
            # --- CVR事前判定 (ここが重要) ---
            # ユーザー属性によるCVR補正
            session_cvr_prob = base_cvr
            session_cvr_prob *= config['device_coeff'][device_type]['cvr']
            session_cvr_prob *= config['channel_coeff'].get(channel, {'cvr': 1.0})['cvr']
            
            is_converting = random.random() < session_cvr_prob
            
            # --- ページ遷移シミュレーション ---
            max_page_reached = 1
            
            if is_converting:
                # CVするユーザーは必ず最後まで（またはフォームまで）行く
                max_page_reached = session_num_pages
            else:
                # CVしないユーザーの離脱ロジック
                # FV離脱判定
                if random.random() < config['fv_exit_rate']:
                    max_page_reached = 1
                else:
                    # 2ページ目以降の遷移
                    for p in range(2, session_num_pages + 1):
                        # ボトルネック判定
                        bottleneck_drop_prob = config.get('bottleneck_pages', {}).get(p - 1, 0.0)
                        
                        # 遷移確率 (基本遷移率 - ボトルネック離脱率)
                        p_trans = norm.rvs(loc=config['transition_mean'], scale=config['transition_sd'])
                        p_trans -= bottleneck_drop_prob # ボトルネックがあれば遷移率ダウン
                        p_trans = np.clip(p_trans, 0.05, 0.99)
                        
                        if random.random() > p_trans:
                            max_page_reached = p - 1
                            break
                        max_page_reached = p

            # --- イベント生成 ---
            session_total_duration_ms = 0
            current_page_events = []
            
            for page_num in range(1, max_page_reached + 1):
                page_location = f"{lp_url_base}#page-{page_num}"
                page_path = f"/tst08/tst08.html#page-{page_num}"
                event_timestamp = session_start_time + timedelta(milliseconds=session_total_duration_ms)
                
                # Load Time
                load_time_ms = gamma.rvs(a=config['load_time_k'], scale=config['load_time_theta_ms'] * config['device_coeff'][device_type]['load'])
                load_time_ms = max(100, load_time_ms)

                # Stay Time
                stay_mu = config['stay_time_mu_base']
                
                # CVするユーザーは滞在時間が長い (1.5倍〜2.0倍)
                if is_converting:
                    stay_mu += 0.5 # log scaleでの加算は乗算効果 (e^0.5 ≈ 1.65倍)
                
                stay_scale_factor = config['device_coeff'][device_type]['stay'] * config['channel_coeff'].get(channel, {'stay': 1.0})['stay']
                scale_ms = np.exp(stay_mu) * stay_scale_factor * 1000
                
                stay_ms = lognorm.rvs(s=config['stay_time_sigma'], scale=scale_ms)
                stay_ms = max(1000, stay_ms) # 最低1秒
                
                # ページ種別補正
                if page_num in [1, 8]: stay_ms *= 1.5 

                # 逆行・スクロール
                direction = 'forward'
                if page_num > 1 and random.random() < config['backflow_base']:
                    direction = 'backward'
                    stay_ms *= (1 + config['backflow_stay_bonus'])
                
                # スクロール率: 長くいるほど、またCVする人は深く読む
                base_scroll = random.uniform(0.2, 0.8)
                if is_converting:
                    base_scroll = random.uniform(0.8, 1.0) # CVする人はほぼ読み切る
                
                scroll_pct = min(1.0, base_scroll + (stay_ms / 20000) * 0.5)

                # Page View Event
                event_data = {
                    "event_date": current_date.date(),
                    "event_timestamp": event_timestamp,
                    "event_timestamp_jst": event_timestamp + timedelta(hours=9),
                    "event_name": 'session_start' if page_num == 1 else 'page_view',
                    "user_pseudo_id": user_pseudo_id,
                    "ga_session_id": ga_session_id,
                    "ga_session_number": ga_session_number,
                    "session_id": session_id,
                    "page_location": page_location,
                    "page_referrer": page_referrer if page_num == 1 else None,
                    "page_path": page_path,
                    "page_num_dom": page_num,
                    "stay_ms": int(stay_ms),
                    "load_time_ms": int(load_time_ms),
                    "max_page_reached": max_page_reached,
                    "total_pages": session_num_pages,
                    "scroll_pct": scroll_pct,
                    "utm_source": utm_source,
                    "utm_medium": utm_medium,
                    "utm_campaign": utm_campaign,
                    "utm_content": utm_content,
                    "device_type": device_type,
                    "direction": direction,
                    "ab_variant": session_variant,
                    "ab_test_target": ab_test_target,
                    # Initialize columns
                    "cv_type": None,
                    "cv_value": None,
                    "value": None,
                    "form_page_number": None,
                    "form_duration_ms": None,
                    "form_direction": None,
                    "click_x_rel": None,
                    "click_y_rel": None,
                    "elem_tag": None,
                    "elem_id": None,
                    "elem_classes": None,
                    "link_url": None,
                    "video_src": None,
                }
                current_page_events.append(event_data)
                
                # Click Event (CVする人はCTAクリック率も高い)
                click_prob = config['cta_click_rate_base']
                if is_converting and page_num == max_page_reached:
                    click_prob = 0.9 # CVするなら最後はほぼクリックする
                
                if random.random() < click_prob:
                    click_event = event_data.copy()
                    click_event['event_name'] = 'click'
                    click_event['event_timestamp'] += timedelta(milliseconds=random.randint(100, int(stay_ms)))
                    click_event['elem_tag'] = 'button'
                    click_event['elem_id'] = 'cta_button'
                    current_page_events.append(click_event)

                session_total_duration_ms += int(stay_ms + load_time_ms)

            # --- CVイベント生成 ---
            if is_converting:
                cv_event = current_page_events[-1].copy()
                cv_event['event_name'] = 'conversion'
                cv_event['event_timestamp'] += timedelta(milliseconds=1000)
                cv_event['cv_type'] = random.choice(["primary", "micro"])
                cv_event['cv_value'] = random.randint(1000, 10000)
                cv_event['value'] = cv_event['cv_value']
                current_page_events.append(cv_event)
            
            data.extend(current_page_events)

    # DataFrameに変換
    df = pd.DataFrame(data)
    
    # --- 意図的なアラートデータを注入 ---
    if num_days >= 10 and not df.empty:
        if scenario == '不調（モバイル課題）':
            # 特定の日にモバイルのCVRが0になるなどの極端な異常値を混ぜる
            pass

    # 日付でソート
    if not df.empty:
        df = df.sort_values("event_timestamp").reset_index(drop=True)
        # total_duration_ms 補完
        df['total_duration_ms'] = df.groupby('session_id')['event_timestamp'].transform(lambda x: (x.max() - x.min()).total_seconds() * 1000)

    return df

if __name__ == "__main__":
    # テスト実行
    df = generate_dummy_data(scenario='標準（ベースライン）', num_days=3)
    print(f"Generated {len(df)} events.")
    if not df.empty:
        print(df[['session_id', 'event_name', 'stay_ms', 'max_page_reached', 'cv_value']].head(20).to_markdown())
        # CVしたセッションの平均滞在時間と、CVしてないセッションの平均滞在時間を比較
        cv_sessions = df[df['event_name'] == 'conversion']['session_id'].unique()
        avg_stay_cv = df[df['session_id'].isin(cv_sessions)]['stay_ms'].mean()
        avg_stay_no_cv = df[~df['session_id'].isin(cv_sessions)]['stay_ms'].mean()
        print(f"\nAvg Stay Time (CV): {avg_stay_cv:.2f} ms")
        print(f"Avg Stay Time (No CV): {avg_stay_no_cv:.2f} ms")
