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
SCENARIO_CONFIGS = {
    '好調': {
        'num_sessions_per_day_range': (450, 550), # Target 15000 sessions/month (500/day)
        'fv_exit_rate': 0.20, # 1P目離脱率 (20%)
        'transition_mean': 0.97, # ページ間遷移確率の平均 (離脱率3%)
        'transition_sd': 0.02, # 遷移確率のばらつき
        'cta_click_rate_base': 0.215, # CTAクリック率の基本値 (18-25%)
        'base_session_cvr': 0.08, # セッション全体のベースCVR (8%)
        'load_time_k': 4, # ガンマ分布の形状パラメータ
        'load_time_theta_ms': 500, # ガンマ分布の尺度パラメータ (小さいほど速い)
        'stay_time_mu_base': 4.5, # 対数正規分布のmu (大きいほど滞在長い)
        'stay_time_sigma': 0.5,
        'backflow_base': 0.05, # 逆行発生確率
        'device_dist': ['mobile', 'desktop', 'tablet'],
        'device_weights': [0.75, 0.22, 0.03],
        'channel_dist': ['Organic Search', 'Paid Search', 'Paid Social', 'Direct', 'Referral', 'Other'],
        'channel_weights': [0.40, 0.25, 0.15, 0.12, 0.05, 0.03],
        'channel_coeff': { # CVR, Stay Time, Load Time への係数
            'Paid Search': {'cvr': 1.15, 'stay': 1.05},
            'Paid Social': {'cvr': 0.9, 'stay': 0.95},
            'Organic Search': {'cvr': 1.1, 'stay': 1.1},
            'Direct': {'cvr': 0.9 + random.uniform(-0.1, 0.1), 'stay': 1.0}, # ばらつき大
            'Referral': {'cvr': 1.05, 'stay': 1.0},
            'Other': {'cvr': 0.8, 'stay': 0.9}
        },
        'device_coeff': { # CVR, Stay Time, Load Time への係数
            'mobile': {'cvr': 0.95, 'stay': 0.95, 'load': 1.12},
            'desktop': {'cvr': 1.05, 'stay': 1.05, 'load': 0.9},
            'tablet': {'cvr': 1.0, 'stay': 1.0, 'load': 1.0}
        },
        'hour_seasonality': {12: 1.15, 20: 1.2, 21: 1.2, 22: 1.15}, # セッション数とCVRへの補正
        'weekday_seasonality': {'Mon': 1.0, 'Tue': 1.02, 'Wed': 1.03, 'Thu': 1.02, 'Fri': 0.98, 'Sat': 0.95, 'Sun': 0.95},
        'load_time_impact_exit_rate': 0.6 / 300, # +0.6pt exit for +300ms load
        'load_time_impact_stay_ms': -0.03 / 300, # -3% stay for +300ms load
        'backflow_stay_bonus': 0.15, # 逆行発生時の滞在時間ボーナス
        'cta_scroll_penalty': 0.20, # CTAが画面1/3スクロール後に出る設計はクリック率 -20%
        'cta_video_bonus': 0.30, # 動画視聴完了でCTAクリック率 +30%
        'fb_depth_bonus': 0.05, # 到達ページ深いほどFBクリック率が増える
        'exit_pop_bounce_bonus': 0.05, # 直帰・逆行強い層ほどExitPOPクリック率高め
        'info_jump_backflow_bonus': 0.50, # 情報ジャンプ区間での逆行確率ボーナス
    },
    '普通': {
        'num_sessions_per_day_range': (300, 400), # Target 10500 sessions/month (350/day)
        'fv_exit_rate': 0.35, # 1P目離脱率 (35%)
        'transition_mean': 0.94, # ページ間遷移確率の平均 (離脱率6%)
        'transition_sd': 0.03,
        'cta_click_rate_base': 0.14, # CTAクリック率の基本値 (10-18%)
        'base_session_cvr': 0.04, # セッション全体のベースCVR (4%)
        'load_time_k': 4,
        'load_time_theta_ms': 600,
        'stay_time_mu_base': 4.2,
        'stay_time_sigma': 0.6,
        'backflow_base': 0.08,
        'device_dist': ['mobile', 'desktop', 'tablet'],
        'device_weights': [0.70, 0.25, 0.05],
        'channel_dist': ['Organic Search', 'Paid Search', 'Paid Social', 'Direct', 'Referral', 'Other'],
        'channel_weights': [0.38, 0.25, 0.15, 0.12, 0.07, 0.03],
        'channel_coeff': {
            'Paid Search': {'cvr': 1.15, 'stay': 1.05},
            'Paid Social': {'cvr': 0.9, 'stay': 0.95},
            'Organic Search': {'cvr': 1.1, 'stay': 1.1},
            'Direct': {'cvr': 0.9 + random.uniform(-0.1, 0.1), 'stay': 1.0},
            'Referral': {'cvr': 1.05, 'stay': 1.0},
            'Other': {'cvr': 0.8, 'stay': 0.9}
        },
        'device_coeff': {
            'mobile': {'cvr': 0.95, 'stay': 0.95, 'load': 1.12},
            'desktop': {'cvr': 1.05, 'stay': 1.05, 'load': 0.9},
            'tablet': {'cvr': 1.0, 'stay': 1.0, 'load': 1.0}
        },
        'hour_seasonality': {12: 1.15, 20: 1.2, 21: 1.2, 22: 1.15},
        'weekday_seasonality': {'Mon': 1.0, 'Tue': 1.02, 'Wed': 1.03, 'Thu': 1.02, 'Fri': 0.98, 'Sat': 0.95, 'Sun': 0.95},
        'load_time_impact_exit_rate': 0.6 / 300,
        'load_time_impact_stay_ms': -0.03 / 300,
        'backflow_stay_bonus': 0.15,
        'cta_scroll_penalty': 0.20,
        'cta_video_bonus': 0.30,
        'fb_depth_bonus': 0.05,
        'exit_pop_bounce_bonus': 0.05,
        'info_jump_backflow_bonus': 0.50,
    },
    '不調': {
        'num_sessions_per_day_range': (180, 220), # Target 6000 sessions/month (200/day)
        'fv_exit_rate': 0.50, # 1P目離脱率 (50%)
        'transition_mean': 0.90, # ページ間遷移確率の平均 (離脱率10%)
        'transition_sd': 0.04,
        'cta_click_rate_base': 0.075, # CTAクリック率の基本値 (5-10%)
        'base_session_cvr': 0.015, # セッション全体のベースCVR (1.5%)
        'load_time_k': 4,
        'load_time_theta_ms': 700,
        'stay_time_mu_base': 3.8,
        'stay_time_sigma': 0.7,
        'backflow_base': 0.12,
        'device_dist': ['mobile', 'desktop', 'tablet'],
        'device_weights': [0.65, 0.30, 0.05],
        'channel_dist': ['Organic Search', 'Paid Search', 'Paid Social', 'Direct', 'Referral', 'Other'],
        'channel_weights': [0.35, 0.20, 0.10, 0.10, 0.15, 0.10],
        'channel_coeff': {
            'Paid Search': {'cvr': 1.15, 'stay': 1.05},
            'Paid Social': {'cvr': 0.9, 'stay': 0.95},
            'Organic Search': {'cvr': 1.1, 'stay': 1.1},
            'Direct': {'cvr': 0.9 + random.uniform(-0.1, 0.1), 'stay': 1.0},
            'Referral': {'cvr': 1.05, 'stay': 1.0},
            'Other': {'cvr': 0.8, 'stay': 0.9}
        },
        'device_coeff': {
            'mobile': {'cvr': 0.95, 'stay': 0.95, 'load': 1.12},
            'desktop': {'cvr': 1.05, 'stay': 1.05, 'load': 0.9},
            'tablet': {'cvr': 1.0, 'stay': 1.0, 'load': 1.0}
        },
        'hour_seasonality': {12: 1.15, 20: 1.2, 21: 1.2, 22: 1.15},
        'weekday_seasonality': {'Mon': 1.0, 'Tue': 1.02, 'Wed': 1.03, 'Thu': 1.02, 'Fri': 0.98, 'Sat': 0.95, 'Sun': 0.95},
        'load_time_impact_exit_rate': 0.6 / 300,
        'load_time_impact_stay_ms': -0.03 / 300,
        'backflow_stay_bonus': 0.15,
        'cta_scroll_penalty': 0.20,
        'cta_video_bonus': 0.30,
        'fb_depth_bonus': 0.05,
        'exit_pop_bounce_bonus': 0.05,
        'info_jump_backflow_bonus': 0.50,
    }
}

def generate_dummy_data(scenario: str = '普通', num_days: int = 30, num_pages: int = 10):
    """
    リアルなスワイプLPイベントデータを生成
    
    Args:
        scenario: '好調', '普通', '不調' のいずれか
        num_days: 過去何日分のデータを生成するか
        num_pages: LPの総ページ数
    
    Returns:
        pd.DataFrame: ダミーデータ
    """
    config = SCENARIO_CONFIGS.get(scenario, SCENARIO_CONFIGS['普通']).copy() # デフォルトは「普通」

    # 基準日時
    end_date = datetime.now()
    start_date = end_date - timedelta(days=num_days)
    
    # LP URL
    lp_url_base = "https://shungene.lm-c.jp/tst08/tst08.html"
    
    # イベント名
    event_names = [
        "page_view",
        "swipe_page",
        "click",
        "form_start",
        "form_submit",
        "form_progress",
        "scroll",
        "video_play",
        "conversion",
        "session_start",
    ]
    
    device_types = config['device_dist']
    device_weights = config['device_weights']
    
    # UTMパラメータ
    traffic_sources = {
        "google": {"mediums": ["organic", "cpc"], "referrer": "https://www.google.com/"},
        "yahoo": {"mediums": ["organic", "cpc"], "referrer": "https://www.yahoo.co.jp/"},
        "bing": {"mediums": ["organic", "cpc"], "referrer": "https://www.bing.com/"},
        "facebook": {"mediums": ["social", "paidsocial", "referral", "display"], "referrer": "https://www.facebook.com/"},
        "instagram": {"mediums": ["social", "paidsocial", "referral", "display"], "referrer": "https://www.instagram.com/"},
        "twitter": {"mediums": ["social", "paidsocial", "referral", "display"], "referrer": "https://t.co/"},
        "youtube": {"mediums": ["paidvideo", "referral"], "referrer": "https://www.youtube.com/"},
        "smartnews": {"mediums": ["display", "referral"], "referrer": "https://www.smartnews.com/"},
        "line": {"mediums": ["social", "paidsocial", "display"], "referrer": "https://line.me/"},
        "news-app": {"mediums": ["display", "referral"], "referrer": "android-app://com.example.news"},
        "direct": {"mediums": ["(none)"], "referrer": None}
    }
    source_keys = list(traffic_sources.keys())
    utm_campaigns = ["spring_sale", "summer_campaign", "brand_awareness", None]
    
    # A/Bテストバリアント
    ab_variants = ["A", "B"]
    
    # A/Bテストターゲット
    ab_test_targets = ['hero_image', 'cta_button', 'headline', 'layout', 'copy', 'form', 'video', None]
    
    # ナビゲーション方法
    navigation_methods = ["swipe", "click", "scroll", "button"]
    
    # 方向
    directions = ["forward", "backward"]

    # ページ種別による滞在時間補正 (例: 動画ページ、比較表ページなど)
    page_type_stay_bonus = {1: 1.2, 8: 1.2, 5: 1.1} # P1, P8は動画、P5は比較表と仮定
    
    # A/Bテストごとのp値を保持する辞書
    test_p_values = {}

    # データ生成
    data = []
    
    # ユーザーIDプール (セッション数の約1/5)
    user_id_pool = [f"user_{i:06d}" for i in range(int(config['num_sessions_per_day_range'][1] * num_days / 5))]

    # 日付ループ
    for i in range(num_days):
        current_date = start_date + timedelta(days=i)
        weekday_name = current_date.strftime('%a') # Mon, Tue, ...

        # 曜日によるセッション数補正
        weekday_factor = config['weekday_seasonality'].get(weekday_name[:3], 1.0)
        
        # その日のセッション数を決定
        num_sessions_today = int(random.uniform(*config['num_sessions_per_day_range']) * weekday_factor)
        
        # その日のユーザーIDをランダムに選択
        daily_user_ids = random.sample(user_id_pool, min(num_sessions_today, len(user_id_pool)))

        for session_idx in range(num_sessions_today):
            user_pseudo_id = random.choice(daily_user_ids)
            ga_session_id = random.randint(1000000000, 9999999999) # GA4 style session ID
            ga_session_number = random.randint(1, 5) # 1-5回のリピートセッションを想定
            session_id = f"{user_pseudo_id}-{ga_session_id}"
            
            # セッション開始時刻 (時間帯の季節性を考慮)
            hour_of_day = random.choices(
                list(range(24)), 
                weights=[config['hour_seasonality'].get(h, 1.0) for h in range(24)]
            )[0]
            session_start_time = datetime.combine(current_date, time(hour_of_day, random.randint(0, 59), random.randint(0, 59)))

            # デバイスとチャネルの決定
            device_type = random.choices(config['device_dist'], weights=config['device_weights'])[0]
            channel = random.choices(config['channel_dist'], weights=config['channel_weights'])[0]
            
            # UTMパラメータの決定
            utm_source = None
            utm_medium = None
            page_referrer = None
            utm_campaign = random.choice(utm_campaigns)
            utm_content = random.choice([f"ad_{k}" for k in range(1, 6)] + [None])

            if channel == 'Direct':
                utm_source = '(direct)'
                utm_medium = '(none)'
                page_referrer = None
            else:
                # channelからutm_sourceとutm_mediumを逆引き的に決定
                possible_sources = [s for s, info in traffic_sources.items() if channel in info['mediums'] or (channel == 'Organic Search' and 'organic' in info['mediums']) or (channel == 'Paid Search' and 'cpc' in info['mediums'])]
                if not possible_sources: # Fallback if no direct match
                    possible_sources = list(traffic_sources.keys())
                
                utm_source_key = random.choice(possible_sources)
                source_info = traffic_sources[utm_source_key]
                utm_source = utm_source_key
                
                if channel == 'Organic Search': utm_medium = 'organic'
                elif channel == 'Paid Search': utm_medium = 'cpc'
                elif channel == 'Organic Social': utm_medium = random.choice(['social', 'referral'])
                elif channel == 'Paid Social': utm_medium = 'paidsocial'
                elif channel == 'Paid Video': utm_medium = 'paidvideo'
                elif channel == 'Display': utm_medium = 'display'
                else: utm_medium = random.choice(source_info["mediums"])

                page_referrer = source_info["referrer"]
                if random.random() < 0.1: # 10%の確率でリファラーが取れないケース
                    page_referrer = None

            # A/Bテストバリアント
            session_variant = random.choice(ab_variants)
            ab_test_target_for_session = random.choice(ab_test_targets)
            ab_test_type = 'layout' # Simplified for now
            if ab_test_target_for_session:
                ab_test_type = random.choice(['presence', 'creative', 'layout'])
            
            # p値の生成ロジック（ab_test_targetが存在する場合に限定）
            p_value = None
            if ab_test_target_for_session:
                test_key = (ab_test_target_for_session, session_variant)
                if test_key not in test_p_values:
                    p_value_options = [
                        random.uniform(0.005, 0.02),  # ★★★
                        random.uniform(0.04, 0.06),   # ★★
                        random.uniform(0.09, 0.11),   # ★
                        random.uniform(0.1, 1.0)      # -
                    ]
                    test_p_values[test_key] = 1.0 if session_variant == 'A' else random.choices(p_value_options, weights=[0.1, 0.2, 0.2, 0.5])[0]
                p_value = test_p_values[test_key]

            # --- ページ進行ファネルのシミュレーション ---
            max_page_reached = 1
            current_page_events = []
            
            # FV離脱
            if random.random() < config['fv_exit_rate']:
                max_page_reached = 1
            else:
                max_page_reached = 2
                for page_num_iter in range(2, num_pages + 1):
                    # ページ n→n+1 の遷移確率 p_n
                    p_n = norm.rvs(loc=config['transition_mean'], scale=config['transition_sd'])
                    p_n = np.clip(p_n, 0.6, 0.95) # 確率を0.6-0.95にクリップ
                    
                    if random.random() > p_n: # 離脱
                        max_page_reached = page_num_iter - 1
                        break
                    max_page_reached = page_num_iter
            
            # --- 各ページイベントの生成 ---
            session_total_duration_ms = 0
            last_stay_ms = 0
            
            for page_num_dom in range(1, max_page_reached + 1):
                page_location = f"{lp_url_base}#page-{page_num_dom}"
                page_path = f"/tst08/tst08.html#page-{page_num_dom}"
                event_timestamp = session_start_time + timedelta(milliseconds=session_total_duration_ms)
                
                # Load Time (ガンマ分布 + デバイス係数)
                load_time_ms = gamma.rvs(a=config['load_time_k'], scale=config['load_time_theta_ms'] * config['device_coeff'][device_type]['load'])
                load_time_ms = np.clip(load_time_ms, 200, 10000) # 200ms - 10s

                # Stay Time (対数正規分布 + デバイス係数 + ページ種別ボーナス + 読込時間影響 + 逆行ボーナス)
                stay_mu = config['stay_time_mu_base'] * config['device_coeff'][device_type]['stay']
                
                stay_mu += (load_time_ms - config['load_time_theta_ms']) * config['load_time_impact_stay_ms'] # 読込時間が長いとstay_muが下がる

                stay_ms = np.exp(lognorm.rvs(s=config['stay_time_sigma'], loc=0, scale=np.exp(stay_mu)))
                # ページ種別ボーナスを乗算で適用
                stay_ms *= page_type_stay_bonus.get(page_num_dom, 1.0)
                stay_ms = np.clip(stay_ms, 1000, 300000) # 1秒 - 5分
                
                
                if page_num_dom > 1 and len(current_page_events) > 0 and current_page_events[-1]['direction'] == 'backward':
                    stay_ms *= (1 + config['backflow_stay_bonus'])

                # フォーム関連のダミー列を初期化
                form_page_number = None
                form_duration_ms = None
                form_direction = None
                # Direction (逆行率)
                direction = 'forward'
                backflow_prob = config['backflow_base']
                if page_num_dom in [6, 9]: # P5からP6、P8からP9への遷移で逆行が増える
                    backflow_prob *= (1 + config['info_jump_backflow_bonus'])
                
                if random.random() < backflow_prob:
                    direction = 'backward'
                
                event_name = 'page_view'
                if page_num_dom == 1: event_name = 'session_start'
                if page_num_dom in [1, 8] and random.random() < 0.3:
                    event_name = 'video_play'
                
                # Always add a page_view event for this page
                click_x_rel, click_y_rel, elem_tag, elem_id, elem_classes, link_url = None, None, None, None, None, None

                video_src = None
                if page_num_dom in [1, 8]:
                    video_src = f"https://example.com/video{page_num_dom}.mp4"

                scroll_pct = np.clip(random.uniform(0.1, 0.5) + (page_num_dom / num_pages) * 0.4, 0.1, 1.0)

                prev_page_path = None
                if page_num_dom > 1:
                    prev_page_path = f"{lp_url_base}#page-{page_num_dom - 1}"
                
                current_page_events.append({
                    "event_date": current_date.date(),
                    "event_timestamp": event_timestamp,
                    "event_timestamp_jst": event_timestamp + timedelta(hours=9),
                    "event_name": event_name,
                    "user_pseudo_id": user_pseudo_id,
                    "ga_session_id": ga_session_id,
                    "ga_session_number": ga_session_number,
                    "session_id": session_id,
                    "page_location": page_location,
                    "page_referrer": page_referrer,
                    "page_path": page_path,
                    "prev_page_path": prev_page_path,
                    "page_num_dom": page_num_dom,
                    "original_page_num": page_num_dom,
                    "stay_ms": int(stay_ms),
                    "total_duration_ms": 0,
                    "load_time_ms": int(load_time_ms),
                    "max_page_reached": max_page_reached,
                    "completion_rate": max_page_reached / num_pages,
                    "total_pages": num_pages,
                    "click_x_rel": click_x_rel,
                    "click_y_rel": click_y_rel,
                    "elem_tag": elem_tag,
                    "elem_id": elem_id,
                    "elem_classes": elem_classes,
                    "scroll_pct": scroll_pct,
                    "utm_source": utm_source,
                    "utm_medium": utm_medium,
                    "utm_campaign": utm_campaign,
                    "utm_content": utm_content,
                    "device_type": device_type,
                    "direction": direction,
                    "navigation_method": random.choice(navigation_methods),
                    "link_url": link_url,
                    "video_src": video_src,
                    "session_variant": session_variant,
                    "presence_test_variant": session_variant if ab_test_type == 'presence' else None,
                    "creative_test_variant": session_variant if ab_test_type == 'creative' else None,
                    "ab_variant": session_variant,
                    "ab_test_target": ab_test_target_for_session,
                    "ab_test_type": ab_test_type,
                    "cv_type": None,
                    "p_value": p_value,
                    "cv_value": None,
                    "value": None,
                    "form_page_number": form_page_number,
                    "form_duration_ms": form_duration_ms,
                    "form_direction": form_direction,
                })

                # クリックイベントを別イベントとして生成
                if random.random() < 0.3: # 30%の確率で何らかのクリックが発生
                    click_event = current_page_events[-1].copy() # page_viewイベントをベースに作成
                    click_event['event_name'] = 'click'
                    click_event['event_timestamp'] = event_timestamp + timedelta(milliseconds=random.randint(100, int(stay_ms/2))) # 滞在時間内にクリック
                    click_event['event_timestamp_jst'] = click_event['event_timestamp'] + timedelta(hours=9)
                    click_event['click_x_rel'] = random.uniform(0.1, 0.9)
                    click_event['click_y_rel'] = random.uniform(0.1, 0.9)
                    click_event['elem_tag'] = random.choice(["button", "a", "div"])

                    click_type_chosen = False
                    # CTAクリック
                    cta_click_prob = config['cta_click_rate_base']
                    if page_num_dom >= num_pages: cta_click_prob *= 1.5
                    if random.random() < cta_click_prob:
                        click_event['elem_classes'] = 'cta btn-primary'
                        click_event['elem_id'] = 'cta-button'
                        click_event['link_url'] = "https://example.com/thank-you"
                        click_type_chosen = True
                    
                    # フローティングバナークリック
                    if not click_type_chosen:
                        fb_click_prob = 0.01 + (page_num_dom / num_pages) * config['fb_depth_bonus']
                        if random.random() < fb_click_prob:
                            click_event['elem_classes'] = 'floating'
                            click_event['elem_id'] = 'floating-banner'
                            click_event['link_url'] = "https://example.com/special-offer"
                            click_type_chosen = True

                    # 離脱POPクリック
                    if not click_type_chosen:
                        exit_pop_click_prob = 0.01
                        if max_page_reached == 1 or direction == 'backward':
                            exit_pop_click_prob += config['exit_pop_bounce_bonus']
                        if random.random() < exit_pop_click_prob:
                            click_event['elem_classes'] = 'exit'
                            click_event['elem_id'] = 'exit-popup'
                            click_event['link_url'] = "https://example.com/exit-offer"
                            click_type_chosen = True
                    
                    if click_type_chosen:
                        current_page_events.append(click_event)


                session_total_duration_ms += int(stay_ms) + int(load_time_ms) + random.randint(100, 500)
                last_stay_ms = int(stay_ms)
            
            for event in current_page_events:
                event['total_duration_ms'] = session_total_duration_ms
            
            is_conversion = False
            
            # --- 新しいCVR計算ロジック ---
            # 1. ベースとなるセッションCVRを設定
            session_cvr_prob = config['base_session_cvr']
            
            # 2. 到達ページ数に応じてCVRをブースト（全ページ到達で1.5倍）
            page_reach_factor = 1.0 + (max_page_reached / num_pages) * 0.5 
            session_cvr_prob *= page_reach_factor

            # 3. チャネル、デバイス、時間帯の係数を適用
            session_cvr_prob *= config['channel_coeff'].get(channel, {}).get('cvr', 1.0)
            session_cvr_prob *= config['device_coeff'].get(device_type, {}).get('cvr', 1.0)
            session_cvr_prob *= config['hour_seasonality'].get(session_start_time.hour, 1.0)
            
            if random.random() < session_cvr_prob:
                is_conversion = True

            if is_conversion and current_page_events:
                cv_event = current_page_events[-1].copy()
                cv_event['event_name'] = 'conversion'
                cv_event['event_timestamp'] = event_timestamp + timedelta(milliseconds=random.randint(1000, 5000))
                cv_event['event_timestamp_jst'] = cv_event['event_timestamp'] + timedelta(hours=9)
                cv_event['cv_type'] = random.choice(["primary", "micro"])
                cv_event['cv_value'] = random.uniform(1000, 50000)
                cv_event['value'] = cv_event['cv_value']
                current_page_events.append(cv_event)
            
            # --- フォームイベントの生成 ---
            if is_conversion and random.random() < 0.8: # CVしたセッションの80%がフォーム経由と仮定
                form_start_event = current_page_events[-1].copy()
                form_start_event['event_name'] = 'form_start'
                form_start_event['event_timestamp'] += timedelta(milliseconds=100)
                form_start_event['event_timestamp_jst'] = form_start_event['event_timestamp'] + timedelta(hours=9)
                current_page_events.append(form_start_event)

                num_form_pages = 5
                total_form_duration = 0
                for form_page in range(1, num_form_pages + 1):
                    form_progress_event = form_start_event.copy()
                    page_duration = random.randint(5000, 20000)
                    total_form_duration += page_duration
                    form_progress_event['event_name'] = 'form_progress'
                    form_progress_event['event_timestamp'] += timedelta(milliseconds=total_form_duration)
                    form_progress_event['event_timestamp_jst'] = form_progress_event['event_timestamp'] + timedelta(hours=9)
                    form_progress_event['form_page_number'] = form_page
                    form_progress_event['form_duration_ms'] = page_duration
                    form_progress_event['form_direction'] = 'forward' if random.random() > 0.1 else 'backward'
                    current_page_events.append(form_progress_event)
                
                # フォーム送信イベントはCVイベントとほぼ同じタイミング

            data.extend(current_page_events)

    # DataFrameに変換
    df = pd.DataFrame(data)
    
    # --- 意図的なアラートデータを注入 ---
    # 期間内にランダムな3日を「異常日」として設定
    if num_days >= 10 and not df.empty: # Ensure df is not empty before accessing columns
        if scenario == '不調':
            # 不調シナリオでは、CVR急落日とセッション急減日をそれぞれ設定
            alert_dates = random.sample(
                [d.date() for d in pd.date_range(start_date, end_date - timedelta(days=3))],
                k=min(4, num_days - 5) # 4日ほど異常日を設定
            )
            
            # 2日をCVR急落日に
            for alert_date in alert_dates[:2]:
                # CVRを意図的に0にする (CVイベントを全て削除)
                cv_indices_on_alert_date = df[(df['event_date'] == alert_date) & (df['cv_type'].notna())].index
                if not cv_indices_on_alert_date.empty:
                    df.drop(cv_indices_on_alert_date, inplace=True)
            
            # 2日をセッション急減日に
            for alert_date in alert_dates[2:]:
                session_drop_rate = random.uniform(0.6, 0.8) # 60-80%減
                sessions_on_alert_date = df[df['event_date'] == alert_date]['session_id'].unique()
                if len(sessions_on_alert_date) > 1:
                    sessions_to_drop_count = int(len(sessions_on_alert_date) * session_drop_rate)
                    sessions_to_drop_ids = random.sample(list(sessions_on_alert_date), sessions_to_drop_count)
                    df = df[~((df['event_date'] == alert_date) & (df['session_id'].isin(sessions_to_drop_ids)))]

        elif scenario == '普通':
            # 普通シナリオでは、CVRを少しだけ下げる日をランダムに1日設定
            if random.random() < 0.5: # 50%の確率でアラート日を設定
                alert_date = (start_date + timedelta(days=random.randint(3, num_days - 3))).date()
                cvr_drop_rate = 0.5 # 50%のCVを削除

                # CVRを意図的に下げる (CVイベントをランダムに削除)
                cv_indices_on_alert_date = df[(df['event_date'] == alert_date) & (df['cv_type'].notna())].index
                if len(cv_indices_on_alert_date) > 1:
                    indices_to_drop = random.sample(list(cv_indices_on_alert_date), k=int(len(cv_indices_on_alert_date) * cvr_drop_rate))
                    df.drop(indices_to_drop, inplace=True)

    # 日付でソート
    if not df.empty:
        df = df.sort_values("event_timestamp").reset_index(drop=True)
        
        # total_duration_msが0のままのイベントを修正 (CVイベントなど)
        # 各セッションの最後のイベントのtotal_duration_msをそのセッションの最大値に設定
        df['total_duration_ms'] = df.groupby('session_id')['event_timestamp'].transform(lambda x: (x.max() - x.min()).total_seconds() * 1000)
    
    return df


if __name__ == "__main__":
    # ダミーデータ生成
    df = generate_dummy_data(scenario='普通', num_days=30, num_pages=10)
    
    # CSV保存
    df.to_csv("/home/ubuntu/swipe_lp_analyzer/app/dummy_data.csv", index=False)
    
    print(f"✅ ダミーデータ生成完了: {len(df)} イベント")
    print(f"📅 期間: {df['event_date'].min()} ～ {df['event_date'].max()}")
    print(f"👥 ユーザー数: {df['user_pseudo_id'].nunique()}")
    print(f"📊 セッション数: {df['session_id'].nunique()}")
