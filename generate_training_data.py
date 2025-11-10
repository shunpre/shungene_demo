import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_training_data(start_date, end_date, scenario='普通'):
    """
    指定されたシナリオと期間に基づいてトレーニングデータを生成する関数。

    Args:
        start_date (datetime.date): データ生成の開始日。
        end_date (datetime.date): データ生成の終了日。
        scenario (str): '好調', '普通', '不調' のいずれか。

    Returns:
        pd.DataFrame: 生成されたトレーニングデータ。
    """
    # シナリオ別のパラメータ設定
    if scenario == '好調':
        cvr_params = (0.05, 0.08)
        fv_exit_prob = 0.15 # FV離脱率を低く
        bottleneck_page = None
        bottleneck_exit_prob = 0
    elif scenario == '不調':
        cvr_params = (0.005, 0.01)
        fv_exit_prob = 0.4
        bottleneck_page = 3 # 3ページ目にボトルネックを設定
        bottleneck_exit_prob = 0.6 # 離脱率を高く
    else: # 普通
        cvr_params = (0.02, 0.03)
        fv_exit_prob = 0.3
        bottleneck_page = None
        bottleneck_exit_prob = 0

    all_events = []
    total_days = (end_date - start_date).days + 1
    
    # 日付ごとにループ
    for i in range(total_days):
        current_date = start_date + timedelta(days=i)
        num_sessions = np.random.randint(300, 401)

        # 1日分のセッションを生成
        for j in range(num_sessions):
            user_pseudo_id = f"user_{np.random.randint(1000):04d}"
            ga_session_id = np.random.randint(1000000, 9999999)
            session_id = f"{user_pseudo_id}-{ga_session_id}"
            ga_session_number = np.random.randint(1, 11)

            # --- ページ到達とCVのシミュレーション ---
            # FV離脱
            if np.random.rand() < fv_exit_prob:
                max_page_reached = 1
            else:
                # ボトルネックページの離脱
                if bottleneck_page and np.random.rand() < bottleneck_exit_prob:
                    max_page_reached = bottleneck_page
                else:
                    # 通常のページ遷移（ページが進むほど離脱しやすくなる）
                    max_page_reached = 2
                    for page_num in range(2, 10):
                        # ページが進むにつれて離脱確率が上がる (例: 10% -> 15% -> ...)
                        if np.random.rand() > (0.1 + page_num * 0.02):
                            max_page_reached += 1
                        else:
                            break
            
            # コンバージョン決定
            is_cv = np.random.rand() < np.random.uniform(*cvr_params)
            cv_type = 'primary' if is_cv else None

            # --- イベントレコードの生成 ---
            page_location = np.random.choice(['https://example.com/lp/product-a', 'https://example.com/lp/product-b', 'https://example.com/lp/service-x'], p=[0.4, 0.3, 0.3])
            page_path = page_location.replace('https://example.com', '')
            
            # 参照元とチャネル
            sources = ['google', 'facebook', 'instagram', 'twitter', 'direct']
            mediums = ['organic', 'cpc', 'social', 'referral', 'none']
            utm_source = np.random.choice(sources, p=[0.4, 0.2, 0.2, 0.1, 0.1])
            utm_medium = np.random.choice(mediums, p=[0.3, 0.3, 0.2, 0.1, 0.1])
            if utm_source == 'direct': utm_medium = 'none'

            # セッション全体の情報を生成
            session_start_time = datetime.combine(current_date, datetime.min.time()) + timedelta(minutes=np.random.randint(0, 1440))
            total_duration_ms = max_page_reached * np.random.randint(15000, 30000)
            device_type = np.random.choice(['mobile', 'desktop', 'tablet'], p=[0.7, 0.25, 0.05])

            # 各ページ到達イベントを生成
            for page_num in range(1, max_page_reached + 1):
                event_timestamp = session_start_time + timedelta(milliseconds=(page_num-1) * np.random.randint(10000, 20000))
                
                # 滞在時間 (最後のページは長く、他は短く)
                if page_num == max_page_reached:
                    stay_ms = np.random.randint(5000, 120000)
                else:
                    stay_ms = np.random.randint(3000, 30000)

                event = {
                    'event_date': current_date.strftime('%Y-%m-%d'),
                    'event_timestamp': event_timestamp,
                    'event_name': 'page_view' if page_num > 1 else 'session_start',
                    'user_pseudo_id': user_pseudo_id,
                    'ga_session_id': ga_session_id,
                    'ga_session_number': ga_session_number,
                    'session_id': session_id,
                    'page_location': page_location,
                    'page_referrer': np.random.choice(['https://www.google.com/', 'https://www.facebook.com/', 'https://twitter.com/', ''], p=[0.4, 0.2, 0.1, 0.3]),
                    'page_path': page_path,
                    'prev_page_path': f"{page_path}#page-{page_num-1}" if page_num > 1 else '',
                    'page_num_dom': page_num,
                    'original_page_num': page_num,
                    'stay_ms': stay_ms,
                    'total_duration_ms': total_duration_ms,
                    'load_time_ms': np.random.randint(500, 4000),
                    'max_page_reached': max_page_reached,
                    'completion_rate': max_page_reached / 10.0,
                    'total_pages': 10,
                    'click_x_rel': np.random.rand(),
                    'click_y_rel': np.random.rand(),
                    'elem_tag': np.random.choice(['div', 'a', 'button', 'img'], p=[0.4, 0.3, 0.2, 0.1]),
                    'elem_id': '',
                    'elem_classes': np.random.choice(['cta', 'link', 'card', ''], p=[0.2, 0.3, 0.3, 0.2]),
                    'scroll_pct': np.random.rand() * 0.3, # 逆行は少なめに
                    'utm_source': utm_source,
                    'utm_medium': utm_medium,
                    'utm_campaign': np.random.choice(['summer_sale', 'spring_sale', 'brand_awareness', '']),
                    'utm_content': np.random.choice(['ad_1', 'ad_2', 'ad_3', 'ad_4', 'ad_5', '']),
                    'device_type': device_type,
                    'direction': 'forward',
                    'navigation_method': np.random.choice(['scroll', 'swipe', 'click']),
                    'link_url': '',
                    'video_src': '',
                    'session_variant': 'A',
                    'presence_test_variant': 'A',
                    'creative_test_variant': 'A',
                    'ab_variant': 'A',
                    'ab_test_target': 'headline',
                    'ab_test_type': 'presence',
                    'cv_type': None,
                    'cv_value': None,
                    'value': None,
                }
                all_events.append(event)

            # CVイベントを追加
            if is_cv:
                event_timestamp = session_start_time + timedelta(milliseconds=total_duration_ms + 1000)
                cv_event = all_events[-1].copy() # 最後のイベントをコピーして一部変更
                cv_event['event_name'] = 'conversion'
                cv_event['event_timestamp'] = event_timestamp
                cv_event['cv_type'] = 'primary'
                cv_event['cv_value'] = np.random.randint(1000, 50000)
                all_events.append(cv_event)

    df = pd.DataFrame(all_events)
    
    # データ型の変換
    df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])
    df['event_date'] = pd.to_datetime(df['event_date'])
    numeric_cols = ['ga_session_id', 'ga_session_number', 'page_num_dom', 'original_page_num', 
                    'stay_ms', 'total_duration_ms', 'load_time_ms', 'max_page_reached', 
                    'completion_rate', 'total_pages', 'click_x_rel', 'click_y_rel', 
                    'scroll_pct', 'cv_value', 'value']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df