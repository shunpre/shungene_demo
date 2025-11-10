"""
ダミーデータ生成スクリプト
BigQueryのevents_flat_tblテーブル構造に対応したリアルなイベントデータを生成
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_dummy_data(num_events=5000, num_days=30):
    """
    リアルなスワイプLPイベントデータを生成
    
    Args:
        num_events: 生成するイベント数
        num_days: 過去何日分のデータを生成するか
    
    Returns:
        pd.DataFrame: ダミーデータ
    """
    
    # 基準日時
    end_date = datetime.now()
    start_date = end_date - timedelta(days=num_days)
    
    # ユーザー数（セッション数の約1/3）
    num_users = num_events // 15
    user_ids = [f"user_{i:06d}" for i in range(num_users)]
    
    # LP URL
    # ユーザーから指定されたURLに固定
    lp_url = "https://shungene.lm-c.jp/tst08/tst08.html"
    
    # イベント名
    event_names = [
        "page_view",
        "swipe_page",
        "click",
        "scroll",
        "video_play",
        "conversion",
        "session_start",
    ]
    
    # デバイスタイプ
    device_types = ["mobile", "desktop", "tablet"]
    device_weights = [0.7, 0.25, 0.05]
    
    # UTMパラメータ
    traffic_sources = {
        "google": {"mediums": ["organic", "cpc"], "referrer": "https://www.google.com/"},
        "yahoo": {"mediums": ["organic", "cpc"], "referrer": "https://www.yahoo.co.jp/"},
        "bing": {"mediums": ["organic", "cpc"], "referrer": "https://www.bing.com/"},
        "facebook": {"mediums": ["social", "paidsocial", "referral"], "referrer": "https://www.facebook.com/"},
        "instagram": {"mediums": ["social", "paidsocial", "referral"], "referrer": "https://www.instagram.com/"},
        "twitter": {"mediums": ["social", "paidsocial", "referral"], "referrer": "https://t.co/"},
        "youtube": {"mediums": ["paidvideo", "referral"], "referrer": "https://www.youtube.com/"},
        "smartnews": {"mediums": ["display", "referral"], "referrer": "https://www.smartnews.com/"},
        "news-app": {"mediums": ["display", "referral"], "referrer": "android-app://com.example.news"},
        "direct": {"mediums": ["(none)"], "referrer": None}
    }
    source_keys = list(traffic_sources.keys())
    utm_campaigns = ["spring_sale", "summer_campaign", "brand_awareness", None]
    
    # A/Bテストバリアント
    ab_variants = ["A", "B"]
    
    # ナビゲーション方法
    navigation_methods = ["swipe", "click", "scroll", "button"]
    
    # 方向
    directions = ["forward", "backward"]
    
    # A/Bテストごとのp値を保持する辞書
    test_p_values = {}

    # データ生成
    data = []
    
    for _ in range(num_events):
        # ランダムな日時
        event_timestamp = start_date + timedelta(
            seconds=random.randint(0, int((end_date - start_date).total_seconds()))
        )
        event_date = event_timestamp.date()
        
        # ユーザーとセッション
        user_pseudo_id = random.choice(user_ids)
        ga_session_id = random.randint(1000000, 9999999)
        ga_session_number = random.randint(1, 10)
        session_id = f"{user_pseudo_id}-{ga_session_id}"
        
        # イベント名
        event_name = random.choice(event_names)
        
        # ページ情報
        page_location = lp_url
        page_path = page_location.split(".com")[1] if ".com" in page_location else "/"
        page_num_dom = random.randint(1, 10)
        original_page_num = page_num_dom
        total_pages = 10
        
        # 滞在時間・パフォーマンス
        stay_ms = random.randint(1000, 120000)  # 1秒～2分
        total_duration_ms = random.randint(stay_ms, 300000)  # 最大5分
        load_time_ms = random.randint(500, 5000)  # 0.5秒～5秒
        
        # 進行状況
        max_page_reached = random.randint(page_num_dom, total_pages)
        completion_rate = max_page_reached / total_pages
        
        # クリック情報（クリックイベントの場合のみ）
        if event_name == "click":
            click_x_rel = random.uniform(0.1, 0.9)
            click_y_rel = random.uniform(0.1, 0.9)
            elem_tag = random.choice(["button", "a", "div", "img"])
            elem_id = random.choice(["cta-button", "nav-link", "video-play", None])
            elem_classes = random.choice(["btn-primary", "link", "card", None])
        else:
            click_x_rel = None
            click_y_rel = None
            elem_tag = None
            elem_id = None
            elem_classes = None
        
        # スクロール
        scroll_pct = random.uniform(0.1, 1.0)
        
        # UTM/トラフィック
        utm_source = random.choice(source_keys)
        source_info = traffic_sources[utm_source]
        utm_medium = random.choice(source_info["mediums"])
        utm_campaign = random.choice(utm_campaigns)
        utm_content = random.choice([f"ad_{i}" for i in range(1, 6)] + [None])
        device_type = random.choices(device_types, weights=device_weights)[0]
        
        # リファラーをUTMソースと連動させる
        page_referrer = source_info["referrer"]
        if utm_source != "direct" and random.random() < 0.1: # 10%の確率でリファラーが取れないケース
            page_referrer = None

        # ナビゲーション
        direction = random.choice(directions)
        navigation_method = random.choice(navigation_methods)
        link_url = random.choice([page_location, "https://example.com/thank-you", None])
        video_src = random.choice(["https://example.com/video1.mp4", None])
        
        # A/Bテスト
        session_variant = random.choice(ab_variants)
        presence_test_variant = random.choice(ab_variants + [None]) # type: ignore
        creative_test_variant = random.choice(ab_variants + [None]) # type: ignore
        ab_variant = session_variant
        ab_test_target = random.choice(["cta_button", "hero_image", "headline", None])
        ab_test_type = random.choice(["presence", "creative", "layout", None])

        # p値の生成ロジック（ab_test_targetが存在する場合に限定）
        p_value = None
        if ab_test_target:
            # テスト種別とバリアントの組み合わせでp値を決定
            test_key = (ab_test_target, ab_variant)
            if test_key not in test_p_values:
                # 0.01, 0.05, 0.1の周辺に偏らせつつ、ランダムなp値を生成
                p_value_options = [
                    random.uniform(0.005, 0.02),  # ★★★
                    random.uniform(0.04, 0.06),   # ★★
                    random.uniform(0.09, 0.11),   # ★
                    random.uniform(0.1, 1.0)      # -
                ]
                # バリアントAはp値1.0（基準）、Bにランダムなp値を割り当て
                test_p_values[test_key] = 1.0 if ab_variant == 'A' else random.choices(p_value_options, weights=[0.1, 0.2, 0.2, 0.5])[0]
            p_value = test_p_values[test_key]
        
        # コンバージョン（5%の確率）
        is_conversion = random.random() < 0.05
        if is_conversion and event_name == "conversion":
            cv_type = random.choice(["primary", "micro"])
            cv_value = random.uniform(1000, 50000)
            value = cv_value
        else:
            cv_type = None
            cv_value = None
            value = None
        
        # 前ページパス
        if page_num_dom > 1:
            # 5%の確率で逆行を発生させる
            if random.random() < 0.05 and page_num_dom < total_pages:
                # 逆行: 現在のページより後のページから来たことにする
                # 例: 現在がpage-3なら、前のページはpage-4やpage-5
                prev_page_num = random.randint(page_num_dom + 1, total_pages)
                prev_page_path = f"{page_path}#page-{prev_page_num}"
            else:
                prev_page_path = f"{page_path}#page-{page_num_dom - 1}"
        else:
            prev_page_path = None
        
        # データ追加
        data.append({
            "event_date": event_date,
            "event_timestamp": event_timestamp,
            "event_timestamp_jst": event_timestamp,
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
            "original_page_num": original_page_num,
            "stay_ms": stay_ms,
            "total_duration_ms": total_duration_ms,
            "load_time_ms": load_time_ms,
            "max_page_reached": max_page_reached,
            "completion_rate": completion_rate,
            "total_pages": total_pages,
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
            "navigation_method": navigation_method,
            "link_url": link_url,
            "video_src": video_src,
            "session_variant": session_variant,
            "presence_test_variant": presence_test_variant,
            "creative_test_variant": creative_test_variant,
            "ab_variant": ab_variant,
            "ab_test_target": ab_test_target,
            "ab_test_type": ab_test_type,
            "cv_type": cv_type,
            "p_value": p_value,
            "cv_value": cv_value,
            "value": value,
        })
    
    # DataFrameに変換
    df = pd.DataFrame(data)
    
    # --- 意図的なアラートデータを注入 ---
    # 期間内にランダムな3日を「異常日」として設定
    if num_days > 10:
        alert_dates = random.sample(
            [end_date - timedelta(days=i) for i in range(3, num_days - 3)], 
            k=min(3, num_days - 6)
        )
        for alert_date in alert_dates:
            alert_date_str = alert_date.strftime('%Y-%m-%d')
            
            # セッション数を意図的に減らす (50-80%減)
            session_drop_rate = random.uniform(0.5, 0.8)
            num_sessions_on_alert_date = len(df[df['event_date'] == alert_date_str])
            sessions_to_drop = int(num_sessions_on_alert_date * session_drop_rate)
            if sessions_to_drop > 0:
                indices_to_drop = df[df['event_date'] == alert_date_str].sample(n=sessions_to_drop).index
                df.drop(indices_to_drop, inplace=True)

            # CVRを意図的に下げる (CVイベントを削除)
            cv_indices_on_alert_date = df[(df['event_date'] == alert_date_str) & (df['cv_type'].notna())].index
            if not cv_indices_on_alert_date.empty:
                df.drop(cv_indices_on_alert_date, inplace=True)

    # 日付でソート
    df = df.sort_values("event_timestamp").reset_index(drop=True)
    
    return df


if __name__ == "__main__":
    # ダミーデータ生成
    df = generate_dummy_data(num_events=10000, num_days=30)
    
    # CSV保存
    df.to_csv("/home/ubuntu/swipe_lp_analyzer/app/dummy_data.csv", index=False)
    
    print(f"✅ ダミーデータ生成完了: {len(df)} イベント")
    print(f"📅 期間: {df['event_date'].min()} ～ {df['event_date'].max()}")
    print(f"👥 ユーザー数: {df['user_pseudo_id'].nunique()}")
    print(f"📊 セッション数: {df['session_id'].nunique()}")
