import streamlit as st
import google.generativeai as genai
import pandas as pd
import json

def get_gemini_model():
    """
    Initialize and return the Gemini model.
    """
    api_key = st.secrets.get("gemini_api_key")
    if not api_key:
        st.error("Gemini API key not found. Please add 'gemini_api_key' to .streamlit/secrets.toml")
        return None
    
    try:
        genai.configure(api_key=api_key)
        # Get selected model from session state, default to gemini-2.5-pro
        model_name = st.session_state.get("selected_gemini_model", "gemini-2.5-pro")
        model = genai.GenerativeModel(model_name)
        return model
    except Exception as e:
        st.error(f"Failed to configure Gemini API: {e}")
        return None

def _safe_generate(prompt):
    """
    Helper to generate content with error handling.
    """
    model = get_gemini_model()
    if not model:
        return "AI model could not be initialized."
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating content: {str(e)}"

def analyze_overall_performance(kpi_data, comparison_data=None):
    """
    Analyze overall KPI performance.
    """
    prompt = f"""
    You are an expert Web Analyst. Analyze the following KPI data for a Landing Page (LP).
    
    Current KPIs:
    {json.dumps(kpi_data, indent=2, default=str)}
    
    Comparison KPIs (Previous Period):
    {json.dumps(comparison_data, indent=2, default=str) if comparison_data else "Not available"}
    
    Task:
    1. Evaluate the overall health of the LP with deep reasoning. Explain *why* the performance is good or bad.
    2. Highlight the most significant changes (positive or negative) if comparison data exists, and hypothesize the causes.
    3. Identify the primary bottleneck (e.g., low FV retention, low CVR) and its potential business impact.
    4. Provide specific, actionable recommendations to improve the weak points.
    
    Output Format:
    Markdown text with clear headings and bullet points. Provide a detailed and comprehensive analysis (around 400-500 words).
    **IMPORTANT: Output must be in Japanese.**
    """
    return _safe_generate(prompt)

def analyze_page_bottlenecks(page_stats_df):
    """
    Analyze page-level statistics to identify bottlenecks.
    """
    # Convert DataFrame to string/dict for prompt
    stats_str = page_stats_df.to_markdown(index=False)
    
    prompt = f"""
    Analyze the following page-level performance data for a multi-page LP (Swipe LP).
    
    Page Statistics:
    {stats_str}
    
    Task:
    1. Identify the page with the highest drop-off rate (excluding the final page) and analyze the user behavior leading up to it.
    2. Analyze the correlation between time spent and drop-off. Does a short time indicate confusion, or does a long time indicate loss of interest?
    3. Suggest a detailed hypothesis for *why* users are leaving at the bottleneck page (e.g., confusing copy, lack of trust signals, technical friction).
    4. Propose 2-3 specific UI/UX or content changes to fix the bottleneck.
    
    Output Format:
    Markdown text. Provide a detailed analysis focusing on the "Why" and "How to Fix".
    **IMPORTANT: Output must be in Japanese.**
    """
    return _safe_generate(prompt)

def analyze_device_performance(device_stats_df):
    """
    Analyze performance by device type.
    """
    stats_str = device_stats_df.to_markdown(index=False)
    
    prompt = f"""
    Analyze the LP performance across different devices.
    
    Device Statistics:
    {stats_str}
    
    Task:
    1. Compare CVR and Session counts across devices in detail.
    2. Identify if there is a significant underperformance on mobile vs desktop. If so, explain potential reasons (e.g., layout issues, load time, navigation difficulty).
    3. Recommend specific device-optimization actions (e.g., "Increase font size on mobile", "Simplify navigation menu").
    
    Output Format:
    Markdown text. Provide specific technical or design recommendations.
    **IMPORTANT: Output must be in Japanese.**
    """
    return _safe_generate(prompt)

def analyze_demographics(age_df, gender_df, region_df):
    """
    Analyze demographic data (Age, Gender, Region).
    """
    prompt = f"""
    Analyze the demographic profile of the LP visitors and their conversion rates.
    
    Age Group Data:
    {age_df.to_markdown(index=False) if not age_df.empty else "No data"}
    
    Gender Data:
    {gender_df.to_markdown(index=False) if not gender_df.empty else "No data"}
    
    Region Data:
    {region_df.to_markdown(index=False) if not region_df.empty else "No data"}
    
    Task:
    1. Define the core persona that converts the best (Age, Gender, Region) and describe their potential motivations.
    2. Identify any untapped demographic segments that show promise (e.g., high engagement but low CVR).
    3. Suggest how to tailor the LP content (images, copy, tone) specifically for the high-performing persona to maximize conversions.
    
    Output Format:
    Markdown text. Provide a deep dive into user psychology and persona analysis.
    **IMPORTANT: Output must be in Japanese.**
    """
    return _safe_generate(prompt)

def generate_improvement_proposal(kpi_data, page_stats_df, device_stats_df, target_customer, other_info):
    """
    Generate a comprehensive improvement proposal.
    """
    prompt = f"""
    Based on the following comprehensive data, generate a detailed improvement proposal for the Landing Page.
    
    Target Customer Context: {target_customer}
    Specific Focus/Notes: {other_info}
    
    Overall KPIs:
    {json.dumps(kpi_data, indent=2, default=str)}
    
    Page Statistics (Bottlenecks):
    {page_stats_df.to_markdown(index=False)}
    
    Device Statistics:
    {device_stats_df.to_markdown(index=False)}
    
    Task:
    Generate a comprehensive, detailed improvement proposal (600+ words):
    1. **Executive Summary**: Brief overview of the current state and main opportunity.
    2. **Immediate Actions (High Priority)**: Quick wins to fix major leaks. Be very specific (e.g., "Change the Hero Image to X", "Add a testimonial section below Y").
    3. **A/B Testing Strategy (Medium Priority)**: Propose 2-3 concrete A/B tests. Define the Hypothesis, Variant A, Variant B, and Success Metric for each.
    4. **Strategic Overhaul (Long-term)**: Structural or content strategy changes based on the target persona. Discuss brand positioning and long-term user engagement.
    
    Output Format:
    Structured Markdown with clear sections. Be extremely specific, actionable, and professional.
    **IMPORTANT: Output must be in Japanese.**
    """
    return _safe_generate(prompt)

def answer_user_question(context_data, question):
    """
    Answer a specific user question based on provided context.
    """
    prompt = f"""
    You are an AI Analyst assistant. Answer the user's question based *only* on the provided data context.
    
    Data Context:
    {context_data}
    
    User Question:
    {question}
    
    Answer (in Japanese):
    """
    return _safe_generate(prompt)

def analyze_lpo_factors(kpi_data, page_stats_df, hearing_sheet_text, lp_text_content, lp_format="縦長"):
    """
    Perform a comprehensive LPO factor analysis based on the user's detailed prompt.
    """
    # Convert data to strings
    kpi_str = json.dumps(kpi_data, indent=2, default=str)
    page_stats_str = page_stats_df.to_markdown(index=False)
    
    # Extract LP content
    # Handle both dictionary (if structured) and string input
    if isinstance(lp_text_content, dict):
        headlines = "\n".join(lp_text_content.get('headlines', []))
        body_copy = "\n".join(lp_text_content.get('body_copy', []))
        ctas = "\n".join(lp_text_content.get('ctas', []))
        lp_content_str = f"Headlines: {headlines}\nBody Copy: {body_copy}\nCTAs: {ctas}"
    else:
        lp_content_str = str(lp_text_content)

    prompt = f"""
    # Role Definition
    あなたは、世界的な実績を持つ**「LPOの最高権威（データサイエンティスト兼行動心理学者）」**でありながら、同時に**「中小企業の現場に寄り添う、説明上手のコンサルタント」**です。

    あなたの役割は以下の2段階です：
    1.  **深層分析フェーズ**: 脳内で最新の行動経済学、UX理論、データ分析手法を駆使し、一切の手加減なく高度で微細な分析を行う。
    2.  **翻訳・提案フェーズ**: その高度な分析結果を、専門用語を知らない現場担当者でも「なるほど！そういうことか」と直感的に理解でき、即座に行動に移せるレベルまで噛み砕いて出力する。

    ---

    # Input Data
    以下の情報を基に分析します。情報が不足している場合は、プロの知見に基づき**「一般的な業界傾向」から論理的に推測**して補完してください。

    1.  **基本情報**:
        * LP形式: {lp_format} (縦長 / スワイプ型 / 記事LPなど)
        * ヒアリングシート情報（商品・ターゲット・課題など）:
        {hearing_sheet_text}

    2.  **現状データ (定性・定量)**:
        * KPIデータ:
        {kpi_str}
        * ページ統計データ (離脱率・滞在時間など):
        {page_stats_str}

    3.  **クリエイティブ**:
        * LPテキストコンテンツ:
        {lp_content_str}

    ---

    # Internal Analysis Framework (AIの脳内思考プロセス)
    ※出力には出さず、以下の高度な視点で分析を実行してください。

    1.  **Behavioral Psychology (行動心理学)**:
        * *Cialdini’s 6 Principles* (返報性、コミットメント、社会的証明、権威、好意、希少性) の欠如特定。
        * *Fogg Behavior Model* (B=MAP) における Motivation/Ability/Trigger のバランス不全分析。
    2.  **Cognitive UX (認知科学・UX)**:
        * *Cognitive Load* (認知負荷) の発生源特定。
        * *Gutenberg Diagram / Z-Pattern / F-Pattern* に基づく視線誘導の断絶分析。
        * *Micro-copy Analysis*: CTA周辺のフリクション（心理的抵抗）分析。
    3.  **Data Logic (データロジック)**:
        * *Message Match*: 流入元（広告）の期待値とLP着地時の整合性乖離。
        * *Funnel Drop-off*: スクロール深度やカード遷移率におけるボトルネック特定。

    ---

    # Output Guidelines (ユーザーへの回答形式)
    分析結果を、以下の構成で**「平易な言葉」**に変換して出力してください。

    ## 1. 専門家からの「診断サマリー」
    * **LPの健康状態**: 「健康・要注意・重症」で判定。
    * **プロの眼**: 
        * 「専門的な視点で見ると、実は『〇〇』が最大の原因です」と、データや心理学の根拠を添えて、しかし平易な言葉で解説。
        * 例：「ボタンの色ではなく、実はお客様が『自分には関係ない』と感じてしまう文章の並び順に根本原因があります」

    ## 2. 劇的改善のための「修正指示書」 (優先度順)

    ### **【最優先】今すぐ直すべき箇所 (Priority High)**
    ※修正コストが低く、成果インパクトが最大のもの。

    * **どこを？**: [対象箇所を具体的に指名]
    * **なぜ？ (翻訳されたロジック)**:
        * 専門用語を使わずに解説。
        * *悪い例*: 「バンドワゴン効果が不足しており、ソーシャルプルーフの提示が必要です」
        * *良い例*: 「『みんなが使っている』という安心感がないため、お客様が購入をためらっています。行列のできているラーメン屋が美味しく見えるのと同じ心理を使いましょう」
    * **どう直す？ (具体的なアクション)**:
        * **Before**: [現状のテキスト/構成]
        * **After**: [改善後のテキスト案/構成案] ※そのままコピペで使えるレベルで。
        * **デザイン指示**: [色、配置、文字サイズなどの具体的指示]

    ### **【推奨】数字をさらに伸ばす一手 (Priority Mid)**
    ※流入経路に合わせた調整や、テストすべき項目。

    * **対象**: [流入元やターゲット属性]
    * **改善案**: [具体的な修正内容]

    ## 3. 今後のための「ワンポイント・レッスン」
    * 今回の分析で用いた**「プロのテクニック（心理学や法則）」**を1つだけ、簡単な言葉で紹介してください。担当者が次回以降、自分で気づけるようになるための教育的コメントです。

    ---

    # Tone & Manner Constraints
    * **専門用語禁止（または即解説）**: 「CTA」「FV」「CVR」などの用語を使う場合は、必ず「CTA（申し込みボタン）」「FV（最初に表示される画面）」のように補足を付けるか、平易な言葉に言い換えること。
    * **共感と論理**: 担当者の努力を否定せず、「こうすればもっと良くなる」というポジティブかつ論理的なトーンで記述すること。
    * **具体性**: 「わかりやすくする」「魅力を伝える」といった抽象的な指示は禁止。具体的な「文言」「色」「位置」を指定すること。
    * **Output must be in Japanese.**
    """
    return _safe_generate(prompt)

def analyze_ad_performance_expert(ad_stats_df, analysis_target):
    """
    Analyze Ad performance using the Expert Consultant persona.
    """
    stats_str = ad_stats_df.to_markdown(index=False)
    
    prompt = f"""
    # Role Definition
    あなたは**「LPOの最高権威」**であり、**「中小企業の現場に寄り添うコンサルタント」**です。
    今回は**「広告パフォーマンス分析」**の専門家として振る舞ってください。

    # Input Data
    分析対象: {analysis_target} (キャンペーン別 / 広告コンテンツ別)
    
    データ:
    {stats_str}

    # Task
    1.  **深層分析**: データから「勝ちパターン」と「負けパターン」を特定し、その要因を仮説立てる。
    2.  **翻訳・提案**: 現場担当者がすぐに広告運用やLP修正に活かせる具体的なアドバイスを行う。

    # Output Guidelines
    ## 1. 専門家からの「診断サマリー」
    * **広告運用の健康状態**: 「順調・要調整・危険」で判定。
    * **プロの眼**: データに見られる最大の特徴と、そこから読み取れるユーザー心理を平易な言葉で解説。

    ## 2. 劇的改善のための「修正指示書」
    ### **【最優先】予算の無駄をなくす一手 (Stop/Fix)**
    * **対象**: パフォーマンスが悪いキャンペーン/コンテンツ
    * **なぜ？**: CVRが低い、クリック率は高いが直帰するなど、具体的なデータに基づく理由。
    * **どうする？**: 「停止する」「ターゲットを変える」「LPのFV（最初の画面）との整合性を見直す」など具体的指示。

    ### **【推奨】成果を最大化する一手 (Scale/Boost)**
    * **対象**: パフォーマンスが良いキャンペーン/コンテンツ
    * **なぜ？**: なぜこの広告はユーザーに刺さっているのかの心理分析。
    * **どうする？**: 「予算を増やす」「類似の訴求で別パターンを作る」など。

    ## 3. 今後のための「ワンポイント・レッスン」
    * 広告とLPの連携（Message Match）や、ユーザー心理に関するプロの知見を1つ紹介。

    # Tone & Manner
    * 専門用語は必ず解説を入れる。
    * 具体的かつアクション可能な指示にする。
    * **Output must be in Japanese.**
    """
    return _safe_generate(prompt)

def analyze_ab_test_expert(ab_stats_df):
    """
    Analyze A/B Test results using the Expert Consultant persona.
    """
    stats_str = ab_stats_df.to_markdown(index=False)
    
    prompt = f"""
    # Role Definition
    あなたは**「LPOの最高権威」**であり、**「中小企業の現場に寄り添うコンサルタント」**です。
    今回は**「A/Bテスト分析」**の専門家として振る舞ってください。

    # Input Data
    A/Bテスト結果:
    {stats_str}

    # Task
    1.  **統計的評価**: 勝敗の判定だけでなく、その結果が「信頼に足るものか（有意差）」を判断。
    2.  **要因分析**: なぜそのバリアントが勝った（負けた）のか、行動心理学的な視点で理由を解明。

    # Output Guidelines
    ## 1. 専門家からの「診断サマリー」
    * **テスト結果**: 「勝者確定・引き分け・データ不足」で判定。
    * **プロの眼**: 勝因（または敗因）となった心理的トリガーを解説。「ボタンの色ではなく、文言が『損をしたくない』という心理を突いたためです」のように。

    ## 2. 劇的改善のための「修正指示書」
    ### **【結論】次はこう動くべき (Next Action)**
    * **判定**: 勝者を採用すべきか、テストを継続すべきか。
    * **実装指示**: 勝者パターンを本番適用する際の注意点。

    ### **【次の一手】さらなるテストの提案 (Next Hypothesis)**
    * 今回の結果から得られた知見を活かし、次にテストすべき具体的な仮説を提案。
    * 例：「『安心感』が効くことがわかったので、次は『お客様の声』の掲載位置をテストしましょう」

    ## 3. 今後のための「ワンポイント・レッスン」
    * A/Bテストの設計や統計的有意差に関するプロの知見を1つ紹介。

    # Tone & Manner
    * 専門用語（有意差、p値など）は必ず平易な言葉で解説する。
    * **Output must be in Japanese.**
    """
    return _safe_generate(prompt)

def analyze_interaction_expert(contribution_df):
    """
    Analyze Interaction data using the Expert Consultant persona.
    """
    stats_str = contribution_df.to_markdown(index=False)
    
    prompt = f"""
    # Role Definition
    あなたは**「LPOの最高権威」**であり、**「中小企業の現場に寄り添うコンサルタント」**です。
    今回は**「インタラクション（ユーザー行動）分析」**の専門家として振る舞ってください。

    # Input Data
    インタラクション別CV貢献度:
    {stats_str}

    # Task
    1.  **行動分析**: どの行動（クリック、視聴など）がコンバージョン（購入/申込）に直結しているかを特定。
    2.  **マイクロコピー/UI分析**: ユーザーがその行動を取りたくなる（または避ける）心理的要因を分析。

    # Output Guidelines
    ## 1. 専門家からの「診断サマリー」
    * **ユーザーの熱量**: 「高い・部分的・低い」で判定。
    * **プロの眼**: 「実は『〇〇』をクリックする人は、ほぼ確実に購入しています。この行動を促すことが鍵です」といった洞察。

    ## 2. 劇的改善のための「修正指示書」
    ### **【最優先】鉄板パターンを強化する (Strengthen)**
    * **対象**: CV貢献度が最も高い行動
    * **どうする？**: その行動をより多くのユーザーに取らせるための具体的なUI/UX改善案（配置、デザイン、マイクロコピー）。

    ### **【要改善】ボトルネックを解消する (Fix Friction)**
    * **対象**: CV貢献度が低い、またはマイナスの行動
    * **どうする？**: その要素がユーザーの邪魔をしている可能性を指摘し、削除または修正を提案。

    ## 3. 今後のための「ワンポイント・レッスン」
    * マイクロインタラクションや行動喚起（Call to Action）に関するプロの知見を1つ紹介。

    # Tone & Manner
    * 専門用語は解説を入れる。
    * **Output must be in Japanese.**
    """
    return _safe_generate(prompt)

def analyze_video_scroll_expert(video_stats, scroll_stats):
    """
    Analyze Video and Scroll data using the Expert Consultant persona.
    """
    video_str = json.dumps(video_stats, indent=2, default=str) if video_stats else "動画データなし"
    scroll_str = scroll_stats.to_markdown(index=False)
    
    prompt = f"""
    # Role Definition
    あなたは**「LPOの最高権威」**であり、**「中小企業の現場に寄り添うコンサルタント」**です。
    今回は**「エンゲージメント（動画・スクロール）分析」**の専門家として振る舞ってください。

    # Input Data
    動画データ:
    {video_str}
    
    スクロールデータ（逆行率など）:
    {scroll_str}

    # Task
    1.  **熟読度分析**: ユーザーがどこで興味を持ち、どこで飽きているか（離脱/スクロール）を分析。
    2.  **コンテンツ評価**: 動画や長文コンテンツが有効に機能しているか評価。

    # Output Guidelines
    ## 1. 専門家からの「診断サマリー」
    * **コンテンツの魅力度**: 「非常に高い・普通・退屈」で判定。
    * **プロの眼**: 「動画は見られていますが、その直後の文章で飽きられています」といった流れの分析。

    ## 2. 劇的改善のための「修正指示書」
    ### **【最優先】離脱ポイントを修復する (Fix Leak)**
    * **対象**: 逆行率が高い、またはスクロールが止まる箇所
    * **なぜ？**: 情報が難解、または自分に関係ないと思われている可能性。
    * **どうする？**: 「見出しを疑問形にする」「画像を配置してリズムを変える」などの具体的指示。

    ### **【推奨】動画/リッチコンテンツの活用 (Optimize Media)**
    * **対象**: 動画や主要コンテンツ
    * **どうする？**: 動画の尺、配置、サムネイル、または動画前後のテキストによる誘導の改善案。

    ## 3. 今後のための「ワンポイント・レッスン」
    * ユーザーの「読む気」を持続させるためのコンテンツ構成テクニックを1つ紹介。

    # Tone & Manner
    * 専門用語は解説を入れる。
    * **Output must be in Japanese.**
    """
    return _safe_generate(prompt)

def analyze_timeseries_expert(timeseries_df):
    """
    Analyze Time Series data using the Expert Consultant persona.
    """
    stats_str = timeseries_df.to_markdown(index=False)
    
    prompt = f"""
    # Role Definition
    あなたは**「LPOの最高権威」**であり、**「中小企業の現場に寄り添うコンサルタント」**です。
    今回は**「時系列トレンド分析」**の専門家として振る舞ってください。

    # Input Data
    日別推移データ:
    {stats_str}

    # Task
    1.  **トレンド特定**: 上昇/下降トレンド、または特定の曜日/日付でのスパイク（急変動）を特定。
    2.  **要因推測**: 外部要因（給料日、週末、季節性）や内部要因（広告変更、メルマガ配信）との関連を推測。

    # Output Guidelines
    ## 1. 専門家からの「診断サマリー」
    * **トレンド状況**: 「上昇気流・安定飛行・下降線・乱気流」で判定。
    * **プロの眼**: 「週末にCVRが落ちる傾向があります。これはBtoB商材特有の動きですが、対策の余地があります」といった洞察。

    ## 2. 劇的改善のための「修正指示書」
    ### **【最優先】機会損失を防ぐ (Stop Loss)**
    * **対象**: パフォーマンスが落ちる曜日や時期
    * **どうする？**: 「週末限定のオファーを出す」「メルマガの配信時間を変える」などの対策。

    ### **【推奨】好調の波に乗る (Ride the Wave)**
    * **対象**: パフォーマンスが良い時期
    * **どうする？**: 「この曜日に広告予算を集中投下する」などの強化策。

    ## 3. 今後のための「ワンポイント・レッスン」
    * 季節性や曜日特性をマーケティングに活かすためのプロの知見を1つ紹介。

    # Tone & Manner
    * 専門用語は解説を入れる。
    * **Output must be in Japanese.**
    """
    return _safe_generate(prompt)

def analyze_demographics_expert(demo_df):
    """
    Analyze Demographics data using the Expert Consultant persona.
    """
    stats_str = demo_df.to_markdown(index=False)
    
    prompt = f"""
    # Role Definition
    あなたは**「LPOの最高権威」**であり、**「中小企業の現場に寄り添うコンサルタント」**です。
    今回は**「ユーザー属性（デモグラフィック）分析」**の専門家として振る舞ってください。

    # Input Data
    ユーザー属性データ:
    {stats_str}

    # Task
    1.  **ペルソナ特定**: 最もCVRが高い「理想の顧客像（ペルソナ）」を明確化。
    2.  **ギャップ分析**: 想定ターゲットと実際の購入層にズレがないか分析。

    # Output Guidelines
    ## 1. 専門家からの「診断サマリー」
    * **ターゲット適合度**: 「バッチリ・ややズレ・見直し必要」で判定。
    * **プロの眼**: 「30代女性を狙っていますが、実際に買っているのは50代女性です。訴求内容が『アンチエイジング』寄りになっている可能性があります」といった洞察。

    ## 2. 劇的改善のための「修正指示書」
    ### **【最優先】メイン層を確実に獲る (Target Core)**
    * **対象**: 最もCVRが高い属性
    * **どうする？**: その属性にさらに響くようなLPの画像選定、言葉選びの修正案。「モデルの写真を同年代にする」など。

    ### **【推奨】新たな可能性を拓く (Expand)**
    * **対象**: 意外に反応が良い属性、または取りこぼしている属性
    * **どうする？**: その層向けの専用LPを作る、あるいは広告のターゲティングを調整する提案。

    ## 3. 今後のための「ワンポイント・レッスン」
    * ペルソナマーケティングやターゲット心理に関するプロの知見を1つ紹介。

    # Tone & Manner
    * 専門用語は解説を入れる。
    * **Output must be in Japanese.**
    """
    return _safe_generate(prompt)
