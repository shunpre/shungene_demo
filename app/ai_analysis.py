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
