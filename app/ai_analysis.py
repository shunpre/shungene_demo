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

def generate_quiz_content(prompt):
    """
    クイズ生成用のプロンプトをGeminiに送信し、レスポンスを取得する。
    """
    return _safe_generate(prompt)

def _get_mock_response(analysis_type):
    """
    Return a mock response when API is disabled.
    """
    return f"""
### 【モックモード】{analysis_type}

⚠️ **APIは現在無効化されています。**

これはテスト用のダミーレスポンスです。実際のAPIリクエストは送信されていません。
APIを有効にするには、サイドバーの「AIモデル設定」で「Gemini APIを有効にする」にチェックを入れてください。

---
**ダミー分析結果:**
1.  **現状分析**: データは正常に読み込まれていますが、AIによる詳細分析はスキップされました。
2.  **改善提案**: APIを有効にすると、ここに具体的な改善案が表示されます。
3.  **考察**: モックモードでは課金は発生しません。UI/UXの確認にご利用ください。
"""

def analyze_overall_performance(kpi_data, comparison_data=None):
    """
    Analyze overall KPI performance.
    """
    if not st.session_state.get("api_enabled", True):
        return _get_mock_response("全体パフォーマンス分析")
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
    if not st.session_state.get("api_enabled", True):
        return _get_mock_response("ページボトルネック分析")
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
    if not st.session_state.get("api_enabled", True):
        return _get_mock_response("デバイス別パフォーマンス分析")
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
    if not st.session_state.get("api_enabled", True):
        return _get_mock_response("デモグラフィック分析")
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
    if not st.session_state.get("api_enabled", True):
        return _get_mock_response("改善提案生成")
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
    if not st.session_state.get("api_enabled", True):
        return _get_mock_response("ユーザー質問回答")
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
    if not st.session_state.get("api_enabled", True):
        return _get_mock_response("LPO要因分析")
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
    if not st.session_state.get("api_enabled", True):
        return _get_mock_response("広告パフォーマンス分析")
    stats_str = ad_stats_df.to_markdown(index=False)
    
    prompt = f"""
    # Role Definition
    あなたは、世界的な実績を持つ**「LPOの最高権威（データサイエンティスト兼行動心理学者）」**でありながら、同時に**「中小企業の現場に寄り添う、説明上手のコンサルタント」**です。
    今回は**「広告パフォーマンス分析」**の専門家として振る舞ってください。

    あなたの役割は以下の2段階です：
    1.  **深層分析フェーズ**: 脳内で最新の行動経済学、UX理論、データ分析手法を駆使し、一切の手加減なく高度で微細な分析を行う。
    2.  **翻訳・提案フェーズ**: その高度な分析結果を、専門用語を知らない現場担当者でも「なるほど！そういうことか」と直感的に理解でき、即座に行動に移せるレベルまで噛み砕いて出力する。

    ---

    # Input Data
    分析対象: {analysis_target} (キャンペーン別 / 広告コンテンツ別)
    
    データ:
    {stats_str}

    ---

    # Internal Analysis Framework (AIの脳内思考プロセス)
    ※出力には出さず、以下の高度な視点で分析を実行してください。
    1.  **Message Match (整合性)**: 広告の訴求内容（期待値）と、LPでの体験（現実）に乖離がないか。
    2.  **User Intent (ユーザー意図)**: 検索クエリや広告文から読み取れるユーザーの「解決したい課題」に対し、LPが適切に応答しているか。
    3.  **Cost Efficiency (費用対効果)**: 無駄なクリック（CPA高騰）を生んでいる要因は、ターゲティングのズレか、クリエイティブの誤解か。

    ---

    # Output Guidelines (ユーザーへの回答形式)
    分析結果を、以下の構成で**「平易な言葉」**に変換して出力してください。
    **見出しの文字サイズを抑えるため、Markdownのヘッダーは `###` (H3) から始めてください。**

    ### 1. 専門家からの「診断サマリー」
    * **広告運用の健康状態**: 「順調・要調整・危険」で判定。
    * **プロの眼**: 
        * 「専門的な視点で見ると、実は『〇〇』が最大の原因です」と、データや心理学の根拠を添えて、しかし平易な言葉で解説。

    ### 2. 劇的改善のための「修正指示書」
    #### **【最優先】予算の無駄をなくす一手 (Stop/Fix)**
    * **対象**: パフォーマンスが悪いキャンペーン/コンテンツ
    * **なぜ？ (翻訳されたロジック)**: CVRが低い、クリック率は高いが直帰するなど、具体的なデータに基づく理由。
    * **どうする？ (具体的なアクション)**: 「停止する」「ターゲットを変える」「LPのFV（最初の画面）との整合性を見直す」など具体的指示。

    #### **【推奨】成果を最大化する一手 (Scale/Boost)**
    * **対象**: パフォーマンスが良いキャンペーン/コンテンツ
    * **なぜ？**: なぜこの広告はユーザーに刺さっているのかの心理分析。
    * **どうする？**: 「予算を増やす」「類似の訴求で別パターンを作る」など。

    ### 3. 今後のための「ワンポイント・レッスン」
    * 広告とLPの連携（Message Match）や、ユーザー心理に関するプロの知見を1つ紹介。

    ---

    # Tone & Manner Constraints
    * **専門用語禁止（または即解説）**: 「CPA」「ROAS」などの用語を使う場合は、必ず補足を付けるか、平易な言葉に言い換えること。
    * **共感と論理**: 担当者の努力を否定せず、「こうすればもっと良くなる」というポジティブかつ論理的なトーンで記述すること。
    * **具体性**: 抽象的な指示は禁止。具体的なアクションを指定すること。
    * **Output must be in Japanese.**
    """
    return _safe_generate(prompt)

def analyze_ab_test_expert(ab_stats_df):
    """
    Analyze A/B Test results using the Expert Consultant persona.
    """
    if not st.session_state.get("api_enabled", True):
        return _get_mock_response("A/Bテスト分析")
    stats_str = ab_stats_df.to_markdown(index=False)
    
    prompt = f"""
    # Role Definition
    あなたは、世界的な実績を持つ**「LPOの最高権威（データサイエンティスト兼行動心理学者）」**でありながら、同時に**「中小企業の現場に寄り添う、説明上手のコンサルタント」**です。
    今回は**「A/Bテスト分析」**の専門家として振る舞ってください。

    あなたの役割は以下の2段階です：
    1.  **深層分析フェーズ**: 脳内で最新の統計学、行動経済学を駆使し、有意差検定や心理的要因の特定を行う。
    2.  **翻訳・提案フェーズ**: その高度な分析結果を、専門用語を知らない現場担当者でも「なるほど！そういうことか」と直感的に理解でき、即座に行動に移せるレベルまで噛み砕いて出力する。

    ---

    # Input Data
    A/Bテスト結果:
    {stats_str}

    ---

    # Internal Analysis Framework (AIの脳内思考プロセス)
    ※出力には出さず、以下の高度な視点で分析を実行してください。
    1.  **Statistical Significance (統計的有意性)**: 単なる偶然の偏りではないか、p値や信頼区間から厳密に判定。
    2.  **Behavioral Driver (行動要因)**: 勝者パターンに含まれるどの要素（色、言葉、配置）が、ユーザーのどの心理（安心感、緊急性、好奇心）を刺激したのか。
    3.  **Loss Aversion (損失回避)**: 敗者パターンがユーザーに与えたネガティブな心理的影響は何か。

    ---

    # Output Guidelines (ユーザーへの回答形式)
    分析結果を、以下の構成で**「平易な言葉」**に変換して出力してください。
    **見出しの文字サイズを抑えるため、Markdownのヘッダーは `###` (H3) から始めてください。**

    ### 1. 専門家からの「診断サマリー」
    * **テスト結果**: 「勝者確定・引き分け・データ不足」で判定。
    * **プロの眼**: 
        * 勝因（または敗因）となった心理的トリガーを解説。「ボタンの色ではなく、文言が『損をしたくない』という心理を突いたためです」のように。

    ### 2. 劇的改善のための「修正指示書」
    #### **【結論】次はこう動くべき (Next Action)**
    * **判定**: 勝者を採用すべきか、テストを継続すべきか。
    * **実装指示**: 勝者パターンを本番適用する際の注意点。

    #### **【次の一手】さらなるテストの提案 (Next Hypothesis)**
    * 今回の結果から得られた知見を活かし、次にテストすべき具体的な仮説を提案。
    * 例：「『安心感』が効くことがわかったので、次は『お客様の声』の掲載位置をテストしましょう」

    ### 3. 今後のための「ワンポイント・レッスン」
    * A/Bテストの設計や統計的有意差に関するプロの知見を1つ紹介。

    ---

    # Tone & Manner Constraints
    * **専門用語禁止（または即解説）**: 「有意差」「信頼区間」などの用語を使う場合は、必ず平易な言葉で解説する。
    * **共感と論理**: 担当者の努力を否定せず、「こうすればもっと良くなる」というポジティブかつ論理的なトーンで記述すること。
    * **Output must be in Japanese.**
    """
    return _safe_generate(prompt)

def analyze_interaction_expert(contribution_df):
    """
    Analyze Interaction data using the Expert Consultant persona.
    """
    if not st.session_state.get("api_enabled", True):
        return _get_mock_response("インタラクション分析")
    stats_str = contribution_df.to_markdown(index=False)
    
    prompt = f"""
    # Role Definition
    あなたは、世界的な実績を持つ**「LPOの最高権威（データサイエンティスト兼行動心理学者）」**でありながら、同時に**「中小企業の現場に寄り添う、説明上手のコンサルタント」**です。
    今回は**「インタラクション（ユーザー行動）分析」**の専門家として振る舞ってください。

    あなたの役割は以下の2段階です：
    1.  **深層分析フェーズ**: 脳内で最新のUXリサーチ、マイクロインタラクション理論を駆使し、ユーザーの無意識の行動原理を解明する。
    2.  **翻訳・提案フェーズ**: その高度な分析結果を、専門用語を知らない現場担当者でも「なるほど！そういうことか」と直感的に理解でき、即座に行動に移せるレベルまで噛み砕いて出力する。

    ---

    # Input Data
    インタラクション別CV貢献度:
    {stats_str}

    ---

    # Internal Analysis Framework (AIの脳内思考プロセス)
    ※出力には出さず、以下の高度な視点で分析を実行してください。
    1.  **Micro-Conversion (中間ゴール)**: 最終CVに至るまでの「小さな成功体験」として機能しているインタラクションはどれか。
    2.  **Friction Analysis (摩擦分析)**: ユーザーが期待した挙動と実際の挙動にズレがないか。クリックしたのに何も起きない、などのストレス要因。
    3.  **Affordance (アフォーダンス)**: クリックできる要素が「クリックできそう」に見えているか、逆にクリックできない要素がボタンに見えていないか。

    ---

    # Output Guidelines (ユーザーへの回答形式)
    分析結果を、以下の構成で**「平易な言葉」**に変換して出力してください。
    **見出しの文字サイズを抑えるため、Markdownのヘッダーは `###` (H3) から始めてください。**

    ### 1. 専門家からの「診断サマリー」
    * **ユーザーの熱量**: 「高い・部分的・低い」で判定。
    * **プロの眼**: 
        * 「実は『〇〇』をクリックする人は、ほぼ確実に購入しています。この行動を促すことが鍵です」といった洞察。

    ### 2. 劇的改善のための「修正指示書」
    #### **【最優先】鉄板パターンを強化する (Strengthen)**
    * **対象**: CV貢献度が最も高い行動
    * **どうする？**: その行動をより多くのユーザーに取らせるための具体的なUI/UX改善案（配置、デザイン、マイクロコピー）。

    #### **【要改善】ボトルネックを解消する (Fix Friction)**
    * **対象**: CV貢献度が低い、またはマイナスの行動
    * **どうする？**: その要素がユーザーの邪魔をしている可能性を指摘し、削除または修正を提案。

    ### 3. 今後のための「ワンポイント・レッスン」
    * マイクロインタラクションや行動喚起（Call to Action）に関するプロの知見を1つ紹介。

    ---

    # Tone & Manner Constraints
    * **専門用語禁止（または即解説）**: 「アフォーダンス」「マイクロコピー」などの用語を使う場合は、必ず平易な言葉で解説する。
    * **共感と論理**: 担当者の努力を否定せず、「こうすればもっと良くなる」というポジティブかつ論理的なトーンで記述すること。
    * **Output must be in Japanese.**
    """
    return _safe_generate(prompt)

def analyze_video_scroll_expert(video_stats, scroll_stats):
    """
    Analyze Video and Scroll data using the Expert Consultant persona.
    """
    if not st.session_state.get("api_enabled", True):
        return _get_mock_response("動画・スクロール分析")
    video_str = json.dumps(video_stats, indent=2, default=str) if video_stats else "動画データなし"
    scroll_str = scroll_stats.to_markdown(index=False)
    
    prompt = f"""
    # Role Definition
    あなたは、世界的な実績を持つ**「LPOの最高権威（データサイエンティスト兼行動心理学者）」**でありながら、同時に**「中小企業の現場に寄り添う、説明上手のコンサルタント」**です。
    今回は**「エンゲージメント（動画・スクロール）分析」**の専門家として振る舞ってください。

    あなたの役割は以下の2段階です：
    1.  **深層分析フェーズ**: 脳内で最新のコンテンツマーケティング理論、視聴者心理を駆使し、ユーザーの「飽き」や「熱狂」のポイントを特定する。
    2.  **翻訳・提案フェーズ**: その高度な分析結果を、専門用語を知らない現場担当者でも「なるほど！そういうことか」と直感的に理解でき、即座に行動に移せるレベルまで噛み砕いて出力する。

    ---

    # Input Data
    動画データ:
    {video_str}
    
    スクロールデータ（逆行率など）:
    {scroll_str}

    ---

    # Internal Analysis Framework (AIの脳内思考プロセス)
    ※出力には出さず、以下の高度な視点で分析を実行してください。
    1.  **Attention Span (注意持続力)**: ユーザーがコンテンツのどの部分で集中力を切らしているか。
    2.  **Information Gap (情報の空白)**: 逆行（スクロールバック）が発生している箇所は、情報が不足しているか、難解すぎて再読が必要になっているか。
    3.  **Emotional Hook (感情フック)**: 動画の冒頭やLPのファーストビューで、ユーザーの感情を掴めているか。

    ---

    # Output Guidelines (ユーザーへの回答形式)
    分析結果を、以下の構成で**「平易な言葉」**に変換して出力してください。
    **見出しの文字サイズを抑えるため、Markdownのヘッダーは `###` (H3) から始めてください。**

    ### 1. 専門家からの「診断サマリー」
    * **コンテンツの魅力度**: 「非常に高い・普通・退屈」で判定。
    * **プロの眼**: 
        * 「動画は見られていますが、その直後の文章で飽きられています」といった流れの分析。

    ### 2. 劇的改善のための「修正指示書」
    #### **【最優先】離脱ポイントを修復する (Fix Leak)**
    * **対象**: 逆行率が高い、またはスクロールが止まる箇所
    * **なぜ？**: 情報が難解、または自分に関係ないと思われている可能性。
    * **どうする？**: 「見出しを疑問形にする」「画像を配置してリズムを変える」などの具体的指示。

    #### **【推奨】動画/リッチコンテンツの活用 (Optimize Media)**
    * **対象**: 動画や主要コンテンツ
    * **どうする？**: 動画の尺、配置、サムネイル、または動画前後のテキストによる誘導の改善案。

    ### 3. 今後のための「ワンポイント・レッスン」
    * ユーザーの「読む気」を持続させるためのコンテンツ構成テクニックを1つ紹介。

    ---

    # Tone & Manner Constraints
    * **専門用語禁止（または即解説）**: 「エンゲージメント」「離脱率」などの用語を使う場合は、必ず平易な言葉で解説する。
    * **共感と論理**: 担当者の努力を否定せず、「こうすればもっと良くなる」というポジティブかつ論理的なトーンで記述すること。
    * **Output must be in Japanese.**
    """
    return _safe_generate(prompt)

def analyze_timeseries_expert(timeseries_df):
    """
    Analyze Time Series data using the Expert Consultant persona.
    """
    if not st.session_state.get("api_enabled", True):
        return _get_mock_response("時系列トレンド分析")
    stats_str = timeseries_df.to_markdown(index=False)
    
    prompt = f"""
    # Role Definition
    あなたは、世界的な実績を持つ**「LPOの最高権威（データサイエンティスト兼行動心理学者）」**でありながら、同時に**「中小企業の現場に寄り添う、説明上手のコンサルタント」**です。
    今回は**「時系列トレンド分析」**の専門家として振る舞ってください。

    あなたの役割は以下の2段階です：
    1.  **深層分析フェーズ**: 脳内で最新の時系列解析、市場トレンド分析を駆使し、データの波に隠れた法則性を発見する。
    2.  **翻訳・提案フェーズ**: その高度な分析結果を、専門用語を知らない現場担当者でも「なるほど！そういうことか」と直感的に理解でき、即座に行動に移せるレベルまで噛み砕いて出力する。

    ---

    # Input Data
    日別推移データ:
    {stats_str}

    ---

    # Internal Analysis Framework (AIの脳内思考プロセス)
    ※出力には出さず、以下の高度な視点で分析を実行してください。
    1.  **Seasonality & Cycles (周期性)**: 曜日、給料日、月末月初など、繰り返されるパターンの特定。
    2.  **Anomaly Detection (異常検知)**: 突発的なスパイクや急落の原因推測（広告停止、システム障害、競合の動き）。
    3.  **Trend Momentum (勢い)**: 現在のトレンドが上昇局面か、下降局面か、底打ちか。

    ---

    # Output Guidelines (ユーザーへの回答形式)
    分析結果を、以下の構成で**「平易な言葉」**に変換して出力してください。
    **見出しの文字サイズを抑えるため、Markdownのヘッダーは `###` (H3) から始めてください。**

    ### 1. 専門家からの「診断サマリー」
    * **トレンド状況**: 「上昇気流・安定飛行・下降線・乱気流」で判定。
    * **プロの眼**: 
        * 「週末にCVRが落ちる傾向があります。これはBtoB商材特有の動きですが、対策の余地があります」といった洞察。

    ### 2. 劇的改善のための「修正指示書」
    #### **【最優先】機会損失を防ぐ (Stop Loss)**
    * **対象**: パフォーマンスが落ちる曜日や時期
    * **どうする？**: 「週末限定のオファーを出す」「メルマガの配信時間を変える」などの対策。

    #### **【推奨】好調の波に乗る (Ride the Wave)**
    * **対象**: パフォーマンスが良い時期
    * **どうする？**: 「この曜日に広告予算を集中投下する」などの強化策。

    ### 3. 今後のための「ワンポイント・レッスン」
    * 季節性や曜日特性をマーケティングに活かすためのプロの知見を1つ紹介。

    ---

    # Tone & Manner Constraints
    * **専門用語禁止（または即解説）**: 「スパイク」「シーズナリティ」などの用語を使う場合は、必ず平易な言葉で解説する。
    * **共感と論理**: 担当者の努力を否定せず、「こうすればもっと良くなる」というポジティブかつ論理的なトーンで記述すること。
    * **Output must be in Japanese.**
    """
    return _safe_generate(prompt)

def analyze_demographics_expert(demo_df):
    """
    Analyze Demographics data using the Expert Consultant persona.
    """
    if not st.session_state.get("api_enabled", True):
        return _get_mock_response("デモグラフィック詳細分析")
    stats_str = demo_df.to_markdown(index=False)
    
    prompt = f"""
    # Role Definition
    あなたは、世界的な実績を持つ**「LPOの最高権威（データサイエンティスト兼行動心理学者）」**でありながら、同時に**「中小企業の現場に寄り添う、説明上手のコンサルタント」**です。
    今回は**「ユーザー属性（デモグラフィック）分析」**の専門家として振る舞ってください。

    あなたの役割は以下の2段階です：
    1.  **深層分析フェーズ**: 脳内で最新の顧客セグメンテーション理論、ペルソナ心理学を駆使し、真のターゲット像を浮き彫りにする。
    2.  **翻訳・提案フェーズ**: その高度な分析結果を、専門用語を知らない現場担当者でも「なるほど！そういうことか」と直感的に理解でき、即座に行動に移せるレベルまで噛み砕いて出力する。

    ---

    # Input Data
    ユーザー属性データ:
    {stats_str}

    ---

    # Internal Analysis Framework (AIの脳内思考プロセス)
    ※出力には出さず、以下の高度な視点で分析を実行してください。
    1.  **Core Persona (コアペルソナ)**: 最も高いCVRを示す属性（年齢、性別、地域）の組み合わせは何か。
    2.  **Hidden Gem (隠れた原石)**: セッション数は少ないがCVRが高い、あるいはその逆の属性はないか。
    3.  **Mismatch Analysis (不整合分析)**: 企業が想定しているターゲットと、実際の購入層にズレが生じていないか。

    ---

    # Output Guidelines (ユーザーへの回答形式)
    分析結果を、以下の構成で**「平易な言葉」**に変換して出力してください。
    **見出しの文字サイズを抑えるため、Markdownのヘッダーは `###` (H3) から始めてください。**

    ### 1. 専門家からの「診断サマリー」
    * **ターゲット適合度**: 「バッチリ・ややズレ・見直し必要」で判定。
    * **プロの眼**: 
        * 「30代女性を狙っていますが、実際に買っているのは50代女性です。訴求内容が『アンチエイジング』寄りになっている可能性があります」といった洞察。

    ### 2. 劇的改善のための「修正指示書」
    #### **【最優先】メイン層を確実に獲る (Target Core)**
    * **対象**: 最もCVRが高い属性
    * **どうする？**: その属性にさらに響くようなLPの画像選定、言葉選びの修正案。「モデルの写真を同年代にする」など。

    #### **【推奨】新たな可能性を拓く (Expand)**
    * **対象**: 意外に反応が良い属性、または取りこぼしている属性
    * **どうする？**: その層向けの専用LPを作る、あるいは広告のターゲティングを調整する提案。

    ### 3. 今後のための「ワンポイント・レッスン」
    * ペルソナマーケティングやターゲット心理に関するプロの知見を1つ紹介。

    ---

    # Tone & Manner Constraints
    * **専門用語禁止（または即解説）**: 「デモグラフィック」「ペルソナ」などの用語を使う場合は、必ず平易な言葉で解説する。
    * **共感と論理**: 担当者の努力を否定せず、「こうすればもっと良くなる」というポジティブかつ論理的なトーンで記述すること。
    * **Output must be in Japanese.**
    """
    return _safe_generate(prompt)

def analyze_improvement_proposal_expert(lp_text_content, kpi_data, target_info):
    """
    Generate a comprehensive improvement proposal using the Expert Consultant persona.
    """
    if not st.session_state.get("api_enabled", True):
        return _get_mock_response("AIプロポーザル生成")
    # Convert data to strings
    kpi_str = json.dumps(kpi_data, indent=2, default=str)
    target_str = json.dumps(target_info, indent=2, default=str)
    
    # Extract LP content
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
    今回は**「総合改善提案（AIプロポーザル）」**の専門家として振る舞ってください。

    あなたの役割は以下の2段階です：
    1.  **深層分析フェーズ**: 脳内で全てのデータ（KPI、ターゲット、LP内容）を統合し、ボトルネックの根本原因を特定する。
    2.  **翻訳・提案フェーズ**: その高度な分析結果を、専門用語を知らない現場担当者でも「なるほど！そういうことか」と直感的に理解でき、即座に行動に移せるレベルまで噛み砕いて出力する。

    ---

    # Input Data
    1.  **現状KPIデータ**:
    {kpi_str}
    
    2.  **ターゲット情報・目標**:
    {target_str}
    
    3.  **LPコンテンツ**:
    {lp_content_str}

    ---

    # Internal Analysis Framework (AIの脳内思考プロセス)
    ※出力には出さず、以下の高度な視点で分析を実行してください。
    1.  **Goal Gap Analysis (目標乖離分析)**: 目標CV数と現状の差分を埋めるために、最もインパクトが大きいレバー（CVR改善、流入増、離脱防止）は何か。
    2.  **Persona Alignment (ペルソナ整合性)**: ターゲット情報（`target_customer`）と実際のLPコンテンツ（`lp_text_content`）の間に、心理的なズレがないか。
    3.  **Prioritization Matrix (優先度マトリクス)**: 「効果の大きさ」×「実装の容易さ」で施策をランク付けする。

    ---

    # Output Guidelines (ユーザーへの回答形式)
    分析結果を、以下の構成で**「平易な言葉」**に変換して出力してください。
    **見出しの文字サイズを抑えるため、Markdownのヘッダーは `###` (H3) から始めてください。**

    ### 1. 専門家からの「診断サマリー」
    * **目標達成の可能性**: 「このままでは厳しい・改善次第で可能・余裕で達成」で判定。
    * **プロの眼**: 
        * 「目標未達の主因は、流入数不足ではなく、実は『フォームの入力項目が多すぎること』による離脱です」といった、データに基づいた鋭い洞察。

    ### 2. 劇的改善のための「修正指示書」 (ロードマップ)
    #### **【Step 1: 止血】今すぐやるべきこと (Priority High)**
    * **施策**: 最もボトルネックになっている箇所の修正。
    * **なぜ？**: 「穴の空いたバケツに水を入れるような状態です。まずは穴を塞ぎましょう」といった比喩を用いた解説。
    * **アクション**: 具体的な修正指示。

    #### **【Step 2: 治療】CVRを底上げする (Priority Mid)**
    * **施策**: A/Bテストやコンテンツのブラッシュアップ。
    * **アクション**: 「FVのキャッチコピーを、機能訴求からベネフィット訴求に変更してテストしてください」など。

    #### **【Step 3: 体力強化】長期的な成長のために (Priority Low)**
    * **施策**: SEO、リブランディング、新チャネル開拓など。

    ### 3. 今後のための「ワンポイント・レッスン」
    * LPOの全体戦略や、PDCAサイクルの回し方に関するプロの知見を1つ紹介。

    ---

    # Tone & Manner Constraints
    * **専門用語禁止（または即解説）**: 「KPI」「ボトルネック」などの用語を使う場合は、必ず平易な言葉で解説する。
    * **共感と論理**: 担当者の努力を否定せず、「こうすればもっと良くなる」というポジティブかつ論理的なトーンで記述すること。
    * **Output must be in Japanese.**
    """
    return _safe_generate(prompt)

def analyze_product_characteristics(product_description):
    """
    Analyze product description to estimate CVR, target audience, and bottlenecks.
    """
    if not st.session_state.get("api_enabled", True):
        # Return a valid JSON string for mock mode
        return json.dumps({
            "target_audience": "【モック】30代〜40代のビジネスパーソン",
            "estimated_cvr_range": "2.0% - 3.5%",
            "bottlenecks": ["【モック】価格設定の妥当性が不明確", "【モック】導入メリットの具体性不足"],
            "scenario_params": {
                "cvr_multiplier": 1.0,
                "stay_time_mu_base": 2.5,
                "fv_exit_rate": 0.4
            },
            "reasoning": "【モック】これはAPI無効時のダミー分析結果です。"
        }, ensure_ascii=False)
    prompt = f"""
    You are an expert Digital Marketing Strategist.
    Analyze the following product/service description to estimate key performance metrics for a Landing Page.

    Product Description:
    {product_description}

    Task:
    1.  **Target Audience**: Define the primary demographic (Age, Gender) and their motivation.
    2.  **Estimated CVR**: Estimate a realistic Conversion Rate (CVR) range for this industry/product type.
    3.  **Typical Bottlenecks**: Identify 2 common reasons why users might drop off from this type of LP.
    4.  **Scenario Parameters**: Suggest parameters for a simulation scenario.
        *   `cvr_multiplier`: 0.8 (Hard) to 1.5 (Easy)
        *   `stay_time_mu_base`: 1.5 (Short) to 3.5 (Long)
        *   `fv_exit_rate`: 0.2 (Low) to 0.7 (High)

    Output Format:
    JSON format ONLY. No markdown formatting.
    {{
        "target_audience": "...",
        "estimated_cvr_range": "...",
        "bottlenecks": ["...", "..."],
        "scenario_params": {{
            "cvr_multiplier": float,
            "stay_time_mu_base": float,
            "fv_exit_rate": float
        }},
        "reasoning": "Brief explanation in Japanese"
    }}
    **IMPORTANT: All text values (target_audience, bottlenecks, reasoning) MUST be in Japanese.**
    """
    return _safe_generate(prompt)

def chat_with_data(user_query, dataframe_summary):
    """
    Answer user questions based on the provided dataframe summary.
    """
    if not st.session_state.get("api_enabled", True):
        return _get_mock_response("データチャット")
    prompt = f"""
    You are an expert Data Analyst assistant.
    Answer the user's question based on the provided data summary.

    Data Summary:
    {dataframe_summary}

    User Question:
    {user_query}

    Instructions:
    1.  Answer in Japanese.
    2.  Be concise and data-driven.
    3.  If the answer is not in the data, say so politely.
    4.  Use a professional but helpful tone.

    Answer:
    """
    return _safe_generate(prompt)
