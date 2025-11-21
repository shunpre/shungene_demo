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

def analyze_lpo_factors(kpi_data, page_stats_df, hearing_sheet_text, lp_text_content):
    """
    Perform a comprehensive LPO factor analysis based on the user's detailed prompt.
    """
    # Convert data to strings
    kpi_str = json.dumps(kpi_data, indent=2, default=str)
    page_stats_str = page_stats_df.to_markdown(index=False)
    
    # Extract LP content
    headlines = "\n".join(lp_text_content.get('headlines', []))
    body_copy = "\n".join(lp_text_content.get('body_copy', []))
    ctas = "\n".join(lp_text_content.get('ctas', []))

    prompt = f"""
    You are an expert Data Analyst and LPO Specialist.
    Perform a comprehensive LPO analysis based on the following "LPO Factor Analysis" framework.
    
    **Input Data:**
    
    1. **Hearing Sheet / Product Info:**
    {hearing_sheet_text}
    
    2. **Current KPIs (Result Data):**
    {kpi_str}
    
    3. **Page Flow / Heatmap Substitute Data:**
    {page_stats_str}
    
    4. **LP Content (Extracted Text):**
    Headlines: {headlines}
    Body Copy: {body_copy}
    CTAs: {ctas}
    
    **Analysis Framework & Prompt:**
    
    【LPO 要因分析】
    
    目的: 限られた情報からでも、CVRが低い原因を可能な限り分析し、効果的な改善施策を立案するための現状分析を行う。
    前提:
    自身は、データ分析に基づき、冷静かつ客観的に課題を特定できる、データ分析の専門家である。
    分析対象のLPは、現状CVRが低く、改善の余地がある。
    分析結果は、具体的な改善施策の立案に活用される。
    
    手順:
    基本情報の確認:
    ターゲット層（顕在層、準顕在層、潜在層、無関心層）: [AIがヒアリングシートと過去のやり取りから推測]
    商品・サービス名: [AIがヒアリングシートから自動記入]
    構成定義: [AIが過去のやり取りから推測]
    目標CVR（数値目標）: [AIがヒアリングシートから自動記入]
    現状のCVR（数値）: [KPIデータから自動記入]
    現状のCV数: [KPIデータから自動記入]
    現状のCPA: [KPIデータから自動記入]
    想定流入経路（広告媒体）: [AIがヒアリングシートから自動記入]
    上記を正確に把握し、分析の軸とする。
    
    現状分析: 現状のLPを、以下の4つの観点から詳細に分析し、CVRが低い原因を特定する。
    
    2-1. ユーザーフロー分析： ユーザーがLP内でどのように行動しているかを分析し、離脱ポイントや問題点を特定する。
    * 2-1-1. ファーストビュー:
        * AIによる分析:
            * FV離脱率: [Page Statisticsのデータから推測 (e.g. 1 - retention at page 1)]
            * FVのクリック状況: [KPIデータのclick rate等から推測]
            * 最初の見出しは？: [LP ContentのHeadlinesから抽出]
        * 人間による評価 (AIがシミュレート):
            * 上記のデータと、LPのテキスト情報を参考に、FVの課題と改善案を記入してください。
    * 2-1-2. コンテンツの流れ:
        * AIによる分析:
            * 最終CTA到達率: [KPIデータのfinal_cta_rateを使用]
        * 人間による評価 (AIがシミュレート):
            * 提供されたデータ、LPのテキスト情報、ヒアリングシート情報を参考に、コンテンツの流れの課題と改善案を記入してください。
            * 特に、ユーザーが離脱しやすい箇所を推測し、その原因と改善案を記述してください。
    * 2-1-3. CTA:
        * AIによる分析:
            * 最終CTA到達後のアクション率: [KPIデータのconversion_rate / final_cta_rate 等から推測]
        * 人間による評価 (AIがシミュレート):
            * 最終CTA到達後のアクション率が低い場合、その原因を推測し、改善案を記述してください。
            * CTAの数、配置、デザイン、文言は適切か、ヒアリングシートの内容と照らし合わせて評価してください。
            
    2-2. コンテンツ分析： LP内のコンテンツ内容を分析し、ユーザーのニーズや課題に合致しているか、行動を促す要素が備わっているかを特定する。
    * 2-2-1. ヘッドライン:
        * AIによる分析:
            * ヘッドラインの文字数、キーワード含有率: [AIが自動算出]
            * ヘッドラインで用いられている心理効果: [AIが推測]
        * 人間による評価 (AIがシミュレート):
            * ヘッドラインは、ターゲット層の興味関心を引くものになっているか？
            * ヘッドラインは、商品・サービスの価値を適切に表現しているか？
            * ヘッドラインの課題と改善案を記入してください。
    * 2-2-2. ボディコピー:
        * AIによる分析:
            * 各パートの文字数、キーワード含有率: [AIが自動算出]
            * 各パートで活用されている心理効果: [AIが推測]
        * 人間による評価 (AIがシミュレート):
            * ボディコピーは、ターゲット層の課題やニーズに共感し、解決策を提示できているか？
            * 商品・サービスの価値やベネフィットを具体的に説明できているか？
            * アピールポイントを訴求できているか？
            * 一貫性のあるストーリーが展開されているか？
            * ボディコピーの課題と改善案を記入してください。
    * 2-2-3. CTA (Content):
        * AIによる分析:
            * CTAボタンの文言: [LP Contentから抽出]
        * 人間による評価 (AIがシミュレート):
            * CTAは、行動喚起を促す表現になっているか？
            * ユーザーにとって魅力的で、クリックしやすいものになっているか？
            * CTAの課題と改善案を記入してください。
    * 2-2-4. FAQ:
        * AIによる分析:
            * FAQの数、質問と回答の文字数: [LP Contentから推測]
        * 人間による評価 (AIがシミュレート):
            * FAQは、ユーザーの疑問や不安を解消できているか？
            * 購入を後押しする内容になっているか？
            * FAQの課題と改善案を記入してください。
            
    2-3. デザイン・レイアウト分析： (テキスト情報から推測できる範囲で分析)
    * 2-3-1. ファーストビュー & 全体構成:
        * 人間による評価 (AIがシミュレート):
            * (テキスト情報から) 伝えたいメッセージが明確に伝わる構成になっているか？
            * 情報の優先順位は適切か？
            * 課題と改善案を記入してください。
            
    2-4. 市場価値分析： 競合との比較から、自社サービスの位置づけと優位性を評価する。
    * 競合との差別化:
        * 人間による評価 (AIがシミュレート):
            * 競合と比較して、自社サービスの独自性や優位性は明確に打ち出されているか？ (ヒアリングシート情報に基づく)
            * 他社との差別化ポイントは、ターゲットにとって魅力的か？
            * 他社との比較、独自性に関する課題と改善案を記入してください。
            
    2-5. 流入経路分析：
    * 2-5-1. 流入元 & キーワード:
        * AIによる分析:
            * ヒアリングシートから流入元・キーワード情報を抽出
        * 人間による評価 (AIがシミュレート):
            * 各流入元のユーザー属性やニーズは、LPのターゲット層と合致しているか？
            * 流入キーワードは、LPの内容と合致しているか？
            * 広告媒体とLPのターゲット層は一致しているか？
            * 課題と改善案を記入してください。

    原因の特定: 上記の分析結果に基づき、CVRが低い原因を、具体的かつ詳細に特定する。
    
    改善の方向性の提示: 特定された原因に基づき、具体的な改善の方向性を提示する。
    
    **Output Format:**
    Generate the report strictly following the "Output Format" specified in the prompt below.
    
    ## LP現状分析レポート
    
    **基本情報**
    ... (Fill in based on analysis)
    
    **現状分析**
    ... (Follow the structure: 1. User Flow, 2. Content, 3. Design, 4. Market Value, 5. Traffic Sources, 6. Ad Analysis)
    
    **CVRが低い原因（結論）**
    ...
    
    **改善の方向性**
    ...
    
    **IMPORTANT: Output must be in Japanese.**
    """
    return _safe_generate(prompt)
