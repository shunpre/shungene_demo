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
