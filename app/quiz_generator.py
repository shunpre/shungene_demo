import pandas as pd
import json
import app.ai_analysis as ai_analysis

def generate_quiz(df: pd.DataFrame, difficulty: str):
    """
    データフレームと難易度に基づいて、3択クイズを10問生成する。
    
    Args:
        df (pd.DataFrame): 分析対象のデータフレーム
        difficulty (str): 難易度 ('初級（穏やかな波）', '中級（乱高下）', '上級（急降下）')
    
    Returns:
        list: クイズのリスト（辞書形式）
        [
            {
                "question": "質問文",
                "options": ["選択肢A", "選択肢B", "選択肢C"],
                "answer": 0, # 正解のインデックス (0, 1, 2)
                "explanation": "解説文"
            },
            ...
        ]
    """
    
    # データの要約を作成（AIに渡すため）
    # データの要約を作成（AIに渡すため）
    # NumPy型をPythonの標準型に変換してJSONシリアライズエラーを回避
    summary = {
        "total_sessions": int(df['session_id'].nunique()),
        "total_cv": int(df[df['cv_type'].notna()]['session_id'].nunique()),
        "avg_cvr": float((df[df['cv_type'].notna()]['session_id'].nunique() / df['session_id'].nunique()) * 100) if df['session_id'].nunique() > 0 else 0.0,
        "data_duration_days": int((df['event_date'].max() - df['event_date'].min()).days + 1),
        "difficulty_setting": difficulty
    }
    
    # トレンド情報の抽出（最初と最後、ピークなど）
    daily_sessions = df.groupby('event_date')['session_id'].nunique()
    summary['trend_start'] = int(daily_sessions.iloc[0])
    summary['trend_end'] = int(daily_sessions.iloc[-1])
    summary['trend_max'] = int(daily_sessions.max())
    summary['trend_min'] = int(daily_sessions.min())
    
    # プロンプト作成
    prompt = f"""
    あなたはWebマーケティングのプロフェッショナル講師です。
    以下のWebサイトアクセスデータ（要約）に基づいて、学習者の理解度を確認する「3択クイズ」を10問作成してください。
    
    ## データ要約
    {json.dumps(summary, ensure_ascii=False, indent=2)}
    
    ## 難易度設定: {difficulty}
    この難易度設定に合わせて、問題のレベルを調整してください。
    - 初級: 基礎的な用語（CVR、セッションなど）や、単純な増減の読み取り。
    - 中級: トレンドの変動要因や、ノイズとトレンドの区別に関する応用問題。
    - 上級: 急激な変化（急落など）への対応策や、原因特定に関する実践的な問題。
    
    ## 出力形式（JSONのみ）
    以下のJSONフォーマットで出力してください。Markdownのコードブロックは不要です。
    [
        {{
            "question": "質問文",
            "options": ["選択肢1", "選択肢2", "選択肢3"],
            "answer": 0, 
            "explanation": "正解の解説"
        }},
        ... (計10問)
    ]
    ※ answerは正解の選択肢のインデックス（0始まり）です。
    """
    
    try:
        # AI分析モジュールの生成関数を利用（モデルは共通のものを使用）
        # ここでは簡易的に ai_analysis.generate_content を呼ぶ想定だが、
        # ai_analysis.py に汎用的な生成関数がない場合は、直接モデルを叩く必要があるかも。
        # ai_analysis.py を確認したところ、analyze_... 関数が主。
        # 汎用関数がない場合、ここで定義するか、ai_analysis.py に追加するのが良い。
        # いったん ai_analysis.py に `generate_quiz_content` を追加する形にするか、
        # ここで `google.generativeai` を呼ぶか。
        # 既存の `ai_analysis.py` を再利用するのが綺麗なので、
        # ai_analysis.py に `generate_quiz_from_summary` を追加する方針でいく。
        # そのため、このファイルはラッパーとして機能させる。
        
        response_text = ai_analysis.generate_quiz_content(prompt)
        
        # JSON抽出とパース
        import re
        match = re.search(r"\[.*\]", response_text, re.DOTALL)
        if match:
            quiz_json = json.loads(match.group(0))
            return quiz_json
        else:
            # フォールバック（JSONが見つからない場合）
            return []
            
    except Exception as e:
        print(f"Quiz Generation Error: {e}")
        return []
