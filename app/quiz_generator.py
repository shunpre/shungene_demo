import pandas as pd
import json
import app.ai_analysis as ai_analysis

def generate_quiz(df: pd.DataFrame, difficulty: str, topic: str = None):
    """
    データフレームと難易度に基づいて、3択クイズを10問生成する。
    
    Args:
        df (pd.DataFrame): 分析対象のデータフレーム
        difficulty (str): 難易度 ('初級', '中級', '鬼マネからの出題')
        topic (str, optional): クイズのテーマ（例: 'ABテスト', 'デモグラフィック'）。指定がない場合は全体。
    
    Returns:
        list: クイズのリスト（辞書形式）
    """
    
    # データの要約を作成（AIに渡すため）
    # NumPy型をPythonの標準型に変換してJSONシリアライズエラーを回避
    summary = {
        "total_sessions": int(df['session_id'].nunique()),
        "total_cv": int(df[df['cv_type'].notna()]['session_id'].nunique()),
        "avg_cvr": float((df[df['cv_type'].notna()]['session_id'].nunique() / df['session_id'].nunique()) * 100) if df['session_id'].nunique() > 0 else 0.0,
        "data_duration_days": int((df['event_date'].max() - df['event_date'].min()).days + 1),
        "difficulty_setting": difficulty,
        "topic": topic if topic else "全体（総合問題）"
    }
    
    # トレンド情報の抽出（最初と最後、ピークなど）
    daily_sessions = df.groupby('event_date')['session_id'].nunique()
    summary['trend_start'] = int(daily_sessions.iloc[0])
    summary['trend_end'] = int(daily_sessions.iloc[-1])
    summary['trend_max'] = int(daily_sessions.max())
    summary['trend_min'] = int(daily_sessions.min())

    # プロンプトの切り替え
    if difficulty == "鬼マネからの出題":
        persona = """
        あなたは、Webマーケティングの現場で若手を指導する「鬼教育マネージャー」です。
        以下のWebサイトアクセスデータ（要約）に基づいて、部下の分析スキルを叩き直すための「示唆に富んだ3択クイズ」を10問作成してください。
        """
        tone_instruction = """
        - **口調は厳しく、現場の緊張感を持たせてください**（例：「こんなことも分からないのか？」「プロ失格だ」など）。
        - 解説は、単に正解を教えるだけでなく、**「プロならどう考えるか」**という視点での厳しいアドバイスを含めてください。
        """
    else:
        persona = """
        あなたは、Webマーケティングのプロフェッショナル講師です。
        以下のWebサイトアクセスデータ（要約）に基づいて、学習者の分析スキルを高めるための「示唆に富んだ3択クイズ」を10問作成してください。
        """
        tone_instruction = """
        - **口調は一般的で丁寧な「です・ます」調**としてください。
        - 解説は、正解の理由だけでなく、**「プロの視点」**でのアドバイスを含めてください。
        """

    # トピック別の追加指示
    topic_instruction = ""
    if topic:
        topic_instruction = f"""
        ## 特記事項: テーマ「{topic}」
        - 今回の問題は、特に**「{topic}」**に関連する内容（指標、分析手法、改善施策など）に焦点を当ててください。
        - データの要約から、{topic}に関連する数値を推測または仮定して問題を作成しても構いません。
        """

    # プロンプト作成
    prompt = f"""
    {persona}
    
    ## データ要約
    {json.dumps(summary, ensure_ascii=False, indent=2)}
    
    ## 難易度設定: {difficulty}
    この難易度設定に合わせて、問題のレベルを調整してください。
    
    {topic_instruction}
    
    ## 出題方針（重要）
    - **単なる数字の読み取り問題は禁止**です（例：「セッション数はいくつ？」など）。
    - **「なぜそうなったのか？（要因仮説）」**や**「次になにをすべきか？（アクション）」**を問う問題を重視してください。
    - 現場で実際に起こりうるシチュエーション（トラブル対応、施策の優先順位付けなど）を想定してください。
    {tone_instruction}
    
    ## 難易度別ガイドライン
    - 初級: 基礎用語の理解だけでなく、それがビジネスにどう影響するかを問う。
    - 中級: 複数の指標（CVRとセッション数など）の関係性から、ボトルネックを特定させる。
    - 鬼マネからの出題（上級相当）: 予期せぬデータ変動（急落など）に対して、冷静かつ論理的に原因を切り分ける手順を問う。
    
    ## 出力形式（JSONのみ）
    以下のJSONフォーマットで出力してください。Markdownのコードブロックは不要です。
    [
        {{
            "question": "質問文（シチュエーションを含めること）",
            "options": ["選択肢1", "選択肢2", "選択肢3"],
            "answer": 0, 
            "explanation": "詳細な解説とアドバイス"
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
