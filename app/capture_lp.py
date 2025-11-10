#!/usr/bin/env python3
"""
LP URLからスクリーンショットやコンテンツ情報を取得する関数
"""
import requests
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
import streamlit as st
import re
import json

@st.cache_data(ttl=3600)  # 1時間キャッシュ
def capture_lp_screenshot(url: str, width: int = 1200, height: int = 1500) -> Image.Image:
    """
    URLからスクリーンショットを取得
    
    Args:
        url: キャプチャするURL
        width: スクリーンショットの幅
        height: スクリーンショットの高さ
    
    Returns:
        PIL Image object
    """
    try:
        # Screenshot APIを使用（無料のサービス）
        # https://api.screenshotmachine.com を使用
        api_url = f"https://api.screenshotmachine.com/?key=demo&url={url}&dimension={width}x{height}"
        
        response = requests.get(api_url, timeout=30)
        
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            return img
        else:
            # エラーの場合はプレースホルダー画像を返す
            return create_placeholder_image(width, height)
    
    except Exception as e:
        print(f"スクリーンショット取得エラー: {e}")
        return create_placeholder_image(width, height)


def create_placeholder_image(width: int, height: int) -> Image.Image:
    """
    プレースホルダー画像を作成
    """
    from PIL import ImageDraw, ImageFont
    
    img = Image.new('RGB', (width, height), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    # テキストを中央に配置
    text = "スクリーンショットを\n取得できませんでした"
    
    # テキストの位置を計算
    text_bbox = draw.textbbox((0, 0), text)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    position = ((width - text_width) // 2, (height - text_height) // 2)
    draw.text(position, text, fill=(150, 150, 150))
    
    return img


@st.cache_data(ttl=3600)
def extract_swipe_lp_images(url: str):
    """
    スワイプLPから各ページの画像を抽出する
    
    このバージョンでは、以下の方法で画像を抽出します:
    1. HTMLから window.lpSettings を抽出
    2. lpSettings から lastPageNum と htmlInsertions を取得
    3. 画像URLのパターンを推測して生成
    4. 動画Pageを認識し、動画URLまたはプレースホルダーを返す
    5. 実際に存在する画像のみを返す
    
    Args:
        url: スワイプLPのURL
    
    Returns:
        画像/動画URLのリスト（辞書形式: {'type': 'image'|'video', 'url': '...'})
    """
    try:
        response = requests.get(url, timeout=30, headers={'User-Agent': 'Mozilla/5.0'})
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # 方法1: window.lpSettings から情報を抽出
        lp_settings = extract_lp_settings(html_content)
        
        if lp_settings:
            images = generate_image_urls_from_settings(url, lp_settings)
            if images:
                return images
        
        # 方法2: 従来の方法（imgタグから抽出）
        images = []
        
        # imgタグから抽出
        for img in soup.find_all('img'):
            src = img.get('src') or img.get('data-src')
            if src and not src.endswith('.svg'):  # SVGは除外
                # 相対パスを絶対パスに変換
                src = convert_to_absolute_url(url, src)
                images.append(src)
        
        # pictureタグから抽出
        for picture in soup.find_all('picture'):
            source = picture.find('source')
            if source:
                srcset = source.get('srcset')
                if srcset:
                    src = srcset.split(',')[0].split(' ')[0]
                    src = convert_to_absolute_url(url, src)
                    if not src.endswith('.svg'):
                        images.append(src)
        
        # 重複を削除
        images = list(dict.fromkeys(images))
        
        return images
    except Exception as e:
        print(f"画像の抽出に失敗しました: {e}")
        return []


def extract_lp_settings(html_content: str) -> dict:
    """
    HTMLから window.lpSettings オブジェクトを抽出
    
    Args:
        html_content: HTMLコンテンツ
    
    Returns:
        lpSettings の辞書、失敗時は None
    """
    try:
        # window.lpSettings = {...}; のパターンを検索
        pattern = r'window\.lpSettings\s*=\s*({.*?});'
        match = re.search(pattern, html_content, re.DOTALL)
        
        if match:
            json_str = match.group(1)
            lp_settings = json.loads(json_str)
            return lp_settings
        
        return None
    except Exception as e:
        print(f"lpSettings の抽出に失敗: {e}")
        return None


def generate_image_urls_from_settings(base_url: str, lp_settings: dict) -> list:
    """
    lpSettings から画像/動画URLのリストを生成
    
    Args:
        base_url: ベーsURL
        lp_settings: lpSettings 辞書
    
    Returns:
        画像/動画URLのリスト（辞書形式）
    """
    try:
        pages = []
        
        # htmlInsertionsから動画ページとカスタムHTMLを抽出
        html_insertions = lp_settings.get('htmlInsertions', {})
        video_pages = {}  # {page_index: video_url}
        html_pages = {}   # {page_index: html_content}
        
        for key, value in html_insertions.items():
            # "0.1": "video: https://..." または "1.1": "html: <div>..."の形式を解析
            page_index = int(float(key))
            
            if 'video:' in value.lower() or '動画:' in value:
                # 動画URLを抽出
                video_url = value.split('video:')[-1].split('動画:')[-1].strip()
                video_pages[page_index] = video_url
            elif 'html:' in value.lower():
                # カスタムHTMLを抽出
                html_content = value.split('html:')[-1].strip()
                html_pages[page_index] = html_content
        
        # 最初のページの画像
        first_image_url = lp_settings.get('firstImageUrl')
        first_page_type = lp_settings.get('firstPageContentType', 'image')
        
        # ページ1が動画かどうかを確認
        if 0 in video_pages:
            # ページ1は動画
            pages.append({'type': 'video', 'url': video_pages[0]})
        elif 0 in html_pages:
            # ページ1はカスタムHTML
            pages.append({'type': 'html', 'content': html_pages[0]})
        elif first_image_url:
            # ページ1は画像
            pages.append({'type': 'image', 'url': first_image_url})
        
        # ページ数を取得
        last_page_num = lp_settings.get('lastPageNum', 0)
        
        if last_page_num > 0 and first_image_url:
            # 画像URLのパターンを推測
            # URLのベース部分とファイル名を分離
            url_parts = first_image_url.rsplit('/', 1)
            if len(url_parts) == 2:
                base_path = url_parts[0]
                first_filename = url_parts[1]
                
                # ファイル名から拡張子を取得
                name_parts = first_filename.rsplit('.', 1)
                if len(name_parts) == 2:
                    extension = name_parts[1]
                    
                    # 画像番号のカウンター（動画ページを除く）
                    # firstImageUrlが01.jpgの場合、カウンターは1から始まる
                    image_counter = int(name_parts[0]) if name_parts[0].isdigit() else 1
                    
                    # ページ1が画像の場合、カウンターは次の画像番号から始める
                    if 0 not in video_pages and 0 not in html_pages:
                        image_counter += 1

                    # 2ページ目以降のページを生成
                    # lastPageNumは画像の最後の番号を意味するため、image_counterがlastPageNumに達するまでループ
                    page_index = 1
                    while image_counter <= last_page_num:
                        if page_index in video_pages:
                            # 動画ページ（画像カウンターはインクリメントしない）
                            pages.append({'type': 'video', 'url': video_pages[page_index]})
                            page_index += 1
                        elif page_index in html_pages:
                            # カスタムHTMLページ（画像カウンターはインクリメントしない）
                            pages.append({'type': 'html', 'content': html_pages[page_index]})
                            page_index += 1
                        else:
                            # 画像ページ
                            # 01, 02, ... または 1, 2, ... の形式に対応
                            if first_filename.startswith('0'):
                                # ゼロパディングあり
                                filename = f"{image_counter:02d}.{extension}"
                            else:
                                # ゼロパディングなし
                                filename = f"{image_counter}.{extension}"
                            
                            image_url = f"{base_path}/{filename}"
                            pages.append({'type': 'image', 'url': image_url})
                            
                            # 画像ページの場合のみカウンターをインクリメント
                            image_counter += 1
                            page_index += 1
        
        # 会社情報ページを最後に追加（このシステムでは必ず最後に1ページの会社情報ページが入る）
        company_info_url = lp_settings.get('companyInfoUrl', '')
        privacy_policy_url = lp_settings.get('privacyPolicyUrl', '')
        sct_law_url = lp_settings.get('sctLawUrl', '')
        
        # 会社情報ページを追加（type='company_info'で識別）
        pages.append({
            'type': 'company_info',
            'urls': {
                'company': company_info_url,
                'privacy': privacy_policy_url,
                'sct_law': sct_law_url
            }
        })
        
        return pages
    
    except Exception as e:
        print(f"画像/動画URL生成エラー: {e}")
        import traceback
        traceback.print_exc()
        return []


def verify_image_exists(url: str, timeout: int = 5) -> bool:
    """
    画像URLが実際に存在するかを確認
    
    Args:
        url: 確認する画像URL
        timeout: タイムアウト秒数
    
    Returns:
        存在する場合 True
    """
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        return response.status_code == 200
    except:
        return False


def convert_to_absolute_url(base_url: str, relative_url: str) -> str:
    """
    相対URLを絶対URLに変換
    
    Args:
        base_url: ベースURL
        relative_url: 相対URL
    
    Returns:
        絶対URL
    """
    if relative_url.startswith('http'):
        return relative_url
    elif relative_url.startswith('/'):
        # ドメインルートからの相対パス
        base_parts = base_url.split('/')
        return f"{base_parts[0]}//{base_parts[2]}{relative_url}"
    else:
        # 現在のディレクトリからの相対パス
        base_parts = base_url.rsplit('/', 1)
        return f"{base_parts[0]}/{relative_url}"


@st.cache_data(ttl=3600)
def extract_lp_text_content(url: str) -> dict:
    """
    LPのURLからテキストコンテンツ（ヘッドライン、ボディコピー、CTA）を抽出する

    Args:
        url: LPのURL

    Returns:
        テキストコンテンツを含む辞書
    """
    try:
        response = requests.get(url, timeout=30, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()  # HTTPエラーチェック
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')

        # ヘッドラインの抽出 (h1, h2タグを優先)
        headlines = [h.get_text(strip=True) for h in soup.find_all(['h1', 'h2'])]
        if not headlines:
            # h1, h2がなければ最初のpタグなどを代替として使うことも可能
            first_p = soup.find('p')
            if first_p:
                headlines.append(first_p.get_text(strip=True))

        # ボディコピーの抽出 (pタグ)
        body_copy = [p.get_text(strip=True) for p in soup.find_all('p') if p.get_text(strip=True)]

        # CTAの抽出 (buttonタグと、'cta'や'btn'クラスを持つaタグ)
        cta_elements = soup.find_all('button')
        cta_elements += soup.find_all('a', class_=re.compile(r'\b(cta|btn|button)\b', re.I))
        
        ctas = [cta.get_text(strip=True) for cta in cta_elements if cta.get_text(strip=True)]
        ctas = list(dict.fromkeys(ctas)) # 重複を削除

        return {
            "headlines": headlines,
            "body_copy": body_copy,
            "ctas": ctas
        }

    except requests.exceptions.RequestException as e:
        print(f"URLからのコンテンツ取得エラー: {e}")
        return {"headlines": [], "body_copy": [], "ctas": []}
    except Exception as e:
        print(f"テキストコンテンツの抽出エラー: {e}")
        return {"headlines": [], "body_copy": [], "ctas": []}
