import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import google.generativeai as genai
from datetime import datetime, timedelta
import json
import time

# -----------------------------------------------------------------------------
# 1. 系統初始化與設定
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI 智能飆股搜尋系統",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. 數據獲取模組 (FinMind API)
# -----------------------------------------------------------------------------
def get_stock_data(symbol, api_token, start_date, end_date, market_type):
    """
    從 FinMind 獲取數據並標準化
    """
    try:
        url = "https://api.finmindtrade.com/api/v4/data"
        dataset = 'TaiwanStockPrice' if market_type == '台股' else 'USStockPrice'
        
        params = {
            "dataset": dataset,
            "data_id": symbol,
            "start_date": start_date.strftime('%Y-%m-%d'),
            "end_date": end_date.strftime('%Y-%m-%d')
        }
        
        headers = {}
        if api_token:
            headers["Authorization"] = f"Bearer {api_token}"

        response = requests.get(url, params=params, headers=headers)
        res_json = response.json()
        
        if res_json.get('msg') != 'success':
            return None, f"API 回傳錯誤: {res_json.get('msg')}"
            
        data = res_json.get('data', [])
        if not data:
            return None, "查無數據"
            
        df = pd.DataFrame(data)
        
        # 欄位標準化
        rename_map = {
            'Trading_Volume': 'volume', 'max': 'high', 'min': 'low',
            'Volume': 'volume', 'Date': 'date', 'Open': 'open',
            'High': 'high', 'Low': 'low', 'Close': 'close'
        }
        df = df.rename(columns=rename_map)
        df.columns = [c.lower() for c in df.columns]
        
        if 'volume' not in df.columns: df['volume'] = 0
        
        df['date'] = pd.to_datetime(df['date'])
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        
        df = df.sort_values('date', ascending=True).reset_index(drop=True)
        return df[['date', 'open', 'high', 'low', 'close', 'volume']], None
        
    except Exception as e:
        return None, f"錯誤: {str(e)}"

# -----------------------------------------------------------------------------
# 3. 技術指標計算 (含評分邏輯)
# -----------------------------------------------------------------------------
def calculate_technical_indicators(df):
    """
    計算 MA, RSI, MACD, 布林通道
    """
    try:
        # MA
        df['MA5'] = df['close'].rolling(window=5).mean()
        df['MA20'] = df['close'].rolling(window=20).mean()
        df['MA60'] = df['close'].rolling(window=60).mean()

        # 布林通道
        std = df['close'].rolling(window=20).std()
        df['BB_Upper'] = df['MA20'] + (std * 2)
        df['BB_Lower'] = df['MA20'] - (std * 2)

        # RSI (14)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        exp12 = df['close'].ewm(span=12, adjust=False).mean()
        exp26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp12 - exp26
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['Hist'] = df['MACD'] - df['Signal']

        return df
    except Exception:
        return df

def calculate_tech_score(df):
    """
    🔥 飆股掃描器核心演算法 (0-100分)
    """
    if len(df) < 30: return 0, "數據不足"
    
    score = 0
    reasons = []
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # 1. RSI 策略 (30%)
    if last['RSI'] < 30:
        score += 30
        reasons.append("RSI超賣")
    elif prev['RSI'] < 30 and last['RSI'] > 30:
        score += 25
        reasons.append("RSI低檔金叉")
    elif last['RSI'] < 45:
        score += 10

    # 2. MACD 策略 (30%)
    if prev['Hist'] < 0 and last['Hist'] > 0:
        score += 30
        reasons.append("MACD翻紅")
    elif last['Hist'] > 0 and last['Hist'] > prev['Hist']:
        score += 15
        reasons.append("多頭增強")

    # 3. 布林通道策略 (30%)
    if last['close'] <= last['BB_Lower'] * 1.02:
        score += 30
        reasons.append("回測布林下軌")
    elif prev['close'] < prev['MA20'] and last['close'] > last['MA20']:
        score += 20
        reasons.append("突破月線")

    # 4. 量能 (10%)
    vol_ma5 = df['volume'].rolling(5).mean().iloc[-1]
    if last['volume'] > vol_ma5 * 1.5:
        score += 10
        reasons.append("爆量")

    return min(100, max(0, score)), ", ".join(reasons)

# -----------------------------------------------------------------------------
# 4. 專業圖表繪製
# -----------------------------------------------------------------------------
def plot_pro_chart(df, symbol):
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f'{symbol} 價量走勢與布林通道', 'RSI 強弱指標', 'MACD 動能指標')
    )

    # K線 + 布林 + MA
    fig.add_trace(go.Candlestick(x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='K線'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['BB_Upper'], line=dict(color='gray', width=1), name='布林上軌', legendgroup='BB', showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['BB_Lower'], line=dict(color='gray', width=1), name='布林下軌', fill='tonexty', fillcolor='rgba(200, 200, 200, 0.1)', legendgroup='BB', showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['MA5'], line=dict(color='orange', width=1), name='MA5'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['MA20'], line=dict(color='purple', width=1), name='MA20'), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df['date'], y=df['RSI'], line=dict(color='#2962FF', width=2), name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # MACD
    colors = np.where(df['Hist'] < 0, '#ef5350', '#26a69a')
    fig.add_trace(go.Bar(x=df['date'], y=df['Hist'], name='MACD柱狀', marker_color=colors), row=3, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['MACD'], line=dict(color='blue', width=1), name='MACD快線'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['Signal'], line=dict(color='orange', width=1), name='MACD慢線'), row=3, col=1)

    fig.update_layout(height=800, xaxis_rangeslider_visible=False, hovermode="x unified", margin=dict(l=30, r=30, t=30, b=30), template="plotly_dark")
    return fig

# -----------------------------------------------------------------------------
# 5. AI 智能決策 (Gemini 2.5 Flash)
# -----------------------------------------------------------------------------
def generate_ai_insights(gemini_key, symbol, df):
    if not gemini_key: return "⚠️ 請輸入 Gemini API Key"

    try:
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel('gemini-2.5-flash')

        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        recent_high = df.tail(30)['high'].max()
        recent_low = df.tail(30)['low'].min()
        
        df_ai = df.tail(20).copy()
        df_ai['date'] = df_ai['date'].dt.strftime('%Y-%m-%d')
        data_json = df_ai.to_json(orient='records', force_ascii=False)

        prompt = f"""
        你是華爾街頂尖交易員。請分析 {symbol} 並給出操作建議。

        ### 關鍵數據 (最新):
        - 收盤: {last_row['close']}
        - RSI(14): {last_row['RSI']:.2f} (前值: {prev_row['RSI']:.2f})
        - MACD柱狀: {last_row['Hist']:.4f}
        - 布林位置: 上 {last_row['BB_Upper']:.2f} / 下 {last_row['BB_Lower']:.2f}
        - 近月高低: {recent_high} / {recent_low}

        ### 近20日數據:
        {data_json}

        ### 請輸出 (繁體中文):
        1. 🎯 **綜合建議**:
           - **可買度評分**: [0-100%]
           - **建議操作**: [積極買進/分批佈局/觀望/賣出]
           - **建議買點**: [具體價格]
           - **建議賣點**: [具體價格]
           - **停損點**: [具體價格]
        
        2. 🔍 **技術支撐**: 簡述 RSI, MACD, 布林狀態。
        3. ⚠️ **風險提示**: 最大隱憂。

        免責聲明: 僅供教學參考。
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 分析失敗: {str(e)}"

# -----------------------------------------------------------------------------
# 6. 主程式 (含掃描器 UI)
# -----------------------------------------------------------------------------
def main():
    # --- 側邊欄：設定區 ---
    st.sidebar.title("🛠️ 系統設定")
    market_type = st.sidebar.selectbox("市場", ["台股", "美股"], index=0)
    
    # 預設代碼
    default_sym = "2330" if market_type == "台股" else "NVDA"
    if 'symbol' not in st.session_state: st.session_state.symbol = default_sym
    
    symbol_input = st.sidebar.text_input("股票代碼", value=st.session_state.symbol).upper()
    
    with st.sidebar.expander("🔑 API 設定"):
        finmind_token = st.text_input("FinMind Token", type="password")
        gemini_key = st.text_input("Gemini Key", type="password")
    
    # 日期
    days_back = st.sidebar.slider("回溯天數", 30, 365, 180)
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days_back)
    
    run_analysis = st.sidebar.button("🚀 執行個股分析", type="primary", use_container_width=True)

    # --- 側邊欄：掃描器區 ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔥 潛力飆股掃描器")
    scan_list_type = st.sidebar.selectbox("掃描清單", ["半導體熱門", "AI 概念", "金融權值", "航運/鋼鐵"])
    
    # 定義掃描清單 (可自行擴充)
    stock_lists = {
        "半導體熱門": ["2330", "2454", "2303", "3711", "3034", "3035", "3443", "6770"],
        "AI 概念": ["2382", "3231", "6669", "2356", "2301", "3017", "2376", "2377"],
        "金融權值": ["2881", "2882", "2891", "2886", "2884", "2890", "5880", "2892"],
        "航運/鋼鐵": ["2603", "2609", "2615", "2002", "2014", "2618"]
    }
    
    run_scan = st.sidebar.button("🔍 開始掃描 (>80分)", use_container_width=True)

    # ==========================
    # 邏輯 A: 執行掃描
    # ==========================
    if run_scan:
        st.title(f"🔥 {scan_list_type} - 強勢股掃描結果")
        target_list = stock_lists[scan_list_type]
        results = []
        
        # 進度條
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, code in enumerate(target_list):
            progress_bar.progress((i + 1) / len(target_list))
            status_text.text(f"正在掃描: {code} ...")
            
            # 取120天數據計算指標
            s_scan = datetime.today() - timedelta(days=120)
            df_scan, _ = get_stock_data(code, finmind_token, s_scan, datetime.today(), "台股")
            
            if df_scan is not None and len(df_scan) > 20:
                df_scan = calculate_technical_indicators(df_scan)
                score, reasons = calculate_tech_score(df_scan)
                
                # 這裡設定分數門檻，建議設 60 以上就能看到東西，80 非常嚴格
                if score >= 60: 
                    last_p = df_scan.iloc[-1]['close']
                    results.append({"代碼": code, "股價": last_p, "評分": score, "訊號": reasons})
            
            time.sleep(0.5) # 避免 API 請求過快
            
        progress_bar.empty()
        status_text.empty()
        
        if results:
            res_df = pd.DataFrame(results).sort_values("評分", ascending=False)
            st.success(f"掃描完成！發現 {len(res_df)} 檔潛力股")
            st.dataframe(res_df.style.background_gradient(subset=['評分'], cmap='Reds'), use_container_width=True)
            
            top_stock = res_df.iloc[0]['代碼']
            st.info(f"💡 建議操作：將 **{top_stock}** 輸入左側欄位，進行 AI 深度分析。")
        else:
            st.warning("⚠️ 掃描完成，但沒有股票符合高分標準 (市場可能偏空)。")

    # ==========================
    # 邏輯 B: 個股分析
    # ==========================
    if run_analysis or (st.session_state.get('analyzed') and not run_scan):
        st.session_state.analyzed = True
        st.session_state.symbol = symbol_input
        
        symbol = st.session_state.symbol
        st.title(f"📈 {symbol} AI 全方位分析")
        
        with st.spinner(f"正在獲取 {symbol} 數據與 AI 運算..."):
            df, error = get_stock_data(symbol, finmind_token, start_date, end_date, market_type)
            
            if error:
                st.error(error)
            else:
                df = calculate_technical_indicators(df)
                
                # 1. 頂部看板
                last = df.iloc[-1]
                change = last['close'] - df.iloc[-2]['close']
                pct = (change / df.iloc[-2]['close']) * 100
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("收盤價", f"{last['close']}", f"{pct:.2f}%")
                c2.metric("RSI (14)", f"{last['RSI']:.1f}")
                macd_txt = "多頭 🔴" if last['Hist'] > 0 else "空頭 🟢"
                c3.metric("MACD", macd_txt)
                score, _ = calculate_tech_score(df)
                c4.metric("技術評分", f"{score} 分")

                st.markdown("---")

                # 2. 圖表
                st.plotly_chart(plot_pro_chart(df, symbol), use_container_width=True)

                # 3. AI 報告
                st.subheader("🤖 AI 操盤手建議")
                if gemini_key:
                    report = generate_ai_insights(gemini_key, symbol, df)
                    st.markdown(report)
                else:
                    st.warning("請輸入 Gemini API Key 以查看買賣建議。")

                with st.expander("查看詳細數據"):
                    st.dataframe(df.sort_values('date', ascending=False).head(50))

if __name__ == "__main__":
    main()