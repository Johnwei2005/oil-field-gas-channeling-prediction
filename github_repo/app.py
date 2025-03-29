import streamlit as st
import streamlit.components.v1 as components
import os

# 设置报告文件路径，请根据实际路径进行修改
report_path = os.path.join("results", "report_20250322_130312.html")

# 读取报告文件内容
with open(report_path, "r", encoding="utf-8") as f:
    html_content = f.read()

st.title("CCUS CO2气窜预测系统报告")
st.markdown("以下为系统生成的预测报告：")

# 嵌入 HTML 报告，调整高度以适应内容
components.html(html_content, height=2000, scrolling=True)

