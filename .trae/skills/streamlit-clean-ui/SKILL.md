---
name: streamlit-clean-ui
description: "课程笔记问答助手的清爽学术风 UI 设计系统。Invoke when building/editing app.py, the Streamlit page, sidebar, chat UI, tabs, buttons, file uploader, or any .streamlit/config.toml / injected CSS for this project."
---

# Streamlit Clean UI（清爽学术风）

> 适用：本项目 `app.py`（Streamlit 单页应用，侧边栏 + 主区三 tabs + 聊天问答）。
> 目标：干净、克制、可信赖的学习工具感，**信息清晰优先，不花哨**。所有界面代码必须先满足本 skill。

## 一、设计基调

- **关键词**：清爽、留白、学术、可信赖、柔和卡片
- **禁止**：渐变背景、深色霓虹、高饱和撞色、大面积阴影、圆角过大（>16px）、emoji 堆砌、花哨动效
- 中文界面、中文占位提示；字体使用系统无衬线，保证中文显示清晰

## 二、设计令牌（颜色 / 字体 / 圆角）

| 令牌 | 值 | 用途 |
|---|---|---|
| primary | `#3B82F6` | 主按钮、链接、强调、用户气泡 |
| primary-hover | `#2563EB` | 主按钮 hover |
| accent | `#8B5CF6` | AI 助手标识、少量点缀 |
| bg-app | `#F6F8FB` | 应用主背景 |
| bg-sidebar | `#FFFFFF` | 侧边栏背景 |
| bg-card | `#FFFFFF` | 卡片/AI 气泡 |
| bg-user | `#3B82F6` | 用户气泡 |
| bg-soft | `#F1F5F9` | 次级块、标签底 |
| border | `#E5E9F0` | 卡片边框、分隔线 |
| text-main | `#1E293B` | 正文/标题 |
| text-sub | `#64748B` | 次要说明、占位 |
| success | `#16A34A` | 成功提示 |
| warning | `#D97706` | 警告 |
| danger | `#DC2626` | 错误 |
| radius-card | `12px` | 卡片、气泡 |
| radius-btn | `8px` | 按钮、输入框 |
| shadow-card | `0 1px 3px rgba(15,23,42,.06), 0 1px 2px rgba(15,23,42,.04)` | 卡片柔和阴影 |

字体栈：`-apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "Helvetica Neue", Arial, sans-serif`
正文字号 15px、行高 1.7；标题层级 28/22/17px。

## 三、主题配置（必须创建）

创建 `.streamlit/config.toml`，内容：

```toml
[theme]
primaryColor = "#3B82F6"
backgroundColor = "#F6F8FB"
secondaryBackgroundColor = "#FFFFFF"
textColor = "#1E293B"
font = "sans serif"

[server]
maxUploadSize = 50
runOnSave = false

[browser]
gatherUsageStats = false
```

## 四、全局 CSS 注入（在 app.py 顶部，set_page_config 之后立即调用）

新建一个 `inject_css()` 函数（可放在 app.py 内或 ui 模块），用 `st.markdown(..., unsafe_allow_html=True)` 注入下列 CSS。**注意：不同 Streamlit 版本 data-testid 可能微调，若某条不生效，以浏览器实际渲染的 DOM 为准调整，不允许因 CSS 报错中断页面。**

```python
import streamlit as st

def inject_css() -> None:
    st.markdown(
        """
        <style>
        /* ===== 全局字体与背景 ===== */
        html, body, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
                         "Microsoft YaHei", "Helvetica Neue", Arial, sans-serif;
        }
        .stApp { background-color: #F6F8FB; }
        /* 顶部留白收紧、内容最大宽度 */
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1180px; }

        /* ===== 隐藏冗余 chrome，保持干净 ===== */
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        header[data-testid="stHeader"] { background: transparent; }
        [data-testid="stToolbar"] { display: none; }

        /* ===== 侧边栏 ===== */
        section[data-testid="stSidebar"] {
            background: #FFFFFF;
            border-right: 1px solid #E5E9F0;
        }
        section[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }
        section[data-testid="stSidebar"] h1 {
            font-size: 18px; color: #1E293B; margin-bottom: 2px;
        }
        /* 侧边栏分组小标题 */
        .side-label {
            font-size: 12px; font-weight: 600; letter-spacing: .04em;
            color: #64748B; text-transform: uppercase; margin: 14px 0 6px;
        }

        /* ===== 主按钮 ===== */
        .stButton > button, button[data-testid="stBaseButton-primary"] {
            border-radius: 8px; font-weight: 600; padding: 8px 16px;
            transition: all .15s ease;
        }
        button[data-testid="stBaseButton-primary"] {
            background: #3B82F6; border-color: #3B82F6;
        }
        button[data-testid="stBaseButton-primary"]:hover {
            background: #2563EB; border-color: #2563EB;
        }
        /* 次要按钮（划重点/提纲）使用浅底 */
        .stButton > button[kind="secondary"] {
            background: #FFFFFF; border: 1px solid #E5E9F0; color: #334155;
        }
        .stButton > button[kind="secondary"]:hover {
            border-color: #3B82F6; color: #2563EB; background: #F8FAFF;
        }

        /* ===== 文件上传 ===== */
        [data-testid="stFileUploader"] section {
            border: 1.5px dashed #CBD5E1; border-radius: 12px;
            background: #F8FAFC; padding: 8px;
        }
        [data-testid="stFileUploader"] section:hover { border-color: #3B82F6; }

        /* ===== 卡片通用（用 st.container(border=True) 或自定义 class） ===== */
        .ui-card {
            background: #FFFFFF; border: 1px solid #E5E9F0;
            border-radius: 12px; padding: 16px 18px;
            box-shadow: 0 1px 3px rgba(15,23,42,.06);
            margin-bottom: 12px;
        }
        div[data-testid="stExpander"] {
            background: #FFFFFF; border: 1px solid #E5E9F0;
            border-radius: 12px; overflow: hidden;
        }

        /* ===== Tabs ===== */
        .stTabs [data-baseweb="tab-list"] { gap: 6px; }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0; padding: 8px 18px; font-weight: 500;
            color: #64748B;
        }
        .stTabs [aria-selected="true"] {
            color: #2563EB; background: #FFFFFF;
        }
        .stTabs [data-baseweb="tab-highlight"] { background-color: #3B82F6; }

        /* ===== 聊天消息 ===== */
        [data-testid="stChatMessage"] {
            background: transparent; border: none; padding: 10px 0;
        }
        [data-testid="stChatMessageContent"] {
            background: #FFFFFF; border: 1px solid #E5E9F0;
            border-radius: 12px; padding: 12px 16px; line-height: 1.7;
            box-shadow: 0 1px 2px rgba(15,23,42,.04);
        }
        /* 用户消息：右侧蓝底 */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
            [data-testid="stChatMessageContent"] {
            background: #3B82F6; color: #FFFFFF; border-color: #3B82F6;
        }
        /* 助手头像圈点缀 */
        [data-testid="stChatMessageAvatarAssistant"] {
            background: #8B5CF6 !important; color: #fff !important;
        }
        /* 参考来源标签 */
        .src-tag {
            display: inline-block; font-size: 12px; color: #2563EB;
            background: #EFF6FF; border: 1px solid #DBEAFE; border-radius: 999px;
            padding: 2px 10px; margin: 6px 6px 0 0;
        }
        /* AI 回答里的来源区块标题 */
        .src-title { font-size: 12px; color: #64748B; margin-top: 10px; }

        /* ===== 聊天输入框 ===== */
        [data-testid="stChatInput"] {
            border: 1px solid #E5E9F0; border-radius: 12px;
            box-shadow: 0 1px 3px rgba(15,23,42,.05); background: #FFFFFF;
        }

        /* ===== 提示条 ===== */
        [data-testid="stAlert"] { border-radius: 10px; }

        /* ===== 状态徽标（知识库状态） ===== */
        .kb-badge {
            display:inline-block; font-size:12px; font-weight:600;
            border-radius:999px; padding:3px 10px;
        }
        .kb-ok { color:#15803D; background:#F0FDF4; border:1px solid #BBF7D0; }
        .kb-no { color:#B45309; background:#FFFBEB; border:1px solid #FDE68A; }
        </style>
        """,
        unsafe_allow_html=True,
    )
```

## 五、页面布局规范

1. **顶部标题区**：主标题 `# 📚 课程笔记智能问答助手`（仅此处允许一个 emoji）+ 一句浅灰说明（`text-sub`）。
2. **侧边栏**（从上到下，分组用 `.side-label`）：
   - 品牌标题「课程笔记问答」
   - 「知识库」分组：`st.file_uploader`（多文件，pdf/docx/pptx）+ 主按钮「构建知识库」
   - 构建成功后：状态徽标（`.kb-ok` 显示 N 个文本块）+ 次要按钮「一键划重点」「生成复习提纲」
   - 底部：重置会话按钮、使用说明（expander）
3. **主区 `st.tabs(["💬 问答", "✨ 重点", "📝 复习提纲"])`**（tab 标题可用 emoji，正文不堆砌）：
   - 问答 tab：聊天消息流 + `st.chat_input`；知识库未就绪时输入框禁用并在顶部放一条 warning 提示先上传课件。
   - 重点 / 提纲 tab：结果放在 `.ui-card` 卡片里；无结果时居中放浅色空状态提示。
4. 所有主按钮在等待时包裹 `st.spinner`，文案中文且具体（如「正在构建知识库…首次加载本地向量模型可能需要几分钟」）。

## 六、空 / 加载 / 错误状态（必须有，禁止白屏）

- 空状态：浅灰卡片 + 简短引导，例如「还没有上传课件，在左侧上传 PDF/Word/PPT 后点击构建知识库」。
- 加载：`st.spinner`，分阶段文案（解析→切分→向量化→建库）。
- 错误：`st.error` 给中文原因与下一步建议；任何异常都 catch，不允许把英文堆栈直接抛给用户。

## 七、参考来源呈现

回答正文之后，用一条 `.src-title` 写「参考来源：」，每个来源用 `st.markdown('<span class="src-tag">📄 第N页 · 文件名</span>', unsafe_allow_html=True)` 输出；最多展示前 3~5 条，避免刷屏。

## 八、不要做

- 不要引入任何前端构建链 / npm / 大型 JS 框架；只用 Streamlit 组件 + 注入 CSS。
- 不要用 `st.set_page_config(layout="wide")` 之外的自定义 HTML 骨架（保证组件行为正常）。
- 不要改变业务逻辑代码，只做呈现层；CSS 统一由 `inject_css()` 一处注入，不要散落多处。
- 不要用高饱和、渐变、玻璃拟态；不要自定义深色调（本应用固定浅色）。

## 九、交付前 QA 清单

- [ ] `.streamlit/config.toml` 存在且配色与令牌一致
- [ ] 顶部无多余空白、无默认菜单/footer 干扰
- [ ] 侧边栏白底、分组清晰；主按钮蓝色、次要按钮浅底
- [ ] 用户气泡蓝底白字、AI 气泡白底卡片；助手头像为紫色
- [ ] 三个 tab 可切换，问答/重点/提纲各自有内容与空状态
- [ ] 上传→构建→成功徽标→提问→流式回答→来源标签 全流程视觉无错位
- [ ] 知识库未就绪时输入框禁用并有中文提示
- [ ] 窗口缩窄（平板宽度）时布局不溢出、不横向滚动
