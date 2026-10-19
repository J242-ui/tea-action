# 🍵 茶园气象服务系统（Tea Garden Weather Service）

基于 **Python + Streamlit** 的轻量化茶园气象服务系统。通过 **和风天气 API** 获取茶园实时与预报气象数据，接入 **百度伏羲/文心大模型**（千帆平台）完成气象灾害研判、病虫害风险分析与农事智能问答；内置 **MySQL 数据库** 支撑用户注册登录与个人茶园管理，并集成了 **风险预警中心** 与 **农事建议** 模块，构建集 **数据监测、风险预警、智能决策、农事指导** 于一体的茶园气象服务系统。

## ✨ 功能特性

- **用户系统**：注册 / 登录 / 退出，密码加盐哈希存储（PBKDF2），游客模式可免登录浏览演示茶园
- **个人茶园**：登录后可添加、编辑、删除自己的茶园，天气与风险分析围绕个人茶园展开
- **气象监测**：实时天气、空气质量（AQI/PM2.5）、未来 7 天预报、生活指数，附趋势图表
- **风险预警**：规则引擎研判气象灾害（低温冻害、高温热害、暴雨渍害、大风、干旱）与茶树病虫害（茶小绿叶蝉、茶饼病、炭疽病、茶蚜、尺蠖等），支持调用大模型做综合研判
- **智能问答**：携带当前气象与风险上下文的农事大模型对话，解答施肥、采摘、打药、灾害防范等问题
- **风险预警（分级机制）**：将每条风险按 低/中/高 归档到 持续观察/提前预防/紧急处置 三档，整园预警档位取最高风险，并按档位给出对应处置动作与预案
- **农事建议**（原智能推送转型）：剥离风险告警，专注基于天气与物候的日常农事操作参考（浇水、施肥、打药、采摘等），按类别筛选并可看历史
- **多茶园**：内置多个产茶区茶园配置（龙井、碧螺春、武夷岩茶、铁观音、信阳毛尖），也可在登录后自助添加
- **智能执行 Agent**：USB 摄像头/演示帧 + YOLO 病害检测 → 自动浇水/自动打药/停止决策（双大模型复核），WebSocket 实时推送执行器动画；支持「持续录像 / 自定义刷新时间」双刷新模式与主站实时画面
- **多摄像头监控台**：可同时接入 USB 摄像头、RTSP 网络摄像头（海康/大华等）、MJPEG/HTTP 视频流，多路画面在动态监控台实时同屏显示，支持运行时添加/删除，与 YOLO 检测、Agent 决策、每日预警全链路打通
- **摄像头监测预警（每日推送）**：风险预警页每日定时（默认 09:00）推送一次「茶园视频监测日报」——当日最新标注画面 + 病害检出情况 + 当日处理动作（灌溉/打药次数），自动落盘可追溯
- **高流畅度性能架构**：摄像头后台线程采集（主线程永不阻塞）、推流与决策双线程解耦、缩略图推流（单帧 ~3KB）、WebSocket 并发广播、天气 10 分钟缓存，杜绝越用越卡
- **演示模式**：未配置 API Key 时自动使用内置演示数据，便于快速体验界面

## 🏗 技术架构

```
┌──────────────────────────────────────────────────┐
│  Streamlit 前端（多页面）                                                                                            │
│  总览 / 气象监测 / 风险预警 / 智能问答 /                                                                      │
│  农事建议 / 我的茶园（用户管理）                                                                                │
└───────────────┬──────────────┬───────────────────┘
                │              │
      ┌─────────▼─────┐  ┌─────▼──────────┐  ┌───────────────┐
      │ 和风天气 API    │  │ 百度千帆(伏羲/文心)│  │   MySQL 数据库  │
      │ 实时/预报/空气  │  │ 灾害研判/农事问答 │  │ 用户/茶园/推送   │
      └───────┬───────┘  └─────┬──────────┘  └───────┬───────┘
              │                │                    │
      ┌───────▼───────────────▼───────────────┐     │
      │  规则引擎风险研判（本地，离线可用）        │─────┘
      └───────────────────────────────────────┘
```

## 📁 项目结构

```
tea-garden-weather-service/
├── app.py                    # 主入口（系统总览，初始化数据库）
├── pages/                    # Streamlit 多页面
│   ├── 1_气象监测.py
│   ├── 2_风险预警.py
│   ├── 3_智能问答.py
│   ├── 4_农事建议.py          # 农事建议：类别筛选 + 参考建议 + 历史
│   ├── 5_我的茶园.py          # 登录用户增删改个人茶园
│   └── 6_智能执行.py          # 🤖 Agent 控制台（新增：启动/停止/单步/状态）
├── src/
│   ├── config.py             # 配置管理（环境变量/茶园数据）
│   ├── weather.py            # 和风天气 API 客户端
│   ├── llm.py                # 大模型客户端（灾害研判/农事问答）
│   ├── risk.py               # 风险研判规则引擎
│   ├── push.py               # 农事建议引擎（生成日常农事建议卡片）
│   ├── db.py                 # MySQL 数据访问层（users/gardens/push_log/...）
│   ├── auth_ui.py            # 登录/注册/退出 UI 组件
│   ├── ui.py                 # 共享 UI 组件 + 游客/登录双模式逻辑
│   ├── agent/                # 🤖 智能执行 Agent（新增）
│   │   ├── decision.py       #   决策引擎：融合视觉+土壤+气象 → 执行指令（含喷雾冷却期）
│   │   ├── llm_decision.py   #   千问大模型复核判断（可同意/调时长/不作业）
│   │   ├── runner.py         #   主循环：推流线程 + 决策线程解耦 → 决策 → 广播
│   │   ├── daily_recorder.py #   每日视频监测汇总（当日帧 + 病害/动作计数）
│   │   └── daily_alert.py    #   摄像头监测预警（每日定时推送日报）
│   ├── sensor/               # 🌱 土壤湿度传感器（新增，模拟+串口接口预留）
│   ├── executor/             # ⚙️ 执行器状态机（新增：滴灌阀门/喷杆喷头）
│   ├── vision/               # 📷 视觉模块（新增：OpenCV采集+YOLO推理+标注）
│   │   ├── capture.py        #   多源采集（USB/RTSP/HTTP）+ 后台采集线程 + 自动降级
│   │   ├── detector.py       #   YOLO 病害检测（缺模型时降级模拟检测）
│   │   └── annotate.py       #   画框标注 + 缩略图/JPEG 编码
│   └── ws_server.py          # 🖥 WebSocket 状态服务（新增，零依赖 ASGI）
├── web/index.html            # 动态展示网页（多摄像头监控台 + 喷杆/阀门动画 + WS 实时推送）
├── training/                 # 🎓 数据集下载与 YOLO 训练（新增）
│   ├── download_data.py      #   下载 figshare 茶叶病害数据集（需网络）
│   └── train_yolo.py         #   YOLOv8 训练（--quick 快速档 / --full 完整档）
├── data/tea_gardens.json     # 系统内置演示茶园配置
├── tests/                    # 单元测试
├── requirements.txt
├── env.example               # 环境变量模板
├── README.md
└── LICENSE.txt               # MIT
```

## 🚀 快速开始

### 1. 环境准备

需要 **Python 3.10 / 3.11**（推荐；3.12+ 部分包可能缺少 wheel）与 **MySQL 5.7 / 8.0**（默认连接本机 root；不配置数据库也能以游客模式体验全部页面）。

#### 完整依赖安装清单

项目全部依赖（含用途）如下：

| 依赖包 | 用途 | 说明 |
| --- | --- | --- |
| `torch`（CPU 版） | YOLO 模型训练/推理底层 | 必装，先装 CPU 版 |
| `ultralytics` | YOLO 病害检测 | 必装 |
| `opencv-python` | USB 摄像头采集与图像编码 | 必装 |
| `streamlit` | 8501 气象主站 | 必装 |
| `uvicorn` / `websockets` | 8510 WebSocket 服务与广播 | 必装 |
| `numpy` | 数值与决策计算 | 必装 |
| `pymysql` | MySQL 数据访问 | 必装 |
| `python-dotenv` | `.env` 配置读取 | 必装 |
| `requests` | 天气 / 大模型 API 调用 | 必装 |
| `pandas` / `plotly` | 页面数据表格与图表 | 必装 |
| `scikit-learn` / `lightgbm` / `statsforecast` / `joblib` | 风险预警三引擎（ML 分类 / 时序 / Bandit） | 必装 |
| `pytest` | 单元测试 | 可选（测试用） |

**安装命令（Windows / Linux 通用）：**

```bash
# 1) 创建独立虚拟环境（推荐，避免污染系统 Python）
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/macOS: source .venv/bin/activate

# 2) 先装 CPU 版 torch（避免 pip 默认拉 2GB+ 的 CUDA 包）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3) 安装其余全部依赖
pip install -r requirements.txt
pip install ultralytics opencv-python
```

> **conda 可选**：依赖均为标准 PyPI 包，与 anaconda 无绑定关系——conda 只是环境管理工具，用 `venv + pip` 完全等价。
> Windows 下如遇 `OMP: Error #15`，启动前设置 `$env:KMP_DUPLICATE_LIB_OK="TRUE"`（项目脚本内已自动处理）。
> 装完可运行 `python smoke_test.py` 自检：18 项全 PASS（尤其 `runner: Agent 主循环单步`、`vision: 采集/检测`、`db: MySQL` 三项）即依赖到位。

### 2. 配置环境变量

复制 `env.example` 为 `.env` 并填写密钥：

```bash
cp env.example .env
```

| 变量 | 说明 |
| --- | --- |
| `QWEATHER_API_KEY` | 和风天气 API Key，[申请地址](https://dev.qweather.com/) |
| `QIANFAN_API_KEY` | **伏羲/文心大模型**（百度千帆）API Key，驱动气象灾害研判、病虫害分析、农事问答，[申请地址](https://console.bce.baidu.com/qianfan/ais/console/onlineService) |
| `QIANFAN_BASE_URL` | 千帆接口地址，默认 `https://qianfan.baidubce.com/v2` |
| `QIANFAN_MODEL` | 千帆模型名称，默认 `ernie-4.0-8k`；使用伏羲气象大模型时改为对应 model |
| `QWEN_API_KEY` | **通义千问**（阿里云百炼）API Key，驱动「智能执行 Agent」的决策判断，[申请地址](https://bailian.console.aliyun.com/) |
| `QWEN_BASE_URL` | 千问接口地址，默认 `https://dashscope.aliyuncs.com/compatible-mode/v1`（OpenAI 兼容） |
| `QWEN_MODEL` | 千问模型名称，默认 `qwen-plus`；可换 `qwen-turbo` / `qwen-max` |
| `DEMO_MODE` | `true` 时使用内置演示数据，无需真实 Key |
| `DEV_MODE` | `true` 时侧边栏显示「开发者控制台」（模型状态 / 一键回测 / 手动重训） |
| `AGENT_WS_HOST` / `AGENT_WS_PORT` | 智能执行动态展示服务监听地址/端口，默认 `127.0.0.1:8510` |
| `AGENT_INTERVAL` | Agent 决策间隔（秒），默认 `5`，最小 `0.5`；可在「智能执行」页运行时调整 |
| `CAMERA_SOURCE` | 默认主摄像头源：`usb:0`（USB 设备号）或 `rtsp://…` / `http://…`（网络摄像头） |
| `ALERT_DAILY_TIME` | 摄像头监测预警每日推送时间（HH:MM），默认 `09:00` |

**本地配置（开发）：**

| 变量 | 说明 |
| --- | --- |
| `DB_HOST` | MySQL 主机，默认 `127.0.0.1` |
| `DB_PORT` | MySQL 端口，默认 `3306` |
| `DB_USER` | MySQL 用户名，默认 `root` |
| `DB_PASSWORD` | MySQL 密码 |
| `DB_NAME` | MySQL 数据库名，默认 `tea_garden` |

**若需要接云服务器：**

| 变量 | 说明 |
| --- | --- |
| `DB_HOST` | 云服务器 IP 或域名 |
| `DB_PORT` | 默认 `3306` |
| `DB_USER` | 数据库账号，不建议用 `root` |
| `DB_PASSWORD` | 密码 |
| `DB_NAME` | `tea_garden` |


> **数据库初始化**：应用启动时自动建表（users / gardens / push_log）。若数据库不存在，请先手动创建：
> ```sql
> CREATE DATABASE tea_garden CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
> ```

> 未配置天气/大模型 Key 时，系统自动进入天气演示模式；不配置数据库也能以游客身份体验所有页面。

### 3. 启动应用

```bash
streamlit run app.py
```

浏览器自动打开 `http://localhost:8501`。

### 4. 运行测试

```bash
# 单元测试
pytest tests/ -v

# 全模块冒烟测试（无需任何配置，两轮自动跑完）
python smoke_test.py                # ROUND 1：无 Key（演示/规则路径）
python smoke_test.py --fake-key     # ROUND 2：假 Key（验证大模型失败回退与页面兜底）

# 端到端验证（需 WS 服务 8510 已在运行）
python e2e_test.py
```

> 冒烟测试覆盖：配置/天气/风险研判/ML分类/时序/三引擎融合/Bandit/大模型客户端/视觉/传感器/执行器/规则决策/决策分发/Agent主循环/数据库/WS服务/首页与各页面编译。

## 🔐 用户与个人茶园

- **游客模式**：未登录时浏览系统内置演示茶园，体验全部功能页面。
- **注册 / 登录**：侧边栏提供注册与登录入口，密码以 PBKDF2 + 随机盐加密存储，不存明文。
- **我的茶园**：登录后在「🌱 我的茶园」页可**添加、编辑、删除**自己的茶园，添加时填写名称、品种、地区、LocationID 或经纬度（二选一）。
- **数据隔离**：登录后侧边栏与各页面只展示当前用户自己的茶园，游客与用户数据互不影响。
## 🧠 三引擎智能风险预警（ML 分类 + 时序 + Contextual Bandit）

在规则引擎之上新增的联动升级，用于更准确的风险预警与推送择优：

```
历史气象 + 茶园环境因子
   ├─ 时序引擎(ts.py)       ：历史窗口特征 + AutoARIMA 未来指标趋势
   ├─ ML分类(ml_risk.py)    ：LightGBM 逐日输出风险等级概率
   └─ 规则引擎(risk.py)     ：兜底，产出具体风险项
             ↓ 融合
   Contextual Bandit(bandit.py / LinUCB)
             ↓ 以环境因子为 context 在线择优
   推送处置动作 → 反馈(采纳/风险是否成真) → 回写 bandit + 落库重训
```

- **数据打底**：首次访问自动为每个茶园生成 180 天模拟历史（`seed_history`），并把每日和风预报归档进 `weather_history` 表；真实数据随使用积累后可重训（`force_retrain=True`）。
- **冷启动兜底**：ML 未训练 / 时序不可用 / 无反馈时，均安全降级到规则引擎，保证任何情况可用。
- **新表**：`weather_history`（逐日气象归档）、`risk_feedback`（推送反馈样本）。
- **页面接入**：`2_风险预警` 顶部「智能融合研判」；`4_农事建议` 顶部「今日智能推荐」均含反馈按钮。
- 新增依赖：`scikit-learn`、`lightgbm`、`statsforecast`、`joblib`。

## 🤖 智能执行 Agent（自动浇水 / 自动打药 / 动态展示）

在气象服务之上新增的**智能执行子系统**，演示完整闭环：

```
USB摄像头/演示帧 → OpenCV → YOLO 推理(病害检测) ─┐
土壤湿度传感器（模拟/串口） ──────────────────┤→ Agent 决策引擎
气象 + 规则引擎风险 ─────────────────────────┘   (滞回防抖 + 连续计数)
                                                   │ 执行指令(浇水/打药/停止)
                                                   ▼
                        WebSocket 状态服务 ──→ 网页动态展示
                                              (喷杆喷头喷雾动画 / 滴灌阀门流水动画)
```

### 功能与决策规则

- **自动浇水（滴灌阀门）**：土壤湿度 < 45% 且未来无大雨 → 开阀滴灌（时长按缺水量估算）；湿度回升到 68% 或出现降水 → 自动关阀（带滞回防抖）；阀门开启期间每轮决策显示「灌溉中」持续状态
- **自动打药（喷杆喷头）**：连续 2 轮检出病害（置信度 ≥ 0.5）且非雨天、风力 < 4 级 → 启动喷雾；连续 3 轮无病害或遇雨/大风 → 自动停止
- **降温喷灌**：气温 ≥ 33°C 且湿度 < 40% 时短时喷灌增湿（复用气象规则）
- 传感器默认**软件模拟**（贴合茶树需水规律），`src/sensor/soil.py` 预留了串口驱动接口；执行器为**状态机模拟**，事件接口可直接转发到 ESP32/继电器驱动真实设备

### 决策判断模式（双大模型分工）

系统接入**两个大模型**，各司其职：

| 职责 | 大模型 | 配置变量 | 无 Key 时 |
| --- | --- | --- | --- |
| **智能执行 Agent 决策**（浇水/打药/停止） | 通义千问 | `QWEN_*` | 完全走规则引擎（防抖/阈值/天气），页面显示模板解释 |
| **气象灾害研判 / 病虫害分析 / 农事问答** | 伏羲（百度千帆） | `QIANFAN_*` | 页面进入演示模式（占位回复） |

**智能执行 Agent 决策流程**：规则引擎先按防抖规则算出候选动作（防线不变）→ 配置了 `QWEN_API_KEY` 时交给**通义千问复核判断**（可同意、可调整时长、可改为不作业，并生成自然语言决策解释推送到网页）；大模型输出经**安全校验**（不得新增规则候选之外的设备动作、时长限幅、JSON 解析失败自动回退规则），防误检触发误操作。入口：`src/agent/llm_decision.py`；决策仍由 `src/agent/decision.py` 规则引擎兜底。

**配置示例（`.env`）：**

```dotenv
# ---- 通义千问（驱动智能执行 Agent 决策判断）----
QWEN_API_KEY=你的千问APIKey
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen-plus

# ---- 百度千帆 · 伏羲大模型（驱动灾害研判 / 病虫害分析 / 农事问答）----
QIANFAN_API_KEY=你的千帆APIKey
QIANFAN_BASE_URL=https://qianfan.baidubce.com/v2
QIANFAN_MODEL=ernie-4.0-8k
```

> 两个 Key 相互独立、可单独配置：只配 `QWEN_API_KEY` → Agent 决策用千问、研判/问答仍是演示；只配 `QIANFAN_API_KEY` → 研判/问答走伏羲、Agent 决策仍是规则。配置后重启 WS 服务（`启动系统.bat`）与 Streamlit 即生效。


### 启动方式

```bash
# 1. 启动动态展示服务（网页 + WebSocket，默认 127.0.0.1:8510）
python -m src.ws_server

# 2. 浏览器打开动态网页
#    http://127.0.0.1:8510/     （执行器动画 + 实时画面）
#    http://localhost:8501      （Streamlit 主站 → 侧边栏「智能执行」控制台）
#    在 Streamlit 控制台点击「启动 Agent 主循环」或在网页 POST /api/agent/start
```

REST API：`GET /api/status`、`GET /api/frame`（最近一帧画面）、`GET /api/history`、`POST /api/agent/start|stop|step|config`（`config` 运行时修改刷新模式/间隔）；WebSocket：`/ws`（`state_update` 完整快照 + `stream_frame` 画面帧）。

### 刷新设置与实时画面

- **双刷新模式**（默认 5 秒，最小 0.5 秒，可在「智能执行」页「⏱ 刷新设置」区运行时切换并应用）：
  - **自定义刷新时间**：整轮（采集→推理→决策→推送）按设定间隔执行
  - **持续录像**：画面约 0.5 秒/帧连续推送（`stream_frame` 事件），决策按间隔节流，画面如监控般流畅
- **主站实时画面**（「智能执行」页「📹 实时画面」开关，默认开启）：
  - **方案 A**：页面内 WebSocket 订阅 `/ws` 缓存画面帧，`@st.fragment(run_every=1.0)` 约 1 秒自动刷新渲染
  - **方案 B**：WS 服务缓存最近一帧（`runner.latest_frame`），`GET /api/frame` 接口兜底拉取，WS 短暂断开时画面仍可用
- **画面来源标识**：网页角标与页面标题实时标注当前帧是「USB摄像头实拍」/「网络摄像头实拍」还是「演示帧（摄像头不可用）」

### 多摄像头源管理（监控台）

动态展示网页（`http://127.0.0.1:8510/`）顶部「＋ 添加摄像头」支持三类源，可运行时添加/删除、多路同屏：

| 类型 | 填写内容 | 示例 |
| --- | --- | --- |
| **USB 摄像头** | 设备号（整数，0 开始） | `0`、`1` |
| **网络摄像头 (RTSP)** | RTSP 地址（海康/大华等监控摄像头） | `rtsp://admin:123456@192.168.1.64:554/Streaming/Channels/102` |
| **MJPEG/HTTP 流** | HTTP 视频流地址 | `http://192.168.1.100:8080/video` |

- 添加的网络摄像头自动进入 YOLO 检测 → Agent 决策 → 推送全链路，与 USB 摄像头一致
- 网络源断线自动 10 秒窗口重连，期间该路降级演示帧，不影响其他路与系统流畅度
- REST：`GET /api/cameras`（源列表）、`POST /api/cameras/add|remove`（增删）；环境变量 `CAMERA_SOURCE` 设定默认主源

### 摄像头监测预警（每日推送）

风险预警页新增「摄像头监测预警」卡片，每天定时（默认 09:00，`ALERT_DAILY_TIME` 可改）推送一次**茶园视频监测日报**：

- **内容**：当日最新标注画面（自动落盘 `data/alerts/`）+ 当日病害检出情况（检出轮数/病害种类）+ 当日处理动作（灌溉/打药次数与详情）
- **数据**：Agent 每轮决策自动写入 `daily_recorder`（当日累计），到点由 `daily_alert` 汇总推送并归档
- **双端展示**：监控台（8510）与风险预警页（8501）均实时渲染

### 🔐 账号绑定与数据隔离

摄像头源与整个项目的历史数据均按账号隔离：

- **摄像头源按账号绑定**：登录用户在数据库 `camera_sources` 表维护自己的源列表（USB/RTSP/HTTP），监控台与智能执行页按 `user_id` 展示；游客（`user_id=0`）无配置时使用环境变量 `CAMERA_SOURCE` 默认源
- **历史数据按账号隔离**：`sensor_log` / `actuator_log` / `agent_decision_log` / `camera_alert_log` 均带 `user_id` 列，旧数据归游客；Agent 决策、传感器、执行器、每日预警的落库与查询全部按当前账号过滤
- **Agent 归属账号**：启动 Agent 时指定 `user_id`（REST `POST /api/agent/start` body 或 8501 控制台自动携带），Agent 运行上下文（茶园、落库、预警）跟随该账号；运行中不可切换账号
- **示例账号**：`admin / 123456`（所有测试账号统一密码 `123456`），系统启动时自动创建/同步
- **游客模式保留**：未登录时全部功能可用，数据归属游客（`user_id=0`）
- **监控台指定账号**：`http://127.0.0.1:8510/?user_id=2`（URL 参数指定账号，默认游客 0）
- **手动建库、系统自动建表**：只需手动执行 `CREATE DATABASE tea_garden`；`camera_sources` 表与各历史表 `user_id` 列由系统启动时自动创建/迁移，无需手动操作
- **物理设备池化**：同一 source（如 `usb:0`）全系统只打开一个采集实例，多账号引用共享，切换账号不抢占、不重复开设备

### 性能与稳定性设计

针对「越用越卡、Agent 启动不了/停不掉」的根因做了专项优化：

- **摄像头后台线程采集**：每个源一个常驻 daemon 采集线程持续刷新帧缓存，主线程读缓存毫秒级返回、永不阻塞在摄像头 IO；摄像头坏句柄只让该源降级，不拖累 Agent
- **推流/决策双线程解耦**：画面推流（0.5s）与完整决策（LLM/天气/写库）互不冻结；决策线程复用推流缓存，不重复读摄像头/跑推理
- **实例级摄像头互斥锁**：同一源任意时刻只允许一个原生调用（构造/读取/释放），杜绝 Windows MSMF 并发构造/释放的原生崩溃（access violation / heap corruption）
- **DSHOW 优先于 MSMF**：DirectShow 在摄像头占用/无设备时失败快速返回、无异步回调崩溃风险
- **缩略图推流**：多路画面以 320px JPEG 缩略图推送（单帧 ~3KB，较全分辨率降 90%+），浏览器解码压力大幅下降
- **WS 并发广播**：慢/死客户端只影响自己；**天气缓存**：10 分钟 TTL，决策每轮不再重复打网络请求

### 病害检测模型（CNN）

- 数据集：**figshare Tea-Leaf Disease and Pest Detection Dataset v1.0**（9,591 张 / 8 类 / YOLO 格式，CC BY 4.0）
- 训练：`python training/download_data.py`（需网络，约 367MB）→ `python training/train_yolo.py --quick`（快速档，产出 `data/models/tea_yolo.pt`）
- **未训练/未装依赖时自动降级**为模拟检测器（`backend=sim`），演示链路完整可用；模型就绪后 `backend=ultralytics` 无缝切换

> ✅ **本机已训练完成**（2026-10）：`data/models/tea_yolo.pt`（快速档，23 轮，验证集 mAP50=0.818）。
> 推理环境需含 torch + ultralytics + opencv（conda 或 venv 皆可，见下「运行环境说明」）；缺少时自动降级为模拟检测。

### 运行环境说明

真实模型推理需要 torch + ultralytics + opencv，可用 **conda 环境** 或 **venv 环境**（二选一，依赖等价，见上文「完整依赖安装清单」）：

```bash
# 方式一：conda 环境（本项目开发机使用）
C:\Users\Lenovo\anaconda3\python.exe training\train_yolo.py --full     # 训练（可选重训）
C:\Users\Lenovo\anaconda3\python.exe -m src.ws_server                  # 启动 8510 动态展示
C:\Users\Lenovo\anaconda3\python.exe -m streamlit run app.py           # 启动 8501 主站

# 方式二：venv 环境（已按上文清单装齐 torch CPU + ultralytics + opencv 后）
python -m src.ws_server
python -m streamlit run app.py
```

> 若环境缺少 torch/ultralytics/opencv，YOLO 检测自动降级为模拟检测器（演示链路仍完整可用），画面角标会显示「演示帧（摄像头不可用）」。

### 新增数据表（自动建表）

`sensor_log`（传感器读数）、`actuator_log`（执行器动作）、`agent_decision_log`（Agent 决策快照）。

## ⛺ 农事建议说明

- **页面定位**：原「智能推送」页转型为专门的日常农事参考建议页，不再混入风险告警（风险内容统一在 [风险预警] 页查看）。
- **类别筛选**：可按 浇水/施肥/打药防治/采摘 四类开关过滤建议。
- **参考建议**：基于实时天气与茶园物候生成非紧急的农事操作建议（含可执行提示），仅作参考、不构成风险告警。
- **历史记录**：登录用户生成的农事建议自动写入数据库，可在页面展开查看最近 20 条。
## 🧠 风险研判说明

- **规则引擎**（本地、离线可用）：依据实时温湿度、风力、逐日预报降水/温度等阈值研判，输出风险等级（低/中/高）与农事建议，覆盖 5 类气象灾害与 7 类常见茶树病虫害。
- **大模型研判**（需千帆 Key）：将实时天气与预报整理为中文气象快照，调用伏羲/文心大模型进行综合研判与农事问答，可作为规则引擎结论的补充与解释。

## 🗺 内置演示茶园

系统内置 5 个演示茶园（龙井、碧螺春、武夷岩茶、铁观音、信阳毛尖），配置位于 `data/tea_gardens.json`，新增条目即可扩展；登录用户则可通过「我的茶园」页面自助添加个人茶园。

```json
{
  "id": "my_garden",
  "name": "我的茶园",
  "location": "省份 城市 区县",
  "location_id": "101010100",
  "lon": 116.4,
  "lat": 39.9,
  "altitude_m": 200,
  "variety": "茶树品种",
  "notes": "备注"
}
```

- `location_id`：和风天气城市 LocationID（可通过 `WeatherClient.search_city("城市名")` 查询）
- 未填写 `location_id` 时，使用 `lon,lat` 作为查询位置

## 📚 项目说明文档

完整的项目说明（整体架构、决策规则、双大模型分工、数据库设计、模型训练、配置与运行、各模块职责）：见 **`项目学习文档.md`**，逐文件解释每个模块的作用与调用关系，适合作为二次开发与部署的参考。

## 📄 许可证

MIT
