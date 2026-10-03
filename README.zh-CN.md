# Pi Web

[English](./README.md) | [日本語](./README.ja.md) | [Русский](./README.ru.md)

[pi 编程智能体](https://github.com/earendil-works/pi)的本地浏览器界面。Pi Web 与 pi 共用本机配置和会话文件，可在浏览器中查找和继续对话、运行智能体、配置模型与资源，并查看项目文件。

**[在线体验演示 →](https://agegr.github.io/pi-web/)**：真实的 Pi Web 界面直接在浏览器里运行，带有示例会话、文件和模型。无需安装；回复都是预设内容，不会调用任何模型。

中文微信群：请查看 [GitHub Discussions 帖子](https://github.com/agegr/pi-web/discussions/271)。

![Pi Web 展示包含结构化 Markdown、工具调用和项目导航的 pi 会话](https://raw.githubusercontent.com/agegr/pi-web/main/docs/screenshot2.png)

## 功能

- **会话工作区**：按项目查找、继续、重命名、导出和删除对话，并查看运行状态、上下文占用、花费和压缩信息。
- **两种分支方式**：**新会话**会从较早的消息创建独立会话文件；**从此处编辑**会在当前会话内创建分支。
- **项目文件工具**：浏览和上传文件、查看 Git Diff，并预览源码、Markdown、图片、音频、PDF 和 DOCX；文件变化后会自动刷新。
- **Git worktree**：从侧边栏切换 checkout，同时把同一仓库不同 worktree 的会话归在一起。
- **网页配置**：无需离开 Pi Web，即可管理 Provider 登录和 API Key、模型、模型测试、插件包及技能。
- **英文、简体中文和繁体中文界面**：Pi Web 首次打开时跟随浏览器语言，也可从顶部栏切换语言。

## 快速开始

Pi Web 要求 Node.js 22.19.0 或更高版本。先用 `node --version` 检查版本，然后运行：

```bash
npx @agegr/pi-web@latest
```

服务就绪后，命令行会尝试自动打开浏览器。如果没有打开，请访问 [http://127.0.0.1:30141](http://127.0.0.1:30141)。Pi Web 默认仅监听 `127.0.0.1`。

如果尚未配置模型 Provider，请打开**模型（Models）**面板登录或添加 API Key。

如需全局安装 `pi-web` 命令：

```bash
npm install -g @agegr/pi-web@latest
pi-web
```

更新前先用 `Ctrl+C` 停止正在运行的进程，再次执行同一条安装命令。卸载时运行 `npm uninstall -g @agegr/pi-web`。

## 配置

端口和主机名以命令行参数为准，优先于对应的环境变量。`--no-open` 与 `PI_WEB_NO_OPEN=1` 中任意一个都会关闭自动打开浏览器。运行 `pi-web --help`（或 `-h`）可打印启动选项并以退出码 0 结束，不会启动服务；未知参数会报错并以退出码 1 结束。

| 参数或环境变量 | 用途 | 默认值 |
| --- | --- | --- |
| `--help`、`-h` | 打印启动选项并退出 | — |
| `--port <端口>`、`-p <端口>` 或 `PORT` | 服务端口 | `30141` |
| `--hostname <主机>`、`-H <主机>` 或 `PI_WEB_HOSTNAME` | 监听主机名 | `127.0.0.1` |
| `--no-open` 或 `PI_WEB_NO_OPEN=1` | 不自动打开浏览器 | 自动打开 |
| `PI_WEB_ALLOWED_HOSTS` | 额外允许的代理或自定义主机名，多个值用逗号分隔，必须精确匹配 | 未设置 |
| `PI_WEB_PASSWORD` | 启用浏览器密码登录；API 客户端可使用用户名为 `pi` 的 Basic Auth | 不启用认证 |

例如：

```bash
pi-web --help
pi-web -p 8080 -H 0.0.0.0 --no-open
```

### 远程访问

监听非回环地址会暴露一个可执行高权限操作的智能体。在可信局域网中使用时，请设置足够长的随机密码：

```bash
PI_WEB_PASSWORD='足够长的随机密码' pi-web --hostname 0.0.0.0
```

密码认证不会加密连接。不要通过明文 HTTP 将 Pi Web 暴露到互联网；远程访问应使用可信反向代理提供 HTTPS，或通过可信 VPN。如果反向代理传递外部主机名，请把该名称精确加入 `PI_WEB_ALLOWED_HOSTS`。这个白名单不会改变 Pi Web 的监听地址。

### HTTP 代理

服务端的模型和 API 请求会读取标准的 `HTTP_PROXY`、`HTTPS_PROXY` 和 `NO_PROXY` 环境变量。

macOS 或 Linux：

```bash
HTTP_PROXY=http://127.0.0.1:7890 \
HTTPS_PROXY=http://127.0.0.1:7890 \
NO_PROXY=localhost,127.0.0.1 \
npx @agegr/pi-web@latest
```

Windows PowerShell：

```powershell
$env:HTTP_PROXY = "http://127.0.0.1:7890"
$env:HTTPS_PROXY = "http://127.0.0.1:7890"
$env:NO_PROXY = "localhost,127.0.0.1"
npx @agegr/pi-web@latest
```

## 注意事项

- **智能体数据**：Pi Web 默认读取 `~/.pi/agent` 下的 pi 数据，包括 `sessions/<编码后的工作目录>/<时间戳>_<uuid>.jsonl` 中的会话文件。可通过 `PI_CODING_AGENT_DIR` 指定其他 pi agent 目录。
- **文件系统访问**：Pi Web 必须能读取智能体数据目录及会话记录中的工作目录。与现有 pi 会话共用数据时，请让 Pi Web 运行在与 pi 相同的文件系统环境中。
- **共享配置**：模型面板使用 pi 的模型、设置和凭据存储，因此两种界面都能看到相关更改。
- **文件访问边界**：文件浏览器仅能访问在 Pi Web 中选择过的工作目录，以及它已识别的项目或会话根目录；它不是通用的文件系统浏览器。
- **Git worktree**：切换器何时显示、如何创建 worktree，以及删除会产生什么影响，见 [Pi Web 里的 Worktree](./docs/worktrees.zh-CN.md)。

## 开发

```bash
npm install
npm run dev
```

开发服务器运行在 [http://127.0.0.1:30141](http://127.0.0.1:30141)。常用检查命令：

```bash
npm test
node_modules/.bin/tsc --noEmit
npm run lint
```

日常开发时不要运行 `next build` 或 `npm run build`。它们会写入 `.next/`，可能干扰开发服务器；仅在发布流程中执行构建。

### Windows：聊天命令 `/update`

在聊天输入框发送 `/update`，确认后，由外部 Windows Web 启动器拉取本仓库的 `origin/gptdot` 并重启。它是内置命令，不会发送给模型，也不使用当前会话的工作目录。

首次装入此功能后，需要在**原来的外部启动窗口**按 `Ctrl+C` 关闭旧启动器，再运行 `.\start-web.cmd` 一次。仅重启 Web 子进程不会升级已经运行的启动器。后续即可使用 `/update`。

- 只接受 `https://github.com/jiah0231/aicpi.git` 的 `origin/gptdot`（HTTPS 地址可省略 `.git`），使用快进更新，不切换分支；会校验 Git `insteadOf` 展开后的实际地址，不接受镜像、凭据、端口或其他协议，也不会改写 `origin`
- 保留不冲突的本地修改及未跟踪文件；冲突、分支分叉、拉取失败都会停止，不会重启，也不会 reset、clean 或 stash
- 重启会中断所有正在执行的任务、待审核内容及内置终端，请先保存工作
- 仅支持启动器的 `dev` 模式；如上游修改依赖文件，请在外部终端手动更新和安装依赖。命令不会安装依赖或运行构建
- “已提交更新请求”不表示已经更新成功。请查看外部窗口，只有 `READY` 表示服务已就绪，然后刷新网页。错误详情在 `.pi-web-run/update.log`
- 重复请求会合并；失败后可排除原因再重试。不要删除本地修改来强行更新

### Windows：终端一键重启

在本仓库目录里，第一次先在**原来启动服务的外部 Windows 终端**按 `Ctrl+C` 停掉旧服务，再运行：

```powershell
.\start-web.cmd
```

保持这个外部窗口打开。它默认启动本仓库的开发服务器，监听 `127.0.0.1:30141`。以后在项目目录下的任意终端（包括网页内置终端）运行：

```powershell
.\restart.cmd
```

等外部窗口显示 `READY`，再刷新原来的网页。重启期间网页内置终端会断开，刷新后重新打开即可。**重启会强制结束该服务及其启动的智能体、终端和子进程，请先结束正在进行的任务并保存工作。**

这次外部启动是必要的一步：内置终端不继承网页登录密码，而且服务退出时会清理内置终端；单独用 `start` 或 `Start-Process` 无法保证重启程序存活。启动器保留外部终端原有的环境变量（包括 `PI_WEB_PASSWORD`、`PI_WEB_ALLOWED_HOSTS` 和代理设置），以相同模式、端口和环境重新启动；不会从旧进程猜启动命令，也不会接管或杀掉已占用端口的未知进程。

使用 Cloudflare 等代理时，沿用原来的密码、隧道和域名白名单设置，再执行 `start-web.cmd`。如果这些变量只在旧启动脚本中设置，把该脚本最后的启动命令换成 `call start-web.cmd`（PowerShell 脚本中用 `& .\start-web.cmd`），保留之前设置变量的部分。不要从网页终端重新创建启动器。重启程序不会改动隧道或域名配置，也不会更改 PowerShell 执行策略、安装服务或计划任务。

可选命令：

```powershell
# 查看是否已就绪（ready 才表示可刷新）
node .\bin\windows-web.mjs status
# 停掉启动器和它管理的服务
node .\bin\windows-web.mjs stop
# 第一次启动时选择其他端口
.\start-web.cmd --port 30142
# 仅适用于已经有生产构建的目录；不会自动构建或更新旧构建
.\start-web.cmd --mode start
```

修改源码后使用默认 `dev` 模式；`start` 模式只运行已有构建。日志在忽略提交的 `.pi-web-run\manager.log` 和 `.pi-web-run\server-*.log` 中。启动器不会把环境变量或密码写入状态文件/管理日志；分享应用日志前仍需检查其中是否有敏感内容。端口被占用、启动失败或 120 秒内未就绪时会明确报错，不会把“请求已提交”当成“重启成功”。若公司设备限制 PowerShell `Add-Type`，请按设备管理要求处理，不要绕过限制。

贡献者文档：[国际化](./docs/i18n.md)和[发布流程](./docs/release.md)。

## 仓库结构

```text
app/             Next.js 界面和 API 路由
components/      React 界面组件
hooks/           客户端状态和交互 hooks
lib/             会话、智能体、模型、文件、Git 和安全逻辑
public/          静态资源和 PWA 文件
bin/             npm CLI 入口及启动参数解析
docs/            面向用户和贡献者的专题文档
demo/            发布到 GitHub Pages 的静态演示站（见 demo/README.md）
```

架构说明和详细文件地图见 [AGENTS.md](./AGENTS.md)。

## 许可证

[MIT](./LICENSE)
