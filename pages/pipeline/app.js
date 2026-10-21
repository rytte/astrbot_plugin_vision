"use strict";

const $ = id => document.getElementById(id);
const stageTitles = ["视频探测", "场景检测", "两路候选", "候选预合并", "抽帧与压缩", "邻近去重", "最终输出", "模型理解"];
const reasonNames = {boundary: "首尾边界", interval: "固定间隔", scene_change: "场景变化", debug_preview: "调试预览"};
const parameterNames = ["overview_interval", "overview_max_frames", "overview_max_edge", "scene_threshold", "overview_output_mode"];
let bridge, config, session = null, view = -1, busy = false, pollTimer = null, previewTimer = null;
let videoURL = null, selectedVideoSession = null, dialogVersion = 0, pairIndex = 0;
let renderKey = "", activeImages = 0;
const imageCache = new Map(), imageQueue = [], pages = new Map();
const modelDetails = new Map();
let modelDraft = {provider_id: "", question: "请按时间顺序描述视频中可见的事件，并标注实际时间戳。", mode: "frames"};
let modelPreview = null, selectedRuns = [], pendingModelRun = null;

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}
function button(text, action, className = "button") {
  const element = node("button", className, text);
  element.type = "button";
  element.addEventListener("click", action);
  return element;
}
function number(value, digits = 3) { return Number(value || 0).toFixed(digits).replace(/\.?0+$/, "") || "0"; }
function time(value) { return number(value) + " s"; }
function bytes(value) { return value >= 1048576 ? number(value / 1048576, 2) + " MiB" : number(value / 1024, 1) + " KiB"; }
function message(text, error = false) {
  $("message").hidden = !text;
  $("message").textContent = text;
  $("message").className = error ? "error" : "";
}
function remember(id) { try { if (id) localStorage.setItem("vision-debug-session", id); else localStorage.removeItem("vision-debug-session"); } catch {} }
function remembered() { try { return localStorage.getItem("vision-debug-session"); } catch { return null; } }
function params() {
  return Object.fromEntries(parameterNames.map(key => [key, key === "overview_output_mode" ? $(key).value : Number($(key).value)]));
}
function setParams(values) { for (const key of parameterNames) $(key).value = values[key]; }
function note(parent, text, warning = false) { parent.append(node("div", "note" + (warning ? " warning" : ""), text)); }
function stats(parent, values) {
  const container = node("div", "stats");
  for (const [label, value] of values) {
    const item = node("div", "stat");
    item.append(node("div", "stat-label", label), node("div", "stat-value" + (String(value).length > 13 ? " compact" : ""), value));
    container.append(item);
  }
  parent.append(container);
}
async function post(action, extra = {}) {
  return bridge.apiPost("debug/" + action, {csrf: config.csrf, id: session.id, ...extra});
}
async function action(work) {
  if (busy) return;
  busy = true;
  message("");
  updateControls();
  try { await work(); }
  catch (error) { message(error.message || "操作失败，请重试", true); }
  finally { busy = false; updateControls(); }
}
function accept(data, selectLatest = false) {
  if (session?.id === data.id && (data.revision < session.revision || (data.revision === session.revision && data.completed < session.completed))) return;
  const changed = !session || session.id !== data.id || session.revision !== data.revision;
  const advanced = session && data.completed > session.completed;
  session = data;
  remember(data.id);
  if (changed) {
    modelDetails.clear(); modelPreview = null; selectedRuns = []; pendingModelRun = null;
    const previous = data.model.prepared || data.model.runs.at(-1)?.input;
    modelDraft = {provider_id: previous?.provider.id || config.default_provider_id || config.providers[0]?.id || "", question: previous?.question || "请按时间顺序描述视频中可见的事件，并标注实际时间戳。", mode: previous?.mode || data.parameters.overview_output_mode};
    pages.clear();
    imageCache.clear();
    imageQueue.splice(0).forEach(job => job.reject(new Error("测试已切换")));
    setParams(data.parameters);
    pairIndex = 0;
    closeDialog();
  }
  if (changed || selectLatest || advanced) view = data.completed - 1;
  render();
  schedulePoll();
}
function schedulePoll() {
  clearTimeout(pollTimer);
  if (session?.running) pollTimer = setTimeout(poll, 650);
}
async function poll() {
  if (!session || busy) { schedulePoll(); return; }
  const id = session.id;
  try {
    const data = await bridge.apiGet("debug/session", {id});
    if (session?.id === id) accept(data);
  } catch (error) { message(error.message || "状态读取失败", true); pollTimer = setTimeout(poll, 2500); }
}
async function loadSessions() {
  config = await bridge.apiGet("debug/state");
  $("sessions").replaceChildren(node("option", "", "选择测试会话"));
  $("sessions").firstChild.value = "";
  for (const item of config.sessions) {
    const option = node("option", "", item.filename);
    option.value = item.id;
    $("sessions").append(option);
  }
  $("overview_max_edge").max = config.max_edge_limit;
  $("edge-hint").textContent = "64–" + config.max_edge_limit + " px";
  if (session) $("sessions").value = session.id;
}
function updateControls() {
  const running = Boolean(session?.running), ready = Boolean(bridge && config);
  $("next").disabled = !ready || !session || busy || running || (session.done && view === 7);
  $("next").textContent = session?.done ? (view === 7 ? "在下方调用模型" : "进入模型理解 →") : "执行" + stageTitles[session?.completed || 0] + " →";
  $("cancel").hidden = !running && !session?.model.preparing;
  $("cancel").disabled = busy;
  $("delete").disabled = !session || busy;
  $("reset").disabled = !session || busy || running;
  $("video-file").disabled = !ready || busy || running;
  $("sessions").disabled = !ready || busy || running;
  $("drop-zone").classList.toggle("busy", busy || running || !ready);
  for (const key of parameterNames) $(key).disabled = busy || running;
  document.querySelectorAll("#model-form input, #model-form select, #model-form textarea, #model-form button").forEach(element => {element.disabled = busy || running || session?.model.preparing;});
  if ($("model-call")) $("model-call").disabled = busy || running || session?.model.preparing || !previewMatches();
}
function render() {
  updateControls();
  $("file-name").textContent = session ? session.filename + " · " + bytes(session.size) : "还没有上传视频";
  $("source-video").hidden = !videoURL || selectedVideoSession !== session?.id;
  if (session) $("sessions").value = session.id;
  $("stages").replaceChildren();
  stageTitles.forEach((title, index) => {
    const completed = Boolean(session && (index < session.completed || (index === 7 && session.model.runs.some(run => run.status === "succeeded"))));
    const element = button("", () => {view = index; render();}, "stage" + (completed ? " completed" : "") + (index === view ? " active" : ""));
    element.disabled = !(completed || (index === 7 && session?.done));
    element.setAttribute("aria-current", index === view ? "step" : "false");
    element.append(node("span", "number", completed ? "✓" : String(index + 1)), node("span", "", title));
    $("stages").append(element);
  });
  const status = $("run-status");
  status.hidden = !session?.running && !session?.error;
  status.className = "run-status" + (session?.running ? " running" : " error");
  status.textContent = session?.running ? "正在执行「" + (session.operation === "model" ? "模型理解" : stageTitles[session.completed]) + "」…" : session?.error || "";
  if (session) {
    $("timings").replaceChildren(node("span", "", "处理累计 " + time(session.processing_seconds)), node("span", "", "额外预览累计 " + time(session.preview_seconds)), node("span", "", "闲置 " + number(config.ttl_seconds / 60, 1) + " 分钟后清理"));
  }
  const key = [session?.id, session?.revision, session?.completed, view, view === 7 ? session?.model.version : ""].join("/");
  if (key === renderKey) return;
  renderKey = key;
  observer.disconnect();
  if (!session || view < 0) {
    $("stage-number").textContent = "准备开始";
    $("stage-title").textContent = session ? "视频已上传，开始探测" : "从一个视频开始";
    if (session) {
      $("content").replaceChildren();
      note($("content"), "视频已上传。点击右上方按钮执行视频探测，之后每次点击只运行下一个阶段。首次使用可先修改左侧参数并应用。");
    }
    return;
  }
  $("stage-number").textContent = "阶段 " + (view + 1) + " / 8 · " + (view === 7 ? "按需调用" : time(session.stages[view].seconds));
  $("stage-title").textContent = stageTitles[view];
  const content = $("content");
  content.replaceChildren();
  for (const warning of session.warnings) note(content, warning, true);
  [renderProbe, renderScenes, renderCandidates, renderMerge, renderExtract, renderDedup, renderOutput, renderModel][view](content);
  updateControls();
}
function renderProbe(parent) {
  const info = session.info;
  stats(parent, [["视频时长", time(info.duration_sec)], ["源分辨率", info.source_width + " × " + info.source_height], ["平均帧率", number(info.source_fps, 2) + " fps"], ["视频大小", bytes(session.size)]]);
  parent.append(node("h3", "", "三种尺寸，各有用途"));
  const tech = node("div", "tech-grid");
  for (const [title, description] of [["≤160 × 160", "场景检测：缩小后逐帧计算变化分数，保持比例。"], ["长边 ≤" + session.parameters.overview_max_edge, "模型图片：候选预合并后正式抽帧，缩放并编码为 JPEG（q=3）。"], ["32 × 18 灰度", "邻近去重：从正式输出图生成指纹，计算归一化像素差异。"]]) {
    const item = node("div", "tech-item");
    item.append(node("strong", "", title), node("span", "", description)); tech.append(item);
  }
  parent.append(tech);
  note(parent, "本次参数：间隔 " + time(session.parameters.overview_interval) + " · 预算 " + session.parameters.overview_max_frames + " 帧 · 场景阈值 " + session.parameters.scene_threshold + "。长视频会自动增大有效采样间隔，保留全片覆盖。");
}
function svgNode(tag, attributes, text) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [name, value] of Object.entries(attributes || {})) element.setAttribute(name, value);
  if (text !== undefined) element.textContent = text;
  return element;
}
function curve(parent) {
  const trace = session.scene_trace, width = 900, height = 220, left = 45, right = 875, top = 18, bottom = 183;
  const svg = svgNode("svg", {viewBox: "0 0 " + width + " " + height, class: "chart", role: "img", "aria-label": "场景变化分数随视频时间的曲线，虚线为阈值"});
  const x = t => left + t / session.info.duration_sec * (right - left), y = s => bottom - Math.min(1, Math.max(0, s)) * (bottom - top);
  for (const score of [0, 0.5, 1]) {
    svg.append(svgNode("line", {x1: left, y1: y(score), x2: right, y2: y(score), stroke: "var(--line)"}), svgNode("text", {x: 10, y: y(score) + 4}, score));
  }
  const points = trace.curve || [];
  if (points.length) svg.append(svgNode("polyline", {points: points.map(p => x(p.timestamp) + "," + y(p.score)).join(" "), fill: "none", stroke: "var(--accent)", "stroke-width": 1.6}));
  const threshold = session.parameters.scene_threshold;
  svg.append(svgNode("line", {x1: left, x2: right, y1: y(threshold), y2: y(threshold), stroke: "var(--amber)", "stroke-dasharray": "5 4"}), svgNode("text", {x: left + 5, y: y(threshold) - 5}, "阈值 " + threshold));
  for (const ratio of [0, 0.25, 0.5, 0.75, 1]) svg.append(svgNode("text", {x: x(ratio * session.info.duration_sec), y: 207, "text-anchor": ratio === 1 ? "end" : "start"}, time(ratio * session.info.duration_sec)));
  parent.append(svg);
}
function renderScenes(parent) {
  const trace = session.scene_trace;
  if (trace.error) note(parent, trace.error, true);
  stats(parent, [["扫描帧数", trace.scanned_frames || 0], ["超过阈值", trace.threshold_hits || 0], ["窗口保留", session.scenes.length], ["筛选窗口", time(trace.bucket_sec)]]);
  curve(parent);
  note(parent, "检测先按比例缩到不超过 160 × 160，再计算 scene 分数。每个窗口保留最高分候选。" + (trace.curve_aggregated ? "曲线按最多 2000 个时间窗取峰值，避免长视频产生过多点。" : "曲线显示逐帧变化分数。") + (trace.cache_hit ? " 本阶段使用相同参数的诊断缓存。" : ""));
  if (session.scenes.length) gallery(parent, "检测尺寸示意", session.scenes, "scene", "detector", 8);
  note(parent, "检测尺寸示意由候选时刻额外抽帧生成，并经过 JPEG 编码；用于观察缩小后的可见细节。", false);
  if (trace.hits?.length) table(parent, "阈值候选与窗口筛选", ["时间", "变化分数", "窗口结果"], trace.hits.map(hit => [time(hit.timestamp), number(hit.score, 6), hit.selected ? "窗口入选" : "淘汰：同窗口有更高分候选"]));
  if (trace.omitted_hits) note(parent, "阈值候选表仅记录前 4000 条，另有 " + trace.omitted_hits + " 条；窗口入选结果和正式流水线仍包含全部筛选结果。");
}
function timeline(parent) {
  const svg = svgNode("svg", {viewBox: "0 0 900 140", class: "timeline", role: "img", "aria-label": "共用视频时间轴：上排固定与首尾候选，下排场景候选"});
  const left = 85, width = 785, duration = session.info.duration_sec;
  [[session.fixed, "固定 / 首尾", 43, "var(--accent)"], [session.scenes, "场景变化", 89, "var(--green)"]].forEach(([items, title, y, color]) => {
    svg.append(svgNode("text", {x: 10, y: y + 4}, title), svgNode("line", {x1: left, x2: left + width, y1: y, y2: y, stroke: "var(--line)"}));
    for (const item of items) {
      const dot = svgNode("circle", {cx: left + item.timestamp / duration * width, cy: y, r: 3.5, fill: color});
      dot.append(svgNode("title", {}, time(item.timestamp) + " · " + reasonNames[item.reason]));
      dot.addEventListener("click", () => seek(item.timestamp));
      svg.append(dot);
    }
  });
  for (const ratio of [0, 0.25, 0.5, 0.75, 1]) svg.append(svgNode("text", {x: left + ratio * width, y: 125, "text-anchor": ratio === 1 ? "end" : "start"}, time(ratio * duration)));
  parent.append(svg);
}
function renderCandidates(parent) {
  stats(parent, [["固定及首尾", session.fixed.length + " 帧"], ["场景候选", session.scenes.length + " 帧"], ["有效采样间隔", time(session.effective_interval_sec)]]);
  timeline(parent);
  note(parent, "下面两排为额外生成的候选预览（长边 ≤240 px），本阶段不进行正式抽帧。场景排已完成检测阶段的窗口筛选。小图按需加载，点击可看时间信息并定位源视频。");
  gallery(parent, "上排 · 固定间隔与首尾", session.fixed, "fixed");
  gallery(parent, "下排 · 场景变化", session.scenes, "scene");
}
function renderMerge(parent) {
  stats(parent, [["预合并输入", session.fixed.length + session.scenes.length], ["合并后候选", session.candidates.length], ["帧数预算", session.parameters.overview_max_frames], ["有效间隔", time(session.effective_interval_sec)]]);
  if (session.effective_interval_sec > session.parameters.overview_interval + 0.000001) note(parent, "固定采样已用满预算，有效间隔已增大。额外场景候选可能因预算被舍弃；时间接近的场景仍可更新已有候选标记。", true);
  table(parent, "场景候选的预合并记录", ["时间", "分数", "处理", "原因"], session.merge_trace.map(item => [time(item.timestamp), number(item.score, 5), {added: "加入", matched: "时间近重合", budget: "预算舍弃", window: "窗口舍弃", outside: "越界舍弃"}[item.action], item.explanation]));
  gallery(parent, "排序后的候选预览", session.candidates, "merged");
  note(parent, "本阶段只决定正式抽取哪些时间点。近重合阈值为 0.04 秒，画面相似度比较在正式抽帧后执行。");
}
function renderExtract(parent) {
  const total = session.frames.reduce((sum, item) => sum + item.bytes, 0);
  stats(parent, [["正式抽帧", session.frames.length + " 帧"], ["长边上限", session.parameters.overview_max_edge + " px"], ["JPEG 总大小", bytes(total)], ["指纹尺寸", "32 × 18"]]);
  note(parent, "缩放保持宽高比，不放大小图，然后编码 JPEG（FFmpeg q=3）。点击任意帧可查看实际输出、灰度指纹，并按需加载原分辨率 JPEG 对照。以实际时间戳为准，尾部可能回退到可读取的帧。");
  gallery(parent, "抽帧与压缩结果", session.frames, "frame");
  table(parent, "逐帧输出信息", ["请求 → 实际", "输出尺寸", "JPEG", "来源"], session.frames.map(item => [time(item.requested_sec) + " → " + time(item.timestamp_sec), item.width + " × " + item.height, bytes(item.bytes), reasonNames[item.reason]]));
}
function renderDedup(parent) {
  stats(parent, [["输入帧", session.frames.length], ["保留帧", session.kept_indices.length], ["去掉帧", session.frames.length - session.kept_indices.length], ["比较次数", session.dedup_trace.length]]);
  note(parent, "按实际时间顺序，将当前帧与上一张保留帧比较。指纹差异 ≤0.035 且时间足够近时，结合质量、场景来源和时间覆盖决定取舍。质量分是对比度与边缘强度的启发式分数。");
  if (!session.dedup_trace.length) { parent.append(node("p", "muted", "只有一张输出帧，无需邻近比较。")); return; }
  const controls = node("div", "pair-controls"), slider = node("input"), label = node("span", "small");
  slider.type = "range"; slider.min = 0; slider.max = session.dedup_trace.length - 1; slider.value = Math.min(pairIndex, Number(slider.max));
  slider.setAttribute("aria-label", "选择去重比较步骤");
  const body = node("div");
  const draw = () => { pairIndex = Number(slider.value); label.textContent = (pairIndex + 1) + " / " + session.dedup_trace.length; drawPair(body, session.dedup_trace[pairIndex]); };
  slider.addEventListener("input", draw);
  controls.append(button("←", () => {slider.value = Math.max(0, Number(slider.value) - 1); draw();}), slider, label, button("→", () => {slider.value = Math.min(Number(slider.max), Number(slider.value) + 1); draw();}));
  parent.append(controls, body); draw();
}
function drawPair(parent, item) {
  parent.querySelectorAll(".frame-image").forEach(element => observer.unobserve(element));
  parent.replaceChildren();
  const pair = node("div", "pair");
  for (const [title, index, dropped] of [["上一张保留帧", item.previous_index, item.action === "replace_previous"], ["当前帧", item.current_index, item.action === "drop_current"]]) {
    const side = node("div"); side.append(node("h3", "", title + (dropped ? " · 本次删除" : " · 本次保留")));
    const card = frameCard(session.frames[index], "frame", index);
    if (dropped) card.classList.add("dropped");
    side.append(card); pair.append(side);
  }
  const verdict = node("div", "verdict");
  verdict.append(node("strong", "", {keep_both: "保留两帧", replace_previous: "当前帧替换上一帧", drop_current: "保留上一帧，删除当前帧"}[item.action]), node("p", "", item.explanation));
  parent.append(pair, verdict);
  stats(parent, [["画面差异 / 阈值", number(item.distance, 5) + " / 0.035"], ["时间差 / 邻近上限", time(item.gap_sec) + " / " + time(item.nearby_limit_sec)], ["上一帧 / 当前质量", number(item.previous_quality, 2) + " / " + number(item.current_quality, 2)], ["覆盖允许最大间隔", time(item.coverage_limit_sec)]]);
  if (item.proposed_max_gap_sec !== null) note(parent, "尝试合并后的最大时间空隙：" + time(item.proposed_max_gap_sec) + (item.same_frame ? "。两次请求落在同一实际帧。" : "。超过覆盖上限时，即使画面相似也保留两帧。"));
}
function renderOutput(parent) {
  const result = session.result, sampling = result.sampling;
  stats(parent, [["最终输出", sampling.returned_count + " 帧"], ["最大时间空隙", time(sampling.actual_max_gap_sec)], ["图片总体积", bytes(result.total_image_bytes)], ["流水线耗时", time(session.processing_seconds)]]);
  const sources = Object.entries(sampling.reason_counts).map(([key, count]) => (reasonNames[key] || key) + " " + count + " 帧").join(" · ");
  note(parent, sources + "。本阶段只生成图片；需要测试模型时，进入下一阶段“模型理解”。");
  const presentation = result.presentation;
  stats(parent, [["概览输出模式", presentation.mode === "grid" ? "九宫格" : "独立帧"], ["实际发送图片", presentation.image_count + " 张"], ["输出图片体积", bytes(presentation.total_bytes)], ["输出总像素", number(presentation.total_pixels / 1000000, 2) + " MP"]]);
  if (presentation.mode === "grid") {
    note(parent, "每组最多 9 帧，按从左到右、从上到下阅读。拼接保留正式抽帧后的像素尺寸，以 PNG 无损保存，编号与实际时间戳位于画面外；不足 9 帧紧凑排列。下方缩略图仅供预览，点击可按原始像素查看发送图片。");
    gallery(parent, "九宫格输出", presentation.images, "grid", "thumb", 6);
    note(parent, "图片数量减少不代表视觉 token 一定减少。下一阶段可以复用同一批帧，分别测试两种输出并查看模型返回的用量。详细回看仍返回独立帧。");
  }
  const items = session.kept_indices.map(index => ({...session.frames[index], assetIndex: index}));
  gallery(parent, "最终保留帧", items, "frame");
  const actions = node("div", "inline-actions"); actions.append(button("导出调试记录 JSON", exportRecord)); parent.append(actions);
  note(parent, "稀疏采样只能帮助定位。采样帧中没有出现某事件，不能证明事件没有发生。");
}
function previewMatches() {
  return modelPreview && modelPreview.id === session?.model.prepared?.id && modelPreview.provider.id === modelDraft.provider_id && modelPreview.question === modelDraft.question && modelPreview.mode === modelDraft.mode;
}
function promptDetails(parent, input) {
  const details = node("details", "model-prompts"); details.append(node("summary", "", "查看完整提示词与图片索引"));
  details.append(node("h3", "", "系统提示词"), node("pre", "model-text", input.system_prompt), node("h3", "", "用户提示词（含时间戳和帧索引）"), node("pre", "model-text", input.prompt));
  parent.append(details);
}
function renderModel(parent) {
  note(parent, "复用已完成的抽帧结果。选择模型、问题和图片模式后，先预览输入，再单独点击调用模型。切换独立帧或九宫格不重新抽帧；提示词和调用入口与聊天中的指定模型概览一致。");
  const form = node("form", "model-form"); form.id = "model-form";
  const row = node("div", "model-form-row"), providerField = node("label", "field", "看图模型"), provider = node("select"); provider.id = "model-provider"; provider.required = true;
  provider.append(new Option("请选择模型", ""));
  for (const item of config.providers) provider.append(new Option(item.id + (item.model ? " · " + item.model : "") + (item.vision_confirmed ? "" : "（图片能力未标注）"), item.id));
  provider.value = modelDraft.provider_id; providerField.append(provider);
  const modeField = node("label", "field", "本次图片模式"), mode = node("select"); mode.id = "model-mode"; mode.append(new Option("独立帧", "frames"), new Option("九宫格", "grid")); mode.value = modelDraft.mode; modeField.append(mode); row.append(providerField, modeField);
  const questionField = node("label", "field", "测试问题"), question = node("textarea"); question.id = "model-question"; question.rows = 4; question.maxLength = 4000; question.required = true; question.value = modelDraft.question; questionField.append(question);
  const onChange = () => {modelDraft = {provider_id: provider.value, mode: mode.value, question: question.value}; updateControls();};
  provider.addEventListener("change", onChange); mode.addEventListener("change", onChange); question.addEventListener("input", onChange);
  const controls = node("div", "inline-actions"), preview = node("button", "button", "预览模型输入"); preview.type = "submit";
  const call = button("调用模型", () => action(startModel), "button primary"); call.id = "model-call";
  controls.append(preview, call, button("刷新模型列表", () => action(async () => {await loadSessions(); renderKey = ""; render();})));
  form.append(row, questionField, controls); parent.append(form);
  form.addEventListener("submit", event => {event.preventDefault(); if (!form.reportValidity()) return; action(async () => {
    modelPreview = null; pendingModelRun = null;
    let prepared;
    try {prepared = await post("model-prepare", {revision: session.revision, ...modelDraft});}
    catch (error) {try {accept(await bridge.apiGet("debug/session", {id: session.id}));} catch {} throw error;}
    modelPreview = prepared; pendingModelRun = null;
    accept(await bridge.apiGet("debug/session", {id: session.id})); renderKey = ""; render();
  });});
  if (!config.providers.length) note(parent, "没有可选择的看图模型。请在 AstrBot 配置并启用支持图片输入的模型，再刷新模型列表。", true);
  note(parent, "只有点击“调用模型”才会提交图片。用量以供应商通过 AstrBot 返回的数据为准，不估算图片专属 token。每次调用单独计时和超时；取消后保留本地帧和调用记录。");
  if (previewMatches()) {
    const p = modelPreview.presentation;
    stats(parent, [["本次输入帧", p.frame_count + " 帧"], ["发送图片", p.image_count + " 张"], ["输入图片体积", bytes(p.total_bytes)], ["输入总像素", number(p.total_pixels / 1000000, 2) + " MP"]]);
    promptDetails(parent, modelPreview);
    gallery(parent, "本次实际输入图片", p.images, "model:" + modelPreview.id, "thumb", 6);
  } else if (session.model.prepared) {
    parent.append(button("加载已准备的输入预览", () => action(async () => {
      const detail = await bridge.apiGet("debug/model-record", {id: session.id, record: session.model.prepared.id});
      modelPreview = detail.input; modelDraft = {provider_id: detail.input.provider.id, question: detail.input.question, mode: detail.input.mode}; renderKey = ""; render();
    })));
    note(parent, "更换模型、问题或图片模式后，请重新预览输入。");
  }
  renderModelHistory(parent);
  const actions = node("div", "inline-actions"); actions.append(button("导出全部调试与模型记录 JSON", exportRecord)); parent.append(actions);
}
async function startModel() {
  if (!previewMatches()) throw new Error("请先预览当前模型、问题和图片模式的输入。");
  if (!pendingModelRun || pendingModelRun.preview_id !== modelPreview.id) pendingModelRun = {preview_id: modelPreview.id, run_id: Array.from(crypto.getRandomValues(new Uint8Array(16)), byte => byte.toString(16).padStart(2, "0")).join("")};
  const run = pendingModelRun;
  const data = await post("model-run", {revision: session.revision, ...run});
  selectedRuns = [...selectedRuns.filter(id => id !== run.run_id), run.run_id].slice(-2);
  pendingModelRun = null; accept(data); renderKey = ""; render();
}
function renderModelHistory(parent) {
  const runs = session.model.runs;
  parent.append(node("h3", "", "调用记录 · " + runs.length + " / " + session.model.max_runs));
  if (!runs.length) {parent.append(node("p", "muted", "还没有调用模型。完成输入预览后即可开始。")); return;}
  if (!selectedRuns.length) selectedRuns = runs.slice(-2).map(run => run.id);
  const list = node("div", "model-run-list");
  for (const [index, run] of runs.entries()) {
    const label = node("label", "model-run-choice"), checkbox = node("input"); checkbox.type = "checkbox"; checkbox.checked = selectedRuns.includes(run.id);
    checkbox.addEventListener("change", () => {selectedRuns = checkbox.checked ? [...selectedRuns.filter(id => id !== run.id), run.id].slice(-2) : selectedRuns.filter(id => id !== run.id); renderKey = ""; render();});
    label.append(checkbox, node("span", "", "#" + (index + 1) + " · " + run.input.provider.id + " · " + (run.input.mode === "grid" ? "九宫格" : "独立帧") + " · " + modelStatus(run.status))); list.append(label);
  }
  parent.append(list);
  note(parent, "选择最多两条记录并排比较。问题、模型和帧相同时，更适合观察两种图片模式的差别。重跑本地流水线会清空本次调用记录，可先导出保留。");
  const comparison = node("div", "model-comparison"); parent.append(comparison);
  const capture = {id: session.id, revision: session.revision};
  for (const run of runs.filter(item => selectedRuns.includes(item.id))) {
    const card = node("section", "model-result"); comparison.append(card);
    card.append(node("h3", "", run.input.provider.id + " · " + (run.input.mode === "grid" ? "九宫格" : "独立帧")), node("p", "small muted", new Date(run.created_at).toLocaleString("zh-CN", {hour12: false}) + " · " + modelStatus(run.status)));
    stats(card, [["调用耗时", time(run.seconds)], ["发送图片", run.input.image_count + " 张"], ["输入 token", run.usage?.input ?? "未提供"], ["输出 token", run.usage?.output ?? "未提供"], ["缓存输入 token", run.usage?.input_cached ?? "未提供"], ["总 token", run.usage?.total ?? "未提供"]]);
    card.append(node("p", "model-question", run.input.question));
    if (run.error) note(card, run.error, true);
    if (["queued", "running"].includes(run.status)) {card.append(node("p", "muted", "等待模型返回…")); continue;}
    const body = node("div"); card.append(body); body.append(node("p", "muted", "加载记录…"));
    if (!modelDetails.has(run.id)) modelDetails.set(run.id, bridge.apiGet("debug/model-record", {id: capture.id, record: run.id}));
    modelDetails.get(run.id).then(detail => {
      if (!body.isConnected || session?.id !== capture.id || session?.revision !== capture.revision) return;
      body.replaceChildren(node("pre", "model-text model-answer", detail.answer || "本次没有返回有效回答。"));
      promptDetails(body, detail.input);
      const pictures = button("查看本次输入图片", () => {pictures.disabled = true; gallery(body, "记录中的输入图片", detail.input.presentation.images, "model:" + run.id, "thumb", 6);}); body.append(pictures);
    }).catch(error => {modelDetails.delete(run.id); if (body.isConnected) body.replaceChildren(node("p", "", error.message || "记录加载失败"));});
  }
}
function modelStatus(status) {return {queued: "等待调用", running: "调用中", succeeded: "成功", failed: "失败", cancelled: "已取消"}[status] || status;}
function table(parent, title, headers, rows) {
  const section = node("section"); parent.append(section);
  const key = session.id + "/" + session.revision + "/table/" + title;
  const draw = () => {
    section.replaceChildren();
    const page = Math.min(pages.get(key) || 0, Math.max(0, Math.ceil(rows.length / 40) - 1));
    const header = node("div", "gallery-header"); header.append(node("h3", "", title));
    header.append(pager(page, rows.length, 40, delta => {pages.set(key, page + delta); draw();})); section.append(header);
    if (!rows.length) { section.append(node("p", "muted small", "没有需要记录的候选。")); return; }
    const wrapper = node("div", "table-scroll"), table = node("table", "data-table"), head = node("thead"), headRow = node("tr");
    headers.forEach(value => headRow.append(node("th", "", value))); head.append(headRow); table.append(head);
    const body = node("tbody");
    for (const row of rows.slice(page * 40, (page + 1) * 40)) { const tr = node("tr"); row.forEach(value => tr.append(node("td", "", value))); body.append(tr); }
    table.append(body); wrapper.append(table); section.append(wrapper);
  }; draw();
}
function pager(page, total, size, change) {
  const controls = node("div", "pager"), count = Math.max(1, Math.ceil(total / size));
  controls.append(node("span", "", total + " 项 · " + (page + 1) + "/" + count));
  if (count > 1) { const prev = button("‹", () => change(-1)), next = button("›", () => change(1)); prev.disabled = page <= 0; next.disabled = page >= count - 1; controls.append(prev, next); }
  return controls;
}
function gallery(parent, title, items, kind, variant = "thumb", size = 24) {
  const section = node("section"), key = session.id + "/" + session.revision + "/gallery/" + title;
  parent.append(section);
  const draw = () => {
    section.querySelectorAll(".frame-image").forEach(item => observer.unobserve(item));
    section.replaceChildren();
    const page = Math.min(pages.get(key) || 0, Math.max(0, Math.ceil(items.length / size) - 1));
    const header = node("div", "gallery-header"); header.append(node("h3", "", title), pager(page, items.length, size, delta => {pages.set(key, page + delta); draw();})); section.append(header);
    if (!items.length) {section.append(node("p", "muted small", "此路没有候选帧。")); return;}
    const strip = node("div", "filmstrip");
    items.slice(page * size, (page + 1) * size).forEach((item, offset) => strip.append(frameCard(item, kind, item.assetIndex ?? page * size + offset, variant)));
    section.append(strip);
  }; draw();
}
function frameCard(item, kind, index, variant = "thumb") {
  if (kind.startsWith("model:")) return gridCard(item, index, kind);
  if (kind === "grid") return gridCard(item, index);
  const card = button("", () => showFrame(kind, index, variant), "frame-card " + (item.reason === "scene_change" ? "scene" : item.reason));
  const imageBox = node("div", "frame-image"), placeholder = node("span", "placeholder", "按需加载预览"), image = node("img");
  image.alt = "视频 " + time(item.timestamp_sec ?? item.timestamp) + " 的画面";
  imageBox.append(placeholder, image);
  const meta = node("div", "frame-meta");
  meta.append(node("strong", "", (kind === "frame" ? "" : "请求 ") + time(item.timestamp_sec ?? item.timestamp)), node("p", "", reasonNames[item.reason] || "候选"));
  if (kind === "frame") meta.append(node("p", "", item.width + "×" + item.height + " · " + bytes(item.bytes)));
  else if (item.score) meta.append(node("p", "", "变化分数 " + number(item.score, 4)));
  card.append(imageBox, meta);
  const capture = {id: session.id, revision: session.revision};
  imageBox._load = () => getImage(capture, kind, index, variant).then(data => {image.src = data.data_url; placeholder.remove(); if (kind !== "frame") meta.append(node("p", "", "实际 " + time(data.frame.timestamp_sec)));}).catch(() => {placeholder.textContent = "预览失败，点击重试";});
  observer.observe(imageBox);
  return card;
}
function gridCard(item, index, kind = "grid") {
  const label = kind === "grid" ? "拼图" : "输入图片";
  const card = button("", () => showGrid(index, kind), "frame-card");
  const imageBox = node("div", "frame-image"), placeholder = node("span", "placeholder", "按需加载拼图"), image = node("img");
  image.alt = label + " " + (index + 1); imageBox.append(placeholder, image);
  const meta = node("div", "frame-meta"), first = item.cells[0], last = item.cells[item.cells.length - 1];
  meta.append(node("strong", "", label + " " + (index + 1) + " · " + item.cells.length + " 帧"), node("p", "", item.width + "×" + item.height + " · " + (item.mime_type === "image/png" ? "PNG" : "JPEG")), node("p", "", time(first.timestamp_sec) + " – " + time(last.timestamp_sec)));
  card.append(imageBox, meta);
  const capture = {id: session.id, revision: session.revision};
  imageBox._load = () => getImage(capture, kind, index, "thumb").then(data => {image.src = data.data_url; placeholder.remove();}).catch(() => {placeholder.textContent = "预览失败，点击重试";});
  observer.observe(imageBox);
  return card;
}
const observer = new IntersectionObserver(entries => {
  for (const entry of entries) if (entry.isIntersecting) {observer.unobserve(entry.target); entry.target._load?.();}
}, {rootMargin: "100px"});
function getImage(capture, kind, index, variant) {
  const key = [capture.id, capture.revision, kind, index, variant].join("/");
  if (!imageCache.has(key)) {
    const promise = new Promise((resolve, reject) => {imageQueue.push({capture, kind, index, variant, resolve, reject}); pumpImages();});
    imageCache.set(key, promise);
    promise.catch(() => {if (imageCache.get(key) === promise) imageCache.delete(key);});
  }
  return imageCache.get(key);
}
function pumpImages() {
  while (activeImages < 3 && imageQueue.length) {
    const job = imageQueue.shift();
    if (job.capture.id !== session?.id || job.capture.revision !== session?.revision) {job.reject(new Error("测试已切换")); continue;}
    activeImages++;
    const [kind, record] = job.kind.split(":");
    bridge.apiGet("debug/image", {id: job.capture.id, revision: job.capture.revision, kind, ...(record ? {record} : {}), index: job.index, variant: job.variant}).then(job.resolve, job.reject).finally(() => {activeImages--; pumpImages(); refreshPreviewTiming();});
  }
}
function refreshPreviewTiming() {
  if (activeImages || imageQueue.length || !session || session.running) return;
  clearTimeout(previewTimer);
  const capture = {id: session.id, revision: session.revision};
  previewTimer = setTimeout(async () => {
    if (busy || capture.id !== session?.id || capture.revision !== session?.revision) return;
    try {
      const data = await bridge.apiGet("debug/session", {id: capture.id});
      if (capture.id === session?.id && capture.revision === session?.revision) accept(data);
    } catch {}
  }, 250);
}
function seek(timestamp) { if (videoURL && selectedVideoSession === session?.id) $("source-video").currentTime = timestamp; }
function closeDialog() { dialogVersion++; if ($("image-dialog").open) $("image-dialog").close(); }
async function showGrid(index, kind = "grid") {
  const version = ++dialogVersion, capture = {id: session.id, revision: session.revision};
  $("image-title").textContent = (kind === "grid" ? "九宫格输出" : "模型输入图片") + " · 第 " + (index + 1) + " 张";
  $("image-description").textContent = "完整图片文件，缩略预览不参与模型调用。";
  const content = $("image-content"); content.replaceChildren(node("p", "muted", "正在加载完整拼图…"));
  if (!$("image-dialog").open) $("image-dialog").showModal();
  try {
    const data = await getImage(capture, kind, index, "full");
    if (version !== dialogVersion) return;
    content.replaceChildren();
    const viewport = node("div", "grid-viewport"), image = node("img"); image.src = data.data_url; image.alt = "带帧编号与实际时间戳的完整九宫格"; viewport.append(image);
    const toggle = button("按原始像素查看", () => {const actual = viewport.classList.toggle("actual-size"); toggle.textContent = actual ? "适应窗口" : "按原始像素查看";});
    const actions = node("div", "inline-actions"); actions.append(toggle); content.append(actions, viewport);
    note(content, data.image.width + " × " + data.image.height + " · " + bytes(data.image.bytes) + "。切换为原始像素后，可滚动查看细节。");
    table(content, "帧与格子对应关系", ["帧编号", "行 / 列", "实际时间", "单帧尺寸", "frame_id"], data.image.cells.map(cell => ["#" + cell.frame_number, (cell.row || 1) + " / " + (cell.column || 1), time(cell.timestamp_sec), (cell.width || data.image.width) + "×" + (cell.height || data.image.height), cell.frame_id]));
  } catch (error) {if (version === dialogVersion) content.replaceChildren(node("p", "", error.message || "拼图加载失败"));}
}
async function showFrame(kind, index, variant) {
  const version = ++dialogVersion, capture = {id: session.id, revision: session.revision};
  const group = {fixed: session.fixed, scene: session.scenes, merged: session.candidates, frame: session.frames}[kind], item = group[index];
  const formal = kind === "frame";
  $("image-title").textContent = (formal ? "正式输出帧" : "候选预览") + " · " + time(item.timestamp_sec ?? item.timestamp);
  $("image-description").textContent = formal ? "点击加载原分辨率对照后，可比较尺寸和细节。" : "额外调试预览，未用于正式输出图的画面去重。";
  const content = $("image-content"); content.replaceChildren(node("p", "muted", "正在加载…"));
  if (!$("image-dialog").open) $("image-dialog").showModal();
  seek(item.timestamp_sec ?? item.timestamp);
  try {
    const result = await getImage(capture, kind, index, formal ? "full" : variant);
    if (version !== dialogVersion) return;
    content.replaceChildren();
    if (!formal) {
      const image = node("img", "detail-image"); image.src = result.data_url; image.alt = "候选帧预览"; content.append(image);
      note(content, "请求 " + time(result.frame.requested_sec) + " → 实际 " + time(result.frame.timestamp_sec) + " · " + result.frame.width + " × " + result.frame.height + " · " + bytes(result.frame.bytes)); return;
    }
    const comparison = node("div", "comparison");
    const figure = node("figure"), image = node("img"); image.src = result.data_url; image.alt = "缩放压缩后的正式 JPEG";
    figure.append(image, node("figcaption", "", "正式 JPEG · " + item.width + " × " + item.height + " · " + bytes(item.bytes) + " · 实际 " + time(item.timestamp_sec)));
    const original = node("figure"), originalButton = button("加载原分辨率 JPEG 对照", async () => {
      originalButton.disabled = true; originalButton.textContent = "正在提取原分辨率画面…";
      try {
        const source = await getImage(capture, kind, index, "source");
        if (version !== dialogVersion) return;
        const img = node("img"); img.src = source.data_url; img.alt = "原分辨率 JPEG 对照";
        original.replaceChildren(img, node("figcaption", "", "原分辨率 JPEG 对照 · " + source.frame.width + " × " + source.frame.height + " · " + bytes(source.frame.bytes) + " · 实际 " + time(source.frame.timestamp_sec)));
      } catch (error) { originalButton.disabled = false; originalButton.textContent = error.message || "提取失败，点击重试"; }
    });
    original.append(originalButton, node("p", "small muted", "保持源分辨率，同样以 JPEG 编码（q=3）。对照提取计入额外预览耗时。"));
    comparison.append(original, figure); content.append(comparison);
    const fingerprint = await getImage(capture, kind, index, "fingerprint");
    if (version !== dialogVersion) return;
    const fingerprintBox = node("div", "fingerprint"), fp = node("img"); fp.src = fingerprint.data_url; fp.alt = "32×18 灰度指纹，放大展示";
    fingerprintBox.append(fp, node("p", "", "实际用于去重的 32 × 18 灰度指纹（放大显示）\n质量分：" + number(item.quality, 3) + "。请求时间：" + time(item.requested_sec) + "；实际时间：" + time(item.timestamp_sec)));
    content.append(fingerprintBox);
  } catch (error) { if (version === dialogVersion) content.replaceChildren(node("p", "", error.message || "图片加载失败")); }
}
function exportRecord() {
  return action(async () => {
    const capture = session, records = await bridge.apiGet("debug/model-records", {id: capture.id});
    const blob = new Blob([JSON.stringify({...capture, model_records: records}, null, 2)], {type: "application/json"}), url = URL.createObjectURL(blob), link = node("a");
    link.href = url; link.download = "vision-pipeline-" + capture.id.slice(0, 8) + ".json"; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
}
async function upload(file) {
  if (!file || session?.running) return;
  await action(async () => {
    if (file.size > config.max_upload_bytes) throw new Error("视频不能超过 256 MiB");
    message("正在上传视频…");
    const data = await bridge.upload("debug/upload/" + encodeURIComponent(config.csrf), file);
    if (videoURL) URL.revokeObjectURL(videoURL);
    videoURL = URL.createObjectURL(file); selectedVideoSession = data.id;
    $("source-video").src = videoURL;
    accept(data, true); await loadSessions(); message("");
  });
  $("video-file").value = "";
}
function bindEvents() {
  $("video-file").addEventListener("change", event => upload(event.target.files[0]));
  $("drop-zone").addEventListener("dragover", event => {event.preventDefault(); $("drop-zone").classList.add("dragging");});
  $("drop-zone").addEventListener("dragleave", () => $("drop-zone").classList.remove("dragging"));
  $("drop-zone").addEventListener("drop", event => {event.preventDefault(); $("drop-zone").classList.remove("dragging"); upload(event.dataTransfer.files[0]);});
  $("next").addEventListener("click", () => {if (session?.done) {view = 7; render(); return;} action(async () => {accept(await post("next", {completed: session.completed, revision: session.revision}));});});
  $("cancel").addEventListener("click", () => action(async () => accept(await post("cancel"))));
  $("parameters").addEventListener("submit", event => {event.preventDefault(); if (!$("parameters").reportValidity()) return; action(async () => {accept(await post("reset", {parameters: params()}), true); message("已应用本次参数。点击执行视频探测开始重跑。");});});
  $("delete").addEventListener("click", () => action(async () => {
    await post("delete"); session = null; view = -1; clearTimeout(pollTimer); remember(null); imageCache.clear(); closeDialog();
    if (videoURL) URL.revokeObjectURL(videoURL); videoURL = null; $("source-video").removeAttribute("src");
    $("content").replaceChildren(node("p", "muted", "测试文件与图片已删除，可上传新视频。"));
    await loadSessions(); render();
  }));
  $("sessions").addEventListener("change", event => {if (event.target.value) action(async () => accept(await bridge.apiGet("debug/session", {id: event.target.value}), true));});
  $("close-dialog").addEventListener("click", closeDialog);
  $("image-dialog").addEventListener("cancel", () => {dialogVersion++;});
  window.addEventListener("pagehide", () => {clearTimeout(pollTimer); clearTimeout(previewTimer); if (videoURL) URL.revokeObjectURL(videoURL); observer.disconnect();});
}
async function initialize() {
  try {
    bridge = window.AstrBotPluginPage;
    if (!bridge || typeof bridge.upload !== "function") throw new Error("请从 AstrBot WebUI 的“视频理解”插件页面打开调试台。");
    await bridge.ready();
    bridge.onContext(context => {document.documentElement.dataset.theme = context.isDark ? "dark" : "light";});
    await loadSessions(); setParams(config.defaults); bindEvents();
    const previous = remembered();
    if (previous && config.sessions.some(item => item.id === previous)) accept(await bridge.apiGet("debug/session", {id: previous}), true);
    else render();
  } catch (error) {message(error.message || "调试台初始化失败", true); updateControls();}
}
initialize();
