# 日志目录总览

项目不再区分"病人"和"健康对照"，实验现在统一由正常参与者做。目前一共有 **7 个日志目录**，
按"这条数据是怎么产生的"分成三大类：

1. **单独测试某个技术**（面板里 "tester une technique"，`usage_mode == "test"`）→ `test_logs`
2. **"control complet" 协议**（面板 Experiment Protocol 页），分两种跑法：
   - **Tester les deux tâches contrôle séparément**（`comparative_run_mode = "test"`）：单独跑 realistic 或 synthetic Fitts 中的一个 → `test_realistic_logs` / `test_fitts_synthetic_logs`
   - **Exécuter le protocole contrôle complet**（`comparative_run_mode = "full"`）：`comparative_session.py` 编排，按参与者编号奇偶反平衡顺序跑完两个任务、中间插入休息 → `control_comparative_logs`（总控）+ `control_realistic_logs`（realistic 子任务）+ `control_fitts_synthetic_logs`（synthetic Fitts 子任务）
3. **定性/试点测试**（`qualitative_baseline.py`）→ `qualitative_logs`

`target_finder_toolkit/logging_utils.py` 里的 `SessionLogger` 是最底层的写日志工具（每条记录自动带上
`t` = 相对会话开始的秒数），各技术子进程（bubble、dynaspot、semantic、ninja_cursors…）和实验控制脚本都在用它。
`control_panel.py` 里的 `EXPERIMENT_LOG_ROOTS` 常量列出了除 `test_logs`/`qualitative_logs` 之外的 5 个"官方"目录，
面板启动时会自动 `mkdir` 出来，即使还没跑过一次也能在文件浏览器里看到。

---

## 1. `test_logs/` —— 单独测试一个技术

面板 "tester une technique" 之后，`logging_utils.make_default_log_path()` 生成的落盘位置：
`{时间戳}_{technique}.jsonl`。跟参与者/实验协议完全无关，纯粹是开发/调试某个技术好不好用时产生的数据。

事件类型主要是 `session_start`（该技术这次运行的参数：滤波器设置、YOLO 置信度/IOU 阈值、
`detection_source` 是 `yolo`（本地推理）还是 `annotations`（读共享标注文件）等）、`cursor_sample`、
`click`、`detection_change`、`session_end`。

---

## 2. `qualitative_logs/` —— 定性测试（trois tâches）

`qualitative_baseline.py` 驱动，两类任务：

- `*_normal_mouse_baseline.jsonl`：普通鼠标基线，`task_order` 依次是三个子任务
  `cursor_stability`（光标稳定性）/`long_distance`（长距离移动）/`dense_interface`（密集界面点击）。
- `*_qualitative_sequence.jsonl` + 配套 `*_{technique}_runtime.jsonl`：多技术连续切换的定性测试序列。

字段含义详见下方"通用字段说明"，`qualitative_sequence` 特有的事件类型：
`phase_start/end`（一个阶段=某技术的一段测试）、`phase_process_*`（该阶段技术子进程生命周期）、
`bubble_ready`、`standard_control_state`/`ninja_control_state`（对照组/ninja 控制状态切换）、
`region_change`（鼠标所在界面区域变化）、`mouse_down/up`、`drag_start/end`、`error`/`*_error`。

判断一条数据是否完整，看最后一行 `session_end.reason` 是不是 `completed`（`user_abort` = 中途手动终止）。

---

## 3. `test_realistic_logs/` —— 单独测试 control complet 的 realistic 任务

面板 "Tester les deux tâches contrôle séparément" → 选 realistic 时的落盘位置：
`{participant}_{时间戳}_realistic/`。跟下面的 `control_realistic_logs` 用的是**同一套任务代码**
（`experimental_session.py` 的 `realistic_screenshot_session`），只是这里是单独测试，不经过总控协议、
不会有 `control_comparative_logs` 里的对应记录。

同一目录下还会有：
- `{technique}.annotations.json`：共享检测框状态文件（不是日志，会被反复覆盖）
- `ninja_cursors.control`：ninja cursors 控制状态文件（`active`/`inactive` + 参考分辨率）
- `*_{technique}_runtime.jsonl`：各技术子进程自己的 `session_start`/`session_end`

---

## 4. `test_fitts_synthetic_logs/` —— 单独测试 control complet 的 synthetic Fitts 任务

面板同一个页面选 synthetic_fitts 时的落盘位置：`{participant}_{时间戳}_synthetic_fitts/`。
对应 `synthetic_fitts_session.py`（`synthetic_fitts_distractors_session` 任务，程序生成目标点，
按 Fitts 定律的 ID/密度/rho 条件做移动实验）。同样不经过总控协议。

---

## 5. `control_comparative_logs/` —— control complet 总控日志

只有跑"Exécuter le protocole contrôle complet"（完整协议模式）才会写。由 `comparative_session.py` 编排，
记录 realistic 和 synthetic Fitts 两个子任务**怎么衔接**，不记录鼠标/点击这类细节：

`{participant}_{时间戳}_comparative/{session_id}_comparative.jsonl`

| `type` | 字段 | 含义 |
|---|---|---|
| `comparative_session_start` | `task_order`, `order_rule`（`odd_participants_synthetic_first_even_participants_realistic_first`）, `trials_per_block`, `fitts_trials_per_condition`, `synthetic_blocks`, `conditions_file`, `*_log_group`/`*_output_root` | 整场协议的顶层设计：两个子任务顺序（按参与者编号奇偶反平衡）、各自输出目录 |
| `comparative_task_start` | `task_index`, `task`（`realistic`/`synthetic`）, `log_group`, `output_dir`, `command` | 某个子任务即将启动 |
| `comparative_task_end` | `task_index`, `task`, `returncode` | 子任务进程退出，`0`=正常，`130`=按 Escape 主动退出 |
| `comparative_pause_start`/`end` | `after_task_index`, `previous_task`, `next_task`, `duration_sec`, `aborted` | 两个子任务之间的休息屏幕 |
| `comparative_session_end` | `reason`（`completed`/`task_failed_or_aborted`/`keyboard_escape_on_between_task_pause`）, `total_duration_sec` | 整场协议结束原因和总时长 |

---

## 6. `control_realistic_logs/` —— control complet 里的 realistic 子任务

完整协议模式下，realistic 子任务的数据写在：
`control_realistic_logs/{participant}_{时间戳}_comparative/0{顺序}_realistic/`
（`0{顺序}` 是这个子任务在协议里排第几，取决于参与者编号奇偶）。

内容结构和 `test_realistic_logs` 一样（`_session.jsonl` + `*_{technique}_runtime.jsonl` +
`{technique}.annotations.json` + `ninja_cursors.control`），区别只在于它是嵌套在完整协议下跑出来的，
`session_start` 里的 `log_group` 会是 `control_realistic_logs` 而不是 `test_realistic_logs`。

---

## 7. `control_fitts_synthetic_logs/` —— control complet 里的 synthetic Fitts 子任务

完整协议模式下，synthetic Fitts 子任务的数据写在：
`control_fitts_synthetic_logs/{participant}_{时间戳}_comparative/0{顺序}_synthetic/`

内容结构和 `test_fitts_synthetic_logs` 一样，`session_start` 字段更偏"实验设计学"：
`id_values`（Fitts 定律难度指数集合）、`densities`（干扰物密度条件）、`rho_values`
（目标与干扰物的比例参数）、`condition_pool`/`condition_sampling`（条件抽样方法）、
`plan_metadata`（含 block 顺序反平衡方法）。

---

## 通用字段说明（跨所有目录）

所有 `.jsonl` 日志都是**一行一个 JSON 对象**，靠 `type` 字段区分事件种类。

| 字段 | 含义 |
|---|---|
| `type` | 事件种类 |
| `t` | 相对时间（秒），从这个文件对应的会话/进程启动算起 |
| `timestamp` | 绝对 Unix 时间戳（不是所有文件都有） |
| `technique` | `bubble`/`dynaspot`/`semantic`/`ninja_cursors`/`normal_mouse`/`mouse` 等 |
| `participant_id` / `session_id` | 参与者编号 / 本次会话唯一 ID |

**`session_start`**：会话/进程开始时写一次。顶层会话会带 `block_order`（每个 block 的技术/难度/试次数）；
技术子进程自己的 `session_start` 带的是滤波器参数（`filter_name`/`min_cutoff`/`beta`/`d_cutoff`，
One-Euro 滤波器超参数）、YOLO 阈值（`confidence`/`iou`）、`detection_source`
（`yolo`=本地推理，`annotations`=读共享标注文件）。

**`session_end`**：`reason` 说明结束原因——`completed`（正常跑完）、`user_abort`（中途手动终止）、
`quit`/`process_exit`/`logger_close`（各种退出方式）。**判断数据完不完整就看这个字段。**

**`cursor_sample`**：`raw`=原始坐标，`filtered`=滤波后坐标（真正用来判定命中的坐标）。

**`click`**：`raw`=实际点击坐标，`effective`=生效坐标（若被技术重定向到目标中心则不同于 `raw`），
`redirected`=是否被重定向，`target`=命中的目标框（`class_name` 取值 `Button`/`ToggleButton`/
`Hyperlink`/`Text`/`TextInput`/`Slider`）。

**`detection_change`**：`detections`=当前所有检测框，`added_ids`/`removed_ids`=新增/消失的框 id。

其余实验编排类事件：`initialization_start/end`、`technique_process_start/stop/output`、
`block_start/end`、`trial_start/end`、`pause_start/end`、`mouse_down/up`、`drag_start/end`、
`region_change`、`ninja_calibration_event`/`ninja_runtime_event`/`calibration_*`、`bubble_ready`、
`error`/`*_error` 系列（各类异常记录，方便事后排查）。
