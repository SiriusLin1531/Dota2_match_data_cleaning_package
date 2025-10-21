# Dota 2 比赛数据集清洗包（含脚本与 Notebook）

本包用于清洗 Kaggle 上的 **Dota 2 Matches** 数据集（https://www.kaggle.com/datasets/devinanzelmo/dota-2-matches ）， 覆盖缺失/重复/异常处理、时间与类别映射、
以及生成结构化日志；并提供分块处理的能力以支持更大规模数据。

## 目录结构
```
Dota2_match_data_cleaning_package/
├─ cleaner.py                # python脚本：可配置的批处理清洗器（含日志/分块/并行处理）
├─ README.md                 # 使用说明（本文件）
├─ requirements.txt          # 依赖列表
└─ clean_dota2_data.ipynb  # Jupyter Notebook文件（与python脚本相同清洗功能，便于交互式探索和调参）
```

> 如果你是从 ZIP 下载，请先解压到一个空目录，然后参考下面的“快速开始”。

## 快速开始

1. **准备环境**
   ```bash
   python -V        # 建议 Python 3.9+
   pip install -r requirements.txt
   ```

2. **准备数据**
   - 将 Kaggle 的 `dota-2-matches` 数据集解压到项目根目录的 `Dota2data/` 目录：
     ```
     Dota2data/
       match.csv
       players.csv
       player_time.csv
       teamfights.csv
       teamfights_players.csv
       objectives.csv
       chat.csv
       test_labels.csv
       test_player.csv
       player_ratings.csv
       match_outcomes.csv
       purchase_log.csv
       ability_upgrades.csv
       cluster_regions.csv
       patch_dates.csv
       ability_ids.csv
       item_ids.csv
       hero_names.csv
     ```

3. **运行清洗**
   - python脚本提供两种运行模式（单文件清洗和目录全体文件清洗），可供选择的清洗参数如下，也可参考下面提供的运行示例。
  <br/>
   
   | 参数 | 类型 | 默认值 | 是否必填 | 说明 |
   |------|------|---------|------------|------|
   | `-i`, `--input` | `str` | 无 | 是 | 输入路径，可为文件或目录 |
   | `-o`, `--output` | `str` | 无 | 是 | 输出目录，若不存在会自动创建 |
   | `--skip-missing` | `bool` (`store_true`) | `False` | 否 | 跳过缺失值填充 |
   | `--log` | `bool` (`store_true`) | `False` | 否 | 启用详细日志输出模式 |
   | `--chunksize` | `int` | `0` | 否 | 每次读取的行数；为 0 表示不分块 |
   | `--jobs` | `int` | `1` | 否 | 并行线程数；为 1 表示单线程 |
   | `-h`, `--help` | Flag | 无 | 否 | 显示帮助信息并退出 |
<br/>

   - 单文件运行示例（清洗players.csv,输出至cleaned文件夹，默认不跳过缺失值补充、不显示详细清洗日志、不启用分块和并行功能）
     ```bash
     python cleaner.py -i ./Dota2data/players.csv -o ./cleaned 
     ```
   - 目录运行示例（清洗目录Dota2data下所有文件,输出至cleaned文件夹，默认不跳过缺失值补充、不显示详细清洗日志、不启用分块和并行功能）：
     ```bash
     python cleaner.py -i ./Dota2data -o ./cleaned
     ```
   - 全参数运行示例：**输入文件目录和输出文件目录**（必须指明），**跳过缺失值补充**，**显示详细清洗日志**，**分块**（例如每块 100,000 行），**并行**（例如4线程并行处理）：
     ```bash
     python cleaner.py -i ./Dota2data -o ./cleaned  --skip-missing --log --chunksize 100000 --jobs 4
     ```
   
5. **清洗结果**
   - 输出位于 `cleaned/` 目录，以 `*_clean.csv` 命名。
   - 日志位于 `logs/` 目录，文件名包含时间戳（例如 `cleaning_2025xxxx_xxxx.log`）。


## 各表数据清洗逻辑说明

### 1. matches.csv（比赛主表）

#### 清洗目标
标准化比赛记录数据，补充时间字段与补丁版本信息，并修正区域映射。

#### 主要清洗步骤
1. **布尔字段转换**  
   - 将 `radiant_win` 从 0/1 转换为布尔类型。
2. **时间戳转换**  
   - 将 `start_time`（UNIX 秒）转换为 `datetime` 类型字段 `start_time_dt`。
3. **长时比赛标记**  
   - 若 `duration > 7200`（超过 2 小时），添加布尔标记列 `long_match = True`。
4. **区域映射**  
   - 若提供 `cluster_regions.csv`，按 `cluster` 合并获取 `region`。
   - 若有未映射记录，输出警告。
5. **补丁版本映射**  
   - 使用 `patch_dates.csv`，基于 `start_time_dt` 与 `patch_date` 的时间先后顺序 (`merge_asof`) 对应补丁号。
6. **去重处理**  
   - 按 `match_id` 删除重复记录。

#### 输出结果
生成 `matches_clean.csv`，增加列：
- `start_time_dt`
- `long_match`
- `region`
- `patch`  
并保证每个 `match_id` 唯一。

---

### 2. players.csv（玩家表）

#### 清洗目标
去除重复玩家记录，填充缺失值，截断异常击杀值，并映射英雄名称。

#### 主要清洗步骤
1. **缺失值填充**  
   - 若未启用 `--skip-missing`，填充所有数值型缺失值为 0。
2. **去重处理**  
   - 按 `(match_id, player_slot)` 删除重复记录。
3. **阵营推断**  
   - 调用 `add_team_from_slot()` 根据 `player_slot` 衍生 `team`（Radiant/Dire）。
4. **异常击杀值修正**  
   - 若 `kills > 100`，将其截断为 100 并添加 `kills_outlier=True` 标记。
5. **英雄名称映射**  
   - 若提供 `hero_names.csv`，根据 `hero_id` 合并英雄名 `hero_name`。

#### 输出结果
生成 `players_clean.csv`，包含新增列：
- `team`
- `kills_outlier`
- `hero_name`

---

### 3. player_time.csv（玩家时间表）

#### 清洗目标
去除无效或异常时间记录。

#### 清洗步骤
1. **时间有效性检查**  
   - 删除 `times < 0` 的记录。

#### 输出结果
生成 `player_time_clean.csv`，仅保留 `times >= 0` 的数据。

---

### 4. teamfights.csv（团战表）

#### 清洗目标
确保团战时间范围有效。

#### 清洗步骤
1. **时间范围过滤**  
   - 保留 `start >= 0`, `end >= 0`, 且 `end > start` 的记录。
   - 删除其他时间异常行。

#### 输出结果
生成 `teamfights_clean.csv`。

---

### 5. teamfights_players.csv（团战-玩家表）

#### 清洗目标
补全缺失值、去重并添加阵营。

#### 清洗步骤
1. **缺失值填充**  
   - 若未启用 `--skip-missing`，填充数值缺失为 0。
2. **去重**  
   - 按 `(match_id, player_slot, damage, deaths)` 去重。
3. **阵营推断**  
   - 使用 `add_team_from_slot()` 生成 `team` 列。

#### 输出结果
生成 `teamfights_players_clean.csv`，含 `team` 列。

---

### 6. objectives.csv（目标事件表）

#### 清洗目标
统一字段名、去除无效时间、补全阵营信息。

#### 清洗步骤
1. **字段统一**  
   - 若存在 `slot` → 重命名为 `player_slot`。  
   - 若存在 `subtype` → 重命名为 `type`。
2. **时间过滤**  
   - 删除 `time < -120` 的记录。
3. **阵营推断**  
   - 若 `team` 不存在，则从 `player_slot` 派生。
4. **去重处理**  
   - 按 `(match_id, time, type, key, team)` 去重。
5. **类型缺失补全**  
   - 若缺失 `type` 列，则设为 `"unknown"`。

#### 输出结果
生成 `objectives_clean.csv`。

---

### 7. chat.csv（聊天记录表）

#### 清洗目标
规范化玩家标识与时间范围。

#### 清洗步骤
1. **字段统一**  
   - 若存在 `slot`，重命名为 `player_slot`。
2. **时间过滤**  
   - 删除 `time < -120` 的记录。
3. **阵营推断**  
   - 从 `player_slot` 派生 `team`。

#### 输出结果
生成 `chat_clean.csv`，含 `player_slot` 与 `team` 列。

---

### 8. test_labels.csv（测试标签表）

#### 清洗目标
标准化比赛结果与唯一性。

#### 清洗步骤
1. **布尔化**  
   - 将 `radiant_win` 转换为布尔值。
2. **去重处理**  
   - 按 `match_id` 去重。

#### 输出结果
生成 `test_labels_clean.csv`。

---

### 9. test_player.csv（测试玩家表）

#### 清洗目标
清理关键字段缺失与重复记录。

#### 清洗步骤
1. **关键字段完整性检查**  
   - 若未启用 `--skip-missing`，删除缺少 `match_id`, `player_slot`, `hero_id` 的行。
2. **去重**  
   - 按 `(match_id, player_slot)` 删除重复。
3. **阵营推断**  
   - 调用 `add_team_from_slot()` 添加 `team` 列。

#### 输出结果
生成 `test_player_clean.csv`。

---

### 10. player_ratings.csv（玩家评分表）

#### 清洗目标
修正异常数值，确保逻辑一致性。

#### 清洗步骤
1. **统计字段约束**  
   - 将 `total_matches`、`total_wins` 设为非负；
   - 若 `total_wins > total_matches`，自动修正为相等。
2. **技能评估值限制**  
   - 对 `trueskill_mu` 与 `trueskill_sigma` 进行下限约束（≥0）。
3. **去重**  
   - 按 `account_id` 删除重复记录。

#### 输出结果
生成 `player_ratings_clean.csv`，所有统计数值非负且逻辑一致。

---

### 11. match_outcomes.csv（比赛结果表）

#### 清洗目标
标准化胜负与数值类型。

#### 清洗步骤
1. **数值转换**  
   - 将 `win`, `rad` 转换为整数（NaN → 0）。
2. **去重处理**  
   - 按 `match_id` 去重。

#### 输出结果
生成 `match_outcomes_clean.csv`。

---

### 12. purchase_log.csv（物品购买表）

#### 清洗目标
统一时间字段、添加分钟列、映射物品名称。

#### 清洗步骤
1. **时间字段处理**  
   - 删除 `time < -120` 的记录；
   - 新增字段 `minute = time // 60`。
2. **物品映射**  
   - 若提供 `item_ids.csv`，按 `item_id` 合并物品名 `item_name`。
3. **阵营推断**  
   - 从 `player_slot` 生成 `team`。

#### 输出结果
生成 `purchase_log_clean.csv`，新增列：
- `minute`
- `item_name`
- `team`

---

### 13. ability_upgrades.csv（技能升级表）

#### 清洗目标
规范字段命名、修正时间范围并映射技能名称。

#### 清洗步骤
1. **字段重命名**  
   - 若存在 `ability`，改为 `ability_id`。
2. **等级修正**  
   - 约束 `level >= 1`。
3. **时间过滤**  
   - 删除 `time < -120` 的记录。
4. **技能映射**  
   - 若提供 `ability_ids.csv`，按 `ability_id` 合并技能名 `ability_name`。
5. **阵营推断**  
   - 从 `player_slot` 衍生 `team`。

#### 输出结果
生成 `ability_upgrades_clean.csv`，包含：
- `ability_name`
- `team`

---

### 14. 映射表加载说明（cluster / patch / hero / item / ability）

| 映射表 | 键字段 | 作用说明 |
|---------|----------|-----------|
| `cluster_regions.csv` | `cluster → region` | 区域映射，用于 `matches.csv` |
| `patch_dates.csv` | `date → patch` | 时间对照补丁版本，用于 `matches.csv` |
| `hero_names.csv` | `hero_id → hero_name` | 英雄名称映射，用于 `players.csv` 与 `test_player.csv` |
| `item_ids.csv` | `item_id → item_name` | 物品名称映射，用于 `purchase_log.csv` |
| `ability_ids.csv` | `ability_id → ability_name` | 技能名称映射，用于 `ability_upgrades.csv` |


> 以上清洗逻辑已在 `cleaner.py` 中实现，并在日志中打印处理情况（缺失、去重、异常等）。

## 鲁棒性测试与性能

- **新数据测试**：若出现 `hero_id`/`item_id`/`ability_id` 未映射的日志，说明映射表需更新。
  请替换 `hero_names.csv / item_ids.csv / ability_ids.csv` 后重跑。
- **分块处理**：对 `players.csv` 等大表可用 `--chunksize` 启用分块，降低内存压力。
  > 注意：跨分块的重复行极少见，如需严格全局去重，可在清洗后对输出再进行去重。
- **并行**：可使用`-jobs` 参数开启并行功能。
- **日志与监控**：日志会汇报每步处理数量与潜在问题，便于快速定位与二次优化。

## 与 Notebook 的关系

本仓库包含（也可单独下载）**`dota2_cleaning_notebook.ipynb`**：
- 适合交互式探索与调参；
- 与脚本共享相同的清洗函数思想。

## 清洗数据结果可用于的 AI 训练任务方向
- 比赛结果预测（基于阵容/前期经济等的二分类）
- 玩家行为聚类（风格画像）
- 团战影响分析与胜率建模
- 物品出装路径研究（序列/时序特征）
- 聊天文本情绪/毒性识别（NLP）

## 常见问题（FAQ）
- **Q:** 运行时提示 `hero_id 未映射`？  
  **A:** 更新 `hero_names.csv`（新英雄），或检查是否启用了ID偏移修正逻辑。

- **Q:** 内存不足？  
  **A:** 对大表启用 `--chunksize`，例如 `100000`；或分表分步执行。

- **Q:** 我能同时使用 `--chunksize` 和 `--jobs` 吗？  
  **A:** 可以。此时每个文件会被分块并行清洗，速度最快。

- **Q:** 想要仅清洗部分表？   
  **A:** 可使用单文件清洗功能进行清洗；也可临时将不需要的源CSV移出 `Dota2data/`目录，脚本会自动识别目录中存在的相关CSV文件进行清洗。


## 许可证
仅供学习与研究用途。请同时遵守 Kaggle 数据集的相应条款。
