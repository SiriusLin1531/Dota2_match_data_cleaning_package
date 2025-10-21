#!/usr/bin/env python
# coding: utf-8
"""
Dota2DataCleaner
-------------------------------------

针对 Kaggle Dota2 Matches 数据集的清洗脚本。
支持命令行运行、日志输出、分块读取和多线程。

"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import datetime

SKIP_MISSING = False  # 是否跳过缺失值处理

# ========================
# 基础辅助函数
# ========================


def read_csv_if_exists(path, chunksize=0, nrows=None):
    """尝试读取 CSV 文件（支持分块）"""
    p = Path(path)
    if not p.exists():
        logging.warning(f"未找到文件: {path}")
        return None
    try:
        if chunksize > 0:
            logging.info(f"以分块模式读取 {p.name}, chunksize={chunksize}")
            return pd.read_csv(p, chunksize=chunksize)
        else:
            df = pd.read_csv(p, nrows=nrows)
            logging.info(f"读取 {p.name}: shape={df.shape}")
            return df
    except Exception as e:
        logging.error(f"读取失败 {p}: {e}")
        return None


def save_df(df, filename, out_dir):
    """保存清洗结果为 CSV 文件"""
    out_path = Path(out_dir) / f"{filename}_clean.csv"
    df.to_csv(out_path, index=False)
    logging.info(f"输出文件 {out_path} (shape={df.shape})")
    return out_path


def boolify01(series):
    """将 '0'/'1'/'True'/'False' 等转换为布尔值"""
    return (
        series.astype(str)
        .str.strip()
        .replace({"True": "1", "False": "0", "true": "1", "false": "0"})
        .astype(float)
        .fillna(0)
        .astype(int)
        .astype(bool)
    )


def ensure_datetime(series, unit=None):
    """将序列转为 datetime 类型"""
    try:
        return pd.to_datetime(series, unit=unit, errors="coerce")
    except Exception:
        return pd.to_datetime(pd.Series([pd.NaT] * len(series)))


def clamp(series, lo=None, hi=None):
    """将数值序列限制在 [lo, hi] 范围内"""
    s = pd.to_numeric(series, errors="coerce")
    if lo is not None:
        s = np.maximum(s, lo)
    if hi is not None:
        s = np.minimum(s, hi)
    return s


def add_team_from_slot(df, slot_col="player_slot"):
    """根据 player_slot 判断 Radiant / Dire 阵营"""
    if slot_col in df.columns:
        slot = pd.to_numeric(df[slot_col], errors="coerce").fillna(0).astype(int)
        df["team"] = np.where(slot < 128, "Radiant", "Dire")
    return df



# ========================
# Dota2 清洗函数
# ========================

def clean_matches(df, cluster_df=None, patch_df=None):
    """清洗 match.csv 比赛主表"""
    df = df.copy()

    if "radiant_win" in df.columns:
        df["radiant_win"] = boolify01(df["radiant_win"])

    if "start_time" in df.columns:
        df["start_time_dt"] = ensure_datetime(df["start_time"], unit="s")

    if "duration" in df.columns:
        df["long_match"] = pd.to_numeric(df["duration"], errors="coerce") > 7200

    if cluster_df is not None and "cluster" in df.columns and "cluster" in cluster_df.columns:
        df = df.merge(cluster_df.drop_duplicates("cluster"), on="cluster", how="left")
        if "region" in df.columns:
            miss = df["region"].isna().sum()
            if miss > 0:
                logging.warning(f"[matches] {miss} 条记录 cluster 未映射 region")

    if patch_df is not None and "start_time_dt" in df.columns and "date" in patch_df.columns:
        pdf = patch_df.sort_values("date")[["patch", "date"]].rename(columns={"date": "patch_date"})
        try:
            df = pd.merge_asof(
                df.sort_values("start_time_dt"),
                pdf.sort_values("patch_date"),
                left_on="start_time_dt",
                right_on="patch_date",
                direction="backward",
            )
        except Exception as e:
            logging.warning(f"[matches] 补丁版本合并失败: {e}")

    if "match_id" in df.columns:
        before = len(df)
        df = df.drop_duplicates("match_id", keep="first")
        dup = before - len(df)
        if dup > 0:
            logging.warning(f"[matches] match_id 重复 {dup} 条，已删除")

    return df


def clean_players(df, hero_map=None):
    """清洗 players.csv 玩家表"""
    df = df.copy()

    if not SKIP_MISSING:
        num_cols = df.select_dtypes(include=[np.number]).columns
        na_count = df[num_cols].isna().sum().sum()
        df[num_cols] = df[num_cols].fillna(0)
        if na_count > 0:
            logging.info(f"[players] 填充数值缺失 {na_count} 个为 0")

    keys = [c for c in ["match_id", "player_slot"] if c in df.columns]
    if keys:
        before = len(df)
        df = df.drop_duplicates(keys, keep="first")
        dup = before - len(df)
        if dup > 0:
            logging.warning(f"[players] 重复 (match_id, player_slot) {dup} 条")

    df = add_team_from_slot(df, "player_slot")

    if "kills" in df.columns:
        kills_num = pd.to_numeric(df["kills"], errors="coerce")
        mask = kills_num > 100
        df["kills_outlier"] = False
        if mask.any():
            df.loc[mask, "kills"] = 100
            df.loc[mask, "kills_outlier"] = True
            logging.warning(f"[players] 截断 kills>100 记录 {mask.sum()} 条")

    if hero_map is not None and "hero_id" in df.columns and "hero_id_key" in hero_map.columns:
        df["hero_id"] = pd.to_numeric(df["hero_id"], errors="coerce")
        df = df.merge(hero_map[["hero_id_key", "hero_name"]], left_on="hero_id", right_on="hero_id_key", how="left")
        if "hero_name" in df.columns:
            miss = df["hero_name"].isna().sum()
            if miss > 0:
                logging.warning(f"[players] hero_id 未映射 {miss} 条")
        df.drop(columns=["hero_id_key"], inplace=True, errors="ignore")

    return df


def clean_player_time(df):
    """清洗 player_time.csv"""
    df = df.copy()
    if "times" in df.columns:
        before = len(df)
        df = df[pd.to_numeric(df["times"], errors="coerce") >= 0]
        removed = before - len(df)
        if removed > 0:
            logging.warning(f"[player_time] 删除 times < 0 的记录 {removed} 条")
    return df

def clean_teamfights(df):
    """清洗 teamfights.csv"""
    df = df.copy()
    if {"start", "end"}.issubset(df.columns):
        s = pd.to_numeric(df["start"], errors="coerce")
        e = pd.to_numeric(df["end"], errors="coerce")
        before = len(df)
        df = df[(s >= 0) & (e >= 0) & (e > s)]
        removed = before - len(df)
        if removed > 0:
            logging.warning(f"[teamfights] 删除时间异常记录 {removed} 条")
    return df


def clean_teamfights_players(df):
    """清洗 teamfights_players.csv"""
    df = df.copy()
    if not SKIP_MISSING:
        num_cols = df.select_dtypes(include=[np.number]).columns
        na_count = df[num_cols].isna().sum().sum()
        df[num_cols] = df[num_cols].fillna(0)
        if na_count > 0:
            logging.info(f"[teamfights_players] 填充数值缺失 {na_count} 个为 0")

    keys = [c for c in ["match_id", "player_slot", "damage", "deaths"] if c in df.columns]
    if keys:
        before = len(df)
        df = df.drop_duplicates(keys, keep="first")
        dup = before - len(df)
        if dup > 0:
            logging.warning(f"[teamfights_players] 删除重复记录 {dup} 条")

    df = add_team_from_slot(df, "player_slot")
    return df


def clean_objectives(df):
    """清洗 objectives.csv"""
    df = df.copy()
    if "slot" in df.columns and "player_slot" not in df.columns:
        df = df.rename(columns={"slot": "player_slot"})
    if "subtype" in df.columns and "type" not in df.columns:
        df = df.rename(columns={"subtype": "type"})

    if "time" in df.columns:
        before = len(df)
        t = pd.to_numeric(df["time"], errors="coerce")
        df = df[(t >= -120) | t.isna()]
        removed = before - len(df)
        if removed > 0:
            logging.warning(f"[objectives] 删除 time 异常记录 {removed} 条")

    if "team" not in df.columns and "player_slot" in df.columns:
        df = add_team_from_slot(df, "player_slot")

    keys = [c for c in ["match_id", "time", "type", "key", "team"] if c in df.columns]
    if keys:
        before = len(df)
        df = df.drop_duplicates(keys, keep="first")
        dup = before - len(df)
        if dup > 0:
            logging.warning(f"[objectives] 删除重复记录 {dup} 条")

    if "type" not in df.columns:
        df["type"] = "unknown"
    return df


def clean_chat(df):
    """清洗 chat.csv"""
    df = df.copy()
    if "slot" in df.columns and "player_slot" not in df.columns:
        df = df.rename(columns={"slot": "player_slot"})
    if "time" in df.columns:
        before = len(df)
        t = pd.to_numeric(df["time"], errors="coerce")
        df = df[(t >= -120) | t.isna()]
        removed = before - len(df)
        if removed > 0:
            logging.warning(f"[chat] 删除 time 异常记录 {removed} 条")
    df = add_team_from_slot(df, "player_slot")
    return df


def clean_test_labels(df):
    """清洗 test_labels.csv"""
    df = df.copy()
    if "radiant_win" in df.columns:
        df["radiant_win"] = boolify01(df["radiant_win"])
    if "match_id" in df.columns:
        before = len(df)
        df = df.drop_duplicates("match_id", keep="first")
        dup = before - len(df)
        if dup > 0:
            logging.warning(f"[test_labels] 删除重复 match_id {dup} 条")
    return df


def clean_test_player(df, hero_map=None):
    """清洗 test_player.csv"""
    df = df.copy()
    if not SKIP_MISSING:
        required = [c for c in ["match_id", "player_slot", "hero_id"] if c in df.columns]
        if required:
            before = len(df)
            df = df.dropna(subset=required)
            removed = before - len(df)
            if removed > 0:
                logging.warning(f"[test_player] 删除缺失关键字段记录 {removed} 条")
    keys = [c for c in ["match_id", "player_slot"] if c in df.columns]
    if keys:
        before = len(df)
        df = df.drop_duplicates(keys, keep="first")
        dup = before - len(df)
        if dup > 0:
            logging.warning(f"[test_player] 删除重复 (match_id, player_slot) {dup} 条")
    df = add_team_from_slot(df, "player_slot")
    return df

def clean_player_ratings(df):
    """清洗 player_ratings.csv 玩家评分表"""
    df = df.copy()

    if {"total_matches", "total_wins"}.issubset(df.columns):
        df["total_matches"] = clamp(df["total_matches"], lo=0)
        df["total_wins"] = clamp(df["total_wins"], lo=0)
        bad = df["total_wins"] > df["total_matches"]
        if bad.any():
            df.loc[bad, "total_wins"] = df.loc[bad, "total_matches"]
            logging.warning(f"[player_ratings] 修正 total_wins>total_matches {bad.sum()} 条")

    for c in ["trueskill_mu", "trueskill_sigma"]:
        if c in df.columns:
            df[c] = clamp(df[c], lo=0)

    if "account_id" in df.columns:
        before = len(df)
        df = df.drop_duplicates("account_id", keep="first")
        dup = before - len(df)
        if dup > 0:
            logging.warning(f"[player_ratings] 删除重复 account_id {dup} 条")
    return df


def clean_match_outcomes(df):
    """清洗 match_outcomes.csv"""
    df = df.copy()
    if "win" in df.columns:
        df["win"] = pd.to_numeric(df["win"], errors="coerce").fillna(0).astype(int)
    if "rad" in df.columns:
        df["rad"] = pd.to_numeric(df["rad"], errors="coerce").fillna(0).astype(int)

    if "match_id" in df.columns:
        before = len(df)
        df = df.drop_duplicates("match_id", keep="first")
        dup = before - len(df)
        if dup > 0:
            logging.warning(f"[match_outcomes] 删除重复 match_id {dup} 条")
    return df


def clean_purchase_log(df, item_df=None):
    """清洗 purchase_log.csv 购买记录"""
    df = df.copy()
    if "time" in df.columns:
        before = len(df)
        t = pd.to_numeric(df["time"], errors="coerce")
        df = df[(t >= -120) | t.isna()]
        removed = before - len(df)
        if removed > 0:
            logging.warning(f"[purchase_log] 删除 time 异常记录 {removed} 条")
        df["minute"] = (t // 60).astype("Int64")

    if item_df is not None and "item_id" in df.columns:
        df = df.merge(item_df[["item_id", "item_name"]], on="item_id", how="left")
        if "item_name" in df.columns:
            miss = df["item_name"].isna().sum()
            if miss > 0:
                logging.warning(f"[purchase_log] item_id 未映射 {miss} 条")

    df = add_team_from_slot(df, "player_slot")
    return df


def clean_ability_upgrades(df, ability_df=None):
    """清洗 ability_upgrades.csv"""
    df = df.copy()
    if "ability" in df.columns and "ability_id" not in df.columns:
        df = df.rename(columns={"ability": "ability_id"})

    if "level" in df.columns:
        df["level"] = clamp(df["level"], lo=1)

    if "time" in df.columns:
        before = len(df)
        t = pd.to_numeric(df["time"], errors="coerce")
        df = df[(t >= -120) | t.isna()]
        removed = before - len(df)
        if removed > 0:
            logging.warning(f"[ability_upgrades] 删除 time 异常记录 {removed} 条")

    if ability_df is not None and "ability_id" in df.columns:
        df = df.merge(ability_df[["ability_id", "ability_name"]], on="ability_id", how="left")
        if "ability_name" in df.columns:
            miss = df["ability_name"].isna().sum()
            if miss > 0:
                logging.warning(f"[ability_upgrades] ability_id 未映射 {miss} 条")

    df = add_team_from_slot(df, "player_slot")
    return df

# ========================
# 通用文件处理函数
# ========================

def process_file(file_path, args, output_dir, func_map,
                 cluster_df, patch_df, hero_map, item_df, ability_df):
    """
    通用文件处理函数（支持分块读取与并行清洗）
    用于单文件或目录模式。
    """
    fname = file_path.stem.lower()
    func = func_map.get(fname)
    if func is None:
        logging.warning(f"跳过未识别文件: {fname}")
        return

    # 特定文件类型传递额外映射参数
    extra_kwargs = {}
    if fname == "players":
        extra_kwargs["hero_map"] = hero_map
    elif fname == "purchase_log":
        extra_kwargs["item_df"] = item_df
    elif fname == "ability_upgrades":
        extra_kwargs["ability_df"] = ability_df
    elif fname == "match":
        extra_kwargs.update({"cluster_df": cluster_df, "patch_df": patch_df})

    reader = read_csv_if_exists(file_path, chunksize=args.chunksize)
    if reader is None:
        return

    # 普通模式：整文件读取
    if isinstance(reader, pd.DataFrame):
        df_clean = func(reader, **extra_kwargs)
        save_df(df_clean, fname, output_dir)
        return

    # 分块并行模式
    logging.info(f"[{fname}] 启用分块并行清洗 (chunksize={args.chunksize}, jobs={args.jobs})")
    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(func, chunk, **extra_kwargs) for chunk in reader]
        for i, fut in enumerate(futures):
            try:
                chunk_result = fut.result()
                results.append(chunk_result)
                logging.info(f"[{fname}] 分块 {i+1} 清洗完成")
            except Exception as e:
                logging.error(f"[{fname}] 分块 {i+1} 失败: {e}")

    if results:
        df_final = pd.concat(results, ignore_index=True)
        save_df(df_final, fname, output_dir)
        logging.info(f"[{fname}] 所有分块清洗完成，总行数={len(df_final)}")

# ========================
# 主函数入口
# ========================

def main():
    parser = argparse.ArgumentParser(description="Dota2 专用数据清洗脚本")
    parser.add_argument("-i", "--input", required=True, help="输入文件或目录路径")
    parser.add_argument("-o", "--output", required=True, help="输出文件目录")
    parser.add_argument("--skip-missing", action="store_true", help="跳过缺失值填充")
    parser.add_argument("--log", action="store_true", help="显示详细清洗日志")
    parser.add_argument("--chunksize", type=int, default=0, help="分块读取行数（0 表示不分块）")
    parser.add_argument("--jobs", type=int, default=1, help="多线程并行数（>1 启用并行）")
    args = parser.parse_args()

    log_level = logging.INFO if args.log else logging.WARNING

    # 创建 .logs 目录
    log_dir = Path(".logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # 日志文件名带时间戳
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"cleaner_{timestamp}.log"

    # 配置 logging 模块：输出到文件 + 控制台
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),   # 写入文件
            logging.StreamHandler(sys.stdout)                  # 同时打印到控制台
        ]
    )

    logging.info(f"日志输出文件: {log_file}")


    global SKIP_MISSING
    SKIP_MISSING = args.skip_missing

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    func_map = {
        "match": clean_matches,
        "players": clean_players,
        "player_time": clean_player_time,
        "teamfights": clean_teamfights,
        "teamfights_players": clean_teamfights_players,
        "objectives": clean_objectives,
        "chat": clean_chat,
        "test_labels": clean_test_labels,
        "test_player": clean_test_player,
        "player_ratings": clean_player_ratings,
        "match_outcomes": clean_match_outcomes,
        "purchase_log": clean_purchase_log,
        "ability_upgrades": clean_ability_upgrades,
    }
    # ========== 映射表载入部分 ==========
    logging.info("正在载入映射表...")

    base_dir = input_path if input_path.is_dir() else input_path.parent

    cluster_df = read_csv_if_exists(base_dir / 'cluster_regions.csv')
    patch_df   = read_csv_if_exists(base_dir / 'patch_dates.csv')
    hero_df    = read_csv_if_exists(base_dir / 'hero_names.csv')
    item_df    = read_csv_if_exists(base_dir / 'item_ids.csv')
    ability_df = read_csv_if_exists(base_dir / 'ability_ids.csv')

    if patch_df is not None:
        patch_df = patch_df.rename(columns={'patch_date':'date','name':'patch'})
        patch_df['date'] = pd.to_datetime(patch_df['date'], errors='coerce')
        patch_df['date'] = patch_df['date'].dt.tz_localize(None)
        patch_df['patch'] = patch_df['patch'].astype(str)
        logging.info(f"[MAP] patch_df rows = {len(patch_df)}")
    else:
        logging.warning("未找到 patch_dates.csv，跳过补丁映射")

    hero_map = None
    if hero_df is not None:
        h = hero_df.copy()
        if 'hero_name' not in h.columns and 'localized_name' in h.columns:
            h = h.rename(columns={'localized_name': 'hero_name'})
        if 'hero_id' in h.columns:
            h['hero_id_key'] = pd.to_numeric(h['hero_id'], errors='coerce')
            hero_map = h[['hero_id_key','hero_name']].drop_duplicates('hero_id_key')
            logging.info(f"[MAP] hero_map rows = {len(hero_map)}")
    else:
        logging.warning("未找到 hero_names.csv，跳过英雄映射")

    if item_df is not None:
        item_df = item_df[['item_id','item_name']].drop_duplicates('item_id')
        logging.info(f"[MAP] item_df rows = {len(item_df)}")
    else:
        logging.warning("未找到 item_ids.csv，跳过物品映射")

    if ability_df is not None:
        ability_df = ability_df[['ability_id','ability_name']].drop_duplicates('ability_id')
        logging.info(f"[MAP] ability_df rows = {len(ability_df)}")
    else:
        logging.warning("未找到 ability_ids.csv，跳过技能映射")

    logging.info("映射表载入完成。")

    # 单文件模式
    if input_path.is_file():
        logging.info("=" * 70)
        logging.info(f"检测到单文件模式: {input_path.name}")
        logging.info(f"chunksize = {args.chunksize}, jobs = {args.jobs}")
        logging.info(f"输出目录: {output_dir}")
        logging.info("=" * 70)

        process_file(input_path, args, output_dir, func_map,
                     cluster_df, patch_df, hero_map, item_df, ability_df)

        logging.info(f"单文件清洗完成: {input_path.name}")
        logging.info("=" * 70)
        return

    # 目录模式
    elif input_path.is_dir():
        files = list(input_path.glob("*.csv"))
        if not files:
            logging.error("未找到 CSV 文件。")
            sys.exit(1)

        if args.jobs > 1:
            with ThreadPoolExecutor(max_workers=args.jobs) as pool:
                futures = [pool.submit(process_file, f, args, output_dir, func_map,
                                       cluster_df, patch_df, hero_map, item_df, ability_df) for f in files]
                for fut in futures:
                    fut.result()
        else:
            for f in files:
                process_file(f, args, output_dir, func_map,
                             cluster_df, patch_df, hero_map, item_df, ability_df)
        logging.info("所有 Dota2 CSV 文件清洗完成。")
    else:
        logging.error("输入路径不是有效的文件或目录。")
        sys.exit(1)



if __name__ == "__main__":
    main()

