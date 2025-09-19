import re
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import ks_2samp
from scipy.stats import wasserstein_distance

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Iterable

# ---------- 表名相似度 ----------
def table_name_similarity(src_table_name: str, tgt_table_name: str, tfidf_model=None) -> float:
    a = (src_table_name or "").replace("_", " ").lower()
    b = (tgt_table_name or "").replace("_", " ").lower()
    jac = jaccard_char_ngrams(a, b, 3)
    if tfidf_model is not None:
        X = tfidf_model.transform([a, b])
        cos = cosine_similarity(X[0:1], X[1:2])[0, 0]
    else:
        cos = jac
    return 0.6 * cos + 0.4 * jac

# ---------- 工具函数 ----------
def jaccard_char_ngrams(a: str, b: str, n: int = 3) -> float:
    if a is None or b is None:
        return 0.0
    A = {a[i:i+n] for i in range(max(1, len(a)-n+1))} if a else set()
    B = {b[i:i+n] for i in range(max(1, len(b)-n+1))} if b else set()
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def safe_is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)

def topk_categories(series: pd.Series, k: int = 20) -> set:
    vc = series.dropna().astype(str).value_counts()
    return set(vc.head(k).index)

def perc_unique(series: pd.Series) -> float:
    s = series.dropna()
    return 0.0 if len(s) == 0 else s.nunique() / len(s)

def perc_missing(series: pd.Series) -> float:
    return series.isna().mean()

def normalize01(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))

# ---------- 名称相似度（TF-IDF + Jaccard） ----------
def name_similarity(src_name: str, tgt_name: str, tfidf_model=None) -> float:
    a = (src_name or "").replace("_", " ")
    b = (tgt_name or "").replace("_", " ")
    # Jaccard over char 3-grams
    jac = jaccard_char_ngrams(a.lower(), b.lower(), 3)
    # Cosine over TF-IDF (fallback到Jaccard)
    if tfidf_model is not None:
        X = tfidf_model.transform([a, b])
        cos = cosine_similarity(X[0:1], X[1:2])[0,0]
    else:
        cos = jac
    return 0.6 * cos + 0.4 * jac

# ---------- 结构相似度 ----------
def structural_similarity(a: pd.Series, b: pd.Series) -> float:
    # 类型相似（数值/非数值）
    type_sim = 1.0 if safe_is_numeric(a) == safe_is_numeric(b) else 0.0
    # 缺失率接近
    miss_a, miss_b = perc_missing(a), perc_missing(b)
    miss_sim = 1.0 - min(1.0, abs(miss_a - miss_b) / 0.5)  # 允许较大差异衰减
    # 唯一率接近
    uniq_a, uniq_b = perc_unique(a), perc_unique(b)
    uniq_sim = 1.0 - min(1.0, abs(uniq_a - uniq_b) / 0.7)
    return 0.5*type_sim + 0.25*miss_sim + 0.25*uniq_sim

# ---------- 分布相似度 ----------
def distribution_similarity(a: pd.Series, b: pd.Series) -> float:
    a_clean = a.dropna()
    b_clean = b.dropna()
    if len(a_clean) < 5 or len(b_clean) < 5:
        return 0.5  # 数据太少，给中性分
    if safe_is_numeric(a) and safe_is_numeric(b):
        # KS 越小越相似；Wasserstein 越小越相似
        ks_stat, _ = ks_2samp(a_clean, b_clean, alternative='two-sided', mode='auto')
        wass = wasserstein_distance(a_clean, b_clean)
        # 简单双指标归一化：经验性裁剪
        ks_sim = 1.0 - min(1.0, ks_stat)          # KS ∈ [0,1]
        wass_sim = 1.0 - min(1.0, normalize01(wass, 0.0, np.nanmax([a_clean.std(), b_clean.std(), 1e-6])*2))
        return 0.6*ks_sim + 0.4*wass_sim
    else:
        # 类别分布：top-k Jaccard
        ja = topk_categories(a_clean)
        jb = topk_categories(b_clean)
        if not ja and not jb:
            return 1.0
        if not ja or not jb:
            return 0.0
        return len(ja & jb)/len(ja | jb)

# ---------- 内容相似度（按规则先验匹配率） ----------
def content_similarity_via_rule(b: pd.Series, rule: Dict[str, Any], sample_n: int = 200) -> float:
    b_sample = b.dropna().sample(min(sample_n, b.dropna().shape[0]), random_state=42) if b.dropna().shape[0] > 0 else pd.Series([], dtype=b.dtype)
    if len(b_sample) == 0:
        return 0.5
    rt = rule.get("rule_type")
    p = rule.get("params", {})
    if rt == "regex":
        pattern = re.compile(p["pattern"])
        matches = b_sample.astype(str).apply(lambda x: bool(pattern.fullmatch(x)))
        return matches.mean()
    elif rt == "range" and safe_is_numeric(b):
        lo, hi = p.get("min", -math.inf), p.get("max", math.inf)
        ok = b_sample.between(lo, hi, inclusive="both")
        return ok.mean()
    elif rt == "enum":
        allowed = set(map(str, p.get("values", [])))
        return b_sample.astype(str).isin(allowed).mean()
    elif rt == "date_format":
        fmt = p.get("format")
        def try_parse(x):
            try:
                pd.to_datetime(x, format=fmt, errors="raise")
                return True
            except Exception:
                return False
        return b_sample.astype(str).apply(try_parse).mean()
    # 其他类型默认中性分
    return 0.5

# ---------- 列匹配器 ----------
@dataclass
class MatchWeights:
    w_table: float = 0.90
    w_name: float = 0.35
    w_struct: float = 0.25
    w_dist: float = 0.20
    w_content: float = 0.20

# ---------- 列匹配器：支持表名 ----------
class ColumnMatcher:
    def __init__(
        self,
        src_df: pd.DataFrame,
        tgt_df: pd.DataFrame,
        rule_hint: Optional[Dict[str, Any]] = None,
        weights: MatchWeights = MatchWeights(),
        src_table_name: str = "source_table",
        tgt_table_name: str = "target_table",
    ):
        self.src_df = src_df
        self.tgt_df = tgt_df
        self.rule_hint = rule_hint
        self.weights = weights
        self.src_table_name = src_table_name
        self.tgt_table_name = tgt_table_name

        # 为列名与表名共同构建 TF-IDF 语料
        corpus = (
            [src_table_name.replace("_"," "), tgt_table_name.replace("_"," ")]
            + [c.replace("_"," ") for c in list(src_df.columns) + list(tgt_df.columns)]
        )
        self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2,4))
        self.vectorizer.fit(corpus)

        # 预先计算表名相似度（整张表都共享）
        self.s_table = table_name_similarity(self.src_table_name, self.tgt_table_name, self.vectorizer)

    def match_column(self, src_col: str, topk: int = 3) -> List[Tuple[str, float, Dict[str,float]]]:
        a = self.src_df[src_col]
        candidates = []
        for tgt_col in self.tgt_df.columns:
            b = self.tgt_df[tgt_col]
            s_name = name_similarity(src_col, tgt_col, self.vectorizer)
            s_struct = structural_similarity(a, b)
            s_dist = distribution_similarity(a, b)
            s_content = content_similarity_via_rule(b, self.rule_hint) if self.rule_hint else 0.5
            s_table = self.s_table  # 新增：表名先验

            total = (self.weights.w_name*s_name +
                     self.weights.w_struct*s_struct +
                     self.weights.w_dist*s_dist +
                     self.weights.w_content*s_content +
                     self.weights.w_table*s_table)

            candidates.append((tgt_col, float(total), {
                "name": float(s_name),
                "struct": float(s_struct),
                "dist": float(s_dist),
                "content": float(s_content),
                "table": float(s_table),   # 新增：在分解里展示
            }))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:topk]

# ---------- 规则引擎（模板化 + 执行） ----------
class RuleEngine:
    def __init__(self, rules: List[Dict[str, Any]]):
        self.rules = rules

    @staticmethod
    def bind_rule(rule: Dict[str, Any], column: str) -> Dict[str, Any]:
        r = dict(rule)
        r["column"] = column
        return r

    @staticmethod
    def apply_rule(df: pd.DataFrame, rule: Dict[str, Any], id_col: Optional[str] = None) -> pd.DataFrame:
        col = rule["column"]
        rt = rule["rule_type"]
        params = rule.get("params", {})
        sev = rule.get("severity", "error")

        s = df[col]
        ok_mask = pd.Series([True]*len(df), index=df.index)

        if rt == "regex":
            pattern = re.compile(params["pattern"])
            ok_mask = s.astype(str).apply(lambda x: bool(pattern.fullmatch(x)) if pd.notna(x) else False)

        elif rt == "range":
            lo = params.get("min", -math.inf)
            hi = params.get("max", math.inf)
            if not safe_is_numeric(s):
                ok_mask = pd.Series([False]*len(df), index=df.index)
            else:
                ok_mask = s.between(lo, hi, inclusive="both")

        elif rt == "enum":
            allowed = set(map(str, params.get("values", [])))
            ok_mask = s.astype(str).isin(allowed)

        elif rt == "date_format":
            fmt = params.get("format")
            def try_parse(x):
                try:
                    pd.to_datetime(x, format=fmt, errors="raise")
                    return True
                except Exception:
                    return False
            ok_mask = s.astype(str).apply(lambda x: try_parse(x) if pd.notna(x) else False)

        else:
            # 未实现的类型，全部置为通过（也可选择全部不通过）
            ok_mask = pd.Series([True]*len(df), index=df.index)

        violations = df.loc[~ok_mask, [col]].copy()
        if id_col and id_col in df.columns:
            violations.insert(0, "row_id", df[id_col])
        violations["rule_type"] = rt
        violations["severity"] = sev
        return violations.reset_index(drop=True)

class MultiTableMatcher:
    """
    在多个候选目标表之间做全局列匹配。
    """
    def __init__(
        self,
        src_df: pd.DataFrame,
        src_table_name: str,
        targets: Iterable[Tuple[pd.DataFrame, str]],  # [(tgt_df, tgt_table_name), ...]
        rule_hint: Optional[Dict[str, Any]] = None,
        weights: MatchWeights = MatchWeights(),
    ):
        self.src_df = src_df
        self.src_table_name = src_table_name
        self.targets = list(targets)
        self.rule_hint = rule_hint
        self.weights = weights

    def match(
        self,
        src_col: str,
        topk_global: int = 5,
        topk_per_table: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        返回跨表全局 top-k 匹配结果。
        每个结果含：target_table, target_column, total_score, breakdown(dict)
        """
        all_candidates: List[Dict[str, Any]] = []
        for tgt_df, tgt_table_name in self.targets:
            # 对每张目标表，构造一个 ColumnMatcher（带表名先验）
            matcher = ColumnMatcher(
                self.src_df, tgt_df,
                rule_hint=self.rule_hint,
                weights=self.weights,
                src_table_name=self.src_table_name,
                tgt_table_name=tgt_table_name,
            )
            # 在该表内取 topk_per_table
            per_table = matcher.match_column(src_col, topk=topk_per_table)
            for col, score, breakdown in per_table:
                all_candidates.append({
                    "target_table": tgt_table_name,
                    "target_column": col,
                    "total_score": float(score),
                    "breakdown": breakdown,  # 含 name/struct/dist/content/table
                })

        # 全局排序
        all_candidates.sort(key=lambda x: x["total_score"], reverse=True)
        return all_candidates[:topk_global]

# ------- 一个简短的使用示例 -------
if __name__ == "__main__":
    # 源表 A
    A = pd.DataFrame({
        "user_id": [1, 2, 3, 4],
        "phone_number": ["13812345678", "15987654321", "18600001111", "13199998888"]
    })

    phone_rule = {
        "rule_type": "regex",
        "template": "<column> must match <pattern>",
        "params": {"pattern": r"^1\d{10}$"},
        "severity": "error"
    }

    # 目标表们（示例）
    B1 = pd.DataFrame({
        "id": [1, 2],
        "mobile": ["15800001234", "81234567"]
    })
    B2 = pd.DataFrame({
        "user_pk": [10, 20],
        "msisdn": ["13912345678", "15622223333"]
    })

    mtm = MultiTableMatcher(
        src_df=A,
        src_table_name="users",
        targets=[(B1, "user_mobile"), (B2, "accounts")],
        rule_hint=phone_rule,
        weights=MatchWeights(),  # 可按需调 w_table
    )

    results = mtm.match(
        src_col="phone_number",
        topk_global=5,      # 全局取前 5
        topk_per_table=3,   # 每张表先取前 3
    )

    print("Global Top-K candidates:")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['target_table']}.{r['target_column']} "
              f"score={r['total_score']:.3f} breakdown={r['breakdown']}")

@dataclass
class BatchMatchResult:
    # DataFrame：每行一个候选（源列 × 目标表.列）
    candidates_df: pd.DataFrame
    # 可选：1-1 匹配结果（若执行了全局分配）
    assignment_df: Optional[pd.DataFrame] = None

class BatchMatcher:
    """
    批量处理多个源列，在多个候选目标表之间做列匹配。
    """
    def __init__(
        self,
        src_df: pd.DataFrame,
        src_table_name: str,
        targets: Iterable[Tuple[pd.DataFrame, str]],  # [(tgt_df, tgt_table_name), ...]
        rule_hint: Optional[Dict[str, Any]] = None,
        weights: MatchWeights = MatchWeights(),
    ):
        self.src_df = src_df
        self.src_table_name = src_table_name
        self.targets = list(targets)
        self.rule_hint = rule_hint
        self.weights = weights

    def match_many(
        self,
        src_cols: List[str],
        topk_per_src: int = 5,
        topk_per_table: int = 5,
        score_threshold: float = 0.0,  # <— 先过滤掉很弱的候选
        do_global_assignment: bool = False,
    ) -> BatchMatchResult:
        # 1) 收集所有候选（独立匹配）
        rows = []
        for src_col in src_cols:
            mtm = MultiTableMatcher(
                src_df=self.src_df,
                src_table_name=self.src_table_name,
                targets=self.targets,
                rule_hint=self.rule_hint,
                weights=self.weights,
            )
            results = mtm.match(
                src_col=src_col,
                topk_global=topk_per_src,
                topk_per_table=topk_per_table,
            )
            for r in results:
                if r["total_score"] >= score_threshold:
                    rows.append({
                        "src_table": self.src_table_name,
                        "src_column": src_col,
                        "tgt_table": r["target_table"],
                        "tgt_column": r["target_column"],
                        "total_score": r["total_score"],
                        # 展开子分数，便于可解释
                        "score_name": r["breakdown"].get("name", np.nan),
                        "score_struct": r["breakdown"].get("struct", np.nan),
                        "score_dist": r["breakdown"].get("dist", np.nan),
                        "score_content": r["breakdown"].get("content", np.nan),
                        "score_table": r["breakdown"].get("table", np.nan),
                    })
        candidates_df = pd.DataFrame(rows).sort_values("total_score", ascending=False).reset_index(drop=True)

        assignment_df = None
        if do_global_assignment and not candidates_df.empty:
            # 2) 构造 1-1 分配的得分矩阵（源列 × 目标全列）
            #    目标列用 “表名.列名” 唯一标识
            tgt_all = []
            for tgt_df, tgt_table in self.targets:
                for col in tgt_df.columns:
                    tgt_all.append(f"{tgt_table}.{col}")
            tgt_all = sorted(set(tgt_all))

            src_all = list(dict.fromkeys(src_cols))  # 去重保序

            # 初始化为 -inf（表示不可选）；用负号做“成本”，便于匈牙利最小化
            cost = np.full((len(src_all), len(tgt_all)), fill_value=1e6, dtype=float)

            # 把已有候选的分数写入矩阵
            best_seen = {}
            for _, row in candidates_df.iterrows():
                src = row["src_column"]
                tgt = f"{row['tgt_table']}.{row['tgt_column']}"
                s = float(row["total_score"])
                i = src_all.index(src)
                j = tgt_all.index(tgt)
                # 如果同一 (src, tgt) 出现多次（不同 per_table 跑法），取最大分
                if (i, j) not in best_seen or s > best_seen[(i, j)]:
                    best_seen[(i, j)] = s

            for (i, j), s in best_seen.items():
                cost[i, j] = -s  # 最大化分数 == 最小化负分

            # 运行匈牙利算法
            row_ind, col_ind = linear_sum_assignment(cost)
            # 只保留那些“可行”的匹配（原始分数> -1e6）
            assigned = []
            for i, j in zip(row_ind, col_ind):
                if cost[i, j] < 1e6:  # 表示有候选
                    src = src_all[i]
                    tgt_table, tgt_col = tgt_all[j].split(".", 1)
                    # 查找对应分数与分解
                    sub = candidates_df[
                        (candidates_df["src_column"] == src) &
                        (candidates_df["tgt_table"] == tgt_table) &
                        (candidates_df["tgt_column"] == tgt_col)
                    ].sort_values("total_score", ascending=False).head(1)
                    if not sub.empty:
                        assigned.append(sub.iloc[0].to_dict())

            if assigned:
                assignment_df = pd.DataFrame(assigned).reset_index(drop=True)

        return BatchMatchResult(candidates_df=candidates_df, assignment_df=assignment_df)

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd

class BatchRuleApplier:
    def __init__(self, targets_dict: Dict[str, pd.DataFrame], id_cols_hint: Optional[Dict[str, str]] = None):
        self.targets = targets_dict
        self.id_cols_hint = id_cols_hint or {}

    @staticmethod
    def _guess_id_col(df: pd.DataFrame) -> Optional[str]:
        # 常见 id 列名优先级
        candidates = ["id", "row_id", "user_id", "user_pk", "pk", "guid"]
        for c in candidates:
            if c in df.columns:
                return c
        # 兜底：若存在唯一整数列，选第一个
        for c in df.columns:
            s = df[c]
            if pd.api.types.is_integer_dtype(s) and s.is_unique:
                return c
        return None

    def _id_col_for_table(self, table: str) -> Optional[str]:
        if table in self.id_cols_hint:
            return self.id_cols_hint[table]
        df = self.targets[table]
        return self._guess_id_col(df)

    def _bind_rules_from_assignment(
        self,
        assignment_df: pd.DataFrame,
        rules_by_src: Dict[str, List[Dict[str, Any]]],
        rule_engine: "RuleEngine",
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        返回 [(tgt_table, bound_rule), ...]
        """
        bound_rules: List[Tuple[str, Dict[str, Any]]] = []
        for _, row in assignment_df.iterrows():
            src_col = row["src_column"]
            tgt_table = row["tgt_table"]
            tgt_col = row["tgt_column"]
            rule_list = rules_by_src.get(src_col, [])
            for tpl in rule_list:
                r = rule_engine.bind_rule(tpl, tgt_col)
                # 给规则补充些可解释元数据（不会影响校验）
                r["_meta"] = {
                    "src_column": src_col,
                    "tgt_table": tgt_table,
                    "tgt_column": tgt_col,
                }
                bound_rules.append((tgt_table, r))
        return bound_rules

    def apply_many(
        self,
        assignment_df: pd.DataFrame,
        rules_by_src: Dict[str, List[Dict[str, Any]]],
        rule_engine: "RuleEngine",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        返回 (violations_df, runs_df)
        """
        # 1) 规则绑定
        bound = self._bind_rules_from_assignment(assignment_df, rules_by_src, rule_engine)

        # 2) 执行 & 汇总
        vio_rows = []
        run_rows = []

        for tgt_table, rule in bound:
            if tgt_table not in self.targets:
                continue  # 防御：不存在的表跳过
            df = self.targets[tgt_table]
            id_col = self._id_col_for_table(tgt_table)

            # 执行
            res = rule_engine.apply_rule(df, rule, id_col=id_col)

            # 统计
            total = len(df)
            bad = len(res)
            run_rows.append({
                "tgt_table": tgt_table,
                "tgt_column": rule.get("column"),
                "src_column": rule.get("_meta", {}).get("src_column"),
                "rule_type": rule.get("rule_type"),
                "severity": rule.get("severity", "error"),
                "total_rows": total,
                "violations": bad,
                "violation_rate": (bad / total) if total else 0.0,
            })

            if bad > 0:
                # 提升可解释性：携带表、源列、目标列、期望描述
                pretty_expect = self._pretty_expect(rule)
                for _, r in res.iterrows():
                    row = {
                        "tgt_table": tgt_table,
                        "tgt_column": rule.get("column"),
                        "src_column": rule.get("_meta", {}).get("src_column"),
                        "rule_type": r.get("rule_type"),
                        "severity": r.get("severity"),
                        "expected": pretty_expect,
                    }
                    # row_id & value
                    if "row_id" in res.columns:
                        row["row_id"] = r["row_id"]
                    # 违规值
                    val_col = rule.get("column")
                    if val_col in r.index:
                        row["value"] = r[val_col]
                    vio_rows.append(row)

        violations_df = pd.DataFrame(vio_rows).reset_index(drop=True)
        runs_df = pd.DataFrame(run_rows).sort_values(["violation_rate","violations"], ascending=False).reset_index(drop=True)
        return violations_df, runs_df

    @staticmethod
    def _pretty_expect(rule: Dict[str, Any]) -> str:
        rt = rule.get("rule_type")
        p = rule.get("params", {})
        if rt == "regex":
            return f"match regex: {p.get('pattern')}"
        if rt == "range":
            lo = p.get("min", "-inf")
            hi = p.get("max", "+inf")
            return f"in range [{lo}, {hi}]"
        if rt == "enum":
            vals = p.get("values", [])
            return f"in set {{{', '.join(map(str, vals))}}}"
        if rt == "date_format":
            return f"date format: {p.get('format')}"
        return rt or "rule"

# ------- 使用示例 -------
if __name__ == "__main__":
    # === 承接上一步 ===
    # 已得到：targets = [(B1,"user_mobile"), (B2,"accounts")]
    targets_dict = {"user_mobile": B1, "accounts": B2}

    # 规则模板：给每个源列一组模板（可多个）
    phone_rule = {
        "rule_type": "regex",
        "template": "<column> must match <pattern>",
        "params": {"pattern": r"^1\d{10}$"},
        "severity": "error"
    }
    date_rule = {
        "rule_type": "date_format",
        "template": "<column> must be date <format>",
        "params": {"format": "%Y-%m-%d"},
        "severity": "warn"
    }
    rules_by_src = {
        "phone_number": [phone_rule],
        "signup_date": [date_rule],
    }

    # 假设这就是上一步 BatchMatcher.match_many(..., do_global_assignment=True) 的输出
    # res.assignment_df（若没有 assignment_df，也可用 res.candidates_df 里每个源列的 top-1 先拼一份）
    assignment_df = res.assignment_df if res.assignment_df is not None else res.candidates_df.groupby("src_column", as_index=False).head(1)

    engine = RuleEngine([])  # 模板执行不需要在构造器里传，后续会 bind
    applier = BatchRuleApplier(targets_dict, id_cols_hint={"user_mobile":"id", "accounts":"user_pk"})

    violations_df, runs_df = applier.apply_many(
        assignment_df=assignment_df,
        rules_by_src=rules_by_src,
        rule_engine=engine,
    )

    print("\n=== Runs Summary ===")
    print(runs_df)

    print("\n=== Violations Detail ===")
    print(violations_df)
