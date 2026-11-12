"""策略工作流：训练集拟合、跨样本评估和可部署规则制品。"""

import json
from functools import reduce
from itertools import islice

import joblib
import pandas as pd

from .analysis import display_table, labeled_data
from .parameters import binning_params, library_options
from .targets import label_map
from .storage import write_json


def bound_frames(workspace, config):
    """读取提交时锁定的数据版本，名称使用角色和标识避免覆盖同名样本。"""
    primary_role = "主评估样本" if config.get("kind") == "ruleset" else "训练集"
    bindings = [(primary_role, {"id": config["dataset_id"], "revision": config["dataset_revision"]})]
    if config.get("test_dataset"):
        bindings.append(("测试集", config["test_dataset"]))
    bindings.extend((f"评估集{i + 1}", item) for i, item in enumerate(config.get("evaluation_datasets", [])))
    return {f"{role} · {workspace.dataset_version(item['id'], item['revision'])['name']}": workspace.read_dataset(item["id"], item["revision"]) for role, item in bindings}


class ReportBundle:
    """登记公共报告计算产生的表格与图片。"""

    def __init__(self, directory, prefix=""):
        self.prefix = prefix
        self.directory, self.tables, self.images, self.artifacts = directory, {}, [], []

    def table(self, title, value):
        if isinstance(value, dict):
            for name, table in value.items():
                self.table(f"{title} / {name}", table)
        elif isinstance(value, pd.DataFrame):
            self.tables[title] = display_table(value)

    def plot(self, title, draw, *, dpi=110):
        from matplotlib import pyplot as plt

        value = draw()
        fig = value if hasattr(value, "savefig") else value.figure if hasattr(value, "figure") else plt.gcf()
        filename = f"{self.prefix}figure_{len(self.images)}.png"
        fig.savefig(self.directory / filename, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        self.images.append({"title": title, "file": filename})
        self.artifacts.append(filename)

    def workbook(self, filename):
        """预览实际 Excel 工作表，包括原报告内嵌图片，保持报告章节顺序。"""
        from openpyxl import load_workbook

        book = load_workbook(self.directory / filename, data_only=True)
        for sheet in book:
            rows = list(islice(sheet.values, 2000))
            self.tables[f"报告预览 / {filename} / {sheet.title}"] = display_table(pd.DataFrame(rows).dropna(axis=1, how="all").rename(columns=lambda c: f"列{c + 1}"))
            for img in sheet._images:
                name = f"{self.prefix}report_image_{len(self.images)}.png"
                (self.directory / name).write_bytes(img._data())
                self.images.append({"title": f"报告预览 / {filename} / {sheet.title}", "file": name})
                self.artifacts.append(name)
        book.close()
        self.artifacts.append(filename)

    def result(self, **extra):
        return {"tables": self.tables, "images": self.images, "artifacts": self.artifacts, **extra}


def rule_payload(rule):
    return {"name": rule.name or "新规则", "expression": rule.expr}


def single_sections(bundle, frames, labels_by_dataset, config, binner, feature):
    from hscredit.report import feature_bin_stats
    from hscredit.core.viz import bin_plot, bin_trend_plot, bin_overdues_plot

    rules = binner.export_rules()
    for name, frame in frames.items():
        prefix = f"{name} / {feature}"
        targets = labels_by_dataset[name]
        table = feature_bin_stats(frame, feature=feature, target=targets, binner=binner, n_jobs=1, **library_options(config, "bin_table"))
        bundle.table(f"{prefix} / 单变量分箱表", table)
        if config.get("amount"):
            bundle.table(f"{prefix} / 金额口径分箱表", feature_bin_stats(frame, feature=feature, target=targets, binner=binner, amount=config["amount"], n_jobs=1, **library_options(config, "bin_table")))
        single_table = feature_bin_stats(frame, feature=feature, target=next(iter(targets.values())), binner=binner, n_jobs=1, **library_options(config, "bin_table"))
        bundle.plot(f"{prefix} / 分箱图", lambda: bin_plot(single_table, **library_options(config, "bin_plot")))
        if len(targets) > 1:
            bundle.plot(f"{prefix} / 多标签分箱图", lambda: bin_overdues_plot(bin_table=table, n_jobs=1, **library_options(config, "target_plot")))
        for key, label in [("date", "时间趋势"), ("group", "分组趋势")]:
            if config.get(key):
                # 每个标签独立计算；完整多标签时间/分组图放在同一字段面板下。
                for target, values in targets.items():
                    plotted = frame.copy()
                    plot_target = target
                    while plot_target in {feature, config[key]}:
                        plot_target = "_" + plot_target
                    plotted[plot_target] = values
                    bundle.plot(f"{prefix} / {label}分箱图 / {target}", lambda: bin_trend_plot(plotted, feature, plot_target, **{key: config[key]}, freq=config["freq"], rules=rules, n_jobs=1, bin_params={"min_bin_size": 1}, **library_options(config, "trend_plot")))

                if len(targets) > 1:
                    dimension = frame[config[key]]
                    if key == "date":
                        dimension = pd.to_datetime(dimension, errors="coerce", utc=True).dt.tz_localize(None).dt.to_period(config["freq"])
                    for group_name, subset in frame.groupby(dimension, sort=True, observed=True, dropna=False):
                        title = "多标签时间趋势分箱图" if key == "date" else "分组多标签分箱图"
                        group_table = feature_bin_stats(subset, feature=feature, target={n: y.loc[subset.index] for n, y in targets.items()}, binner=binner, n_jobs=1, **library_options(config, "bin_table"))
                        bundle.plot(f"{prefix} / {title} / {group_name}", lambda: bin_overdues_plot(bin_table=group_table, n_jobs=1, **{"title": f"{feature} · {group_name}", **library_options(config, "target_plot")}))


def strategy_workflow(workspace, config, directory, progress):
    from hscredit.core.binning import OptimalBinning, OptimalBinning2D
    from hscredit.core.binning.spec import BinSpec
    from hscredit.core.rules import Rule
    from hscredit.report import feature_bin_stats_2d, feature_bin_stats
    from hscredit.core.viz import bin_plot, bin_2d_plot
    from hscredit.report.mining import ManualTreeExtractor, MultiwayTreeExtractor

    frames = bound_frames(workspace, config)
    labels_by_dataset = {name: label_map(frame, config, evaluation=True) for name, frame in frames.items()}
    training_labels = labels_by_dataset[next(iter(frames))]
    primary_name = next(iter(training_labels))
    training, features, target, _ = labeled_data(next(iter(frames.values())), config, training_labels)
    bundle, candidates, branches = ReportBundle(directory), [], []
    params = binning_params(config)
    progress(10, "在训练集拟合分箱与策略")
    binner = OptimalBinning(**{"n_jobs": 1, **params}).fit(training[features], training[target])
    mode = config["strategy_mode"]
    if mode == "single":
        for spec in BinSpec.from_binner(binner, features[0]).values():
            candidates.append(rule_payload(Rule(spec.to_expression(), name=f"{features[0]} · {spec.label()}")))
    elif mode == "pair":
        pair = OptimalBinning2D(method=config["method"], max_n_bins=config["max_n_bins"], min_bin_size=config["min_bin_size"], **{"n_jobs": 1, **library_options(config, "pair_binning")}).fit(training[features], training[target])
        specs = [BinSpec.from_binner(b, f) for b, f in zip([pair.binner_x_, pair.binner_y_], features)]
        # 二维区域可能包含多个网格，按实际二维分箱编号合并，绝不解析展示标签。
        for region, cells in pair.get_cross_table().groupby("二维分箱", sort=False):
            rules = [Rule(f"({specs[0][row['特征1分箱']].to_expression()}) & ({specs[1][row['特征2分箱']].to_expression()})") for _, row in cells.iterrows()]
            rule = reduce(lambda a, b: a | b, rules)
            rule.name = f"二维分箱 {region}"
            candidates.append(rule_payload(rule))
        for name, frame in frames.items():
            table = feature_bin_stats_2d(frame, features=features, target=labels_by_dataset[name], binner=pair, n_jobs=1, **library_options(config, "bin_table"))
            bundle.table(f"{name} / 2D分箱表", table)
            bundle.plot(f"{name} / 2D分箱图", lambda: bin_2d_plot(frame, features=features, target=next(iter(labels_by_dataset[name].values())), binner=pair, **library_options(config, "pair_plot")))
        joblib.dump(pair, directory / "binner.joblib")
        bundle.artifacts.append("binner.joblib")
    else:
        options = {"target": target, "features": features, "max_depth": config["tree_depth"], "min_samples_leaf": config["tree_min_samples"], "n_jobs": 1}
        tree = (ManualTreeExtractor(**options, random_state=config["random_state"], **library_options(config, "binary_tree")) if mode == "binary" else MultiwayTreeExtractor(**options, max_n_bins=config["max_n_bins"], **library_options(config, "multiway_tree"))).fit(training)
        for edit in config.get("tree_edits", []):
            if mode == "binary":
                tree.manual_split(training, feature=edit["feature"], threshold=edit["threshold"], node=edit["node"])
            else:
                tree.manual_split(training, feature=edit["feature"], thresholds=edit.get("thresholds"), category_groups=edit.get("category_groups"), node=edit["node"])
        candidates = [rule_payload(r) for r in tree.get_rules()]
        leaves = {r["name"] for r in candidates}
        branches = [rule_payload(r) for r in tree.get_rules(leaf_only=False) if r.name not in leaves and r.expr != "True"]
        bundle.plot("树图", lambda: tree.plot(backend="matplotlib", **library_options(config, "tree_plot")))
        # 训练标签映射后，其他评估标签保留各自缺失值。
        mapped = {}
        leaf_rules = tree.get_rules()
        node_training = pd.Series(0, index=training.index)
        for index, rule in enumerate(leaf_rules):
            node_training.loc[rule.predict(training).astype(bool)] = index + 1
        node_binner = OptimalBinning(user_splits={"树叶节点": [i + 0.5 for i in range(len(leaf_rules))]}, min_n_bins=1, min_bin_size=1, n_jobs=1).fit(node_training.to_frame("树叶节点"), training[target])
        columns = set().union(*(frame.columns for frame in frames.values()))
        bindings = {}
        for label in training_labels:
            column = label
            while column in columns:
                column = "_" + column
            bindings[label] = column
            columns.add(column)
        for name, frame in frames.items():
            values = labels_by_dataset[name]
            mapped[name] = frame.assign(**{column: values[label] for label, column in bindings.items()})
        labels = {label: (lambda frame, c=column: frame[c]) for label, column in bindings.items()}
        bundle.table("树规则表", tree.report(mapped, target=labels, save=str(directory / "tree_report.xlsx"), date=config.get("date"), group=config.get("group"), amount=config.get("amount"), freq=config["freq"], methods=[config["method"]], bin_params=params, **library_options(config, "tree_report")))
        bundle.workbook("tree_report.xlsx")
        for name, frame in frames.items():
            # 叶节点作为固定类别分箱，复用 feature_bin_stats 与 bin_plot。
            node_values = pd.Series(0, index=frame.index)
            for index, rule in enumerate(leaf_rules):
                node_values.loc[rule.predict(frame).astype(bool)] = index + 1
            plotted = frame.assign(树叶节点=node_values)
            table = feature_bin_stats(plotted, feature="树叶节点", target=labels_by_dataset[name], binner=node_binner, n_jobs=1, **library_options(config, "bin_table"))
            bundle.table(f"{name} / 多变量分箱表", table)
            primary = feature_bin_stats(plotted, feature="树叶节点", target=next(iter(labels_by_dataset[name].values())), binner=node_binner, n_jobs=1, **library_options(config, "bin_table"))
            bundle.plot(f"{name} / 多变量分箱图", lambda: bin_plot(primary, **library_options(config, "bin_plot")))
        joblib.dump(tree, directory / "tree.joblib")
        bundle.artifacts.append("tree.joblib")
    for index, feature in enumerate(features):
        progress(35 + int(55 * index / len(features)), f"评估 {feature} 的分箱、标签与稳定性")
        single_sections(bundle, frames, labels_by_dataset, config, binner, feature)
    # 候选规则在全部绑定数据上评估，页面选择时同时显示其命中报告。
    for index, item in enumerate([*candidates, *branches]):
        item["id"] = str(index)
        rule = Rule(item["expression"], name=item["name"])
        item["metrics"] = {}
        for name, frame in frames.items():
            effect = rule.report(frame, target=labels_by_dataset[name], long_format=True)
            bundle.table(f"候选规则 / {index + 1} / {name}", effect)
            primary = effect.loc[(effect["分箱"] == "命中") & (effect["逾期标签"] == primary_name), ["样本总数", "样本占比", "坏样本率", "LIFT值"]]
            if not primary.empty:
                item["metrics"][name] = display_table(primary.reset_index(drop=True))[0]
    write_json(directory / "rules.json", candidates)
    bundle.artifacts.append("rules.json")
    return bundle.result(candidates=candidates, branches=branches, datasets=list(frames), features=features, strategy_mode=mode, note="所有分箱与树仅在训练集拟合；评估样本使用同一规则。未配置日期、分组或多标签时不生成相应图表。")


def ruleset_workflow(workspace, config, directory, progress):
    from hscredit.core.rules import Rule
    from hscredit.core.models import RuleSet, RulesClassifier
    from hscredit.report import ruleset_analysis, swap_out_report

    frames, bundle = bound_frames(workspace, config), ReportBundle(directory)
    rules = [Rule(r["expression"], name=r["name"]) for r in config["selected_rules"]]
    combined = reduce((lambda a, b: a & b) if config["rule_logic"] == "and" else (lambda a, b: a | b), rules)
    # 单条规则的 reduce 会返回原对象，不要覆盖用户命名。
    combined = Rule(combined.expr, name="组合规则")
    workbooks = {}
    for index, (name, frame) in enumerate(frames.items()):
        progress(10 + int(70 * index / len(frames)), f"正在评估 {name} 的规则集")
        target = label_map(frame, config, evaluation=True)
        for rule in rules:
            bundle.table(f"{name} / 单规则 / {rule.name}", rule.report(frame, target=target, **library_options(config, "rule_report")))
        bundle.table(f"{name} / 多规则组合效果", combined.report(frame, target=target, **library_options(config, "rule_report")))
        bundle.table(f"{name} / 顺序累积与增量效果", ruleset_analysis(frame, rules, target=target, n_jobs=1, amount=config.get("amount"), **library_options(config, "ruleset_analysis")))
        filename = f"swap_out_report_{index + 1}.xlsx"
        workbooks[name] = filename
        swap_out_report(frame, rules if config["rule_logic"] == "or" else combined, target=target, save=str(directory / filename), date=config.get("date"), group=config.get("group"), amount=config.get("amount"), freq=config["freq"], methods=config["method"], bin_params={key: value for key, value in binning_params(config).items() if key != "method"}, n_jobs=1, **library_options(config, "swap_report"))
        bundle.workbook(filename)
    model = RulesClassifier(RuleSet(name=config.get("rule_name", "规则集"), rules=rules, logic=config["rule_logic"], n_jobs=1), n_jobs=1)
    joblib.dump(model, directory / "model.joblib")
    write_json(directory / "rules.json", config["selected_rules"])
    bundle.artifacts += ["model.joblib", "rules.json"]
    return bundle.result(datasets=list(frames), workbooks=workbooks, note="规则顺序影响 ruleset_analysis 的累积和增量结果；组合逻辑独立控制最终命中。")
