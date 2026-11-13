import {option, choices} from './workflow-parameters.js'

// 类别名称与 hscredit.core.models 各子包的 API 文档一致。
const familyLabels = {
  classical: '经典模型',
  boosting: '提升模型',
  net: '神经网络',
  scorecard: '评分卡',
  rules: '规则模型',
  calibration: '概率校准',
}

export function modelOptions(catalog) {
  const groups = new Map(Object.values(familyLabels).map(label => [label, []]))
  for (const [key, model] of Object.entries(catalog)) {
    const family = model.module?.match(/^hscredit\.core\.models\.([^.]+)/)?.[1]
    const label = familyLabels[family] ?? model.group ?? '其他模型'
    if (!groups.has(label)) groups.set(label, [])
    groups.get(label).push({
      value: key,
      label: model.name,
      disabled: !model.available,
      reason: model.reason,
    })
  }
  return Array.from(groups, ([label, children]) => ({value: label, label, children}))
    .filter(group => group.children.length > 0)
}

const parameterLabels={max_epochs:'最大训练轮数',batch_size:'批量大小',num_leaves:'最大叶节点数',iterations:'提升迭代次数',depth:'树深度',device:'计算设备',kernel:'核函数',method:'校准方法',calib_ratio:'校准样本比例',pdo:'分数翻倍间隔 PDO',base_score:'基准分',min_lift:'最小 LIFT',top_n:'最多规则数',C:'正则化强度 C',max_iter:'最大迭代次数',n_estimators:'树的数量',max_depth:'树的最大深度',min_samples_leaf:'叶节点最少样本',learning_rate:'学习率'}
const parameterHelp={C:'数值越小，正则化约束越强',max_iter:'限制优化器迭代次数',n_estimators:'控制参与集成的估计器数量',max_depth:'限制单棵树的分裂层数',depth:'限制每棵提升树的分裂层数',min_samples_leaf:'叶节点至少覆盖的样本数',learning_rate:'每次更新使用的步长',max_epochs:'遍历训练数据的最大轮数',batch_size:'每次参数更新使用的样本数',num_leaves:'限制每棵树的叶节点数量',iterations:'提升算法的最大迭代次数',device:'模型训练和推理使用的设备',kernel:'支持向量机的核函数',method:'将原始概率映射为校准概率的方法',calib_ratio:'训练数据中用于概率校准的比例',pdo:'赔率翻倍对应的评分变化',base_score:'评分刻度的基准分',min_lift:'仅保留达到该提升度的规则',top_n:'最多保留的规则数量'}

const descriptions = {
  LogisticRegression:'用特征的线性组合估计坏样本概率，回归系数可用于解释变量影响。',
  RandomForest:'在不同样本和特征子集上训练多棵决策树，综合各树的预测概率。',
  ExtraTrees:'通过随机选取分裂阈值构建多棵决策树并汇总预测。',
  GradientBoosting:'逐轮拟合决策树，修正前面各轮的预测误差。',
  DecisionTreeClassifier:'按特征条件递归划分样本，使用叶节点的类别分布预测。',
  SVM:'通过核函数学习分类边界，并输出类别概率。',
  XGBoost:'使用带正则化的梯度提升树拟合特征间的非线性关系。',
  LightGBM:'采用直方图分箱和按叶生长的梯度提升树。',
  CatBoost:'使用提升树学习非线性关系，原生支持类别特征。',
  NGBoost:'利用自然梯度提升估计预测分布。',
  ResNet:'使用带残差连接的神经网络学习表格特征。',
  FTTransformer:'将特征编码为向量，通过注意力机制学习特征交互。',
  TabM:'使用共享网络的多个预测成员对表格数据进行集成预测。',
  TabNet:'使用逐步注意力机制选择用于每一步预测的特征。',
  TabPFN:'使用预训练表格模型，根据训练样本进行上下文预测。',
  ScoreCard:'将分箱后的 WOE 与回归系数组合为评分卡。',
  RoundScoreCard:'按设定精度对分箱分值取整，生成整数评分卡。',
  ProbabilityScoreCard:'将基础模型输出的坏样本概率映射为评分。',
  ProbabilityCalibrator:'在校准样本上调整基础模型输出的预测概率。',
  RulesClassifier:'挖掘并组合风险规则，根据命中情况进行分类。',
}

export function modelDescription(model) {
  return model.description || descriptions[model.class_name] || model.note || model.group
}

export function modelParameterSpecs(model) {
  return Object.entries(model?.params || {}).map(([key,value]) =>
    option(key,parameterLabels[key] || key,parameterHelp[key] || '训练时传给模型的参数',value,{
      group:'模型超参数', jsonValue:['max_depth','min_samples_leaf'].includes(key), ...(model.choices?.[key] ? {choices:choices(model.choices[key])} : {}),
    }))
}

export function selectedModelParameters(keys,catalog,overrides={}) {
  return Object.fromEntries(keys.map(key=>[key,{...catalog[key]?.params,...overrides[key]}]))
}
