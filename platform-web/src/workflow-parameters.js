// 页面参数只描述交互；拟合和报告使用服务端的 hscredit 接口。
export const option=(key,label,description,defaultValue,settings={})=>({key,label,description,default:defaultValue,...settings})
export const choices=values=>values.map(value=>({value,label:String(value)}))
export const assets=datasets=>datasets.map(d=>({value:d.id,label:d.name}))
const methodNames={quantile:'等频',uniform:'等距',tree:'决策树',chi:'卡方',mdlp:'MDLP',best_iv:'最优 IV',best_ks:'最优 KS',best_lift:'最优 LIFT',or_tools:'运筹优化',cp_sat:'约束规划',cart:'CART 决策树',kmeans:'K 均值聚类',monotonic:'单调分箱',genetic:'遗传算法',smooth:'平滑分箱',kernel_density:'核密度',target_bad_rate:'目标坏样本率'}
export const named=map=>Object.entries(map).map(([value,label])=>({value,label}))
export const running=job=>job&&['排队中','运行中','停止中'].includes(job.status)
export async function pollJob(job,api){
  if(!running(job.value))return
  const id=job.value.id
  const latest=await api('/jobs/'+id)
  if(job.value?.id===id&&running(job.value))job.value=latest
}

const parameterGroups = {
  '训练与验证':['features','selector','selector_threshold','cv','search','n_iter','search_spaces'],
  '预测与评分':['threshold','scorecard','base_score','pdo'],
  '分箱与规则':['method','min_n_bins','max_n_bins','min_bin_size','tree_depth','tree_min_samples'],
  '报告统计':['date','freq','amount','group','dropna'],
  '报告输出':['pictures','corr','margins','long_format','explain'],
}
export function binningOptions(group,methods=Object.keys(methodNames)) {
  return [
    option('method','分箱方法','在拟合样本上确定数值特征的分箱边界','quantile',{group,choices:methods.map(value=>({value,label:(methodNames[value]||value)+'（'+value+'）'}))}),
    option('min_n_bins','最少分箱数','允许低变异字段只保留一个箱',1,{group,min:1,max:50}),
    option('max_n_bins','最多分箱数','限制单个数值字段的分箱粒度',5,{group,min:2,max:50}),
    option('min_bin_size','每箱最少样本','小于 1 时为比例，达到 1 时按样本数计算',1,{group,min:0.000001,max:1000000}),
  ]
}
export function parameterGroup(spec){
  if(spec.group)return spec.group
  if(spec.key.includes(':'))return '模型超参数'
  return Object.keys(parameterGroups).find(group=>parameterGroups[group].includes(spec.key))||'分析参数'
}
