export const taskModules = {
  explore: '数据探索',
  strategy: '策略挖掘',
  model: '模型构建',
  third_party: '三方测试',
}

const modulesByKind = {interaction:'explore', ruleset:'strategy', explain:'model'}
export const taskModule = job => modulesByKind[job.kind] || job.kind

export function taskName(job) {
  const config = job.config || {}
  if(job.kind==='explore'&&config.workflow&&config.features?.length)return '组合变量探索'
  if (job.kind === 'strategy' && config.workflow) {
    return {single:'单变量策略', pair:'双变量策略', binary:'二叉树策略', multiway:'多叉树策略'}[config.strategy_mode] || '策略挖掘'
  }
  if (job.kind === 'model') return config.model_mode === 'compare' ? '模型效果对比' : '模型实验'
  return {explore:'单变量探索', interaction:'多变量交互', ruleset:'规则集评估', explain:'样本预测解释'}[job.kind] || taskModules[job.kind]
}

export const taskStatusType = status => ({成功:'success', 失败:'danger', 已停止:'info'}[status] || 'warning')
export const taskUrl = (module, id) => '#tasks/' + module + (id ? '/' + id : '')
