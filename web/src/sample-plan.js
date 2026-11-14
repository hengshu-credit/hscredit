import {computed, reactive, watch} from 'vue'
import {datasetDraftKey,readDraft,writeDraft} from './form-drafts.js'

export const sampleUses = {
  summary: {primary:'分析样本', batch:'批量变量探索', help:'按所选字段分别分析每份样本。'},
  interaction: {primary:'分析样本', batch:'批量交互分析', help:'每份样本分别计算联合分布和交互图。'},
  third_party: {primary:'报告样本', batch:'批量生成报告', help:'每份样本独立分箱，生成一份三方报告。'},
  strategy: {primary:'拟合样本', batch:'批量独立挖掘', help:'在拟合样本上确定分箱和规则，其他评估样本沿用相同的边界。'},
  ruleset: {primary:'主评估样本', batch:'批量运行规则集', help:'将同一组已选规则应用到各份样本，分别计算命中和置换效果。'},
  model: {primary:'训练样本', batch:'批量独立训练', help:'模型仅使用训练部分拟合，测试和其他评估样本共用该模型。'},
}

export function useSamplePlan(datasets, primaryId, context, tree) {
  const defaults = () => ({batch:[], evaluations:[], test:'', psi:'', testMode:'holdout', fraction:0.25})
  const drafts = new Map()
  const assets = computed(() => new Map(datasets.value.map(d => [d.id, d])))
  const draftKey = computed(() => {
    const asset = assets.value.get(primaryId.value)
    return asset && sampleUses[context.value] ? datasetDraftKey(asset,'samples/'+context.value) : ''
  })
  const draft = computed(() => {
    const key = draftKey.value
    if (!drafts.has(key)) drafts.set(key,reactive({...defaults(),...(key ? readDraft(key) : null)}))
    return drafts.get(key)
  })
  const groupParents = computed(() => {
    const parents = new Map()
    function visit(row) {
      for (const child of row.children || []) {
        if (row.group_path) parents.set(child.id, row)
        visit(child)
      }
    }
    tree?.value.forEach(visit)
    return parents
  })
  watch([assets, primaryId, draft, () => draft.value.testMode], () => {
    const d = draft.value
    const available = id => primaryId.value && id !== primaryId.value && assets.value.has(id)
    d.batch = d.batch.filter(available)
    const secondary = id => available(id) && !d.batch.includes(id)
    if (!secondary(d.test) || d.evaluations.includes(d.test)) d.test = ''
    if (!secondary(d.psi)) d.psi = ''
    d.evaluations = d.evaluations.filter(id => secondary(id) && (d.testMode !== 'external' || id !== d.test))
  }, {flush:'sync'})
  watch(draft, value=>{if(draftKey.value) writeDraft(draftKey.value,value)}, {deep:true})
  const runDatasets = computed(() => [primaryId.value, ...draft.value.batch].map(id => assets.value.get(id)).filter(Boolean))
  function binding(id) {
    const asset = assets.value.get(id)
    return asset ? {id, revision:asset.revision || 1} : null
  }
  const error = computed(() => {
    const d = draft.value
    if (context.value === 'model' && d.testMode === 'external' && !binding(d.test)) return '请选择独立测试样本'
    if (context.value === 'model' && d.testMode === 'holdout' && !(d.fraction >= 0.1 && d.fraction <= 0.5)) return '留出比例须在 10% 至 50% 之间'
    return ''
  })
  const sharedRows = computed(() => {
    if (!['model','strategy'].includes(context.value)) return []
    const d = draft.value
    const evaluationIds = context.value === 'model' && d.testMode === 'external' ? [d.test, ...d.evaluations] : d.evaluations
    const pairs = []
    function contains(parent, child) {
      while (child?.parent_id) {
        const group = groupParents.value.get(child.id)
        const source = group || assets.value.get(child.parent_id)
        if (!source || (!group && (source.revision || 1) !== child.splitting?.source_revision)) return false
        if (source.id === parent.id) return true
        child = source
      }
      return false
    }
    for (const train of runDatasets.value) {
      for (const id of evaluationIds) {
        const evaluation = assets.value.get(id)
        if (evaluation && (contains(train,evaluation) || contains(evaluation,train))) {
          pairs.push('「'+train.name+'」与「'+evaluation.name+'」')
        }
      }
    }
    return pairs
  })
  const payload = computed(() => {
    const d = draft.value
    if (context.value === 'summary') return d.psi ? {psi_dataset:binding(d.psi)} : {}
    if (!['model','strategy','ruleset'].includes(context.value)) return {}
    const roles = {evaluation_datasets:d.evaluations.map(binding)}
    if (context.value !== 'model') return roles
    return {...roles, test_dataset:d.testMode === 'external' ? binding(d.test) : null,
      use_holdout:d.testMode === 'holdout', ...(d.testMode === 'holdout' ? {test_size:d.fraction} : {})}
  })
  return reactive({draft, runDatasets, payload, error, sharedRows})
}
