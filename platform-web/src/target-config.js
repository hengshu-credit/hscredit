import {computed, reactive, watch} from 'vue'

export function copyTargetDraft(to, from) {
  const snapshot = JSON.parse(JSON.stringify(from))
  const {positives,...fields} = snapshot
  Object.assign(to, fields)
  // fields 的同步 watcher 会重建映射；保留编辑前选定的具体坏样本值。
  to.positives = positives
}

export function parseThresholds(text) {
  const tokens = text.trim().split(/[,，\s]+/).filter(Boolean)
  if (!tokens.length) return {error: '请填写至少一个逾期阈值'}
  const values = tokens.map(Number)
  if (tokens.some((t, i) => !/^\+?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i.test(t) || !Number.isFinite(values[i]))) {
    return {error: '阈值须为非负有限数值，多个阈值请用逗号或空格分隔'}
  }
  const dpds = [...new Set(values)]
  return dpds.length > 20 ? {error: '最多支持 20 个不同阈值'} : {dpds}
}

export function useTargetConfig(dataset, purpose) {
  const role = computed(()=>typeof purpose==='function'?purpose():purpose)
  const summary = role.value === 'explore'
  let nextId = 0
  function newEntry(){
    nextId = Math.max(nextId,...draft.entries.map(row=>row.id))+1
    return {id:nextId, name:'', field:'', operator:'raw', value:null}
  }
  const draft = reactive({mode: 'target', fields: [], entries: [], positives: {}, overdue: [], text: '7', operator: '>', grey: false, dictionaryGrey: false, missing: 'drop', primary: ''})
  const columns = computed(() => dataset()?.columns || [])
  const binary = computed(() => columns.value.filter(c => c.unique === 2))
  const numeric = computed(() => columns.value.filter(c => c.kind === 'numeric'))
  const values = name => (binary.value.find(c => c.name === name)?.values || []).map(v => v.value).filter(v => v != null)
  watch(() => [dataset()?.id, dataset()?.revision], ([id, rev], previous) => {
    if (previous && id === previous[0] && rev === previous[1]) return
    const preferred = binary.value.find(c => c.name === dataset()?.default_target) || binary.value.find(c => /target|标签|FPD|是否逾期/i.test(c.name)) || binary.value[0]
    const overdue = (dataset()?.default_overdue || []).filter(n => numeric.value.some(c => c.name === n))
    Object.assign(draft, {mode: summary ? 'none' : preferred || !numeric.value.length ? 'target' : 'overdue',
      positives: {}, fields: !summary && preferred ? [preferred.name] : [], entries: [newEntry()], overdue, text: '7', operator: '>', grey: false, dictionaryGrey: false, missing: 'drop', primary: ''})
  }, {immediate: true, flush: 'sync'})
  watch(() => draft.fields.slice(), names => {
    draft.positives = Object.fromEntries(names.map(name => {
      const options = values(name), old = draft.positives[name]
      const positive = options.includes(old) ? old : [1, true, '1'].find(v => options.includes(v))
      return [name, positive]
    }))
  }, {immediate: true, flush: 'sync'})
  const thresholds = computed(() => parseThresholds(draft.text))
  const sources = computed(() => {
    if (draft.mode === 'none') return []
    if (draft.mode === 'dict') return [...new Set(draft.entries.map(row => row.field).filter(Boolean))]
    return draft.mode === 'target' ? draft.fields : draft.overdue
  })
  const error = computed(() => {
    if (draft.mode === 'none') return ''
    if (draft.mode === 'overdue') return !draft.overdue.length ? '请选择逾期天数字段' : thresholds.value.error || ''
    if (draft.mode === 'dict') {
      if (!draft.entries.length) return '请添加一个标签'
      const names = draft.entries.map(row => row.name.trim())
      if (names.some(n => !n || n === '分箱详情' || (summary && names.length > 1 && ['统计详情','分布详情'].includes(n)))) return '请填写有效的标签名称，避开报告保留名称'
      if (new Set(names).size !== names.length) return '字典中的标签名称不能重复'
      if (draft.entries.some(row => !row.field)) return '请为每个标签选择来源字段'
      if (draft.entries.some(row => row.operator !== 'raw' && (row.value == null || (typeof row.value === 'number' && !Number.isFinite(row.value))))) return '请填写每个条件的比较值'
      return ''
    }
    if (!draft.fields.length) return binary.value.length ? '请选择标签字段' : '当前数据集没有二分类字段，可按逾期天数定义'
    return draft.fields.some(n => !values(n).includes(draft.positives[n])) ? '请选择每个标签的坏样本取值' : ''
  })
  const conditions = computed(() => {
    if (draft.mode === 'overdue') return draft.overdue.flatMap(n => (thresholds.value.dpds || []).map(d => n+' '+draft.operator+' '+d+' 天'))
    if (draft.mode === 'dict') return draft.entries.map(row => row.name.trim())
    return sources.value.map(n => n+' = '+(draft.positives[n] == null ? '待选取值' : String(draft.positives[n])))
  })
  const keys = computed(() => draft.mode === 'overdue'
    ? draft.overdue.flatMap(n => (thresholds.value.dpds || []).map(d => JSON.stringify([n, d])))
    : draft.mode === 'dict' ? draft.entries.map(row => 'dict:'+row.id) : draft.fields.map(n => JSON.stringify([n])))
  const primaryIndex = computed(() => Math.max(0, keys.value.indexOf(draft.primary)))
  const primary = computed({
    get: () => conditions.value[primaryIndex.value],
    set: value => { draft.primary = keys.value[conditions.value.indexOf(value)] },
  })
  const payload = computed(() => {
    if (draft.mode === 'none') return {targets: [], positive_labels: {}}
    const common = {label_mode: draft.mode === 'dict' ? 'target' : draft.mode, handle_missing_target: draft.missing}
    if (draft.mode === 'dict') {
      const definitions = draft.entries.map(row => [row.name.trim(), row.operator === 'raw' ? {field: row.field} : {field: row.field, operator: row.operator, value: row.value}])
      return {...common, target: Object.fromEntries(definitions), target_order: definitions.map(([name]) => name),
        del_grey: draft.dictionaryGrey && draft.entries.some(row=>row.operator==='raw'), ...(role.value === 'training' ? {primary_index: primaryIndex.value} : {})}
    }
    if (draft.mode === 'overdue') return {...common, overdue: draft.overdue, dpds: thresholds.value.dpds || [], overdue_operator: draft.operator, del_grey: draft.grey && ['>', '>='].includes(draft.operator),
      ...(role.value === 'training' ? {primary_index: primaryIndex.value} : {})}
    if (summary) return {...common, targets: draft.fields, positive_labels: draft.positives}
    const target = draft.fields[role.value === 'training' ? primaryIndex.value : 0]
    const evaluation_targets = draft.fields.filter(n => n !== target)
    return {...common, target, positive_label: draft.positives[target], evaluation_targets,
      positive_labels: Object.fromEntries(evaluation_targets.map(n => [n, draft.positives[n]]))}
  })
  return reactive({draft, purpose:role, columns, newEntry, binary, numeric, values, sources, error, payload, conditions, primary, primaryIndex})
}
