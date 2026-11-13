import {computed, ref} from 'vue'

export function useTableFilters() {
  const drafts = ref([])
  const conditions = computed(previous => {
    const active = drafts.value.filter(row => row.key && row.values.length)
    // 只添加空条件时保留已生效的筛选，避免重置页码或重复查询整表。
    return previous?.length === active.length && active.every((row, i) => row === previous[i]) ? previous : active
  })
  return {drafts, conditions}
}

export function valueLabel(value) {
  if (value == null) return '缺失值'
  if (value === '') return '空字符串'
  if (value === '缺失值' || value === '空字符串') return '「' + value + '」'
  if (typeof value === 'boolean') return value ? '真' : '假'
  return String(value)
}

export function fieldValue(row, field) {
  return (field.read ? field.read(row) : row[field.key]) ?? null
}

export function filterRows(rows, fields, conditions) {
  if (!conditions.length) return rows
  const tests = conditions.map(({key, values}) => ({
    field: fields.find(field => field.key === key), values: new Set(values),
  }))
  return rows.filter(row => tests.every(({field, values}) => field && values.has(fieldValue(row, field))))
}

export function summaryFields(columns) {
  return columns.flatMap(column => column.children
    ? column.children.map(child => ({key: child.key, label: column.label + ' / ' + child.label}))
    : [column])
}
