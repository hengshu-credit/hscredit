// 将一行草稿转换为一次树操作，保留校验错误所在行。
export function usesCategoryGroups(mode, field) {
  return mode === 'multiway' && field && (field.kind !== 'numeric' || /bool/i.test(field.dtype))
}

export function treeEditPayload(row, mode, fields) {
  if (!Number.isSafeInteger(row.node) || row.node < 0) throw Error('请输入非负整数节点编号')
  const edit = {action: row.action, node: row.node}
  if (row.action === 'prune') return edit
  const field = fields.find(field => field.name === row.feature)
  if (!field) throw Error('请从当前分析字段中选择分裂字段')
  edit.feature = field.name
  if (usesCategoryGroups(mode, field)) {
    let groups
    try { groups = JSON.parse(row.categories) } catch { throw Error('请输入有效的类别组 JSON') }
    if (!Array.isArray(groups) || !groups.length || groups.some(group => !Array.isArray(group) || !group.length)) {
      throw Error('请使用嵌套数组填写非空类别组')
    }
    return {...edit, category_groups: groups}
  }
  const parts = row.values.split(/[,，]/).map(value => value.trim())
  const thresholds = parts.map(Number)
  if (parts.includes('') || !thresholds.every(Number.isFinite)) throw Error('请输入有效的数值阈值')
  if (mode === 'binary') {
    if (thresholds.length !== 1) throw Error('二叉树只需一个阈值')
    return {...edit, threshold: thresholds[0]}
  }
  if (thresholds.some((value, i) => i > 0 && value <= thresholds[i - 1])) throw Error('请按递增顺序填写阈值')
  return {...edit, thresholds}
}
