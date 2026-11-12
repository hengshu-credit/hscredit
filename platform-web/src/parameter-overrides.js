// 参数覆盖只保存用户添加的项，省略与显式 null 使用不同的表示。
export function scopeOptions(catalog, context) {
  const scopes = catalog?.scopes?.[context] || []
  return Object.fromEntries(scopes.map(key => [key, catalog.groups[key]]))
}

export function availableParameters(group, method, sampler='random') {
  return {...group?.methods?.[method], ...group?.variants?.[sampler], ...group?.params}
}

export function parameterRows(values) {
  return Object.entries(values || {}).flatMap(([scope, params]) =>
    Object.entries(params).map(([name, value]) => ({scope, name, text: JSON.stringify(value)})))
}

export function parseRows(rows, groups, method) {
  const result = Object.create(null)
  for (const {scope, name, text} of rows) {
    const group = groups[scope]
    if (!group || !/^[A-Za-z][A-Za-z0-9_]*$/.test(name) || (!group.custom && !Object.hasOwn(availableParameters(group, method), name))) {
      throw Error(name ? `当前配置不支持参数 ${name}` : '请选择或输入参数名')
    }
    const values = result[scope] ||= Object.create(null)
    if (Object.hasOwn(values, name)) throw Error(`参数 ${name} 重复`)
    try { values[name] = JSON.parse(text) }
    catch { throw Error(`${name} 需要有效的 JSON 值，文本请加双引号`) }
  }
  return result
}
