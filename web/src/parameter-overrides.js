// 参数覆盖只保存用户添加的项，省略与显式 null 使用不同的表示。
import {visit} from 'jsonc-parser'

export function scopeOptions(catalog, context) {
  const scopes = catalog?.scopes?.[context] || []
  return Object.fromEntries(scopes.map(key => [key, catalog.groups[key]]))
}

export function availableParameters(group, method, sampler='random') {
  return {...group?.methods?.[method], ...group?.variants?.[sampler], ...group?.params}
}

export function parameterRows(values) {
  return Object.entries(values || {}).flatMap(([scope, params]) =>
    Object.entries(params).map(([name, value]) => ({scope, name, text: parameterText(value)})))
}

function parseJson(text) {
  return JSON.parse(text, (_, value) => {
    if (typeof value === 'number' && !Number.isFinite(value)) throw Error('数值超出范围')
    return value
  })
}

export function parseParameterValue(text, name) {
  const input = text.trim()
  if (!input) throw Error(`请填写 ${name} 的参数值`)
  if (/^true$/i.test(input)) return true
  if (/^false$/i.test(input)) return false
  if (input === 'None') return null
  try { return parseJson(input) }
  catch (error) {
    if (!(error instanceof SyntaxError)) throw error
    if (['[', '{', '"'].includes(input[0])) throw Error(`${name} 需要有效的 JSON 值`)
    return input
  }
}

// 普通文本直接回显；空串、"true" 等可能改变类型的字符串保留引号。
export function parameterText(value) {
  if (typeof value === 'string') {
    try { if (parseParameterValue(value, '') === value) return value } catch {}
  }
  return JSON.stringify(value)
}

export function parseRows(rows, groups, method) {
  const result = Object.create(null)
  const samplerRow = rows.find(row => row.scope === 'tuner' && row.name === 'sampler')
  const sampler = samplerRow && parseParameterValue(samplerRow.text, 'sampler')
  for (const {scope, name, text} of rows) {
    const group = Object.hasOwn(groups, scope) && groups[scope]
    const params = availableParameters(group, method, sampler)
    if (!group || !/^[A-Za-z][A-Za-z0-9_]*$/.test(name) || (!group.custom && !Object.hasOwn(params, name))) {
      throw Error(name ? `当前配置不支持参数 ${name}` : '请选择或输入参数名')
    }
    const values = result[scope] ||= Object.create(null)
    if (Object.hasOwn(values, name)) throw Error(`参数 ${name} 重复`)
    values[name] = parseParameterValue(text, name)
    if (params[name]?.choices && !params[name].choices.includes(values[name])) throw Error(`${name} 的值不在可选范围内`)
  }
  return result
}

// 弹窗中的 JSON 以参数名为键，底层调用重名时由“用途.参数名”区分。
const specName = spec => spec.apiName ? 'model.' + spec.apiName : spec.key

function parameterNames(specs, groups, method, sampler) {
  const names = new Map()
  function add(name, key) {
    if (!names.has(name)) names.set(name, [])
    names.get(name).push(key)
  }
  for (const s of specs) add(s.apiName || s.key, specName(s))
  for (const [scope, group] of Object.entries(groups)) {
    for (const name of Object.keys(availableParameters(group, method, sampler))) add(name, scope + '.' + name)
  }
  return names
}

export function parameterDocument(specs, values, overrides, groups) {
  const names = parameterNames(specs, groups, values.method, overrides.tuner?.sampler)
  const keyFor = (name, key) => names.get(name)?.length === 1 && names.get(name)[0] === key ? name : key
  const entries = specs.map(s => [keyFor(s.apiName || s.key, specName(s)), s.type === 'json' ? parseJson(values[s.key]) : values[s.key]])
  for (const [scope, params] of Object.entries(overrides)) {
    for (const [name, value] of Object.entries(params)) entries.push([keyFor(name, scope + '.' + name), value])
  }
  return Object.fromEntries(entries)
}

function readDocument(text) {
  let doc
  try { doc = parseJson(text) } catch { throw Error('请检查参数 JSON 的格式和数值范围') }
  if (!doc || typeof doc !== 'object' || Array.isArray(doc)) throw Error('参数 JSON 需要是一个对象')
  const objects = []
  visit(text, {
    onObjectBegin: () => objects.push(new Set()),
    onObjectEnd: () => objects.pop(),
    onObjectProperty(name) {
      const names = objects.at(-1)
      if (names.has(name)) throw Error(`JSON 中的参数 ${name} 重复`)
      names.add(name)
    },
  })
  return doc
}

export function readParameterDocument(text, specs, groups) {
  const doc = readDocument(text)
  const byKey = new Map(specs.map(s => [specName(s), s]))
  const values = Object.fromEntries(specs.map(s => [s.key, s.default]))
  const names = parameterNames(specs, groups, doc.method ?? values.method, doc['tuner.sampler'] ?? doc.sampler)
  const rows = []
  const seen = new Set()
  for (const [name, value] of Object.entries(doc)) {
    const candidates = names.get(name)
    let key = name
    if (!byKey.has(name) && candidates) {
      if (candidates.length > 1) throw Error(`${name} 有多个用途，请使用 ${candidates.join(' 或 ')}`)
      key = candidates[0]
    }
    if (seen.has(key)) throw Error(`参数 ${key} 重复`)
    seen.add(key)
    const spec = byKey.get(key)
    if (spec) {
      validateFormValue(spec, value)
      values[spec.key] = spec.type === 'json' ? JSON.stringify(value, null, 2) : value
    } else {
      const parts = key.split('.')
      if (parts.length !== 2) throw Error(`当前配置不支持参数 ${name}`)
      rows.push({scope: parts[0], name: parts[1], text: parameterText(value)})
    }
  }
  return {values, overrides: parseRows(rows, groups, values.method)}
}

function validateFormValue(s, value) {
  if (s.jsonValue && value === null) return
  if (s.type === 'json') {
    if (!value || typeof value !== 'object' || Array.isArray(value)) throw Error(`${s.label} 需要 JSON 对象`)
  } else if (s.multi || s.fields || s.choices) {
    if (s.fields && !s.multi && value === null) return
    const items = s.multi ? value : [value]
    if (!Array.isArray(items)) throw Error(`${s.label} 需要数组`)
    const allowed = s.fields ? s.fields.map(f => f.name) : s.choices.map(c => c.value)
    if (items.some(v => !(s.custom && typeof v === 'string') && !allowed.includes(v))) throw Error(`${s.label} 的值不在可选范围内`)
    if (s.limit && items.length > s.limit) throw Error(`${s.label} 最多可选 ${s.limit} 项`)
  } else if (s.jsonValue || typeof s.default === 'number') {
    if (typeof value !== 'number' || value < s.min || value > s.max) throw Error(`${s.label} 需要有效范围内的数值`)
  } else if (typeof value !== typeof s.default) throw Error(`${s.label} 的参数类型有误`)
}
