// 解析正在输入的表达式，只替换光标所在的字段、函数名或列名字符串。
import {getLocation} from 'jsonc-parser'

const identifier = /^[\p{ID_Start}_][\p{ID_Continue}]*$/u
const reserved = new Set('False None True and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield'.split(' '))
const wordEnd = /[\p{ID_Continue}]*$/u
const wordStart = /^[\p{ID_Continue}]*/u
const methods = [
  ['isna', '判断缺失'], ['notna', '判断非缺失'], ['isin', '集合包含'], ['between', '区间判断'],
]

function completeExpression(text, offset, context) {
  const {fields = [], functions = [], processing = false} = context
  const before = text.slice(0, offset)
  let start = offset - before.match(wordEnd)[0].length
  let end = offset + text.slice(offset).match(wordStart)[0].length
  let quote = '', colArgument = /\bcol\(\s*$/.test(text.slice(0, start))
  // 跳过字符串内的普通文本。col 的参数与规则中的反引号列名接受字段提示。
  for (const match of text.matchAll(/(["'`])(?:\\[\s\S]|(?!\1)[^\\\n])*?(?:\1|(?=\n|$))/g)) {
    if (match.index >= offset) break
    const part = match[0], closed = part.length > 1 && part.endsWith(match[1])
    const contentEnd = match.index + part.length - Number(closed)
    if (offset > contentEnd) continue
    quote = match[1]
    colArgument = processing && /\bcol\(\s*$/.test(text.slice(0, match.index))
    if (!(colArgument || (!processing && quote === '`'))) return null
    start = match.index + 1
    end = contentEnd
    break
  }
  const preceding = text.slice(0, start)
  if (!quote && preceding.endsWith('.')) {
    const items = processing ? [] : methods.map(([label, detail]) => ({
      label, detail, insertText: label + (text[end] === '(' ? '' : '($0)'), snippet: true,
    }))
    return {start, end, items}
  }
  const items = []
  for (const {name, dtype, derived} of fields) {
    if (!processing && name.includes('`')) continue
    let insertText
    if (quote === '`') insertText = name
    else if (quote) {
      const escaped = JSON.stringify(name).slice(1, -1)
      insertText = quote === "'" ? escaped.replace(/'/g, "\\'") : escaped
    } else if (processing && colArgument) insertText = JSON.stringify(name)
    else if (processing) insertText = 'col(' + JSON.stringify(name) + ')'
    else insertText = identifier.test(name) && !reserved.has(name) ? name : '`' + name + '`'
    items.push({label: name, detail: derived ? '前序加工字段' : dtype || '数据集字段', insertText})
  }
  if (!quote && !colArgument && processing) {
    items.push({label: 'col', detail: '引用字段', snippet: true, insertText: text[end] === '(' ? 'col' : 'col("$1")'})
    for (const fn of functions) items.push({
      label: fn.name, detail: fn.label + '（' + fn.args.join('，') + '）', snippet: true,
      insertText: fn.name + (text[end] === '(' ? '' : '(' + fn.args.map((_, i) => '$' + (i + 1)).join(', ') + ')'),
    })
  }
  return {start, end, items}
}

export function expressionSuggestions(text, offset, context) {
  if (!context) return null
  if (!context.embedded) return completeExpression(text, offset, context)
  const {path, previousNode: node} = getLocation(text, offset)
  const rulePath = context.ruleProperty ? path.slice(1) : path
  if (context.ruleProperty && path[0] !== context.ruleProperty) return null
  if (rulePath.length !== 2 || typeof rulePath[0] !== 'number' || rulePath[1] !== 'expression' || node?.type !== 'string') return null
  // JSON 转义前后的列数不同。把表达式中的替换范围映射回原始 JSON。
  const boundaries = [node.offset + 1]
  const raw = text.slice(node.offset + 1, node.offset + node.length)
  for (const match of raw.matchAll(/\\(?:u[\da-fA-F]{4}|[^\n])|[\s\S]/g)) {
    if (boundaries.length > node.value.length) break
    boundaries.push(boundaries.at(-1) + match[0].length)
  }
  const decodedOffset = boundaries.indexOf(offset)
  if (decodedOffset < 0) return null
  const result = completeExpression(node.value, decodedOffset, context)
  if (!result) return null
  return {
    start: boundaries[result.start], end: boundaries[result.end],
    items: result.items.map(item => ({...item, insertText: JSON.stringify(item.insertText).slice(1, -1)})),
  }
}
