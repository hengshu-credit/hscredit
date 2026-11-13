// 批量选择和名称反查都只在当前入口允许的字段范围内执行。
export function addFields(selected, candidates, limit) {
  const names = new Set(selected)
  let overflow = 0
  for (const name of new Set(candidates)) {
    if (names.has(name)) continue
    if (limit > 0 && names.size >= limit) overflow++
    else names.add(name)
  }
  return {names:[...names], overflow}
}

export function fieldNameIndex(available) {
  const exact = new Set(available)
  const folded = new Map()
  for (const name of exact) {
    const key = name.toLowerCase()
    folded.set(key, folded.has(key) ? null : name)
  }
  return {exact,folded}
}
// 保留输入位置，多行粘贴后可以把光标移回任意一项查看候选。
export function fieldTerms(text, {exact,folded}) {
  const input = text.trim()
  if (!input) return []
  const offset = text.indexOf(input)
  if (exact.has(input)) return [{text:input,start:offset,end:offset+input.length}]
  const bracketed = input.startsWith('[') && input.endsWith(']')
  if (bracketed) {
    try {
      const parsed = JSON.parse(input)
      if (Array.isArray(parsed) && parsed.every(v => typeof v === 'string')) {
        return Array.from(input.matchAll(/"(?:\\.|[^"\\])*"/g), (m,i)=>({
          text:parsed[i], start:offset+m.index, end:offset+m.index+m[0].length,
        }))
      }
    } catch { /* 同样接受从 Python 复制的单引号字段列表。 */ }
  }
  const terms = []
  const plain = bracketed ? input.slice(1,-1) : input
  for (const piece of plain.matchAll(/[^\r\n\t,，、;；]+/g)) {
    const raw = piece[0].trim()
    if (!raw) continue
    const token = exact.has(raw) ? raw : raw.replace(/^(["'])(.*)\1$/, '$2')
    const start = offset + Number(bracketed) + piece.index + piece[0].indexOf(raw) + raw.indexOf(token)
    if (!folded.has(token.toLowerCase()) && /\s/.test(token) && token.split(/\s+/).some(w=>folded.has(w.toLowerCase()))) {
      for (const word of token.matchAll(/\S+/g)) terms.push({text:word[0],start:start+word.index,end:start+word.index+word[0].length})
    } else {
      terms.push({text:token,start,end:start+token.length})
    }
  }
  return terms
}

export function suggestedFieldNames(term, index) {
  if (!term) return []
  const query = term.toLowerCase(), same = [], prefix = [], contains = []
  for (const name of index.exact) {
    const lower = name.toLowerCase()
    if (lower === query) same.push(name)
    else if (lower.startsWith(query)) prefix.push(name)
    else if (lower.includes(query)) contains.push(name)
  }
  if (index.exact.has(term)) return [term,...same.filter(n=>n!==term),...prefix,...contains]
  return [...same,...prefix,...contains]
}

export function resolveFieldTerms(terms, {exact,folded}, choices=new Map()) {
  const matched = new Set(), unknown = new Set(), ambiguous = new Set()
  for (const {text} of terms) {
    if (choices.has(text)) {
      for (const name of choices.get(text)) if (exact.has(name)) matched.add(name)
    } else if (exact.has(text)) matched.add(text)
    else if (!folded.has(text.toLowerCase())) unknown.add(text)
    else {
      const name = folded.get(text.toLowerCase())
      if (name===null) ambiguous.add(text)
      else matched.add(name)
    }
  }
  return {matched:[...matched], unknown:[...unknown], ambiguous:[...ambiguous]}
}
export function matchFieldNames(text,index) {
  return resolveFieldTerms(fieldTerms(text,index),index)
}
