// 首次打开代码输入框时加载。编辑器、语言服务和 worker 都随平台分发。
import * as monaco from 'monaco-editor/esm/vs/editor/editor.api.js'
import 'monaco-editor/esm/vs/editor/browser/coreCommands.js'
import 'monaco-editor/esm/vs/editor/contrib/bracketMatching/browser/bracketMatching.js'
import 'monaco-editor/esm/vs/editor/contrib/clipboard/browser/clipboard.js'
import 'monaco-editor/esm/vs/editor/contrib/format/browser/formatActions.js'
import 'monaco-editor/esm/vs/editor/contrib/hover/browser/hoverContribution.js'
import 'monaco-editor/esm/vs/editor/contrib/placeholderText/browser/placeholderText.contribution.js'
import 'monaco-editor/esm/vs/editor/contrib/suggest/browser/suggestController.js'
import 'monaco-editor/esm/vs/editor/contrib/snippet/browser/snippetController2.js'
import 'monaco-editor/esm/vs/language/json/monaco.contribution.js'
import EditorWorker from 'monaco-editor/esm/vs/editor/editor.worker.js?worker'
import JsonWorker from 'monaco-editor/esm/vs/language/json/json.worker.js?worker'
import {expressionSuggestions} from './expression-suggestions.js'

self.MonacoEnvironment = {
  getWorker: (_, language) => language === 'json' ? new JsonWorker() : new EditorWorker(),
}
const schemas = new Map()
function syncSchemas() {
  monaco.languages.json.jsonDefaults.setDiagnosticsOptions({
    validate: true, allowComments: false, trailingCommas: 'error', enableSchemaRequest: false,
    schemas: [...schemas.values()],
  })
}
syncSchemas()

monaco.languages.register({id: 'expression'})
monaco.languages.setMonarchTokensProvider('expression', {
  tokenizer: {
    root: [
      [/"(?:\\.|[^"\\])*"?|'(?:\\.|[^'\\])*'?/, 'string'],
      [/`[^`]*`?/, 'identifier'],
      [/\b(?:and|or|not|in|is|True|False|None|nan)\b/, 'keyword'],
      [/[A-Za-z_]\w*(?=\s*\()/, 'function'],
      [/[A-Za-z_\u0080-\uFFFF][\w\u0080-\uFFFF]*/, 'identifier'],
      [/(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?/, 'number'],
      [/[+*/%<>=!&|~^-]+/, 'operator'],
    ],
  },
})
monaco.languages.setLanguageConfiguration('expression', {
  brackets: [['(', ')'], ['[', ']']],
  autoClosingPairs: ['()', '[]', '""', "''"].map(([open, close]) => ({open, close})),
})

const contexts = new WeakMap()
monaco.languages.registerCompletionItemProvider(['expression', 'json'], {
  triggerCharacters: ['"', "'", '`', '.', '('],
  provideCompletionItems(model, position) {
    const context = contexts.get(model)
    if (!context) return
    const result = expressionSuggestions(model.getValue(), model.getOffsetAt(position), context)
    if (!result) return
    const range = monaco.Range.fromPositions(model.getPositionAt(result.start), model.getPositionAt(result.end))
    return {suggestions: result.items.map(({snippet, ...item}) => ({
      ...item, range, sortText: (snippet ? '1' : '0') + item.label,
      kind: snippet ? monaco.languages.CompletionItemKind.Function : monaco.languages.CompletionItemKind.Field,
      insertTextRules: snippet ? monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet : undefined,
    }))}
  },
})

export function setEditorSchema(editor, schema) {
  const resource = editor.getModel().uri.toString()
  // 模型目录中的候选数组可能是 Vue 代理，JSON worker 只接收可克隆的 Schema 数据。
  if (schema) schemas.set(resource, {uri: resource + '.schema.json', fileMatch: [resource], schema: JSON.parse(JSON.stringify(schema))})
  else if (!schemas.delete(resource)) return
  syncSchemas()
}

export function setExpressionContext(editor, context) {
  contexts.set(editor.getModel(), context)
}

const themeName = 'hscredit'
function syncTheme() {
  const root = document.documentElement
  const style = getComputedStyle(root)
  const color = name => style.getPropertyValue(name).trim()
  const accent = color('--accent-text'), secondary = color('--chart-secondary')
  const surface = color('--surface'), muted = color('--muted'), text = color('--text')
  monaco.editor.defineTheme(themeName, {
    base: root.dataset.theme === 'dark' ? 'vs-dark' : 'vs',
    inherit: true,
    rules: [
      {token: 'string.key.json', foreground: accent.slice(1), fontStyle: 'bold'},
      {token: 'string.value.json', foreground: secondary.slice(1)},
      {token: 'keyword.json', foreground: accent.slice(1), fontStyle: 'bold'},
      {token: 'delimiter', foreground: muted.slice(1)},
      {token: 'string', foreground: secondary.slice(1)},
      {token: 'keyword', foreground: accent.slice(1), fontStyle: 'bold'},
      {token: 'function', foreground: accent.slice(1)},
      {token: 'number', foreground: color('--success').slice(1)},
      {token: 'operator', foreground: text.slice(1)},
    ],
    colors: {
      'editor.background': surface, 'editor.foreground': text,
      'editorGutter.background': surface, 'editorLineNumber.foreground': muted,
      'editorLineNumber.activeForeground': accent, 'editorCursor.foreground': accent,
      'editor.lineHighlightBackground': color('--soft'), 'editor.selectionBackground': accent + '40',
      'editor.inactiveSelectionBackground': accent + '26', 'editorBracketMatch.border': accent,
      'editorError.foreground': color('--danger'), 'editorWidget.background': surface,
      'editorWidget.foreground': text, 'editorWidget.border': color('--border'),
      'editorSuggestWidget.background': surface, 'editorSuggestWidget.foreground': text,
      'editorSuggestWidget.border': color('--border'), 'editorSuggestWidget.selectedBackground': color('--hover'),
      'editorSuggestWidget.selectedForeground': text, 'editorSuggestWidget.highlightForeground': accent,
      'editorSuggestWidget.focusHighlightForeground': accent,
    },
  })
}
syncTheme()
window.addEventListener('hscredit-theme-change', syncTheme)
if (import.meta.hot) import.meta.hot.dispose(() => window.removeEventListener('hscredit-theme-change', syncTheme))

export function createCodeEditor(element, options) {
  const editor = monaco.editor.create(element, {
    theme: themeName, automaticLayout: true, minimap: {enabled: false},
    lineNumbersMinChars: 2, lineDecorationsWidth: 8, folding: false, glyphMargin: false,
    fontFamily: 'Consolas, "Microsoft YaHei Mono", monospace', fontSize: 13, lineHeight: 21,
    padding: {top: 7, bottom: 7}, wordWrap: 'on', scrollBeyondLastLine: false,
    scrollbar: {alwaysConsumeMouseWheel: false, verticalScrollbarSize: 8, horizontalScrollbarSize: 8},
    overviewRulerBorder: false, overviewRulerLanes: 0, stickyScroll: {enabled: false},
    contextmenu: false, tabFocusMode: true,
    quickSuggestions: {other: true, strings: true, comments: false}, wordBasedSuggestions: 'off',
    fixedOverflowWidgets: true,
    ...options,
  })
  editor.getModel().updateOptions({
    tabSize: 2, insertSpaces: true,
    bracketColorizationOptions: {enabled: false, independentColorPoolPerBracketType: false},
  })
  if (options.language === 'json') {
    editor.onDidType(text => {
      if (['{', '[', ','].includes(text)) editor.trigger('keyboard', 'editor.action.triggerSuggest', {})
    })
  }
  return editor
}
