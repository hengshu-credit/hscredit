<script setup>
import {computed, onBeforeUnmount, onMounted, ref, watch} from 'vue'

const props = defineProps({
  modelValue: {type: String, default: ''},
  language: {type: String, default: 'json'},
  label: {type: String, required: true},
  rows: {type: Number, default: 4},
  placeholder: {type: String, default: ''},
  disabled: Boolean,
  schema: Object,
  expressionContext: Object,
})
const emit = defineEmits(['update:modelValue'])
const host = ref(null), fallback = ref(null), ready = ref(false), failed = ref(false)
const contentHeight = ref(0)
const height = computed(() => Math.max(props.rows * 21 + 14, Math.min(contentHeight.value, 8 * 21 + 14)) + 'px')
let editor, runtime, focusRequested = false

function focus() {
  focusRequested = true
  if (editor) editor.focus()
  else fallback.value?.focus()
}
defineExpose({focus})

onMounted(async () => {
  try {
    runtime = await import('../code-editor.js')
    if (!host.value) return
    editor = runtime.createCodeEditor(host.value, {
      value: props.modelValue, language: props.language, ariaLabel: props.label,
      placeholder: props.placeholder, readOnly: props.disabled,
    })
    runtime.setEditorSchema(editor, props.schema)
    runtime.setExpressionContext(editor, props.expressionContext)
    contentHeight.value = editor.getContentHeight()
    editor.onDidContentSizeChange(event => { contentHeight.value = event.contentHeight })
    editor.onDidChangeModelContent(() => emit('update:modelValue', editor.getValue()))
    ready.value = true
    if (focusRequested) editor.focus()
  } catch (error) {
    console.error('代码编辑器加载失败', error)
    failed.value = true
  }
})
watch(() => props.modelValue, value => {
  if (editor && value !== editor.getValue()) editor.setValue(value)
})
watch(() => props.disabled, readOnly => editor?.updateOptions({readOnly}))
watch(() => props.schema, schema => { if (editor) runtime.setEditorSchema(editor, schema) })
watch(() => props.expressionContext, context => { if (editor) runtime.setExpressionContext(editor, context) })
onBeforeUnmount(() => {
  if (editor) {
    runtime.setEditorSchema(editor, null)
    editor.dispose()
  }
})
</script>

<template>
  <div class="code-editor">
    <div v-if="language === 'json' && ready" class="code-editor-toolbar">
      <span>JSON</span>
      <button type="button" :disabled="disabled" title="格式化 JSON（Shift + Alt + F）"
              @click="editor.getAction('editor.action.formatDocument').run()">格式化</button>
    </div>
    <div v-show="!failed" ref="host" class="code-editor-host" :style="{height}">
      <span v-if="!ready" class="code-editor-loading" role="status">正在加载编辑器…</span>
    </div>
    <template v-if="failed">
      <span role="status">编辑器加载失败，可继续以纯文本编辑。</span>
      <textarea ref="fallback" :aria-label="label" :value="modelValue" :rows="rows" :placeholder="placeholder"
                :disabled="disabled" @input="emit('update:modelValue', $event.target.value)"/>
    </template>
  </div>
</template>

<style scoped>
.code-editor{width:100%;min-width:0;box-sizing:border-box;border:1px solid var(--border);background:var(--surface);color:var(--text);line-height:normal;text-align:left}
.code-editor:focus-within{border-color:var(--accent-text)}
.code-editor-toolbar{display:flex;align-items:center;justify-content:space-between;padding:3px 8px 3px 10px;border-bottom:1px solid var(--line-soft);background:var(--soft);color:var(--muted);font-size:11px}
.code-editor-toolbar button{border:0;background:transparent;color:var(--accent-text);padding:2px 4px;cursor:pointer;font-size:12px}
.code-editor-host{position:relative;min-width:0;overflow:hidden}
.code-editor-loading{position:absolute;inset:8px auto auto 10px;color:var(--muted);font-size:12px}
.code-editor textarea{box-sizing:border-box;width:100%;background:var(--surface);color:var(--text)}
</style>
