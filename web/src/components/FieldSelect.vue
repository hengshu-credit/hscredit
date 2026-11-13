<script setup>
import {computed, onBeforeUnmount, ref, watch} from 'vue'
import {ArrowDown} from '@element-plus/icons-vue'
import FieldContentPicker from './FieldContentPicker.vue'
import SelectedFieldNames from './SelectedFieldNames.vue'

const props = defineProps({
  modelValue: [String, Array], fields: {type:Array, required:true}, label: {type:String, required:true},
  placeholder: {type:String, default:'选择字段'}, multiple:Boolean, limit:Number, clearable:Boolean
})
const emit = defineEmits(['update:modelValue','change'])
const open = ref(false), initialText=ref(''), display=ref(null)
const names = computed(() => Array.isArray(props.modelValue) ? props.modelValue : props.modelValue ? [props.modelValue] : [])
function show(text='') {
  initialText.value=text
  open.value=true
}
function paste(event) {
  const text=event.clipboardData?.getData('text/plain')
  if(text){event.preventDefault();show(text)}
}
function typeName(event) {
  if(open.value||event.isComposing||event.ctrlKey||event.metaKey||event.altKey||event.key.length!==1||!event.key.trim())return
  event.preventDefault()
  show(event.key)
  // 弹窗接管焦点前，快速键入的后续字符仍属于同一次输入。
  window.addEventListener('keydown',bufferOpeningText,true)
}
function bufferOpeningText(event) {
  if(event.target instanceof HTMLTextAreaElement||!open.value){stopBuffer();return}
  if(event.isComposing||event.ctrlKey||event.altKey||event.metaKey)return
  if(event.key.length!==1&&event.key!=='Backspace'){stopBuffer();return}
  event.preventDefault();event.stopPropagation()
  initialText.value=event.key==='Backspace'?initialText.value.slice(0,-1):initialText.value+event.key
}
function stopBuffer(){window.removeEventListener('keydown',bufferOpeningText,true)}
watch(open,value=>{if(!value)stopBuffer()})
onBeforeUnmount(stopBuffer)
function openFromClick(event) {
  const box=display.value?.$el
  if(event.detail && event.target===box && box.scrollHeight>box.clientHeight &&
      event.clientX-box.getBoundingClientRect().left>=box.clientWidth)return
  show()
}
function apply(value) {
  if (JSON.stringify(value) === JSON.stringify(props.modelValue)) return
  emit('update:modelValue',value)
  emit('change',value)
}
</script>
<template>
  <div class="field-select">
    <el-button class="field-trigger" :aria-label="label" aria-haspopup="dialog" :aria-expanded="open"
      :aria-description="names.length?'已选 '+names.length+' 个字段。可滚动查看，或直接输入和粘贴字段名。':placeholder"
      @click="openFromClick" @paste="paste" @keydown="typeName" @keydown.down.prevent="show()"
      @keydown.page-down.prevent="display.page(1)" @keydown.page-up.prevent="display.page(-1)">
      <SelectedFieldNames ref="display" :names="names" :placeholder="placeholder"/><el-icon><ArrowDown/></el-icon>
    </el-button>
    <FieldContentPicker v-if="open" v-model="open" :fields="fields" :selected="modelValue" :multiple="multiple"
      :limit="limit" :clearable="clearable" :initial-text="initialText" @select="apply"/>
  </div>
</template>
<style>
.field-select{width:100%;min-width:0}
.field-select .field-trigger.el-button{width:100%;height:auto;min-height:var(--control-height);font-weight:400;padding:5px 10px;justify-content:start}
.field-trigger>span{display:flex;gap:6px;width:100%;min-width:0;align-items:center}
.field-trigger .el-icon{flex:none;color:var(--muted)}
</style>
