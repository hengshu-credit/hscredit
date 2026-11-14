<script>
// 同一数据集的选字段入口共享本次会话的使用记录。
const recentByDataset = new Map()
</script>
<script setup>
import {computed, inject, nextTick, onBeforeUnmount, onMounted, ref, watch} from 'vue'
import {Check, Search} from '@element-plus/icons-vue'
import {fieldKind} from '../processing'
import {pageSizes, useFieldPickerPages} from '../field-picker-pages'
import FieldProfileFloat from './FieldProfileFloat.vue'
import SelectedFieldNames from './SelectedFieldNames.vue'
import ManualFieldSuggestions from './ManualFieldSuggestions.vue'
import {addFields,fieldNameIndex,fieldTerms,resolveFieldTerms,suggestedFieldNames} from '../field-selection'

const props = defineProps({
  modelValue: Boolean, fields: {type: Array, required: true}, selected: [String, Array],
  multiple: Boolean, limit: Number, clearable: Boolean, confirmLabel: {type: String, default: '确定'}, initialText:{type:String,default:''}
})
const emit = defineEmits(['update:modelValue', 'select'])
const dataset = inject('fieldPickerDataset',ref(null))
const search = ref(''), category = ref('all'), draft = ref([]), recent = ref([])
const finder = ref(), workspace = ref(), preview = ref(null), pinned = ref(false)
const floatLeft = ref(false), floatAbove = ref(false)
const types = {numeric:'数值', string:'文本', boolean:'布尔', date:'日期', unknown:'待计算'}
const typeOf = field => field.valueType || fieldKind(field)
const byName = computed(() => new Map(props.fields.map(field => [field.name,field])))
const selectedNames = computed(() => new Set(draft.value))
const validDraft = computed(() => draft.value.filter(name => byName.value.has(name)))
const selectedPageNames = ref([])
const recentFields = computed(() => recent.value.map(name => byName.value.get(name)).filter(Boolean))
const categories = computed(() => {
  const counts = new Map()
  for (const field of props.fields) {
    const type = typeOf(field)
    counts.set(type, (counts.get(type) || 0) + 1)
  }
  const derived = props.fields.filter(field => field.derived).length
  return [
    ['all','全部字段',props.fields.length], ['selected','已选字段',validDraft.value.length],
    ['recent','最近使用',recentFields.value.length],
    ...Object.entries(types).filter(([key]) => counts.has(key)).map(([key,label]) => [key,label+'字段',counts.get(key)]),
    ...(derived ? [['derived','前序步骤结果',derived]] : [])
  ]
})
const filtered = computed(() => {
  const query = search.value.trim().toLowerCase()
  const candidates = category.value === 'recent' ? recentFields.value :
    category.value === 'selected' ? selectedPageNames.value.map(name => byName.value.get(name)).filter(Boolean) : props.fields
  return candidates.filter(field => {
    if (!field.name.toLowerCase().includes(query)) return false
    switch (category.value) {
      case 'all': case 'recent': case 'selected': return true
      case 'derived': return field.derived
      default: return typeOf(field) === category.value
    }
  })
})
const {list, size, current, rows, before, after, jump, resize, scroll, wheel} = useFieldPickerPages(filtered)
const canConfirm = computed(() => (props.multiple || props.clearable || validDraft.value.length > 0) &&
  (!props.limit || validDraft.value.length <= props.limit))
const bulk = computed(()=>props.multiple && props.limit!==1)
const selectedResultCount = computed(() => filtered.value.filter(field => selectedNames.value.has(field.name)).length)
const allResultsSelected = computed(() => filtered.value.length > 0 && selectedResultCount.value === filtered.value.length)
const someResultsSelected = computed(() => selectedResultCount.value > 0 && !allResultsSelected.value)
const limitReached = computed(() => props.limit > 0 && validDraft.value.length >= props.limit)
const clearResults = computed(() => allResultsSelected.value || (limitReached.value && someResultsSelected.value))
const manual=ref(false), input=ref(''), writer=ref(), notice=ref('')
const index=computed(()=>fieldNameIndex(byName.value.keys()))
const terms=computed(()=>fieldTerms(input.value,index.value))
const cursor=ref(null), choices=ref(new Map())
const activeTerm=computed(()=>{
  const pos=cursor.value??input.value.length
  return terms.value.find(t=>pos<=t.end)?.text??terms.value.at(-1)?.text??''
})
const suggestions=computed(()=>suggestedFieldNames(activeTerm.value,index.value))
const matches=computed(()=>resolveFieldTerms(terms.value,index.value,choices.value))
const activeChoices=computed(()=>new Set(resolveFieldTerms([{text:activeTerm.value}],index.value,choices.value).matched))
const unknown=computed(()=>matches.value.unknown.filter(t=>t!==activeTerm.value||!suggestions.value.length))
function trackCursor(){cursor.value=writer.value?.textarea?.selectionStart??null}
function chooseSuggestion(name){
  if(!bulk.value){
    input.value=name===name.trim()?name:JSON.stringify([name])
    cursor.value=null
    choices.value.clear()
    return
  }
  const selected=new Set(activeChoices.value)
  if(selected.has(name))selected.delete(name)
  else selected.add(name)
  choices.value.set(activeTerm.value,[...selected])
}
watch(terms,items=>{
  const remaining=new Set(items.map(t=>t.text))
  for(const term of choices.value.keys())if(!remaining.has(term))choices.value.delete(term)
})
const canMatch=computed(()=>matches.value.matched.length>0 && (bulk.value||matches.value.matched.length===1))
const overflow=computed(()=>bulk.value&&props.limit ? addFields(validDraft.value,matches.value.matched,props.limit).overflow : 0)
const positions=computed(()=>new Map(filtered.value.map((field,i)=>[field.name,i])))
const drag=ref(null)
let anchor=null, ignoreClick=false
function rangeNames(first,last){
  const a=positions.value.get(first), b=positions.value.get(last)
  if(a===undefined||b===undefined)return []
  return filtered.value.slice(Math.min(a,b),Math.max(a,b)+1).map(f=>f.name)
}
function changeMany(names, on) {
  dismissProfile()
  if(on){
    const result=addFields(validDraft.value,names,props.limit)
    draft.value=result.names
    notice.value=result.overflow?'选择上限为 '+props.limit+' 个，另有 '+result.overflow+' 个字段未加入。':''
  }else{
    const removed=new Set(names)
    draft.value=validDraft.value.filter(n=>!removed.has(n))
    notice.value=''
  }
}
function toggleResults(event) {
  anchor = null
  changeMany(filtered.value.map(field => field.name), !clearResults.value)
  // 浏览器点击后会重置半选标记；有选择上限时，结果可能仍为半选。
  event.target.checked = allResultsSelected.value
  event.target.indeterminate = someResultsSelected.value
}
function enterManual(text=''){
  dismissProfile();drag.value=null;notice.value=''
  cursor.value=null;choices.value.clear()
  input.value=text;manual.value=true
  nextTick(()=>writer.value?.focus())
}
function applyMatches(){
  if(!canMatch.value)return
  if(bulk.value)changeMany(matches.value.matched,true)
  else draft.value=matches.value.matched.slice()
  input.value='';manual.value=false;category.value='selected';search.value='';anchor=null;jump()
}
function pasteSearch(event){
  const text=event.clipboardData?.getData('text/plain')||''
  if(/[\r\n\t,，;；、]/.test(text)){event.preventDefault();enterManual(text)}
}
function beginDrag(field,event){
  if(!bulk.value||event.pointerType!=='mouse'||event.button!==0||event.shiftKey)return
  drag.value={from:field.name,to:field.name,on:!selectedNames.value.has(field.name)}
}
function enterRow(field,event){
  if(drag.value && event.buttons===1){drag.value.to=field.name;hideTransient()}
}
function inRange(name){
  if(!drag.value||drag.value.from===drag.value.to)return false
  const a=positions.value.get(drag.value.from),b=positions.value.get(drag.value.to),i=positions.value.get(name)
  return i>=Math.min(a,b)&&i<=Math.max(a,b)
}
function endDrag(event){
  const gesture=drag.value
  drag.value=null
  if(!gesture||gesture.from===gesture.to||!list.value?.contains(event.target))return
  changeMany(rangeNames(gesture.from,gesture.to),gesture.on)
  anchor={name:gesture.from,on:gesture.on}
  ignoreClick=true;setTimeout(()=>{ignoreClick=false},0)
}
function cancelDrag(){drag.value=null}

let timer, pendingField
function stopTimer() {
  clearTimeout(timer)
  pendingField = null
}
function dismissProfile() {
  stopTimer()
  preview.value = null
  pinned.value = false
}
function hideTransient() {
  stopTimer()
  if (!pinned.value) preview.value = null
}
function leaveList() {
  stopTimer()
  if (!pinned.value) timer = setTimeout(hideTransient, 250)
}
function showProfile(field, x, y) {
  stopTimer()
  if (pinned.value || !workspace.value || preview.value === field) return
  const box = workspace.value.getBoundingClientRect()
  // 在指针另一侧打开，避免浮层遮住触发行，使双击和连续滚动保持可用。
  floatLeft.value = x > document.documentElement.clientWidth / 2
  floatAbove.value = y > box.top + box.height / 2
  preview.value = field
}
function hover(field,event) {
  if (event.pointerType === 'touch' || drag.value || pinned.value || pendingField === field) return
  stopTimer()
  if (preview.value === field) return
  pendingField = field
  const {clientX,clientY} = event
  timer = setTimeout(() => showProfile(field,clientX,clientY), 320)
}
function previewAt(field,event) {
  if(drag.value)return
  const rect = event.currentTarget.getBoundingClientRect()
  showProfile(field,event.clientX || rect.left + 12,event.clientY || rect.top + rect.height/2)
}
function atLimit(name) {
  return props.multiple && props.limit > 1 && validDraft.value.length >= props.limit && !selectedNames.value.has(name)
}
function select(field,event) {
  if(ignoreClick)return
  const name = field.name
  if(bulk.value && event.shiftKey && anchor && positions.value.has(anchor.name)){
    changeMany(rangeNames(anchor.name,name),anchor.on)
    return
  }
  previewAt(field,event)
  if (atLimit(name)) return
  notice.value=''
  if (!props.multiple || (props.limit === 1 && !selectedNames.value.has(name))) draft.value = [name]
  else if (selectedNames.value.has(name)) draft.value = draft.value.filter(f => f !== name)
  else draft.value.push(name)
  anchor={name,on:draft.value.includes(name)}
}
function confirm() {
  if (!canConfirm.value) return
  const names = validDraft.value
  const original=new Set(Array.isArray(props.selected)?props.selected:[])
  const added = props.multiple ? names.filter(name => !original.has(name)).slice(-30).reverse() : names
  if (added.length) recentByDataset.set(dataset.value?.id, [...new Set([...added, ...recent.value])].slice(0, 30))
  emit('select', props.multiple ? names : names[0] ?? null)
  emit('update:modelValue', false)
}
function quickSelect(field) {
  if (props.multiple) return
  draft.value = [field.name]
  confirm()
}
function examples(field) {
  return field.values?.slice(0,4).map(item => item.value === null ? '缺失' : String(item.value)).join('、') ||
    (field.derived ? '前序步骤执行后生成' : '—')
}
watch(() => props.modelValue, open => {
  dismissProfile()
  if (!open) return
  cursor.value=null;choices.value.clear()
  anchor=null;drag.value=null;notice.value='';manual.value=!!props.initialText;input.value=props.initialText
  search.value = ''; category.value = 'all'; jump()
  draft.value = Array.isArray(props.selected) ? props.selected.slice() : props.selected ? [props.selected] : []
  recent.value = recentByDataset.get(dataset.value?.id) || []
}, {immediate:true})
watch([category,search], () => { anchor=null;cancelDrag();jump();hideTransient() })
// 进入已选列表时固定行顺序，以便取消后仍能在原位置重新勾选。
watch([category,manual], ([key, isManual]) => {
  if (key === 'selected' && !isManual) selectedPageNames.value = validDraft.value
})
watch(writer,el=>{if(el&&manual.value)el.focus()},{flush:'post'})
watch(()=>props.initialText,text=>{if(props.modelValue&&manual.value)input.value=text},{flush:'sync'})
watch([() => dataset.value?.id, () => dataset.value?.revision], () => {
  if (props.modelValue) emit('update:modelValue', false)
})
onMounted(()=>{window.addEventListener('pointerup',endDrag);window.addEventListener('blur',cancelDrag);window.addEventListener('pointercancel',cancelDrag)})
onBeforeUnmount(()=>{stopTimer();window.removeEventListener('pointerup',endDrag);window.removeEventListener('blur',cancelDrag);window.removeEventListener('pointercancel',cancelDrag)})
</script>
<template>
  <el-dialog :model-value="modelValue" class="field-picker-dialog" width="min(940px, 96vw)" top="5vh"
    append-to-body destroy-on-close @open-auto-focus="manual?writer?.focus():finder?.focus()" @update:model-value="emit('update:modelValue',$event)">
    <template #header="{titleId,titleClass}">
      <h2 :id="titleId" :class="titleClass">内容选择器</h2>
      <el-button v-if="!manual" class="picker-manual-button" @click="enterManual()">手工输入</el-button>
    </template>
    <div ref="workspace" class="picker-workspace" :class="{'manual-workspace':manual}">
      <nav v-if="!manual" class="picker-categories" aria-label="字段分类">
        <button v-for="[key,label,count] in categories" :key="key" type="button" :aria-pressed="category===key"
          @click="category=key"><span>{{label}}</span><small>{{count}}</small></button>
      </nav>
      <section v-if="manual" class="picker-manual" aria-label="手工匹配字段">
        <label>输入或粘贴字段名</label>
        <el-input ref="writer" v-model="input" type="textarea" :rows="4" aria-label="输入或粘贴字段名" placeholder="输入部分名称查找字段，或粘贴以换行、逗号分隔的字段名。" @input="trackCursor" @click="trackCursor" @keyup="trackCursor" @select="trackCursor" @keydown.ctrl.enter.prevent="applyMatches"/>
        <p class="muted">完整字段名自动匹配。输入部分名称后可在下方选择；粘贴多项时，点击输入框中的某一项可查看它的候选字段。</p>
        <ManualFieldSuggestions :term="activeTerm" :names="suggestions" :selected="activeChoices" :multiple="bulk" @choose="chooseSuggestion"/>
        <div v-if="input.trim()" class="match-results" aria-live="polite">
          <div><strong>已匹配 {{matches.matched.length}}</strong><SelectedFieldNames :names="matches.matched" :placeholder="suggestions.length?'请选择候选字段':'没有匹配的字段'"/></div>
          <div v-if="unknown.length"><strong>未匹配 {{unknown.length}}</strong><SelectedFieldNames :names="unknown"/></div>
          <div v-if="matches.ambiguous.length"><strong>大小写有歧义 {{matches.ambiguous.length}}</strong><SelectedFieldNames :names="matches.ambiguous"/></div>
          <p v-if="!bulk&&matches.matched.length>1" class="picker-warning">此处只允许选择一个字段，请缩小输入范围。</p>
          <p v-if="overflow" class="picker-warning">达到 {{limit}} 个上限后，余下 {{overflow}} 个字段不会加入。</p>
        </div>
      </section>
      <div v-else class="picker-main">
        <div class="picker-search">
          <el-input ref="finder" v-model="search" :prefix-icon="Search" placeholder="搜索字段名称" aria-label="搜索字段名称" clearable @paste="pasteSearch"/>
        </div>
        <div ref="list" class="picker-list" tabindex="0" aria-label="字段列表" :style="{'--batch-size':size}"
          @scroll.passive="scroll(); hideTransient()" @wheel.passive="wheel">
          <table aria-label="可选字段">
            <thead><tr>
              <th class="picker-name" scope="col"><div class="picker-name-heading">
                <input v-if="bulk" type="checkbox" :checked="allResultsSelected" :indeterminate="someResultsSelected"
                  :disabled="!filtered.length || (limitReached && !selectedResultCount)"
                  :aria-label="clearResults ? '取消选中当前结果' : '选中全部当前结果'"
                  :title="clearResults ? '取消选中当前结果' : '选中全部当前结果'" @change="toggleResults">
                字段名称
              </div></th>
              <th class="picker-dtype" scope="col">数据类型</th><th class="picker-example" scope="col">示例值</th>
            </tr></thead>
            <tbody @pointerleave="leaveList">
              <tr v-if="before" class="field-spacer" aria-hidden="true"><td colspan="3" :style="{height:before+'px'}"/></tr>
              <tr v-for="field in rows" :key="field.name" data-field-row :class="{chosen:selectedNames.has(field.name),'in-range':inRange(field.name)}"
                @pointerdown="beginDrag(field,$event)" @pointerenter="enterRow(field,$event)"
                @pointermove="hover(field,$event)" @click="select(field,$event)" @dblclick="quickSelect(field)">
                <td><button type="button" class="picker-option" :aria-disabled="atLimit(field.name)" :aria-pressed="selectedNames.has(field.name)"
                  :title="atLimit(field.name) ? '已达选择上限，取消一个已选字段后可更换' : field.name"
                  @focus="previewAt(field,$event)" @click.stop="select(field,$event)">
                  <span class="picker-mark" :class="{multi:multiple}" aria-hidden="true"><el-icon v-if="selectedNames.has(field.name)"><Check/></el-icon></span>
                  <span>{{field.name}}</span>
                </button></td>
                <td class="picker-dtype" :title="field.dtype">{{field.dtype}}</td>
                <td class="picker-example" :title="examples(field)">{{examples(field)}}</td>
              </tr>
              <tr v-if="after" class="field-spacer" aria-hidden="true"><td colspan="3" :style="{height:after+'px'}"/></tr>
            </tbody>
          </table>
          <p v-if="!rows.length" class="picker-empty">{{category==='selected'?'没有匹配的已选字段':category==='recent'?'没有匹配的最近使用字段':'没有匹配的字段'}}</p>
        </div>
        <div class="picker-paging">
          <small v-if="bulk" class="picker-hint">拖选或 Shift 连选</small>
          <span>共 {{filtered.length}} 条</span>
          <select :value="size" aria-label="每页字段数" @change="resize(Number($event.target.value)); hideTransient()">
            <option v-for="n in pageSizes" :key="n" :value="n">{{n}} 条/页</option>
          </select>
          <el-pagination :current-page="current" :page-size="size" :total="filtered.length" :pager-count="5" size="small"
            layout="prev,pager,next" @update:current-page="jump($event); hideTransient()"/>
        </div>
      </div>
    </div>
    <FieldProfileFloat v-if="modelValue && preview" v-model:pinned="pinned" :field="preview" :dataset="dataset"
      :left="floatLeft" :above="floatAbove" @close="dismissProfile" @enter="stopTimer" @leave="leaveList"/>
    <template #footer>
      <span class="picker-status" aria-live="polite">{{multiple ? '已选 '+validDraft.length+(limit?' / '+limit:'')+' 个字段' : '单选'}}<span v-if="notice&&!manual" class="picker-warning"> · {{notice}}</span></span>
      <div v-if="manual" class="picker-actions"><el-button @click="manual=false">返回列表</el-button><el-button type="primary" :disabled="!canMatch" @click="applyMatches">{{bulk?'加入已选':'使用匹配字段'}}</el-button></div>
      <div v-else class="picker-actions">
        <el-button v-if="(multiple||clearable)&&draft.length" text @click="changeMany(validDraft,false);anchor=null">清空</el-button>
        <el-button @click="emit('update:modelValue',false)">取消</el-button><el-button type="primary" :disabled="!canConfirm" @click="confirm">{{confirmLabel}}</el-button>
      </div>
    </template>
  </el-dialog>
</template>
<style>
.field-picker-dialog.el-dialog{--el-dialog-padding-primary:0px;--picker-height:min(510px,67dvh);--picker-sidebar:148px;padding:0}
.field-picker-dialog .el-dialog__header{display:grid;grid-template-columns:var(--picker-sidebar) minmax(0,1fr);align-items:center;min-height:50px;box-sizing:border-box;margin:0;padding:6px 0;border-bottom:1px solid var(--border)}
.field-picker-dialog .el-dialog__title{margin:0 0 0 16px;white-space:nowrap}
.picker-manual-button{justify-self:center}
.field-picker-dialog .el-dialog__headerbtn{height:44px;width:44px}
.field-picker-dialog .el-dialog__body{position:relative}
.picker-workspace{resize:vertical;overflow:hidden;min-height:170px;max-height:74dvh;display:grid;grid-template-columns:var(--picker-sidebar) minmax(0,1fr);height:var(--picker-height);color:var(--text)}
.picker-categories{background:var(--soft);border-right:1px solid var(--border);padding:10px 6px;min-height:0;overflow:auto}
.picker-categories button{width:100%;display:flex;align-items:center;justify-content:space-between;gap:4px;min-height:35px;padding:6px 9px;
  border:0;border-left:2px solid transparent;background:transparent;color:var(--text);font:inherit;font-size:13px;cursor:pointer;text-align:left}
.picker-categories button:nth-child(4){margin-top:12px}
.picker-categories button:hover{background:var(--hover)}
.picker-categories button[aria-pressed=true]{background:var(--accent-soft);color:var(--accent-text);border-left-color:var(--accent);font-weight:600}
.picker-categories small{font-size:11px;font-variant-numeric:tabular-nums;color:var(--muted)}
.picker-main{display:grid;grid-template-rows:auto minmax(0,1fr) auto;min-width:0;min-height:0;padding:0 12px}
.picker-search{padding:10px 0}
.picker-workspace.manual-workspace{grid-template-columns:minmax(0,1fr)}
.picker-manual{overflow:auto;padding:16px;min-width:0}.picker-manual>label{display:block;margin-bottom:8px;font-weight:600}.picker-manual .muted{margin:8px 0}
.match-results>div{display:grid;grid-template-columns:minmax(100px,auto) minmax(0,1fr);align-items:start;gap:12px;padding:8px 0;border-top:1px solid var(--border);font-size:13px}
.picker-warning{color:var(--warning)}.picker-list .field-spacer>td{padding:0;border:0}.picker-list tr.in-range{background:var(--table-hover);outline:1px solid var(--accent-text);outline-offset:-1px}.picker-list [data-field-row]{user-select:none}
.picker-list{--picker-row-height:var(--table-row-height);min-height:0;overflow:auto;overscroll-behavior:contain;overflow-anchor:none;scrollbar-gutter:stable;
  max-height:calc((var(--batch-size) + 1) * var(--picker-row-height) - 2px)}
.picker-list table{width:100%;border-collapse:separate;border-spacing:0;table-layout:fixed;font-size:13px;text-align:left;color:var(--text)}
.picker-list th,.picker-list td{box-sizing:border-box;padding:4px 8px;height:var(--picker-row-height);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.picker-list th{position:sticky;top:0;z-index:1;border-top:2px solid var(--table-rule);background:var(--table-head);font-weight:600}
.picker-name{width:36%}.picker-dtype{width:18%}
.picker-name-heading{display:flex;align-items:center;gap:7px}
.picker-name-heading input{flex:none;width:14px;height:14px;margin:0;accent-color:var(--accent);cursor:pointer}
.picker-name-heading input:disabled{cursor:not-allowed}
.picker-list tbody tr{background:var(--table-row);cursor:pointer}
.picker-list tbody tr:hover,.picker-list tbody tr:focus-within{background:var(--table-hover)}
.picker-list tbody tr.chosen{background:var(--accent-soft)}
.picker-option{width:100%;display:flex;align-items:center;gap:7px;background:transparent;border:0;color:inherit;font:inherit;padding:0;text-align:left;cursor:pointer}
.picker-option[aria-disabled=true]{color:var(--muted);cursor:not-allowed}
.picker-option>span:last-child{overflow:hidden;text-overflow:ellipsis}
.picker-mark{width:14px;height:14px;flex:none;display:grid;place-items:center;color:var(--accent-text)}
.picker-mark.multi{border:1px solid var(--border);border-radius:var(--radius)}
.chosen .picker-mark.multi{background:var(--accent);color:var(--on-brand);border-color:var(--accent)}
.picker-empty{padding:24px 8px;text-align:center;color:var(--muted)}
.picker-paging{border-top:1px solid var(--border);display:flex;flex-wrap:wrap;align-items:center;justify-content:flex-end;gap:8px;padding:6px 0;font-size:12px;color:var(--muted)}
.picker-paging select{font:inherit;color:var(--text);background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);height:26px;padding:0 4px}
.picker-paging .el-pagination{margin:0}
.picker-hint{margin-right:auto;font-size:inherit}
.field-picker-dialog .el-dialog__footer{padding:8px 14px;margin:0;gap:8px}
.picker-status{margin-right:auto;color:var(--muted);font-size:12px}
.picker-actions{display:flex;align-items:center;gap:6px}.picker-actions .el-button+.el-button{margin:0}
@media(max-width:640px){
  .field-picker-dialog.el-dialog{--picker-sidebar:88px}
  .picker-categories{padding:6px 2px}.picker-categories button{padding:6px 4px;font-size:12px}.picker-categories small{font-size:10px}
  .picker-main{padding:0 6px}.picker-search{padding:6px 0}
  .picker-list table{min-width:410px}.picker-name{width:145px}.picker-dtype{width:92px}
  .picker-list td,.picker-list th{padding-inline:4px}
  .picker-paging{gap:4px}.picker-paging .el-pagination{margin-left:auto}
  .field-picker-dialog .el-dialog__footer{padding:6px 8px}
}
</style>
