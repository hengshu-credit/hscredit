<script setup>
import {computed,ref} from 'vue'
import {Close,ArrowDown} from '@element-plus/icons-vue'
import {placeField} from '../explore-fields'
import FieldContentPicker from './FieldContentPicker.vue'

const props=defineProps({columns:{type:Array,required:true},axes:{type:Array,required:true},group:{type:String,default:''}})
const emit=defineEmits(['change'])
const active=ref(0),dragging=ref(''),over=ref(null),message=ref(''),picker=ref(false)
const roles=[{key:0,name:'X 轴',hint:'选择分析字段'},{key:1,name:'Y 轴',hint:'可选 · 两字段交叉'},{key:2,name:'Z 轴',hint:'可选 · 三字段组合'},{key:'group',name:'分组标签',hint:'可选 · 最多 30 类'}]
const value=key=>key==='group'?props.group:props.axes[key]
const activeRole=computed(()=>roles.find(role=>role.key===active.value))
function selectRole(key){active.value=key;message.value=''}
function place(key,name){
  const next=placeField('multi',props.axes,props.group,key,name,props.columns)
  if(next.error){message.value=next.error;return}
  emit('change',next)
  active.value=key
  const following=roles.find(role=>role.key===next.axes.length)
  if(following)active.value=following.key
  message.value=''
}
function remove(key){
  const axes=[...props.axes]
  if(key!=='group')axes.splice(key,1)
  emit('change',{axes,group:key==='group'?'':props.group})
  selectRole(key==='group'?key:Math.min(key,axes.length))
}
function start(event,name){
  dragging.value=name;message.value=''
  event.dataTransfer.effectAllowed='move'
  event.dataTransfer.setData('text/plain',name)
}
function end(){dragging.value='';over.value=null}
function dragOver(event,key){
  if(!dragging.value)return
  event.preventDefault();event.dataTransfer.dropEffect='move';over.value=key
}
function drop(event,key){
  if(dragging.value){event.preventDefault();place(key,dragging.value)}
  end()
}
</script>

<template>
  <div class="explore-board">
    <FieldContentPicker v-if="picker" v-model="picker" :fields="active==='group'?columns.filter(c=>c.unique+(c.missing>0?1:0)<=30):columns" :selected="value(active)" :confirm-label="'选为'+activeRole.name" @select="name=>place(active,name)"/>
    <div class="board-main">
      <section class="panel field-layout" aria-label="字段布局">
        <header><h2>字段布局</h2><div class="board-actions"><slot name="actions"/></div></header>
        <p class="muted board-hint">放入一个字段查看分布，放入两至三个字段可同时分析各对组合。点击位置打开内容选择器，已选字段可拖动换位。</p>
        <div class="field-shelves">
          <div v-for="role in roles" :key="role.key" class="field-shelf" :class="{active:active===role.key,hovered:over===role.key,filled:value(role.key)}"
            @dragover="dragOver($event,role.key)" @dragleave="!$event.currentTarget.contains($event.relatedTarget)&&(over=null)" @drop="drop($event,role.key)">
            <button type="button" class="shelf-target" :aria-label="'选择'+role.name" aria-haspopup="dialog" :disabled="role.key!=='group'&&role.key>axes.length"
              :draggable="!!value(role.key)" @dragstart="start($event,value(role.key))" @dragend="end" @click="selectRole(role.key);picker=true">
              <span class="shelf-role">{{role.name}}</span>
              <span class="shelf-value" :title="value(role.key)">{{value(role.key)||role.hint}} <el-icon><ArrowDown/></el-icon></span>
            </button>
            <button v-if="value(role.key)" type="button" class="remove-field" :aria-label="'移除'+role.name" @click="remove(role.key)"><el-icon><Close/></el-icon></button>
            <div v-if="role.key!=='group'&&value(role.key)" class="shelf-display"><slot name="format" :field="value(role.key)"/></div>
          </div>
        </div>
        <slot name="combinations"/>
        <p v-if="message" class="field-message" role="alert">{{message}}</p>
        <div class="board-settings"><slot name="settings"/></div>
      </section>
      <slot/>
    </div>
  </div>
</template>

<style scoped>
.field-layout header,.board-actions{display:flex;align-items:center;justify-content:space-between;gap:8px}.field-layout h2{margin:0}
.board-main{min-width:0}.board-actions{flex-wrap:wrap}.board-hint{margin:6px 0 12px}
.field-shelves{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(170px,100%),1fr));gap:8px}
.field-shelf{display:flex;flex-wrap:wrap;align-content:start;align-items:center;min-width:0;border:1px dashed var(--border);border-radius:var(--radius);color:var(--text)}
.shelf-display{width:100%;padding:0 10px 8px}.shelf-display :deep(select){width:100%;padding:3px;background:var(--surface);color:var(--text);border:1px solid var(--border);font:inherit;font-size:12px}
.field-shelf.filled{border-style:solid}.field-shelf.active{border-color:var(--accent)}.field-shelf.hovered{outline:2px solid var(--accent);background:var(--el-color-primary-light-9)}
.shelf-target,.remove-field{background:transparent;border:0;color:inherit}.shelf-target{display:grid;gap:5px;text-align:left;flex:1;min-width:0;padding:9px 12px}
.filled .shelf-target{cursor:pointer}.shelf-role{font-weight:600;font-size:12px}.shelf-value{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--muted);font-size:12px}.filled .shelf-value{color:var(--accent-text)}
.remove-field{padding:8px;display:grid;place-items:center}.board-settings{border-top:1px solid var(--line-soft);margin-top:12px;padding-top:8px}.board-settings summary{cursor:pointer;width:fit-content;font-size:13px}
.field-message{color:var(--danger);margin:8px 0 0;font-size:13px}
@media(max-width:800px){.field-shelves{grid-template-columns:repeat(2,minmax(0,1fr))}}
</style>
