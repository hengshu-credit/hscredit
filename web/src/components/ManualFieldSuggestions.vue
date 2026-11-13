<script setup>
import {computed,watch} from 'vue'
import {Check} from '@element-plus/icons-vue'
import {useFieldPickerPages} from '../field-picker-pages'

const props=defineProps({names:{type:Array,required:true},selected:{type:Set,required:true},term:String,multiple:Boolean})
const emit=defineEmits(['choose'])
const names=computed(()=>props.names)
const {list,rows,before,after,jump,scroll,wheel}=useFieldPickerPages(names)
watch(names,()=>jump())
</script>
<template>
  <section v-if="names.length" class="manual-candidates" aria-label="候选字段">
    <header><strong>候选字段</strong><span>「{{term}}」 · {{names.length}} 项 · {{multiple?'可多选':'单选'}}</span></header>
    <div ref="list" class="picker-list suggestion-list" :style="{'--suggestions':names.length}" tabindex="0" @scroll.passive="scroll" @wheel.passive="wheel">
      <table aria-label="匹配候选列表"><tbody>
        <tr v-if="before" class="field-spacer" aria-hidden="true"><td :style="{height:before+'px'}"/></tr>
        <tr v-for="name in rows" :key="name" data-field-row :class="{chosen:selected.has(name)}"><td>
          <button type="button" class="picker-option" :aria-pressed="selected.has(name)" :title="name" @click="emit('choose',name)">
            <span class="picker-mark" :class="{multi:multiple}" aria-hidden="true"><el-icon v-if="selected.has(name)"><Check/></el-icon></span>
            <span>{{name}}</span>
          </button>
        </td></tr>
        <tr v-if="after" class="field-spacer" aria-hidden="true"><td :style="{height:after+'px'}"/></tr>
      </tbody></table>
    </div>
  </section>
</template>
<style scoped>
.manual-candidates{margin:10px 0;border-block:1px solid var(--table-rule)}
.manual-candidates header{display:flex;align-items:baseline;flex-wrap:wrap;gap:8px;padding:6px 10px;background:var(--table-head)}
.manual-candidates header span{color:var(--muted);font-size:12px;overflow-wrap:anywhere}
.manual-candidates .suggestion-list{height:min(190px,26dvh,calc(var(--picker-row-height) * var(--suggestions)));max-height:52dvh;resize:vertical}
.manual-candidates .suggestion-list table{min-width:0}
.manual-candidates td{padding:0}.manual-candidates .picker-option{min-height:var(--picker-row-height);padding:4px 10px}
</style>
