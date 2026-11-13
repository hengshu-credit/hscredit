<script setup>
import {computed,ref,watch} from 'vue'
import {ElMessageBox} from 'element-plus'
import {datasetNode,leafNames} from '../split-tree'
import {editorMethods,sampleRows,renameSample,makeSampleSplit,appendSample,removeSample,balanceSamples,replaceSample,availableNames,sampleAncestors,sampleSource,dismissSample,restoreSample} from '../split-editor'
import SplitCondition from './SplitCondition.vue'
import SampleHierarchy from './SampleHierarchy.vue'
import {Check,ArrowLeft} from '@element-plus/icons-vue'
import SplitParameters from './SplitParameters.vue'
const props=defineProps({node:Object,dataset:Object,existing:{type:Array,default:()=>[]},disabled:Boolean})
const emit=defineEmits(['replace'])
const selected=ref('[]'),history=ref([])
const rows=computed(()=>sampleRows(props.node)),current=computed(()=>rows.value.find(row=>row.key===selected.value)||rows.value[0]),active=computed(()=>current.value.node)
const ancestors=computed(()=>sampleAncestors(props.node,current.value.path))
const children=computed(()=>rows.value.filter(row=>row.parent===active.value))
const ratio=computed(()=>active.value.type==='split'&&active.value.form.method!=='oot'&&!(active.value.form.method==='time'&&active.value.form.time_mode==='cutoff'))
const method=computed(()=>active.value.type==='condition'?'condition':active.value.type==='split'?active.value.form.method:'')
const canSplit=computed(()=>current.value.depth<7)
watch(()=>props.dataset?.id,()=>{selected.value='[]';history.value=[]})
function remember(){history.value.push({tree:JSON.parse(JSON.stringify(props.node)),selected:selected.value});if(history.value.length>20)history.value.shift()}
function undo(){const previous=history.value.pop();if(previous){emit('replace',previous.tree);selected.value=previous.selected}}
function replace(value){emit('replace',replaceSample(props.node,current.value.path,value))}
async function confirmChange(message){try{await ElMessageBox.confirm(message,'调整样本划分',{confirmButtonText:'确认调整',cancelButtonText:'保留当前配置',type:'warning'});return true}catch{return false}}
async function choose(value){
  if(value===method.value||!canSplit.value)return
  const node=active.value
  if(node.type==='split'&&value!=='condition'){
    if(value==='oot'&&node.children.length>2&&!await confirmChange('时间外划分只保留前两份样本，第三份及之后的配置会移除。可用“撤销”恢复。'))return
    remember();node.form.method=value
    if(value==='oot'){node.form.parts=node.form.parts.slice(0,2);node.children=node.children.slice(0,2);balanceSamples(node)}
    return
  }
  if(['split','condition'].includes(node.type)&&!await confirmChange('切换到此方法将替换当前样本下的划分配置，其他样本不受影响。可用“撤销”恢复。'))return
  remember();replace(makeSampleSplit(value,props.dataset,current.value.depth?current.value.name:(props.dataset?.name||'样本'),availableNames(props.node,props.existing)))
}
async function keep(){
  if(['split','condition'].includes(active.value.type)&&!await confirmChange(current.value.depth?'取消这份样本下面的所有拆分，将其作为一份结果保存。可用“撤销”恢复。':'清空全部划分配置并重新选择方法。可用“撤销”恢复。'))return
  remember();replace(datasetNode(current.value.name))
}
async function dismiss(row=current.value){
  if(props.disabled||!row.parent||row.node.type==='exclude')return
  const count=leafNames(row.node).length
  if(!await confirmChange(`移除「${row.name}」${count>1?`及下方 ${count} 份结果`:''}？这部分样本将不保存，其他分支的比例、时间边界和样本归属保持不变。可恢复或撤销。`))return
  remember();emit('replace',dismissSample(props.node,row.path));selected.value=row.key
}
function restore(row=current.value){if(props.disabled||!row.node.removed)return;remember();emit('replace',restoreSample(props.node,row.path));selected.value=row.key}
function add(){remember();appendSample(active.value,availableNames(props.node,props.existing))}
async function removeRule(index){if(!await confirmChange('删除这条条件及其下方配置后，原本命中的样本会继续匹配后续条件或进入其余样本。可用“撤销”恢复。'))return;remember();removeSample(active.value,index)}
function move(index,offset){remember();const rules=active.value.rules;[rules[index],rules[index+offset]]=[rules[index+offset],rules[index]]}
function balance(){remember();balanceSamples(active.value)}
</script>
<template>
  <div class="sample-workspace">
    <aside class="sample-map" aria-label="样本划分概览">
      <div class="sample-map-heading"><strong>样本去向</strong><el-button text size="small" :disabled="disabled||!history.length" @click="undo">撤销</el-button></div>
      <p class="muted">选择任意一份，独立配置或继续拆分</p>
      <SampleHierarchy :root="node" :selected="current.key" :disabled="disabled" @select="selected=$event" @remove="dismiss" @restore="restore"/>
      <div v-if="node.type==='dataset'" class="sample-map-total">选择方法后，在这里查看样本去向</div>
      <div v-else class="sample-map-total">最终保存 <strong>{{leafNames(node).length}}</strong> 份数据集</div>
    </aside>
    <section class="sample-editor" aria-label="当前样本配置">
      <nav v-if="ancestors.length" class="sample-breadcrumbs" aria-label="样本来源路径"><template v-for="row in ancestors" :key="row.key"><el-button text :disabled="disabled" :title="row.depth?row.name:'全部样本'" @click="selected=row.key">{{row.depth?row.name:'全部样本'}}</el-button><span aria-hidden="true">/</span></template><span aria-current="page">{{current.name}}</span></nav>
      <div v-if="current.depth" class="sample-parent"><el-button text :icon="ArrowLeft" :disabled="disabled" @click="selected=ancestors.at(-1).key">返回上级</el-button><span>第 {{current.depth}} 层 · {{sampleSource(current)}}</span></div>
      <div class="sample-editor-heading"><div><span class="muted">正在编辑{{current.depth?' · 仅影响这份样本':' · 从全部样本开始'}}</span><h3>{{current.depth?current.name:'全部样本'}}</h3></div><el-tag effect="plain">{{!current.depth&&active.type==='dataset'?'尚未划分':current.status}}</el-tag></div>
      <el-form-item v-if="current.depth" :label="active.type==='dataset'?'结果名称':'样本名称'"><el-input :model-value="current.name" maxlength="150" aria-label="当前样本名称" @update:model-value="renameSample(current,$event)"/></el-form-item>
      <p v-if="!active.removed" class="sample-instruction">{{method?'可更换当前方法；下方每份结果都可以继续拆分。':'选择如何拆分这份样本，方法和先后顺序由你决定。'}}</p>
      <div v-if="!active.removed" class="sample-methods" role="group" aria-label="选择划分方式">
        <button v-for="(option,value) in editorMethods" :key="value" type="button" :class="{active:method===value}" :aria-pressed="method===value" :disabled="disabled||!canSplit||(value==='condition'&&current.depth>5)" @click="choose(value)"><strong><el-icon v-if="method===value" aria-hidden="true"><Check/></el-icon>{{option.label}}</strong><span>{{option.hint}}</span></button>
      </div>
      <p v-if="!canSplit" class="muted">已达到最大划分层数，这份样本可以保存或排除。</p>
      <template v-if="active.type==='split'">
        <SplitParameters :key="current.key" :form="active.form" :columns="dataset?.columns||[]" :dataset="dataset"/>
        <div class="sample-section-title"><strong>拆分为 {{children.length}} 份</strong><div><el-button v-if="ratio" text @click="balance">均分比例</el-button><el-button v-if="active.form.method!=='oot'" :disabled="children.length>=10" @click="add">＋ 添加一份</el-button></div></div>
        <div v-for="(row,index) in children" :key="row.key" class="sample-result" :class="{'is-removed':row.node.type==='exclude'}">
          <div class="sample-result-row"><span class="sample-result-index">{{index+1}}</span><el-input :model-value="row.name" :aria-label="`第 ${index+1} 份名称`" maxlength="150" placeholder="样本名称" @update:model-value="renameSample(row,$event)"/><div v-if="ratio" class="sample-percent"><el-input-number v-model="active.form.parts[index].percent" :controls="false" :min="0.01" :max="99.99" :precision="2" :aria-label="`第 ${index+1} 份比例`"/><span>%</span></div><el-button :aria-label="`编辑第 ${index+1} 份样本`" @click="selected=row.key">{{row.node.type==='exclude'?'查看详情':['split','condition'].includes(row.node.type)?'编辑划分':'继续拆分'}}</el-button><el-button v-if="row.node.removed" text :aria-label="`恢复第 ${index+1} 份`" @click="restore(row)">恢复结果</el-button><el-button v-else-if="row.node.type!=='exclude'" text type="danger" :aria-label="`移除第 ${index+1} 份`" @click="dismiss(row)">移除结果</el-button></div>
          <div class="sample-result-caption"><span class="sample-number">{{row.number}}</span>{{row.status}}{{row.node.type==='dataset'?' · 将以此名称保存为数据集':''}}<span v-if="active.form.method==='oot'"> · {{index===0?'起始时间之前':'指定的时间外范围'}}</span></div>
          <el-form-item v-if="active.form.method==='time'&&active.form.time_mode==='cutoff'&&index<children.length-1" :label="`第 ${index+1} / ${index+2} 份的时间边界`" required><el-date-picker v-model="active.form.cutoffs[index]" type="datetime" value-format="YYYY-MM-DDTHH:mm:ss" placeholder="此时刻起进入后一份" :aria-label="`时间边界 ${index+1}`"/></el-form-item>
        </div>
        <p v-if="ratio" class="muted">合计 {{active.form.parts.reduce((sum,p)=>sum+(p.percent||0),0).toFixed(2)}}% · 比例只针对当前这份{{active.form.method==='group'?'样本的分组数':'样本'}}</p>
      </template>
      <template v-else-if="active.type==='condition'">
        <p class="muted">按下方顺序匹配，每行只进入第一个符合的结果；未命中条件的进入“其余样本”。</p>
        <div v-for="(rule,index) in active.rules" :key="index" class="sample-condition-card">
          <div class="sample-section-title"><strong>条件 {{index+1}}</strong><div><el-button text size="small" :disabled="index===0" @click="move(index,-1)">上移</el-button><el-button text size="small" :disabled="index===active.rules.length-1" @click="move(index,1)">下移</el-button><el-button text size="small" type="danger" :disabled="active.rules.length<=1" @click="removeRule(index)">删除条件</el-button></div></div>
          <SplitCondition :node="rule.condition" :columns="dataset?.columns||[]"/>
          <div class="sample-condition-output"><span>符合条件 →</span><el-input :model-value="children[index].name" :aria-label="`条件 ${index+1} 结果名称`" maxlength="150" @update:model-value="renameSample(children[index],$event)"/><el-button @click="selected=children[index].key">{{children[index].node.type==='exclude'?'查看详情':'编辑 / 继续拆分'}}</el-button><el-button v-if="children[index].node.removed" text @click="restore(children[index])">恢复结果</el-button><el-button v-else-if="children[index].node.type!=='exclude'" text type="danger" @click="dismiss(children[index])">移除结果</el-button></div>
        </div>
        <el-button :disabled="active.rules.length>=20" @click="add">＋ 添加条件分支</el-button>
        <div class="sample-condition-output sample-remainder"><strong>其余样本 →</strong><el-input :model-value="children.at(-1).name" aria-label="其余样本名称" maxlength="150" @update:model-value="renameSample(children.at(-1),$event)"/><el-button @click="selected=children.at(-1).key">{{children.at(-1).node.type==='exclude'?'查看详情':'编辑 / 继续拆分'}}</el-button><el-button v-if="children.at(-1).node.removed" text @click="restore(children.at(-1))">恢复结果</el-button><el-button v-else-if="children.at(-1).node.type!=='exclude'" text type="danger" @click="dismiss(children.at(-1))">移除结果</el-button></div>
      </template>
      <div v-else-if="active.removed" class="sample-leaf-note"><strong>这份结果已移除</strong><p>本分支及下方结果不再生成数据集。保留来源位置用于核对样本去向；其他分支不受影响。</p><el-button @click="restore()">恢复结果及下级划分</el-button></div>
      <div v-else-if="active.type==='exclude'" class="sample-leaf-note"><el-form-item label="排除原因" required><el-input v-model="active.reason" maxlength="150" aria-label="排除原因" placeholder="例如：不在本次建模范围"/></el-form-item></div>
      <div v-else class="sample-leaf-note">{{current.depth?'这份样本将直接保存。需要细分时，选择上方任意方法。':'例如：先按时间拆分，再只对其中一份随机拆分；也可以先随机，再按条件细分。'}}</div>
      <div v-if="current.depth||method" class="sample-disposition"><el-button v-if="active.type!=='dataset'&&!active.removed" text @click="keep">{{!current.depth?'清空划分，重新开始':method?'取消下方拆分，整份保存':'恢复为保存结果'}}</el-button><el-button v-if="current.depth&&active.type!=='exclude'" text type="danger" @click="dismiss()">移除这份结果</el-button></div>
    </section>
  </div>
</template>
