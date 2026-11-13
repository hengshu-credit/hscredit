<script setup>
import {computed} from 'vue'
import {splitMethods} from '../splitting'
import {actionTypes,actionNode,datasetNode,emptyCondition} from '../split-tree'
import SplitCondition from './SplitCondition.vue'
import SplitParameters from './SplitParameters.vue'
const props=defineProps({node:Object,dataset:Object,depth:{type:Number,default:1},path:{type:String,default:'全部样本'}})
const emit=defineEmits(['replace'])
const columns=computed(()=>props.dataset?.columns||[])
const ratio=computed(()=>props.node.type==='split'&&props.node.form.method!=='oot'&&!(props.node.form.method==='time'&&props.node.form.time_mode==='cutoff'))
function changeType(type){emit('replace',actionNode(type,props.dataset,props.node.name))}
function changeMethod(method){if(method==='oot'){props.node.form.parts=props.node.form.parts.slice(0,2);props.node.children=props.node.children.slice(0,2)}}
function addBranch(){const n=props.node.form.parts.length+1;props.node.form.parts.push({name:`分支${n}`,percent:10});props.node.children.push(datasetNode(`${(props.dataset?.name||'数据集').slice(0,115)}_结果${n}`))}
function removeBranch(index){props.node.form.parts.splice(index,1);props.node.children.splice(index,1);props.node.form.cutoffs.splice(Math.max(0,index-1),1)}
function moveRule(index,offset){const rows=props.node.rules;[rows[index],rows[index+offset]]=[rows[index+offset],rows[index]]}
</script>
<template>
  <section class="split-tree-node" :data-node-type="node.type">
    <div class="split-tree-node-head"><span class="split-tree-path" :title="path">{{path}}</span><el-select :model-value="node.type" aria-label="分支动作" @update:model-value="changeType"><el-option v-for="(label,value) in actionTypes" :key="value" :value="value" :label="label" :disabled="depth>=7&&['condition','split'].includes(value)"/></el-select></div>
    <div v-if="node.type==='dataset'" class="split-leaf-content"><el-form-item label="结果数据集名称" required><el-input v-model="node.name" maxlength="150" aria-label="结果数据集名称" placeholder="每个叶子单独命名"/></el-form-item></div>
    <div v-else-if="node.type==='exclude'" class="split-leaf-content"><el-form-item label="排除原因" required><el-input v-model="node.reason" maxlength="150" aria-label="排除原因" placeholder="预览会汇总排除行数与原因"/></el-form-item></div>
    <div v-else-if="node.type==='condition'" class="split-decision-table">
      <p class="muted">从上到下首次命中；未命中的样本进入“其他”。每个动作都可以继续嵌套。</p>
      <div v-for="(rule,index) in node.rules" :key="index" class="split-rule-card">
        <div class="split-rule-toolbar"><strong>#{{index+1}}</strong><el-button text size="small" :disabled="index===0" @click="moveRule(index,-1)">上移</el-button><el-button text size="small" :disabled="index===node.rules.length-1" @click="moveRule(index,1)">下移</el-button><el-button text size="small" type="danger" :disabled="node.rules.length<=1" @click="node.rules.splice(index,1)">删除规则</el-button></div>
        <div class="split-rule-grid"><SplitCondition :node="rule.condition" :columns="columns"/><div class="split-rule-action"><strong>动作 THEN</strong><SplitTreeNode :node="rule.action" :dataset="dataset" :depth="depth+1" :path="`条件 ${index+1} 命中样本`" @replace="rule.action=$event"/></div></div>
      </div>
      <el-button :disabled="node.rules.length>=20" @click="node.rules.push({condition:emptyCondition(),action:datasetNode('')})">＋ 添加规则行</el-button>
      <div class="split-otherwise"><strong>其他 ELSE · 前述条件均未命中</strong><SplitTreeNode :node="node.otherwise" :dataset="dataset" :depth="depth+1" path="其他样本" @replace="node.otherwise=$event"/></div>
    </div>
    <div v-else class="split-node-config">
      <el-form-item label="划分方式"><el-select v-model="node.form.method" aria-label="节点划分方式" @change="changeMethod"><el-option v-for="(label,value) in splitMethods" :key="value" :value="value" :label="label"/></el-select></el-form-item>
      <SplitParameters :form="node.form" :columns="columns" :dataset="dataset"/>
      <div class="split-branch-heading"><strong>划分结果与后续动作</strong><el-button v-if="node.form.method!=='oot'" :disabled="node.children.length>=10" @click="addBranch">＋ 添加划分分支</el-button></div>
      <div v-for="(part,index) in node.form.parts" :key="index" class="split-action-branch">
        <div class="split-branch-settings"><el-form-item label="分支名称"><el-input v-model="part.name" maxlength="150" aria-label="分支名称"/></el-form-item><el-form-item v-if="ratio" label="占当前节点样本（%）"><el-input-number :controls="false" v-model="part.percent" :min="0.01" :max="99.99" :precision="2" aria-label="节点分支比例"/></el-form-item><el-button v-if="node.form.method!=='oot'" text type="danger" :disabled="node.children.length<=2" @click="removeBranch(index)">移除分支</el-button></div>
        <el-form-item v-if="node.form.method==='time'&&node.form.time_mode==='cutoff'&&index<node.children.length-1" label="本分支与下一分支之间的时间边界"><el-input v-model="node.form.cutoffs[index]" placeholder="例如 2025-06-01 00:00:00" aria-label="节点时间边界"/></el-form-item>
        <SplitTreeNode :node="node.children[index]" :dataset="dataset" :depth="depth+1" :path="part.name||`分支 ${index+1}`" @replace="node.children[index]=$event"/>
      </div>
      <p v-if="ratio" class="muted">本节点比例合计：{{node.form.parts.reduce((sum,p)=>sum+(p.percent||0),0).toFixed(2)}}%。分组划分按组数分配。</p>
    </div>
  </section>
</template>
