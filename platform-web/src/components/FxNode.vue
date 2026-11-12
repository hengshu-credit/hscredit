<script setup>
import {computed,ref,watch} from 'vue'
import {ArrowDown} from '@element-plus/icons-vue'
import {emptyNode,literalNode,functionNode,changeNodeKind,parseNumericLiteral,removeOperationTerm} from '../processing'
const props=defineProps({modelValue:{type:Object,required:true},fields:{type:Array,default:()=>[]},catalog:{type:Object,required:true},path:{type:Array,default:()=>[]},depth:{type:Number,default:0}})
const emit=defineEmits(['update:modelValue','pick','remove'])
const node=computed(()=>props.modelValue),label=computed(()=>props.path.length?props.path.join('.'):'根表达式')
const numericText=ref(String(node.value.value??''))
watch(()=>node.value.value,value=>{if(value!==parseNumericLiteral(numericText.value))numericText.value=String(value??'')})
function setNumericText(value){numericText.value=value;patch({value:parseNumericLiteral(value)})}
const selectedField=computed(()=>props.fields.find(f=>f.name===node.value.name))
const args=computed(()=>props.catalog.functions?.find(f=>f.name===node.value.name)?.args||[])
const typeLabel={numeric:'数值',string:'文本',boolean:'布尔',date:'日期',unknown:'待计算'}
function patch(values){emit('update:modelValue',{...node.value,...values})}
function setTerm(i,operand){patch({terms:node.value.terms.map((t,index)=>index===i?{...t,operand}:t)})}
function setOperator(i,operator){patch({terms:node.value.terms.map((t,index)=>index===i?{...t,operator}:t)})}
function removeTerm(i){emit('update:modelValue',removeOperationTerm(node.value,i))}
function setArg(i,arg){patch({args:node.value.args.map((a,index)=>index===i?arg:a)})}
</script>
<template>
  <div class="fx-node">
    <div class="fx-node-head">
      <el-select :model-value="node.kind" :aria-label="`${label}内容类型`" class="fx-kind" @update:model-value="emit('update:modelValue',changeNodeKind($event,fields,catalog.functions))">
        <el-option label="未配置" value="EMPTY"/>
        <el-option label="字段" value="FIELD"/><el-option label="常量" value="LITERAL"/>
        <el-option label="ƒx 组合表达式" value="OPERATION" :disabled="depth>=8"/><el-option label="函数" value="FUNCTION" :disabled="depth>=8"/>
        <el-option label="取负 / 取反" value="UNARY" :disabled="depth>=8"/>
      </el-select>
      <el-input v-if="node.kind==='FIELD'" class="fx-field-input" :model-value="node.name||''" readonly placeholder="选择字段" :aria-label="`选择字段：${node.name||'未选择'}`" aria-haspopup="dialog" @click="emit('pick',path)" @keydown.enter.prevent="emit('pick',path)" @keydown.space.prevent="emit('pick',path)" @keydown.down.prevent="emit('pick',path)"><template #prefix><span class="fx-field-tag">{{typeLabel[selectedField?.valueType]||'字段'}}</span></template><template #suffix><el-icon><ArrowDown/></el-icon></template></el-input>
      <template v-else-if="node.kind==='LITERAL'"><el-select :model-value="node.value_type" aria-label="常量类型" class="fx-literal-type" @update:model-value="emit('update:modelValue',literalNode($event==='STRING'?'':$event==='BOOLEAN'?false:$event==='MISSING'?null:0,$event))"><el-option label="数字" value="NUMBER"/><el-option label="文本" value="STRING"/><el-option label="布尔" value="BOOLEAN"/><el-option label="缺失值" value="MISSING"/></el-select>
        <el-input v-if="node.value_type==='NUMBER'" :model-value="numericText" inputmode="decimal" aria-label="数值常量" placeholder="输入数字，例如 0.0001" @update:model-value="setNumericText"/>
        <el-input v-else-if="node.value_type==='STRING'" :model-value="node.value" aria-label="文本常量" placeholder="输入文本（可留空）" maxlength="500" @update:model-value="patch({value:$event})"/>
        <el-select v-else-if="node.value_type==='BOOLEAN'" :model-value="node.value" aria-label="布尔常量" @update:model-value="patch({value:$event})"><el-option label="真" :value="true"/><el-option label="假" :value="false"/></el-select>
        <el-tag v-else type="info">数值缺失 NaN</el-tag>
      </template>
      <el-select v-else-if="node.kind==='FUNCTION'" :model-value="node.name" aria-label="选择函数" filterable class="fx-function" @update:model-value="emit('update:modelValue',functionNode($event,catalog.functions))"><el-option v-for="f in catalog.functions" :key="f.name" :label="`${f.label} · ${f.name}`" :value="f.name"/></el-select>
      <el-select v-else-if="node.kind==='UNARY'" :model-value="node.operator" aria-label="一元运算符" @update:model-value="patch({operator:$event})"><el-option label="取负" value="-"/><el-option label="取正" value="+"/><el-option label="布尔取反" value="not"/></el-select>
      <span v-else class="muted">{{node.kind==='EMPTY'?'选择内容类型开始配置':'括号组合 · 乘除优先，可继续嵌套'}}</span>
      <el-button v-if="['OPERATION','FUNCTION','UNARY'].includes(node.kind)" text type="danger" class="fx-remove-group" :aria-label="depth?'移除当前组合及全部子项':'清空整个表达式'" @click="emit('remove')">{{depth?'移除组合':'清空表达式'}}</el-button>
    </div>
    <div v-if="node.kind==='OPERATION'" class="fx-terms">
      <div v-for="(term,index) in node.terms" :key="index" class="fx-term">
        <el-select v-if="index" :model-value="term.operator" :aria-label="`第${index+1}项运算符`" class="fx-operator" @update:model-value="setOperator(index,$event)"><el-option v-for="o in catalog.operators" :key="o.value" :label="`${o.value} ${o.label}`" :value="o.value"/></el-select>
        <span v-else class="fx-start">起始项</span>
        <FxNode :model-value="term.operand" :fields="fields" :catalog="catalog" :path="[...path,'terms',index,'operand']" :depth="depth+1" @update:model-value="setTerm(index,$event)" @pick="emit('pick',$event)" @remove="removeTerm(index)"/>
        <el-button text type="danger" :aria-label="`删除第${index+1}个运算项及其子项`" @click="removeTerm(index)">移除</el-button>
      </div>
      <p v-if="!node.terms.length" class="muted">组合为空，可追加运算项或双击组合预览输入表达式。</p>
      <el-button class="append-control fx-append" :disabled="node.terms.length>=20" @click="patch({terms:[...node.terms,{...(node.terms.length?{operator:'+'}:{}),operand:emptyNode()}]})">追加运算项</el-button>
    </div>
    <div v-else-if="node.kind==='FUNCTION'" class="fx-arguments"><div v-for="(arg,index) in node.args" :key="index" class="fx-argument"><label>{{args[index]||`参数 ${index+1}`}}</label><FxNode :model-value="arg" :fields="fields" :catalog="catalog" :path="[...path,'args',index]" :depth="depth+1" @update:model-value="setArg(index,$event)" @pick="emit('pick',$event)" @remove="setArg(index,emptyNode())"/><el-button text type="danger" :aria-label="`清空${args[index]}及其子项`" @click="setArg(index,emptyNode())">移除</el-button></div></div>
    <div v-else-if="node.kind==='UNARY'" class="fx-arguments"><FxNode :model-value="node.operand" :fields="fields" :catalog="catalog" :path="[...path,'operand']" :depth="depth+1" @update:model-value="patch({operand:$event})" @pick="emit('pick',$event)" @remove="patch({operand:emptyNode()})"/></div>
  </div>
</template>
