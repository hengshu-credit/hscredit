<script setup>
import {computed,nextTick,ref,watch} from 'vue'
import {FolderOpened,Document,Remove,Delete,RefreshLeft} from '@element-plus/icons-vue'
import {sampleHierarchy,sampleAncestors,sampleSource} from '../split-editor'
const props=defineProps({root:Object,selected:String,disabled:Boolean})
const emit=defineEmits(['select','remove','restore'])
const outline=ref(null),items=computed(()=>sampleHierarchy(props.root))
watch(()=>[props.selected,items.value],async(_,previous)=>{
  await nextTick()
  outline.value?.setCurrentKey(props.selected,false)
  if(props.selected!==previous?.[0])for(const row of sampleAncestors(props.root,JSON.parse(props.selected))){const node=outline.value?.getNode(row.key);if(node)node.expanded=true}
},{immediate:true})
function select(row){if(!props.disabled)emit('select',row.key)}
function onKeydown(event){
  const item=event.target
  if(item.getAttribute('role')!=='treeitem'||!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Home','End','Enter',' '].includes(event.key))return
  event.preventDefault();event.stopPropagation()
  if(props.disabled)return
  const node=outline.value.getNode(item.dataset.key),elements=[...outline.value.$el.querySelectorAll('[role="treeitem"]')].filter(el=>el.getClientRects().length)
  if(event.key==='ArrowLeft'){if(node.expanded&&node.childNodes.length)node.collapse();else elements.find(el=>el.dataset.key===node.parent?.key)?.focus()}
  else if(event.key==='ArrowRight'){if(!node.expanded&&node.childNodes.length)node.expand();else elements.find(el=>el.dataset.key===node.childNodes[0]?.key)?.focus()}
  else if(event.key==='ArrowUp'||event.key==='ArrowDown')elements[Math.max(0,Math.min(elements.length-1,elements.indexOf(item)+(event.key==='ArrowUp'?-1:1)))]?.focus()
  else if(event.key==='Home')elements[0]?.focus()
  else if(event.key==='End')elements.at(-1)?.focus()
  else select(node.data)
}
</script>
<template>
  <el-tree ref="outline" class="sample-hierarchy" :data="items" node-key="key" :current-node-key="selected" :indent="16" default-expand-all highlight-current :expand-on-click-node="false" :props="{label:'name',children:'children'}" aria-label="样本层级" @node-click="select" @keydown.capture="onKeydown">
    <template #default="{data}">
      <div class="sample-tree-content" :class="{'is-removed':data.node.type==='exclude'}">
        <el-icon aria-hidden="true"><Remove v-if="data.node.type==='exclude'"/><FolderOpened v-else-if="data.children.length"/><Document v-else/></el-icon>
        <div class="sample-tree-copy" :title="`${data.depth?data.name:'全部样本'} · ${sampleSource(data)}`">
          <span class="sample-map-name"><span v-if="data.number" class="sample-number">{{data.number}}</span>{{data.depth?data.name:'全部样本'}}</span>
          <span class="sample-map-status">{{!data.depth&&data.node.type==='dataset'?'选择划分方式':data.status}}<span v-if="data.children.length"> · {{data.children.length}} 个子分支</span></span>
        </div>
        <el-button v-if="data.node.removed" text size="small" :icon="RefreshLeft" :disabled="disabled" :aria-label="`恢复结果：${data.name}`" title="恢复结果及下级划分" @click.stop="emit('restore',data)" @keydown.stop/>
        <el-button v-else-if="data.depth&&data.node.type!=='exclude'" text size="small" :icon="Delete" :disabled="disabled" :aria-label="`移除结果：${data.name}`" title="移除结果及下级划分" @click.stop="emit('remove',data)" @keydown.stop/>
      </div>
    </template>
  </el-tree>
</template>
