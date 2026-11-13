<script>
// 每列有独立的浮层，关闭后接收焦点或鼠标的表头可能属于另一列。
const closingPanels = new WeakSet()
const coveredHeaders = new WeakMap()
</script>
<script setup>
import {nextTick,ref,watch} from 'vue'
import {useProfileDrag} from '../profile-drag'
import {profileTypes} from '../column-profile'
import ColumnProfileBody from './ColumnProfileBody.vue'
const props=defineProps({datasetId:{type:String,required:true},revision:{type:Number,required:true},column:{type:String,required:true},themeKey:String})
const visible=ref(false),pinned=ref(false),popover=ref(),dismissed=ref(false)
const panel = ref(null)
const {position, dragging, floatingStyle, startDrag, moveDrag, stopDrag, resetPosition, keepInViewport} =
  useProfileDrag(pinned,'.column-profile-popper')
let lastPointer
const popperOptions={modifiers:[{name:'preventOverflow',options:{altAxis:true,tether:false,padding:12}}]}
function updateVisible(value){if(value&&dismissed.value)return;if(value||(!pinned.value&&!dragging.value))visible.value=value}
function rememberPointer(event){lastPointer={x:event.clientX,y:event.clientY}}
function enterReference(event){
  const point = coveredHeaders.get(event.currentTarget)
  coveredHeaders.delete(event.currentTarget)
  dismissed.value = Boolean(point && Math.hypot(event.clientX-point.x,event.clientY-point.y)<2)
}
function openFromReference(){dismissed.value=false;visible.value=true}
function focusReference(event){
  // 组件会在关闭期间同步还原焦点，不能等下一次 DOM 更新再识别它。
  if(closingPanels.has(event.relatedTarget?.closest?.('.column-profile')))return
  openFromReference()
}
function close(event){
  if(!visible.value)return
  if(panel.value)closingPanels.add(panel.value)
  const point = event.type==='click' && event.detail>0 ? {x:event.clientX,y:event.clientY} : lastPointer
  if(point){
    const header = document.elementsFromPoint(point.x,point.y)
      .find(element=>element.matches('.column-profile-trigger'))
    if(header)coveredHeaders.set(header,point)
  }
  stopDrag()
  dismissed.value=true;pinned.value=false;visible.value=false
}
watch(visible,open=>{
  if(open){
    closingPanels.delete(panel.value)
    // 快速重新打开可能打断退场动画，此时也要从当前表头的位置开始。
    resetPosition()
  }else stopDrag()
})
async function updatePosition(){
  await nextTick()
  if(visible.value){keepInViewport();popover.value?.popperRef?.popperInstanceRef?.update()}
}
watch(visible,updatePosition,{flush:'post'})
</script>

<template>
  <el-popover ref="popover" :visible="visible" trigger="hover" :trigger-keys="[]" role="dialog" placement="bottom-start" :show-after="300" :hide-after="180" :width="560" :popper-options="popperOptions" :popper-class="['column-profile-popper',{'profile-moved':position,'profile-dragging':dragging}]" :popper-style="floatingStyle" @update:visible="updateVisible" @after-leave="resetPosition">
    <template #reference><button type="button" class="column-profile-trigger" :aria-label="`查看${column}的字段概况与分布`" @mouseenter="enterReference" @mouseleave="coveredHeaders.delete($event.currentTarget)" @pointermove="rememberPointer" @focus="focusReference" @keydown.esc.stop="close" @click.stop="openFromReference"><span class="column-profile-name">{{column}}</span><span aria-hidden="true">⌁</span></button></template>
    <section ref="panel" class="column-profile" :aria-label="`${column}的字段详情`" @pointermove="rememberPointer" @keydown.esc.stop="close">
      <ColumnProfileBody v-bind="props" :visible="visible" @resize="updatePosition">
        <template #header="{kind}">
          <header title="拖拽标题栏移动面板" @pointerdown="startDrag" @pointermove="moveDrag" @pointerup="stopDrag" @pointercancel="stopDrag" @lostpointercapture="stopDrag"><span class="profile-grip" aria-hidden="true">⠿</span><strong>{{column}}</strong><el-tag v-if="kind" size="small" effect="plain">{{profileTypes[kind]}}</el-tag><div class="profile-actions" title=""><el-checkbox v-model="pinned" size="small">固定面板</el-checkbox><el-button text size="small" aria-label="关闭字段详情" @click="close">关闭</el-button></div></header>
        </template>
      </ColumnProfileBody>
    </section>
  </el-popover>
</template>

<style>
.column-profile-trigger{border:0;background:transparent;color:inherit;font:inherit;font-weight:inherit;padding:0;width:100%;text-align:left;display:flex;align-items:center;gap:6px;min-height:24px}
.column-profile-name{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.column-profile-trigger>[aria-hidden]{flex:none;color:var(--muted);font-size:16px}
.column-profile-popper.el-popover{width:min(560px,calc(100vw - 24px))!important;max-height:82vh;overflow:auto;padding:12px 16px;color:var(--text);background:var(--surface);border-color:var(--border)}
.column-profile-popper.profile-moved{position:fixed!important;left:var(--profile-left)!important;top:var(--profile-top)!important;right:auto!important;bottom:auto!important;transform:none!important}.profile-moved>.el-popper__arrow{display:none}
.column-profile header{display:flex;align-items:center;flex-wrap:wrap;gap:8px;position:sticky;top:-12px;z-index:1;background:var(--surface);min-height:30px;padding:0 0 4px;cursor:grab;touch-action:none;user-select:none}.profile-dragging .column-profile header{cursor:grabbing}.profile-grip{font-size:18px;color:var(--muted);line-height:1}.column-profile header>strong{font-size:15px;overflow-wrap:anywhere;min-width:0}.profile-actions{--control-height:26px;margin-left:auto;display:flex;align-items:center;gap:8px;flex-shrink:0;cursor:default}.profile-actions .el-checkbox{height:26px}.column-profile .profile-actions .el-button{height:26px;min-height:0;padding:4px 6px}
</style>
