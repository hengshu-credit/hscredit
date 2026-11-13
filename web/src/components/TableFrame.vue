<script setup>
import {computed, ref} from 'vue'

const props = defineProps({height:[String,Number], maxHeight:[String,Number]})
const viewport = ref(null)
const chosen = defineModel({type:Number, default:null})
const minimum = 72
const tableSize = computed(() => ({
  height: chosen.value ?? props.height ?? '',
  maxHeight: chosen.value === null ? props.maxHeight ?? '' : '',
}))
const viewportStyle = computed(() => chosen.value === null ? {} : {height:chosen.value+'px', maxHeight:'none'})
let drag = null
function start(event) {
  if (event.button !== 0 || !event.isPrimary) return
  drag = {id:event.pointerId, y:event.clientY, height:viewport.value.offsetHeight}
  event.currentTarget.setPointerCapture(event.pointerId)
  event.currentTarget.focus({preventScroll:true})
  event.preventDefault()
}
function move(event) {
  if (drag?.id === event.pointerId) chosen.value = Math.max(minimum, drag.height + event.clientY - drag.y)
}
function adjust(event) {
  if (event.key === 'Home') chosen.value = null
  else if (event.key === 'ArrowUp' || event.key === 'ArrowDown') {
    chosen.value = Math.max(minimum, viewport.value.offsetHeight + (event.key === 'ArrowUp' ? -24 : 24))
  } else return
  event.preventDefault()
}
</script>

<template>
  <div class="table-frame">
    <div ref="viewport" class="table-frame-content"><slot :table-size="tableSize" :viewport-style="viewportStyle"/></div>
    <button type="button" class="table-height-handle" aria-label="调整表格高度" aria-keyshortcuts="ArrowUp ArrowDown Home"
      title="上下拖动或按上下方向键调整表格高度，双击或按 Home 恢复默认高度"
      @pointerdown="start" @pointermove="move" @lostpointercapture="drag=null"
      @keydown="adjust" @dblclick="chosen=null"/>
  </div>
</template>

<style scoped>
.table-frame,.table-frame-content{min-width:0}
.table-height-handle{display:grid;place-items:center;width:100%;height:14px;padding:0;touch-action:none;cursor:ns-resize;border:0;background:var(--soft);color:var(--muted)}
.table-height-handle::after{content:'';width:32px;height:4px;border-block:1px solid currentColor}
.table-height-handle:is(:hover,:focus-visible,:active){color:var(--accent-text);background:var(--accent-soft)}
</style>
