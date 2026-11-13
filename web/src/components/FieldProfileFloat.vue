<script setup>
import {Close} from '@element-plus/icons-vue'
import {watch} from 'vue'
import {useProfileDrag} from '../profile-drag'
import ColumnProfileBody from './ColumnProfileBody.vue'

defineProps({field: Object, dataset: Object, left: Boolean, above: Boolean})
const pinned = defineModel('pinned',{type:Boolean})
const emit = defineEmits(['close','enter','leave'])
const {position, dragging, floatingStyle, startDrag, moveDrag, stopDrag, resetPosition} = useProfileDrag(pinned,'.picker-float')
watch(pinned, value => { if (!value) resetPosition() })
</script>
<template>
  <aside class="picker-float column-profile" :class="{'on-left':left,'above-row':above,'is-moved':position,'is-dragging':dragging}" :style="floatingStyle"
    :aria-label="field.name+'的字段详情'" @pointerenter="emit('enter')" @pointerleave="emit('leave')"
    @focusin="emit('enter')" @keydown.esc.stop.prevent="emit('close')">
    <div class="picker-float-head" title="拖动标题栏移动面板" @pointerdown="startDrag" @pointermove="moveDrag"
      @pointerup="stopDrag" @pointercancel="stopDrag" @lostpointercapture="stopDrag">
      <span class="profile-grip" aria-hidden="true">⠿</span>
      <strong :title="field.name">{{field.name}}</strong>
      <div class="profile-actions" title=""><el-checkbox v-model="pinned" size="small">固定面板</el-checkbox>
        <el-button text :icon="Close" aria-label="关闭字段详情" @click="emit('close')"/></div>
    </div>
    <div :key="field.name" class="picker-float-content">
      <p v-if="field.derived" class="profile-note">前序步骤结果，保存加工后可查看字段概况。</p>
      <ColumnProfileBody v-else-if="dataset" :dataset-id="dataset.id" :revision="dataset.revision||1" :column="field.name" visible/>
    </div>
  </aside>
</template>
<style>
/* 浮层留在对话框的焦点范围内，独立滚动，优先使用对话框外侧的空间。 */
.picker-float{
  --float-width:min(390px,calc(50vw - 12px));
  --float-offset:min(calc(100% + 6px),calc(50% + 50vw - var(--float-width) - 12px));
  position:absolute;z-index:6;left:var(--float-offset);top:0;width:var(--float-width);max-height:calc(var(--picker-height) - 42px);
  display:flex;flex-direction:column;background:var(--surface);color:var(--text);border:1px solid var(--border);
  border-radius:var(--radius);box-shadow:0 8px 32px var(--shadow)
}
.picker-float.on-left{left:auto;right:var(--float-offset)}
.picker-float-head{flex:none;display:flex;align-items:center;gap:8px;padding:6px 10px;border-bottom:1px solid var(--line-soft);cursor:grab;touch-action:none;user-select:none}
.picker-float.is-dragging .picker-float-head{cursor:grabbing}
.picker-float-head strong{flex:1;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.picker-float-head .el-button{padding:4px;height:26px}
.picker-float-content{min-height:0;overflow:auto;overscroll-behavior:contain;padding:10px 14px}
@media(max-width:640px){
  .picker-float,.picker-float.on-left{left:auto;right:6px;top:auto;bottom:4px;width:min(390px,calc(96vw - 12px));max-height:calc(var(--picker-height) * .46)}
  .picker-float.above-row{top:4px;bottom:auto}
}
.picker-float.is-moved{position:fixed;left:var(--profile-left);top:var(--profile-top);right:auto;bottom:auto}
</style>
