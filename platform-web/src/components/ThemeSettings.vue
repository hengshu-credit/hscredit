<script setup>
import {computed} from 'vue'
import {defaults,presets,resolveTheme} from '../theme'
const props=defineProps({modelValue:Boolean,theme:Object})
const emit=defineEmits(['update:modelValue','preview','save','cancel'])
const set=(key,value)=>emit('preview',{[key]:value})
const preview=computed(()=>resolveTheme(props.theme))
const choices=[
  ['scheme','界面模式',[['light','浅色'],['dark','深色'],['auto','跟随系统']]],
  ['layout','导航位置',[['left','左侧导航'],['top','顶部导航']]],
  ['sidebar','导航配色',[['dark','深色'],['light','浅色']]],
  ['table','表格滚动',[['auto','随页面滚动'],['fixed','表格内滚动']]],
]
</script>
<template>
  <el-drawer :model-value="modelValue" title="主题配置" size="min(440px, 100vw)" class="theme-drawer" @update:model-value="emit('update:modelValue',$event)" @close="emit('cancel')">
    <p class="muted theme-intro">点选即可预览；保存后保留设置，取消可恢复原主题。</p>
    <section class="theme-section">
      <h3>显示与配色</h3>
      <div class="theme-setting"><label>界面模式</label><div class="theme-choices" role="group" aria-label="界面模式"><button v-for="[value,label] in choices[0][2]" :key="value" :aria-pressed="theme.scheme===value" @click="set('scheme',value)">{{label}}</button></div></div>
      <div class="theme-setting"><label>配色方式</label><div class="theme-choices" role="group" aria-label="配色方式"><button :aria-pressed="!theme.custom" @click="set('custom',false)">主题预设</button><button :aria-pressed="theme.custom&&!theme.gradient" @click="emit('preview',{custom:true,gradient:false})">自选纯色</button><button :aria-pressed="theme.custom&&theme.gradient" @click="emit('preview',{custom:true,gradient:true})">自选渐变</button></div></div>
      <div v-if="!theme.custom" class="theme-swatches" role="group" aria-label="主题预设"><button v-for="(p,i) in presets" :key="p[0]" :aria-pressed="theme.preset===i" @click="emit('preview',{preset:i,gradient:true})"><span :style="{background:`linear-gradient(135deg,${p[1]},${p[2]})`}"/><small>{{p[0]}}</small><b v-if="theme.preset===i" aria-hidden="true">✓</b></button></div>
      <template v-else>
        <div class="theme-color-fields"><label>主题色<input type="color" :value="theme.primary" aria-label="主题色" @input="set('primary',$event.target.value)"></label><label v-if="theme.gradient&&theme.threeColor">中间色<input type="color" :value="theme.middle" aria-label="中间色" @input="set('middle',$event.target.value)"></label><label v-if="theme.gradient">结束色<input type="color" :value="theme.secondary" aria-label="结束色" @input="set('secondary',$event.target.value)"></label></div>
        <template v-if="theme.gradient"><div class="theme-gradient-preview" :style="{background:preview.rawBackground}" aria-label="渐变效果预览"/><div class="theme-setting"><el-checkbox class="option-toggle" :model-value="theme.threeColor" @update:model-value="set('threeColor',$event)">使用三色渐变</el-checkbox></div><div class="theme-choices" role="group" aria-label="渐变类型"><button :aria-pressed="theme.gradientType==='linear'" @click="set('gradientType','linear')">线性渐变</button><button :aria-pressed="theme.gradientType==='radial'" @click="set('gradientType','radial')">径向渐变</button></div><div v-if="theme.gradientType==='linear'" class="theme-setting"><label for="theme-angle">渐变方向 · {{theme.angle}}°</label><input id="theme-angle" type="range" min="0" max="360" :value="theme.angle" @input="set('angle',Number($event.target.value))"><div class="theme-choices"><button v-for="[angle,label] in [[90,'水平'],[0,'垂直'],[135,'对角']]" :key="angle" :aria-pressed="theme.angle===angle" @click="set('angle',angle)">{{label}}</button></div></div></template>
      </template>
    </section>
    <section class="theme-section"><h3>页面布局</h3><div v-for="[key,label,options] in choices.slice(1)" :key="key" class="theme-setting"><label>{{label}}</label><div class="theme-choices" role="group" :aria-label="label"><button v-for="[value,name] in options" :key="value" :aria-pressed="theme[key]===value" @click="set(key,value)">{{name}}</button></div></div></section>
    <section class="theme-section"><h3>辅助选项</h3><div class="theme-toggle"><span>固定导航<small>滚动页面时保持菜单可见</small></span><el-checkbox class="option-toggle" :model-value="theme.fixedSidebar" aria-label="固定导航" @update:model-value="set('fixedSidebar',$event)">启用</el-checkbox></div><div class="theme-toggle"><span>色弱辅助<small>以灰度显示界面</small></span><el-checkbox class="option-toggle" :model-value="theme.weak" aria-label="色弱辅助" @update:model-value="set('weak',$event)">启用</el-checkbox></div></section>
    <template #footer><div class="theme-footer"><el-button @click="emit('preview',{...defaults})">恢复默认</el-button><el-button @click="emit('cancel')">取消</el-button><el-button type="primary" @click="emit('save')">保存主题</el-button></div></template>
  </el-drawer>
</template>
