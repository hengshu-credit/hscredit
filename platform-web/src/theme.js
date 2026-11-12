// 风控平台主题契约：沿用观星配色、对比度与直角组件规范，兼容已有浏览器设置。
import {SIDEBAR_MIN_WIDTH,SIDEBAR_DEFAULT_WIDTH,SIDEBAR_COMPACT_THRESHOLD,clampSidebarWidth} from './layout.js'
export const presets = [
  ['主题蓝', '#2639e9', '#2639e9'], ['凝液紫', '#873ff2', '#873ff2'],
  ['桃粉红', '#ff6b9d', '#ff6b9d'], ['霓虹粉', '#e14eca', '#e14eca'],
  ['深夜紫', '#19103b', '#19103b'], ['靛青紫', '#5b3df5', '#5b3df5'],
  ['星云蓝', '#536dfe', '#536dfe'], ['晶石蓝', '#2f80ed', '#2f80ed'],
  ['主题蓝渐变', '#2639e9', '#536dfe'], ['凝液紫渐变', '#19103b', '#873ff2'],
  ['桃粉渐变', '#873ff2', '#e14eca'], ['星云渐变', '#536dfe', '#e14eca'],
]
export const THEME_KEY = 'hscredit-theme'
export const defaults = Object.freeze({scheme:'light', preset:0, custom:false, primary:'#2639e9', secondary:'#873ff2', middle:'#e14eca', threeColor:false, gradient:true, gradientType:'linear', angle:135, layout:'left', sidebar:'dark', width:'fluid', table:'auto', fixedSidebar:true, weak:false, collapsed:false, sidebarWidth:SIDEBAR_DEFAULT_WIDTH, sidebarExpandedWidth:SIDEBAR_DEFAULT_WIDTH})
export function normalizeTheme(value) {
  const input = value && typeof value === 'object' ? value : {}
  const t = Object.fromEntries(Object.keys(defaults).map(key => [key, input[key] ?? defaults[key]]))
  for (const [key, options] of Object.entries({scheme:['light','dark','auto'], layout:['left','top'], sidebar:['dark','light'], width:['fixed','fluid'], table:['fixed','auto'], gradientType:['linear','radial']})) {
    if (!options.includes(t[key])) t[key] = defaults[key]
  }
  if (!Number.isInteger(t.preset) || !presets[t.preset]) t.preset = defaults.preset
  for (const key of ['primary','secondary','middle']) if (typeof t[key] !== 'string' || !/^#[\da-f]{6}$/i.test(t[key])) t[key] = defaults[key]
  for (const key of ['custom','threeColor','gradient','fixedSidebar','weak','collapsed']) if (typeof t[key] !== 'boolean') t[key] = defaults[key]
  // 兼容旧版定宽/折叠设置，主内容不再保留居中宽度上限。
  t.width='fluid'
  t.sidebarWidth=clampSidebarWidth(input.sidebarWidth??(t.collapsed?SIDEBAR_MIN_WIDTH:SIDEBAR_DEFAULT_WIDTH))
  t.collapsed=t.sidebarWidth<SIDEBAR_COMPACT_THRESHOLD
  t.sidebarExpandedWidth=Math.max(SIDEBAR_COMPACT_THRESHOLD,clampSidebarWidth(input.sidebarExpandedWidth??(t.collapsed?SIDEBAR_DEFAULT_WIDTH:t.sidebarWidth)))
  if (!Number.isInteger(t.angle) || t.angle < 0 || t.angle > 360) t.angle = defaults.angle
  return t
}
const rgb = color => [1,3,5].map(i => parseInt(color.slice(i,i+2),16))
export const mix = (a,b,weight) => '#' + rgb(a).map((v,i) => Math.round(v*(1-weight)+rgb(b)[i]*weight).toString(16).padStart(2,'0')).join('')
const luminance = color => rgb(color).map(v => v/255).map(v => v<=0.04045?v/12.92:((v+0.055)/1.055)**2.4).reduce((sum,v,i) => sum+v*[0.2126,0.7152,0.0722][i],0)
export const contrast = (a,b) => (Math.max(luminance(a),luminance(b))+0.05)/(Math.min(luminance(a),luminance(b))+0.05)
export function readableAccent(color, surface) {
  const destination = luminance(surface)>0.4?'#111827':'#ffffff'
  while (contrast(color,surface)<4.5) color = mix(color,destination,0.12)
  return color
}
export function resolveTheme(value, systemDark=false) {
  const config=normalizeTheme(value), dark=config.scheme==='dark'||(config.scheme==='auto'&&systemDark)
  const [,a,b]=presets[config.preset]
  const colors=config.custom?[config.primary,...(config.gradient?(config.threeColor?[config.middle,config.secondary]:[config.secondary]):[])]:config.gradient&&a!==b?[a,b]:[a]
  const surface=dark?'#151d31':'#ffffff'
  const rawBackground=colors.length===1?colors[0]:config.gradientType==='radial'?`radial-gradient(circle,${colors.join(',')})`:`linear-gradient(${config.angle}deg,${colors.join(',')})`
  const samples=colors.flatMap((color,i)=>i===colors.length-1?[color]:Array.from({length:21},(_,j)=>mix(color,colors[i+1],j/20)))
  const white=Math.min(...samples.map(c=>contrast(c,'#ffffff'))), black=Math.min(...samples.map(c=>contrast(c,'#111827')))
  const needsScrim=Math.max(white,black)<4.5
  return {config,dark,surface,accent:colors[0],accentText:readableAccent(colors[0],surface),secondary:readableAccent(colors.at(-1),surface),rawBackground,
    background:needsScrim?`linear-gradient(#00000099,#00000099), ${rawBackground}`:rawBackground,
    onBrand:needsScrim||white>=black?'#ffffff':'#111827',
    brandSamples:needsScrim?samples.map(c=>mix(c,'#000000',0.6)):samples}
}
export function applyTheme(value, win=window) {
  const state=resolveTheme(value,win.matchMedia('(prefers-color-scheme: dark)').matches), t=state.config, root=win.document.documentElement
  root.classList.toggle('dark',state.dark)
  Object.assign(root.dataset,{theme:state.dark?'dark':'light',sidebar:t.sidebar,table:t.table,weak:String(t.weak),fixedSidebar:String(t.fixedSidebar)})
  const onPrimary=contrast(state.accentText,'#ffffff')>=contrast(state.accentText,'#111827')?'#ffffff':'#111827'
  const variables={'--el-color-primary':state.accentText,'--on-primary':onPrimary,'--accent':state.accent,'--accent-text':state.accentText,'--brand':state.background,'--on-brand':state.onBrand,'--chart-accent':state.accentText,'--chart-secondary':state.secondary}
  for(let i=1;i<=9;i++) variables[`--el-color-primary-light-${i}`]=mix(state.accentText,state.surface,i/10)
  variables['--el-color-primary-dark-2']=mix(state.accentText,state.dark?'#ffffff':'#111827',0.2)
  for(const [key,color] of Object.entries(variables)) root.style.setProperty(key,color)
  win.dispatchEvent(new win.CustomEvent('hscredit-theme-change'))
  return state
}
