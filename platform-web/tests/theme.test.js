import test from 'node:test'
import assert from 'node:assert/strict'
import {normalizeTheme,presets,defaults,resolveTheme,contrast,applyTheme} from '../src/theme.js'
test('损坏的浏览器设置恢复为安全有效的主题',()=>{
  const t=normalizeTheme({scheme:'invalid',preset:99,primary:'url(evil)',layout:'other'})
  assert.equal(t.scheme,'light');assert.equal(t.layout,'left');assert.equal(t.primary,defaults.primary);assert.ok(presets[t.preset])
})
test('旧版偏好保留，非法布尔值和渐变参数不会进入主题',()=>{
  const t=normalizeTheme({preset:10,primary:'#873ff2',gradient:true,table:'fixed',weak:'false',collapsed:1,angle:361,threeColor:'yes',middle:'red',unknown:true})
  assert.equal(t.preset,10);assert.equal(t.table,'fixed');assert.equal(t.primary,'#873ff2')
  assert.equal(t.weak,false);assert.equal(t.collapsed,false);assert.equal(t.angle,135);assert.equal(t.threeColor,false);assert.equal(t.middle,defaults.middle);assert.equal(t.unknown,undefined)
  assert.deepEqual(normalizeTheme(null),defaults)
})
test('全部预设及极端自定义渐变在深浅界面上保持可读文字',()=>{
  const variants=[...presets.map((_,preset)=>({preset})),{custom:true,primary:'#ffffff',gradient:false},{custom:true,primary:'#000000',gradient:false},{custom:true,primary:'#ffffff',secondary:'#000000',threeColor:true,middle:'#ffff00'},{custom:true,primary:'#ff0000',secondary:'#00ffff',gradientType:'radial'}]
  for(const scheme of ['light','dark'])for(const variant of variants){
    const state=resolveTheme({...variant,scheme})
    assert.ok(contrast(state.accentText,state.surface)>=4.5,`${scheme} ${JSON.stringify(variant)} 链接文字`)
    for(const color of state.brandSamples)assert.ok(contrast(state.onBrand,color)>=4.5,`${scheme} ${JSON.stringify(variant)} 按钮文字`)
  }
})
test('系统模式更新所有组件的主题令牌并通知已挂载图表',()=>{
  const styles=new Map(),classes=new Set(),events=[],root={dataset:{},style:{setProperty:(k,v)=>styles.set(k,v)},classList:{toggle:(name,on)=>on?classes.add(name):classes.delete(name)}}
  let dark=true
  const win={document:{documentElement:root},matchMedia:()=>({matches:dark}),CustomEvent:class{constructor(type){this.type=type}},dispatchEvent:event=>events.push(event.type)}
  applyTheme({scheme:'auto',sidebar:'light',table:'fixed',weak:true,fixedSidebar:false},win)
  assert.equal(root.dataset.theme,'dark');assert.ok(classes.has('dark'));assert.equal(root.dataset.table,'fixed');assert.equal(root.dataset.weak,'true');assert.equal(root.dataset.fixedSidebar,'false')
  const previous=styles.get('--chart-accent')
  dark=false;applyTheme({scheme:'auto'},win)
  assert.equal(root.dataset.theme,'light');assert.ok(!classes.has('dark'));assert.notEqual(styles.get('--chart-accent'),previous)
  assert.equal(styles.get('--el-color-primary'),styles.get('--accent-text'));assert.deepEqual(events,['hscredit-theme-change','hscredit-theme-change'])
})
test('三色渐变保留颜色顺序和方向，径向渐变不使用角度',()=>{
  const value={custom:true,threeColor:true,primary:'#112233',middle:'#445566',secondary:'#778899',angle:90}
  assert.equal(resolveTheme(value).rawBackground,'linear-gradient(90deg,#112233,#445566,#778899)')
  assert.equal(resolveTheme({...value,gradientType:'radial'}).rawBackground,'radial-gradient(circle,#112233,#445566,#778899)')
})
test('保留深色顶部导航与自定义颜色',()=>{
  const t=normalizeTheme({scheme:'dark',layout:'top',primary:'#123456',custom:true})
  assert.equal(t.scheme,'dark');assert.equal(t.layout,'top');assert.equal(t.primary,'#123456');assert.equal(t.custom,true)
})

test('旧版定宽与折叠菜单迁移到铺满布局，并保留展开宽度',()=>{
  const legacy=normalizeTheme({width:'fixed',collapsed:true})
  assert.equal(legacy.width,'fluid');assert.equal(legacy.sidebarWidth,64);assert.equal(legacy.sidebarExpandedWidth,220)
  const expanded=normalizeTheme({sidebarWidth:290,sidebarExpandedWidth:290})
  const collapsed=normalizeTheme({...expanded,sidebarWidth:64})
  assert.equal(collapsed.collapsed,true);assert.equal(collapsed.sidebarExpandedWidth,290)
  const restored=normalizeTheme({...collapsed,sidebarWidth:collapsed.sidebarExpandedWidth})
  assert.equal(restored.sidebarWidth,290);assert.equal(restored.collapsed,false)
})

test('浏览器侧栏尺寸限制在可用范围，损坏的尺寸不会撑破工作区',()=>{
  for(const invalid of [NaN,Infinity,'999px',{},false])assert.equal(normalizeTheme({sidebarWidth:invalid}).sidebarWidth,220)
  assert.equal(normalizeTheme({sidebarWidth:-20}).sidebarWidth,64)
  assert.equal(normalizeTheme({sidebarWidth:99999}).sidebarWidth,320)
  assert.equal(normalizeTheme({sidebarWidth:167}).collapsed,true)
  assert.equal(normalizeTheme({sidebarWidth:168}).collapsed,false)
  assert.equal(normalizeTheme({sidebarExpandedWidth:64}).sidebarExpandedWidth,168)
})
