import test from 'node:test'
import assert from 'node:assert/strict'
import {nextTick, ref, effectScope} from 'vue'
import {useFieldPickerPages} from '../src/field-picker-pages.js'

test('滚动逐页接续且保留行序，页码跟随可见位置，手动跳页与筛选重新定位', async () => {
  const all = Array.from({length:43}, (_, i) => '字段'+(i+1))
  const fields = ref(all), scope = effectScope()
  const pages = scope.run(() => useFieldPickerPages(fields))
  try {
    pages.resize(10)
    pages.list.value = {
      clientHeight:180, scrollTop:0,
      get scrollHeight() { return (pages.rows.value.length + 1) * 30 },
      querySelector:() => ({getBoundingClientRect:() => ({height:30})})
    }
    const list = pages.list.value
    list.scrollTop = 150
    pages.scroll()
    assert.deepEqual(pages.rows.value, all.slice(0,20))
    assert.equal(list.scrollTop,150)
    list.scrollTop = 320
    pages.scroll()
    assert.equal(pages.current.value,2)
    list.scrollTop = 20
    pages.scroll()
    assert.equal(pages.current.value,1)

    pages.jump(4)
    assert.equal(list.scrollTop,0)
    assert.deepEqual(pages.rows.value,all.slice(30,40))
    list.scrollTop = 150
    pages.scroll()
    list.scrollTop = list.scrollHeight-list.clientHeight
    pages.scroll()
    assert.equal(pages.current.value,5)
    assert.deepEqual(pages.rows.value,all.slice(30))

    fields.value = all.slice(0,3)
    await nextTick()
    assert.deepEqual(pages.rows.value,fields.value)
    assert.equal(pages.current.value,1)
    assert.equal(list.scrollTop,0)

    fields.value = all
    pages.resize(20)
    list.clientHeight = list.scrollHeight
    pages.wheel({deltaY:120})
    assert.deepEqual(pages.rows.value,all.slice(0,40))
  } finally {
    scope.stop()
  }
})

test('浏览大量字段时只挂载附近批次，已读范围用占位保留，仍可回滚',()=>{
  const scope=effectScope()
  try{
    const fields=ref(Array.from({length:5000},(_,i)=>i))
    const pages=scope.run(()=>useFieldPickerPages(fields))
    pages.resize(20)
    pages.list.value={clientHeight:400,scrollTop:0,
      get scrollHeight(){return pages.before.value+pages.after.value+(pages.rows.value.length+1)*30},
      querySelector(){return {getBoundingClientRect:()=>({height:30})}}}
    for(let i=0;i<80;i++){
      pages.list.value.scrollTop=pages.list.value.scrollHeight-400
      pages.scroll()
      assert.ok(pages.rows.value.length<=60)
    }
    assert.ok(pages.before.value>0)
    pages.list.value.scrollTop=0;pages.scroll()
    assert.equal(pages.rows.value[0],0)
    assert.ok(pages.after.value>0)
  }finally{scope.stop()}
})

