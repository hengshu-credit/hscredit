import test from 'node:test'
import assert from 'node:assert/strict'
import {effectScope, nextTick, ref} from 'vue'
import {datasetDraftKey,rememberForm} from '../src/form-drafts.js'

test('同一数据版本重新进入时恢复子页与未完成的分裂草稿，不同数据或版本各自独立', async () => {
  const values = new Map()
  const original = Object.getOwnPropertyDescriptor(globalThis,'localStorage')
  Object.defineProperty(globalThis,'localStorage',{configurable:true,value:{
    getItem:key=>values.get(key) ?? null, setItem:(key,value)=>values.set(key,value),
  }})
  function open(id,revision) {
    const scope=effectScope(), form=ref({mode:'single',rows:[]})
    scope.run(()=>rememberForm(datasetDraftKey({id,revision},'strategy'),()=>form.value,saved=>{form.value=saved}))
    return {scope,form}
  }
  try {
    const first=open('训练',1)
    first.form.value={mode:'multiway',rows:[{node:0,values:'30, '}]}
    await nextTick()
    first.scope.stop()
    for(const [id,revision] of [['训练',2],['另一份训练',1]]) {
      const other=open(id,revision)
      assert.deepEqual(other.form.value,{mode:'single',rows:[]})
      other.scope.stop()
    }
    const restored=open('训练',1)
    assert.deepEqual(restored.form.value,first.form.value)
    restored.scope.stop()
  } finally {
    if(original) Object.defineProperty(globalThis,'localStorage',original)
    else delete globalThis.localStorage
  }
})
