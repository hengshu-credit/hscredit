import {onScopeDispose, watch} from 'vue'

// 只保存调用方明确选取的表单字段，不缓存报告、请求状态或凭据。
const prefix = 'hscredit-form-v1:'
export function datasetDraftKey(dataset, section) {
  return JSON.stringify([dataset.id, dataset.revision || 1, section])
}

export function readDraft(key) {
  try { return JSON.parse(localStorage.getItem(prefix + key)) } catch { return null }
}

export function writeDraft(key, value) {
  try { localStorage.setItem(prefix + key, JSON.stringify(value)) } catch {
    // 存储不可用时仍允许编辑与提交。
  }
}

export function rememberForm(key, snapshot, restore) {
  const saved = readDraft(key)
  if (saved) restore(saved)
  const save = () => writeDraft(key, snapshot())
  watch(snapshot, save, {deep:true})
  onScopeDispose(save)
}
