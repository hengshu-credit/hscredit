// 每份选中样本都有自己的任务、数据版本和结果文件。
export async function submitDatasetRuns(api, config, datasets) {
  const bindings = datasets.map(d => ({id: d.id, revision: d.revision ?? 1}))
  if (!bindings.length) throw new Error('请选择运行任务的数据集')
  const first = bindings[0]
  const body = {...config, dataset_id: first.id, dataset_revision: first.revision}
  const post = payload => ({method: 'POST', body: JSON.stringify(payload)})
  if (bindings.length === 1) return [await api('/jobs', post(body))]
  const result = await api('/jobs/batch', post({config: body, datasets: bindings}))
  return result.jobs
}
