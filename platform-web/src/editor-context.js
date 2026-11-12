// 用模块当前选择的模型、服务和字段生成编辑器的补全上下文。
import {modelParameterSpecs} from './model-options.js'

export function modelParametersSchema(selected, catalog, search = true) {
  const models = selected.filter(id => typeof id === 'string' && Object.hasOwn(catalog, id)).map(id => {
    const model = catalog[id]
    const params = modelParameterSpecs(model).map(({key: name, default: value, label, description}) => {
      const candidates = model.search[name] ?? [value]
      const item = model.choices[name] ? {enum: model.choices[name]} : {type: typeof value, examples: candidates}
      const schema = search ? {type: 'array', default: candidates, items: item} : {...item, default: value}
      return [name, {...schema, description: label + ' · ' + description}]
    })
    for(const [key,spec] of Object.entries(model.extra_params||{})){
      const item={description:spec.description,...(Object.hasOwn(spec,'default')?{default:spec.default}:{})}
      params.push([key,search?{type:'array',items:item,description:spec.description}:item])
    }
    return [id, {type: 'object', description: model.name, properties: Object.fromEntries(params), additionalProperties: false}]
  })
  return {type: 'object', properties: Object.fromEntries(models), additionalProperties: false}
}

export const rulesSchema = {
  type: 'array', items: {type: 'object', properties: {
    name: {type: 'string', description: '规则名称'},
    expression: {type: 'string', description: '使用字段比较和布尔运算编写规则'},
  }},
}

export function categoryGroupsSchema(field) {
  const examples = (field?.values || []).map(item => item.value).filter(value => value != null)
  return {type: 'array', items: {type: 'array', items: {examples}}}
}

export function requestSchema(features) {
  const properties = Object.fromEntries(features.map(name => [name, {description: '服务输入字段', default: 0}]))
  return {type: 'object', properties: {records: {type: 'array', items: {type: 'object', properties}}}}
}

export function taskConfigSchema(config, columns, datasets, catalog, models) {
  const field = {examples: columns.map(c => c.name)}
  const properties = Object.fromEntries(Object.entries(config).map(([key, value]) => [key, {default: value}]))
  const hints = {
    dataset_id: {examples: datasets.map(d => d.id)},
    features: {type: 'array', items: field}, targets: {type: 'array', items: field},
    feature: field, other: field, group: field, date: field, amount: field, target: field,
  }
  if (config.kind === 'model') Object.assign(hints, {
    models: {type: 'array', items: {examples: Object.keys(catalog).filter(key => catalog[key].available)}},
    search_spaces: modelParametersSchema(models, catalog), model_params: modelParametersSchema(models, catalog, false),
  })
  if (config.kind === 'ruleset') hints.selected_rules = rulesSchema
  return {type: 'object', properties: {...properties, ...hints}}
}
