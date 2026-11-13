<script setup>
import FieldSelect from "./FieldSelect.vue"
defineProps({form:Object,columns:Array,dataset:Object})
</script>
<template>
      <div class="split-fields">
        <el-form-item v-if="form.method==='random'" label="分层字段（可选）"><FieldSelect v-model="form.stratify" :fields="columns" clearable label="分层字段" placeholder="不分层"/></el-form-item>
        <el-form-item v-if="form.method==='group'" label="分组字段" required><FieldSelect v-model="form.group_col" :fields="columns" label="分组字段" placeholder="例如客户编号"/></el-form-item>
        <template v-if="['time','oot'].includes(form.method)">
          <el-form-item label="时间字段" required><FieldSelect v-model="form.time_col" :fields="columns.filter(c=>c.kind!=='numeric')" label="时间字段"/></el-form-item>
        </template>
        <el-form-item v-if="form.method==='time'" label="时间划分依据"><el-radio-group v-model="form.time_mode"><el-radio value="ratio">样本比例</el-radio><el-radio value="cutoff">指定时间边界</el-radio></el-radio-group></el-form-item>
        <template v-if="form.method==='oot'">
          <el-form-item label="时间外起始时间（包含）" required><el-date-picker v-model="form.oot_start" type="datetime" value-format="YYYY-MM-DDTHH:mm:ss" placeholder="选择起始时间" aria-label="时间外起始时间"/></el-form-item>
          <el-form-item label="时间外结束时间（包含，可选）"><el-date-picker v-model="form.oot_end" type="datetime" value-format="YYYY-MM-DDTHH:mm:ss" placeholder="留空使用全部后续样本" aria-label="时间外结束时间"/></el-form-item>
        </template>
      </div>
      <details class="split-advanced"><summary>高级设置 · {{['random','group'].includes(form.method)?`随机种子 ${form.random_state}`:`边界间隔 ${form.gap} 行`}}</summary><div class="split-fields">
        <el-form-item v-if="['random','group'].includes(form.method)" label="随机种子"><el-input-number :controls="false" v-model="form.random_state" :min="0" :max="2147483647" :precision="0" aria-label="划分随机种子"/></el-form-item>
        <el-form-item v-else label="每个边界前排除的间隔行数"><el-input-number :controls="false" v-model="form.gap" :min="0" :max="dataset?.rows||0" :precision="0" aria-label="边界间隔行数"/></el-form-item>
      </div></details>
      <p v-if="form.method==='random'" class="muted">固定种子可复现结果；分层时尽量保持各类别的比例，分层字段不能包含缺失值。</p>
      <p v-if="form.method==='group'" class="muted">比例按不同分组的数量分配，实际行数占比可能不同；同一分组只会进入一个结果数据集。</p>
      <p v-if="form.method==='time'" class="muted">结果按时间从早到晚排列。比例按扣除间隔后的样本数分配；指定边界时，边界时刻归入后一份。按比例划分可能将相同时间的行分到相邻数据集。</p>
      <p v-if="form.method==='oot'" class="muted">第一份为起始时间之前的开发集，第二份为时间外样本；间隔行和结束时间之后的样本会排除。</p>
</template>

<style scoped>
.split-fields{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:0 20px}.split-fields .el-select,.split-fields :deep(.el-date-editor){width:100%}
</style>
