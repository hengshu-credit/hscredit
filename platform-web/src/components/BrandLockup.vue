<script setup>
import {BRAND} from '../brand'
import logo from '../../../docs/_static/hengshucredit_logo.svg?raw'
</script>

<template>
  <div class="brand" :aria-label="`${BRAND.title} · ${BRAND.product}`" :title="`${BRAND.title} · ${BRAND.subtitle}`">
    <!-- 固定的仓库品牌资源随 JS 内嵌，避免旧 SVG HTTP 响应缓存导致空白。 -->
    <span class="brand-logo" role="img" :aria-label="`${BRAND.title} Logo`" v-html="logo"/>
    <div class="brand-word"><strong>{{BRAND.title}}</strong><span>{{BRAND.subtitle}}</span></div>
  </div>
</template>

<style scoped>
.brand-logo :deep(svg) { display:block; width:100%; height:100%; }
</style>
