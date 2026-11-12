import {defineConfig} from 'vite'
import vue from '@vitejs/plugin-vue'
import {BRAND} from './src/brand.js'
export default defineConfig({plugins:[vue(),{name:'platform-brand',transformIndexHtml:html=>html.replace('__PLATFORM_TITLE__',`${BRAND.title} · ${BRAND.product}`)}], build:{outDir:'../hscredit/resources/platform', emptyOutDir:true, chunkSizeWarningLimit:1600, rollupOptions:{output:{manualChunks:{charts:['echarts'], ui:['element-plus','@element-plus/icons-vue']}}}}, server:{proxy:{'/api':'http://127.0.0.1:8765','/services':'http://127.0.0.1:8765'}}})
