<script setup>
import wechatQr from '../../../docs/assets/readme/wechat.svg'
import officialAccountQr from '../../../docs/assets/readme/wechat-official-account.svg'
const year=new Date().getFullYear()
const contacts=[
  {label:'微信：itlubber',image:wechatQr,alt:'微信 itlubber 二维码'},
  {label:'公众号：hscreditlab',image:officialAccountQr,alt:'衡枢风控实验室（hscreditlab）微信公众号二维码'},
]
</script>
<template>
  <footer class="page-footer" role="contentinfo" aria-label="版权与联系方式">
    <span class="footer-copyright">© {{year}} <a href="https://hengshucredit.com/" target="_blank" rel="noopener noreferrer">上海衡枢真信科技有限公司</a> · ALL RIGHTS RESERVED</span>
    <span>开源地址：<a href="https://github.com/hengshu-credit" target="_blank" rel="noopener noreferrer">github.com/hengshu-credit</a></span>
    <a href="https://github.com/itlubber" target="_blank" rel="noopener noreferrer">github.com/itlubber</a>
    <a v-for="contact in contacts" :key="contact.label" class="footer-contact" :href="contact.image" target="_blank" rel="noopener noreferrer" :aria-label="`${contact.label}，查看二维码`">
      {{contact.label}}<img :src="contact.image" :alt="contact.alt" width="160" height="160" loading="lazy">
    </a>
  </footer>
</template>
<style scoped>
.page-footer{display:flex;flex-wrap:wrap;align-items:baseline;justify-content:center;gap:6px 16px;margin-top:auto;padding:14px 0 2px;border-top:1px solid var(--border);color:var(--text);font-size:12px;line-height:1.8;text-align:center}
.page-footer>span{max-width:100%;overflow-wrap:anywhere}
.page-footer a{color:var(--accent-text);text-decoration:underline;text-underline-offset:4px;text-decoration-color:color-mix(in srgb,var(--accent-text) 40%,transparent)}
.page-footer a:hover{ text-decoration-color:var(--accent-text) }
.footer-contact{position:relative;white-space:nowrap}
.footer-contact img{visibility:hidden;position:absolute;z-index:10;bottom:100%;right:0;display:block;width:176px;height:176px;max-width:none;padding:8px;box-sizing:border-box;background:#fff;border:1px solid var(--border);box-shadow:0 4px 18px var(--shadow)}
.footer-contact:hover img,.footer-contact:focus-visible img{visibility:visible}
@media(max-width:800px){.page-footer{gap:8px 12px;padding-top:12px;font-size:11px}.footer-copyright{flex-basis:100%}}
</style>
