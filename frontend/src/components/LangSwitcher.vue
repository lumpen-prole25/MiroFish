<template>
  <div class="lang-switcher">
    <button
      v-for="lang in languages"
      :key="lang.code"
      :class="['lang-btn', { active: currentLocale === lang.code }]"
      @click="switchLang(lang.code)"
    >
      {{ lang.label }}
    </button>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { setLocale } from '../i18n'

const { locale } = useI18n()

const languages = [
  { code: 'en', label: 'EN' },
  { code: 'ko', label: 'KR' }
]

const currentLocale = computed(() => locale.value)

function switchLang(code) {
  setLocale(code)
}
</script>

<style scoped>
.lang-switcher {
  display: flex;
  gap: 2px;
  background: #f0f0f0;
  border-radius: 4px;
  padding: 2px;
}

.lang-btn {
  background: none;
  border: none;
  padding: 4px 10px;
  font-family: inherit;
  font-size: 11px;
  font-weight: 600;
  color: #888;
  cursor: pointer;
  border-radius: 3px;
  transition: all 0.2s;
}

.lang-btn:hover {
  color: #333;
}

.lang-btn.active {
  background: #000;
  color: #fff;
}
</style>
