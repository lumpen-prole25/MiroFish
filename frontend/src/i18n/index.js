import { createI18n } from 'vue-i18n'
import en from './en.json'
import ko from './ko.json'

const savedLocale = localStorage.getItem('geronimo-locale') || 'en'

const i18n = createI18n({
  legacy: false,
  locale: savedLocale,
  fallbackLocale: 'en',
  messages: { en, ko }
})

export function setLocale(locale) {
  i18n.global.locale.value = locale
  localStorage.setItem('geronimo-locale', locale)
  document.documentElement.lang = locale
}

export function getLocale() {
  return i18n.global.locale.value
}

export default i18n
