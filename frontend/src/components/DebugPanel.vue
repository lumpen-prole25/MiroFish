<template>
  <div class="debug-panel" :class="{ expanded: isExpanded }">
    <!-- Collapsed bar -->
    <div class="debug-bar" @click="toggle">
      <div class="bar-left">
        <span class="bar-icon">⚙</span>
        <span class="bar-title">Debug</span>
        <span class="status-dot" :class="connected ? 'connected' : 'disconnected'"></span>
        <span v-if="status" class="bar-stats">
          MEM {{ status.memory?.percent }}% · CPU {{ status.cpu_percent }}% · SIM {{ status.active_simulations }}
        </span>
      </div>
      <div class="bar-right">
        <span class="toggle-icon">{{ isExpanded ? '▼' : '▲' }}</span>
      </div>
    </div>

    <!-- Expanded panel -->
    <div v-show="isExpanded" class="debug-content">
      <!-- Drag handle -->
      <div class="drag-handle" @mousedown="startResize"></div>

      <div class="panels">
        <!-- Log viewer -->
        <div class="log-panel">
          <div class="panel-header">
            <span>Logs ({{ logs.length }})</span>
            <div class="panel-actions">
              <button @click="autoScroll = !autoScroll" :class="{ active: autoScroll }">Auto⬇</button>
              <button @click="clearLogs">Clear</button>
            </div>
          </div>
          <div class="log-content" ref="logContainer">
            <div v-for="(log, i) in logs" :key="i" class="log-entry" :class="log.level?.toLowerCase()">
              <span class="log-time">{{ log.timestamp }}</span>
              <span class="log-level">{{ log.level }}</span>
              <span class="log-msg">{{ log.message }}</span>
            </div>
            <div v-if="logs.length === 0" class="log-empty">Waiting for logs...</div>
          </div>
        </div>

        <!-- Command terminal -->
        <div class="cmd-panel">
          <div class="panel-header">
            <span>Terminal</span>
          </div>
          <div class="cmd-content" ref="cmdContainer">
            <div v-for="(entry, i) in cmdHistory" :key="i" class="cmd-entry">
              <div class="cmd-input-line">> {{ entry.command }}</div>
              <pre class="cmd-output" :class="entry.error ? 'error' : ''">{{ entry.output }}</pre>
            </div>
          </div>
          <div class="cmd-input-wrap">
            <span class="cmd-prompt">></span>
            <input
              ref="cmdInput"
              v-model="cmdText"
              @keydown.enter="executeCommand"
              @keydown.up.prevent="historyUp"
              @keydown.down.prevent="historyDown"
              @keydown.tab.prevent="autoComplete"
              placeholder="Type /help for commands..."
              spellcheck="false"
            />
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, nextTick, watch } from 'vue'

const API = 'http://localhost:5001/api/debug'

// Panel state
const isExpanded = ref(false)
const panelHeight = ref(300)

// Log state
const logs = ref([])
const logContainer = ref(null)
const autoScroll = ref(true)
const connected = ref(false)
let eventSource = null

// Command state
const cmdText = ref('')
const cmdHistory = ref([])
const cmdContainer = ref(null)
const cmdInput = ref(null)
const inputHistory = ref([])
const historyIdx = ref(-1)

// Status
const status = ref(null)
let statusInterval = null

// Available commands for auto-complete
const COMMANDS = ['/kill', '/restart', '/status', '/clear', '/config', '/help']

function toggle() {
  isExpanded.value = !isExpanded.value
  if (isExpanded.value) {
    connectSSE()
    fetchStatus()
    nextTick(() => cmdInput.value?.focus())
  } else {
    disconnectSSE()
  }
}

// SSE connection
function connectSSE() {
  if (eventSource) return
  try {
    eventSource = new EventSource(`${API}/logs`)
    eventSource.onmessage = (e) => {
      try {
        const entry = JSON.parse(e.data)
        logs.value.push(entry)
        if (logs.value.length > 1000) logs.value.splice(0, 100)
        if (autoScroll.value) {
          nextTick(() => {
            if (logContainer.value) {
              logContainer.value.scrollTop = logContainer.value.scrollHeight
            }
          })
        }
      } catch {}
    }
    eventSource.onopen = () => { connected.value = true }
    eventSource.onerror = () => { connected.value = false }
  } catch {
    connected.value = false
  }
}

function disconnectSSE() {
  if (eventSource) {
    eventSource.close()
    eventSource = null
    connected.value = false
  }
}

function clearLogs() {
  logs.value = []
  executeRemoteCommand('/clear')
}

// Commands
async function executeCommand() {
  const cmd = cmdText.value.trim()
  if (!cmd) return

  inputHistory.value.unshift(cmd)
  historyIdx.value = -1
  cmdText.value = ''

  await executeRemoteCommand(cmd)
}

async function executeRemoteCommand(cmd) {
  try {
    const res = await fetch(`${API}/command`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ command: cmd }),
    })
    const data = await res.json()
    const output = data.error
      ? `Error: ${data.error}`
      : JSON.stringify(data.result, null, 2)

    cmdHistory.value.push({ command: cmd, output, error: !!data.error })
  } catch (e) {
    cmdHistory.value.push({ command: cmd, output: `Connection error: ${e.message}`, error: true })
  }

  nextTick(() => {
    if (cmdContainer.value) {
      cmdContainer.value.scrollTop = cmdContainer.value.scrollHeight
    }
  })
}

function historyUp() {
  if (inputHistory.value.length === 0) return
  historyIdx.value = Math.min(historyIdx.value + 1, inputHistory.value.length - 1)
  cmdText.value = inputHistory.value[historyIdx.value]
}

function historyDown() {
  if (historyIdx.value <= 0) {
    historyIdx.value = -1
    cmdText.value = ''
    return
  }
  historyIdx.value--
  cmdText.value = inputHistory.value[historyIdx.value]
}

function autoComplete() {
  const text = cmdText.value.trim()
  const matches = COMMANDS.filter(c => c.startsWith(text))
  if (matches.length === 1) {
    cmdText.value = matches[0]
  } else if (matches.length > 1) {
    cmdHistory.value.push({
      command: text,
      output: 'Matches: ' + matches.join(', '),
      error: false,
    })
  }
}

// Status polling
async function fetchStatus() {
  try {
    const res = await fetch(`${API}/status`)
    status.value = await res.json()
  } catch {}
}

// Resize
function startResize(e) {
  e.preventDefault()
  const startY = e.clientY
  const startH = panelHeight.value
  const onMove = (ev) => {
    panelHeight.value = Math.max(150, Math.min(startH + (startY - ev.clientY), window.innerHeight * 0.8))
  }
  const onUp = () => {
    document.removeEventListener('mousemove', onMove)
    document.removeEventListener('mouseup', onUp)
  }
  document.addEventListener('mousemove', onMove)
  document.addEventListener('mouseup', onUp)
}

// Keyboard shortcut: Ctrl+`
function onKeydown(e) {
  if (e.ctrlKey && e.key === '`') {
    e.preventDefault()
    toggle()
  }
}

onMounted(() => {
  document.addEventListener('keydown', onKeydown)
  statusInterval = setInterval(fetchStatus, 5000)
})

onUnmounted(() => {
  document.removeEventListener('keydown', onKeydown)
  disconnectSSE()
  if (statusInterval) clearInterval(statusInterval)
})

watch(isExpanded, (val) => {
  if (val) fetchStatus()
})
</script>

<style scoped>
.debug-panel {
  position: fixed;
  bottom: 0;
  left: 0;
  right: 0;
  z-index: 9999;
  font-family: 'JetBrains Mono', 'Courier New', monospace;
  font-size: 12px;
}

.debug-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 4px 12px;
  background: #1a1a2e;
  color: #a0a0b0;
  cursor: pointer;
  user-select: none;
  border-top: 1px solid #333;
}
.debug-bar:hover {
  background: #22223a;
}

.bar-left {
  display: flex;
  align-items: center;
  gap: 8px;
}
.bar-icon { font-size: 14px; }
.bar-title { font-weight: 600; color: #e0e0e0; }
.bar-stats { color: #6b7280; font-size: 11px; }

.status-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  display: inline-block;
}
.status-dot.connected { background: #22c55e; }
.status-dot.disconnected { background: #ef4444; }

.toggle-icon { font-size: 10px; }

.debug-content {
  background: #1a1a2e;
  height: v-bind(panelHeight + 'px');
  border-top: 1px solid #333;
  display: flex;
  flex-direction: column;
}

.drag-handle {
  height: 4px;
  background: #333;
  cursor: ns-resize;
  flex-shrink: 0;
}
.drag-handle:hover {
  background: #555;
}

.panels {
  display: flex;
  flex: 1;
  overflow: hidden;
}

/* Log panel */
.log-panel {
  flex: 6;
  display: flex;
  flex-direction: column;
  border-right: 1px solid #333;
}

.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 4px 10px;
  background: #16162a;
  color: #8888a0;
  font-size: 11px;
  flex-shrink: 0;
}

.panel-actions {
  display: flex;
  gap: 6px;
}
.panel-actions button {
  background: none;
  border: 1px solid #444;
  color: #8888a0;
  padding: 1px 6px;
  border-radius: 3px;
  cursor: pointer;
  font-family: inherit;
  font-size: 10px;
}
.panel-actions button:hover {
  background: #333;
  color: #e0e0e0;
}
.panel-actions button.active {
  border-color: #22c55e;
  color: #22c55e;
}

.log-content {
  flex: 1;
  overflow-y: auto;
  padding: 4px 10px;
}

.log-entry {
  display: flex;
  gap: 8px;
  line-height: 1.5;
  white-space: nowrap;
}
.log-time { color: #555; min-width: 80px; }
.log-level { min-width: 50px; font-weight: 600; }
.log-msg { color: #d0d0d0; white-space: pre-wrap; word-break: break-all; }

.log-entry.info .log-level { color: #22c55e; }
.log-entry.warning .log-level { color: #eab308; }
.log-entry.error .log-level { color: #ef4444; }
.log-entry.debug .log-level { color: #6b7280; }
.log-empty { color: #555; padding: 20px; text-align: center; }

/* Command panel */
.cmd-panel {
  flex: 4;
  display: flex;
  flex-direction: column;
}

.cmd-content {
  flex: 1;
  overflow-y: auto;
  padding: 4px 10px;
}

.cmd-entry {
  margin-bottom: 8px;
}
.cmd-input-line {
  color: #22c55e;
  line-height: 1.5;
}
.cmd-output {
  color: #d0d0d0;
  margin: 2px 0 0 0;
  font-family: inherit;
  font-size: 12px;
  white-space: pre-wrap;
  word-break: break-all;
  line-height: 1.4;
}
.cmd-output.error {
  color: #ef4444;
}

.cmd-input-wrap {
  display: flex;
  align-items: center;
  padding: 4px 10px;
  background: #16162a;
  border-top: 1px solid #333;
  flex-shrink: 0;
}
.cmd-prompt {
  color: #22c55e;
  margin-right: 6px;
  font-weight: 600;
}
.cmd-input-wrap input {
  flex: 1;
  background: none;
  border: none;
  outline: none;
  color: #e0e0e0;
  font-family: inherit;
  font-size: 12px;
  caret-color: #22c55e;
}
.cmd-input-wrap input::placeholder {
  color: #444;
}

/* Scrollbar */
.log-content::-webkit-scrollbar,
.cmd-content::-webkit-scrollbar {
  width: 5px;
}
.log-content::-webkit-scrollbar-track,
.cmd-content::-webkit-scrollbar-track {
  background: transparent;
}
.log-content::-webkit-scrollbar-thumb,
.cmd-content::-webkit-scrollbar-thumb {
  background: #333;
  border-radius: 3px;
}
</style>
