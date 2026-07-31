<template>
  <div class="control-panel">
    <h2 class="title">CityFlow</h2>
    
    <!-- 状态显示 -->
    <div class="status-section">
      <div class="status-item">
        <span class="label">车辆数量</span>
        <span class="value">{{ carNum }}</span>
      </div>
      <div class="status-item">
        <span class="label">当前步数</span>
        <span class="value">{{ currentStep }}</span>
      </div>
      <div class="status-item">
        <span class="label">选中对象</span>
        <span class="value">{{ selectedEntity || '-' }}</span>
      </div>
    </div>

    <!-- 文件上传 -->
    <div class="file-section">
      <div class="file-input">
        <input
          type="file"
          id="roadnet-file"
          accept=".json"
          @change="handleRoadnetChange"
        />
        <label for="roadnet-file">
          {{ roadnetFileName || '路网文件 (*.json)' }}
        </label>
      </div>

      <div class="file-input">
        <input
          type="file"
          id="replay-file"
          accept=".txt"
          @change="handleReplayChange"
        />
        <label for="replay-file">
          {{ replayFileName || '回放文件 (*.txt)' }}
        </label>
      </div>

      <div class="file-input">
        <input
          type="file"
          id="chart-file"
          accept=".txt"
          @change="handleChartChange"
        />
        <label for="chart-file">
          {{ chartFileName || '图表文件 (可选)' }}
        </label>
      </div>

      <div class="actions">
        <label class="checkbox">
          <input type="checkbox" v-model="debugMode" />
          <span>调试模式</span>
        </label>
        <button class="btn-primary" @click="emitStart" :disabled="loading">
          {{ loading ? '加载中...' : '开始' }}
        </button>
      </div>
    </div>

    <!-- 控制区 -->
    <div class="control-section">
      <h5>控制面板</h5>
      <div class="speed-control">
        <div class="speed-label">
          <span>回放速度</span>
          <span class="speed-value">{{ replaySpeed.toFixed(2) }}</span>
        </div>
        <div class="speed-slider">
          <button @click="decreaseSpeed" class="speed-btn">−</button>
          <input
            type="range"
            min="0"
            max="100"
            :value="replaySpeed * 100"
            @input="handleSpeedChange"
          />
          <button @click="increaseSpeed" class="speed-btn">+</button>
        </div>
      </div>
      <button class="btn-secondary" @click="$emit('pause')">
        {{ paused ? '继续' : '暂停' }}
      </button>
    </div>

    <!-- 信息框 -->
    <div class="info-section">
      <h5>信息</h5>
      <div class="info-box">
        <div v-for="(msg, index) in infoMessages" :key="index" class="info-msg">
          - {{ msg }}
        </div>
      </div>
    </div>

    <!-- 使用说明 -->
    <div class="guide-section" v-if="!loading && infoMessages.length === 0">
      <h5>使用说明</h5>
      <ul>
        <li>上传路网文件 (*.json)</li>
        <li>上传回放文件 (*.txt)</li>
        <li>（可选）上传图表文件</li>
        <li>点击"开始"按钮开始回放</li>
        <li>按 P 键暂停/继续</li>
        <li>按 [ / ] 键单步前进/后退</li>
      </ul>
    </div>
  </div>
</template>

<script setup>
import { ref, watch } from 'vue'

const props = defineProps({
  carNum: { type: Number, default: 0 },
  currentStep: { type: Number, default: 0 },
  totalStep: { type: Number, default: 0 },
  selectedEntity: { type: String, default: '' },
  replaySpeed: { type: Number, default: 0.5 },
  paused: { type: Boolean, default: false },
  loading: { type: Boolean, default: false },
  infoMessages: { type: Array, default: () => [] }
})

const emit = defineEmits(['upload-files', 'start', 'pause', 'update-speed'])

const roadnetFile = ref(null)
const replayFile = ref(null)
const chartFile = ref(null)
const roadnetFileName = ref('')
const replayFileName = ref('')
const chartFileName = ref('')
const debugMode = ref(false)

function handleRoadnetChange(e) {
  roadnetFile.value = e.target.files[0]
  roadnetFileName.value = roadnetFile.value?.name || ''
  emitFiles()
}

function handleReplayChange(e) {
  replayFile.value = e.target.files[0]
  replayFileName.value = replayFile.value?.name || ''
  emitFiles()
}

function handleChartChange(e) {
  chartFile.value = e.target.files[0]
  chartFileName.value = chartFile.value?.name || ''
  emitFiles()
}

function emitFiles() {
  emit('upload-files', {
    roadnet: roadnetFile.value,
    replay: replayFile.value,
    chart: chartFile.value,
    debugMode: debugMode.value
  })
}

function emitStart() {
  emit('start')
}

function handleSpeedChange(e) {
  const speed = parseFloat(e.target.value) / 100
  emit('update-speed', speed)
}

function decreaseSpeed() {
  emit('update-speed', props.replaySpeed - 0.1)
}

function increaseSpeed() {
  emit('update-speed', props.replaySpeed + 0.1)
}

watch(debugMode, () => {
  emitFiles()
})
</script>

<style scoped>
.control-panel {
  width: 300px;
  height: 100vh;
  background: #f8f9fa;
  padding: 20px;
  overflow-y: auto;
  box-shadow: 2px 0 8px rgba(0,0,0,0.1);
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.title {
  font-size: 24px;
  font-weight: bold;
  color: #333;
  margin: 0;
  padding-bottom: 15px;
  border-bottom: 2px solid #dee2e6;
}

.status-section {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding-bottom: 15px;
  border-bottom: 1px solid #dee2e6;
}

.status-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.label {
  color: #666;
  font-size: 14px;
}

.value {
  color: #333;
  font-weight: 500;
}

.file-section {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.file-input {
  position: relative;
}

.file-input input[type="file"] {
  position: absolute;
  opacity: 0;
  width: 0;
  height: 0;
}

.file-input label {
  display: block;
  padding: 10px 15px;
  background: white;
  border: 1px solid #ced4da;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  color: #495057;
  transition: all 0.2s;
  text-overflow: ellipsis;
  overflow: hidden;
  white-space: nowrap;
}

.file-input label:hover {
  border-color: #007bff;
  background: #f0f8ff;
}

.actions {
  display: flex;
  align-items: center;
  gap: 10px;
  padding-top: 5px;
}

.checkbox {
  display: flex;
  align-items: center;
  gap: 5px;
  cursor: pointer;
  font-size: 14px;
  color: #666;
}

.checkbox input {
  cursor: pointer;
}

.btn-primary,
.btn-secondary {
  padding: 8px 20px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  font-weight: 500;
  transition: all 0.2s;
}

.btn-primary {
  background: #007bff;
  color: white;
  flex: 1;
}

.btn-primary:hover:not(:disabled) {
  background: #0056b3;
}

.btn-primary:disabled {
  background: #6c757d;
  cursor: not-allowed;
}

.btn-secondary {
  background: #6c757d;
  color: white;
  width: 100%;
}

.btn-secondary:hover {
  background: #545b62;
}

.control-section {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding-top: 10px;
  border-top: 1px solid #dee2e6;
}

.control-section h5 {
  margin: 0;
  font-size: 16px;
  color: #333;
}

.speed-control {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.speed-label {
  display: flex;
  justify-content: space-between;
  font-size: 14px;
  color: #666;
}

.speed-value {
  color: #333;
  font-weight: 500;
}

.speed-slider {
  display: flex;
  align-items: center;
  gap: 8px;
}

.speed-btn {
  width: 30px;
  height: 30px;
  border: 1px solid #ced4da;
  background: white;
  border-radius: 4px;
  cursor: pointer;
  font-size: 18px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;
}

.speed-btn:hover {
  background: #e9ecef;
}

.speed-slider input[type="range"] {
  flex: 1;
  height: 6px;
  border-radius: 3px;
  background: #dee2e6;
  outline: none;
  -webkit-appearance: none;
}

.speed-slider input[type="range"]::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: #007bff;
  cursor: pointer;
}

.speed-slider input[type="range"]::-moz-range-thumb {
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: #007bff;
  cursor: pointer;
  border: none;
}

.info-section {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.info-section h5 {
  margin: 0;
  font-size: 16px;
  color: #333;
}

.info-box {
  background: white;
  border: 1px solid #dee2e6;
  border-radius: 4px;
  padding: 10px;
  max-height: 200px;
  overflow-y: auto;
  font-size: 13px;
  color: #495057;
}

.info-msg {
  margin-bottom: 5px;
}

.guide-section {
  display: flex;
  flex-direction: column;
  gap: 10px;
  background: #e7f3ff;
  padding: 15px;
  border-radius: 4px;
  border: 1px solid #b3d9ff;
}

.guide-section h5 {
  margin: 0 0 10px 0;
  font-size: 16px;
  color: #004085;
}

.guide-section ul {
  margin: 0;
  padding-left: 20px;
  color: #004085;
  font-size: 13px;
}

.guide-section li {
  margin-bottom: 5px;
}
</style>
