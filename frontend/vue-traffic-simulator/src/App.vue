<template>
  <div class="traffic-simulator noselect">
    <ControlPanel
      :carNum="state.carNum"
      :currentStep="state.currentStep"
      :totalStep="state.totalStep"
      :selectedEntity="state.selectedEntity"
      :replaySpeed="state.replaySpeed"
      :paused="state.paused"
      :loading="state.loading"
      :infoMessages="state.infoMessages"
      @upload-files="handleUploadFiles"
      @start="handleStart"
      @pause="togglePause"
      @update-speed="updateReplaySpeed"
    />
    <SimulatorCanvas
      ref="canvasRef"
      :loading="state.loading"
    />
  </div>
</template>

<script setup>
import { reactive, ref, onMounted, onUnmounted } from 'vue'
import ControlPanel from './components/ControlPanel.vue'
import SimulatorCanvas from './components/SimulatorCanvas.vue'
import { loadFile } from './utils/helpers.js'
import { KEY_CODES } from './utils/constants.js'

const canvasRef = ref(null)

const state = reactive({
  carNum: 0,
  currentStep: 0,
  totalStep: 0,
  selectedEntity: '',
  replaySpeed: 0.5,
  paused: false,
  loading: false,
  infoMessages: [],
  roadnetData: null,
  replayData: null,
  chartData: null,
  debugMode: false
})

const files = reactive({
  roadnet: null,
  replay: null,
  chart: null
})

function addInfo(msg) {
  state.infoMessages.push(msg)
}

function resetInfo() {
  state.infoMessages = []
}

function handleUploadFiles({ roadnet, replay, chart, debugMode }) {
  files.roadnet = roadnet
  files.replay = replay
  files.chart = chart
  state.debugMode = debugMode
}

async function handleStart() {
  if (state.loading) return
  if (!files.roadnet || !files.replay) {
    addInfo('请先上传路网文件和回放文件')
    return
  }

  state.loading = true
  resetInfo()

  try {
    // 加载路网文件
    addInfo('加载 ' + files.roadnet.name)
    const roadnetText = await loadFile(files.roadnet)
    addInfo(files.roadnet.name + ' 加载完成')
    state.roadnetData = JSON.parse(roadnetText)

    // 加载回放文件
    addInfo('加载 ' + files.replay.name)
    const replayText = await loadFile(files.replay)
    addInfo(files.replay.name + ' 加载完成')
    state.replayData = replayText.split('\n').filter(line => line.trim())

    // 加载图表文件（可选）
    if (files.chart) {
      addInfo('加载 ' + files.chart.name)
      const chartText = await loadFile(files.chart)
      addInfo(files.chart.name + ' 加载完成')
      state.chartData = chartText
    }

    state.totalStep = state.replayData.length
    state.currentStep = 0
    state.paused = false

    // 初始化模拟器
    addInfo('绘制路网')
    await canvasRef.value.initSimulator(
      state.roadnetData,
      state.replayData,
      state.debugMode
    )
    addInfo('开始回放')
    
    canvasRef.value.startSimulation((stepData) => {
      state.carNum = stepData.carNum
      state.currentStep = stepData.currentStep
      state.selectedEntity = stepData.selectedEntity
    })

  } catch (error) {
    addInfo('错误: ' + error.message)
    console.error(error)
  } finally {
    state.loading = false
  }
}

function togglePause() {
  state.paused = !state.paused
  if (canvasRef.value) {
    canvasRef.value.setPaused(state.paused)
  }
}

function updateReplaySpeed(speed) {
  state.replaySpeed = Math.max(0, Math.min(1, speed))
  if (canvasRef.value) {
    canvasRef.value.setReplaySpeed(state.replaySpeed)
  }
}

function handleKeyDown(e) {
  if (e.keyCode === KEY_CODES.P) {
    togglePause()
  } else if (e.keyCode === KEY_CODES.ONE) {
    updateReplaySpeed(Math.max(state.replaySpeed / 1.5, 0.01))
  } else if (e.keyCode === KEY_CODES.TWO) {
    updateReplaySpeed(Math.min(state.replaySpeed * 1.5, 1))
  } else if (e.keyCode === KEY_CODES.LEFT_BRACKET && canvasRef.value) {
    canvasRef.value.stepBackward()
  } else if (e.keyCode === KEY_CODES.RIGHT_BRACKET && canvasRef.value) {
    canvasRef.value.stepForward()
  }
}

onMounted(() => {
  document.addEventListener('keydown', handleKeyDown)
})

onUnmounted(() => {
  document.removeEventListener('keydown', handleKeyDown)
})
</script>

<style scoped>
.traffic-simulator {
  display: flex;
  height: 100vh;
  width: 100vw;
}
</style>
