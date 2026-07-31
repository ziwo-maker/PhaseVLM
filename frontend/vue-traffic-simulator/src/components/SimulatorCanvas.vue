<template>
  <div class="simulator-canvas" ref="containerRef">
    <div v-if="loading" class="spinner">
      <div class="rect1"></div>
      <div class="rect2"></div>
      <div class="rect3"></div>
      <div class="rect4"></div>
      <div class="rect5"></div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import * as PIXI from 'pixi.js'
import { Viewport } from 'pixi-viewport'
import { Point } from '../utils/Point.js'
import { transCoord, statusToColor, stringHash, pointToSegmentDistance, getTrafficColor } from '../utils/helpers.js'
import * as CONSTANTS from '../utils/constants.js'

const props = defineProps({
  loading: { type: Boolean, default: false }
})

const containerRef = ref(null)
let app = null
let viewport = null
let simulatorContainer = null
let carContainer = null
let trafficLightContainer = null
let turnSignalContainer = null
let carPool = []
let trafficLightsG = {}
let nodes = {}
let edges = {}
let edgeGraphics = {}  // 存储每条道路的Graphics对象，用于动态更新颜色
let edgeLaneCarCount = {}   // 存储每条道路每个车道的停留车辆数量 {edgeId: [lane0Count, lane1Count, ...]}
let carPositionHistory = {}  // 存储每辆车的位置历史 {carId: [{x, y, step}, ...]}
let logs = []
let currentStep = 0
let totalStep = 0
let frameElapsed = 0
let replaySpeed = 0.5
let paused = false
let animationId = null
let debugMode = false
let updateCallback = null
let turnSignalTextures = []

defineExpose({
  initSimulator,
  startSimulation,
  setPaused,
  setReplaySpeed,
  stepForward,
  stepBackward
})

function initSimulator(roadnetData, replayData, debug) {
  return new Promise((resolve) => {
    debugMode = debug
    logs = replayData
    totalStep = logs.length
    currentStep = 0
    
    setupPixi()
    drawRoadnet(roadnetData.static)
    resolve()
  })
}

function setupPixi() {
  if (app) {
    app.destroy(true)
  }

  app = new PIXI.Application({
    width: containerRef.value.offsetWidth,
    height: containerRef.value.offsetHeight,
    backgroundColor: CONSTANTS.BACKGROUND_COLOR,
    antialias: true
  })

  containerRef.value.appendChild(app.view)

  // 正确配置viewport - 使用PIXI v7的事件系统
  viewport = new Viewport({
    screenWidth: containerRef.value.offsetWidth,
    screenHeight: containerRef.value.offsetHeight,
    worldWidth: 10000,
    worldHeight: 10000,
    events: app.renderer.events
  })

  // 配置交互功能
  viewport
    .drag({
      mouseButtons: 'left',
      pressDrag: false
    })
    .pinch()
    .wheel({
      smooth: 3,
      percent: 0.1
    })
    .decelerate({
      friction: 0.95
    })
    .clampZoom({
      minScale: 0.1,
      maxScale: 5
    })

  app.stage.addChild(viewport)
  simulatorContainer = new PIXI.Container()
  viewport.addChild(simulatorContainer)
}

function drawRoadnet(roadnet) {
  nodes = {}
  edges = {}
  trafficLightsG = {}
  edgeGraphics = {}
  edgeLaneCarCount = {}

  // 处理节点
  for (let i = 0; i < roadnet.nodes.length; i++) {
    const node = roadnet.nodes[i]
    node.point = new Point(transCoord(node.point))
    nodes[node.id] = node
  }

  // 处理边
  for (let i = 0; i < roadnet.edges.length; i++) {
    const edge = roadnet.edges[i]
    edge.from = nodes[edge.from]
    edge.to = nodes[edge.to]
    for (let j = 0; j < edge.points.length; j++) {
      edge.points[j] = new Point(transCoord(edge.points[j]))
    }
    edges[edge.id] = edge
    // 初始化每个车道的车辆计数
    edgeLaneCarCount[edge.id] = new Array(edge.nLane).fill(0)
  }

  // 绘制地图
  trafficLightContainer = new PIXI.ParticleContainer(CONSTANTS.MAX_TRAFFIC_LIGHT_NUM, { tint: true })
  const mapGraphics = new PIXI.Graphics()
  simulatorContainer.addChild(mapGraphics)

  // 绘制节点
  for (const nodeId in nodes) {
    if (!nodes[nodeId].virtual) {
      drawNode(nodes[nodeId], mapGraphics)
    }
  }

  // 绘制边 - 为每条边创建独立的Graphics对象
  for (const edgeId in edges) {
    const edgeG = new PIXI.Graphics()
    simulatorContainer.addChild(edgeG)
    edgeGraphics[edgeId] = edgeG
    drawEdge(edges[edgeId], edgeG, app.renderer)
  }

  // 居中显示
  const bounds = simulatorContainer.getBounds()
  simulatorContainer.pivot.set(bounds.x + bounds.width / 2, bounds.y + bounds.height / 2)
  simulatorContainer.position.set(app.renderer.width / 2, app.renderer.height / 2)
  simulatorContainer.addChild(trafficLightContainer)

  // 创建车辆池
  setupCarPool()
}

function drawNode(node, graphics) {
  graphics.beginFill(CONSTANTS.LANE_COLOR)
  const outline = node.outline
  for (let i = 0; i < outline.length; i += 2) {
    outline[i + 1] = -outline[i + 1]
    if (i === 0) {
      graphics.moveTo(outline[i], outline[i + 1])
    } else {
      graphics.lineTo(outline[i], outline[i + 1])
    }
  }
  graphics.endFill()
}

function drawEdge(edge, graphics, renderer) {
  const from = edge.from
  const to = edge.to
  const points = edge.points

  let roadWidth = 0
  edge.laneWidths.forEach(l => roadWidth += l)

  for (let i = 1; i < points.length; i++) {
    let pointA, pointAOffset, pointB, pointBOffset
    let prevPointBOffset = null

    if (i === 1) {
      pointA = points[0].moveAlongDirectTo(points[1], from.virtual ? 0 : from.width)
      pointAOffset = points[0].directTo(points[1]).rotate(CONSTANTS.ROTATE)
    } else {
      pointA = points[i - 1]
      pointAOffset = prevPointBOffset
    }

    if (i === points.length - 1) {
      pointB = points[i].moveAlongDirectTo(points[i - 1], to.virtual ? 0 : to.width)
      pointBOffset = points[i - 1].directTo(points[i]).rotate(CONSTANTS.ROTATE)
    } else {
      pointB = points[i]
      pointBOffset = points[i - 1].directTo(points[i + 1]).rotate(CONSTANTS.ROTATE)
    }
    prevPointBOffset = pointBOffset

    // 绘制交通灯
    if (i === points.length - 1 && !to.virtual) {
      const edgeTrafficLights = []
      let prevOffset = 0
      let offset = 0

      const lightG = new PIXI.Graphics()
      lightG.lineStyle(CONSTANTS.TRAFFIC_LIGHT_WIDTH, 0xFFFFFF)
      lightG.moveTo(0, 0)
      lightG.lineTo(1, 0)
      const lightTexture = renderer.generateTexture(lightG)

      for (let lane = 0; lane < edge.nLane; lane++) {
        offset += edge.laneWidths[lane]
        const light = new PIXI.Sprite(lightTexture)
        light.anchor.set(0, 0.5)
        light.scale.set(offset - prevOffset, 1)
        const point = pointB.moveAlong(pointBOffset, prevOffset)
        light.position.set(point.x, point.y)
        light.rotation = pointBOffset.getAngleInRadians()
        edgeTrafficLights.push(light)
        prevOffset = offset
        trafficLightContainer.addChild(light)
      }
      trafficLightsG[edge.id] = edgeTrafficLights
    }

    // 绘制道路
    graphics.lineStyle(CONSTANTS.LANE_BORDER_WIDTH, CONSTANTS.LANE_BORDER_COLOR, 1)
    graphics.moveTo(pointA.x, pointA.y)
    graphics.lineTo(pointB.x, pointB.y)

    const pointA1 = pointA.moveAlong(pointAOffset, roadWidth)
    const pointB1 = pointB.moveAlong(pointBOffset, roadWidth)

    graphics.lineStyle(0)
    graphics.beginFill(CONSTANTS.LANE_COLOR)
    graphics.drawPolygon([pointA.x, pointA.y, pointB.x, pointB.y, pointB1.x, pointB1.y, pointA1.x, pointA1.y])
    graphics.endFill()

    // 绘制车道线
    let offset = 0
    for (let lane = 0; lane < edge.nLane - 1; lane++) {
      offset += edge.laneWidths[lane]
      graphics.lineStyle(CONSTANTS.LANE_BORDER_WIDTH, CONSTANTS.LANE_INNER_COLOR)
      drawDashLine(graphics, pointA.moveAlong(pointAOffset, offset), pointB.moveAlong(pointBOffset, offset))
    }
  }
}

function drawDashLine(graphics, pointA, pointB, dash = 16, gap = 8) {
  const direct = pointA.directTo(pointB)
  const distance = pointA.distanceTo(pointB)
  let currentPoint = pointA
  let currentDistance = 0

  while (true) {
    graphics.moveTo(currentPoint.x, currentPoint.y)
    const length = currentDistance + dash >= distance ? distance - currentDistance : dash
    currentPoint = currentPoint.moveAlong(direct, length)
    graphics.lineTo(currentPoint.x, currentPoint.y)
    
    if (currentDistance + dash >= distance) break
    currentDistance += length

    if (currentDistance + gap >= distance) break
    currentPoint = currentPoint.moveAlong(direct, gap)
    currentDistance += gap
  }
}

// 根据每个车道的车辆数量更新道路颜色
function updateEdgeColors() {
  for (const edgeId in edges) {
    const edge = edges[edgeId]
    const graphics = edgeGraphics[edgeId]
    const laneCarCounts = edgeLaneCarCount[edgeId] || []
    
    // 清除旧的绘制内容
    graphics.clear()
    
    const from = edge.from
    const to = edge.to
    const points = edge.points
    
    let roadWidth = 0
    edge.laneWidths.forEach(l => roadWidth += l)
    
    // 重新绘制道路
    for (let i = 1; i < points.length; i++) {
      let pointA, pointAOffset, pointB, pointBOffset
      let prevPointBOffset = null
      
      if (i === 1) {
        pointA = points[0].moveAlongDirectTo(points[1], from.virtual ? 0 : from.width)
        pointAOffset = points[0].directTo(points[1]).rotate(CONSTANTS.ROTATE)
      } else {
        pointA = points[i - 1]
        pointAOffset = prevPointBOffset
      }
      
      if (i === points.length - 1) {
        pointB = points[i].moveAlongDirectTo(points[i - 1], to.virtual ? 0 : to.width)
        pointBOffset = points[i - 1].directTo(points[i]).rotate(CONSTANTS.ROTATE)
      } else {
        pointB = points[i]
        pointBOffset = points[i - 1].directTo(points[i + 1]).rotate(CONSTANTS.ROTATE)
      }
      prevPointBOffset = pointBOffset
      
      // 绘制道路外边框
      graphics.lineStyle(CONSTANTS.LANE_BORDER_WIDTH, CONSTANTS.LANE_BORDER_COLOR, 1)
      graphics.moveTo(pointA.x, pointA.y)
      graphics.lineTo(pointB.x, pointB.y)
      
      const pointA1 = pointA.moveAlong(pointAOffset, roadWidth)
      const pointB1 = pointB.moveAlong(pointBOffset, roadWidth)
      
      graphics.moveTo(pointA1.x, pointA1.y)
      graphics.lineTo(pointB1.x, pointB1.y)
      
      // 为每个车道单独绘制颜色
      let prevOffset = 0
      for (let lane = 0; lane < edge.nLane; lane++) {
        const laneWidth = edge.laneWidths[lane]
        const currentOffset = prevOffset + laneWidth
        
        // 获取该车道的车辆数量和对应颜色
        const laneCarCount = laneCarCounts[lane] || 0
        const laneColor = getTrafficColor(laneCarCount)
        
        // 计算该车道的四个角点
        const lanePointA1 = pointA.moveAlong(pointAOffset, prevOffset)
        const lanePointA2 = pointA.moveAlong(pointAOffset, currentOffset)
        const lanePointB1 = pointB.moveAlong(pointBOffset, prevOffset)
        const lanePointB2 = pointB.moveAlong(pointBOffset, currentOffset)
        
        // 绘制该车道的颜色填充
        graphics.lineStyle(0)
        graphics.beginFill(laneColor)
        graphics.drawPolygon([
          lanePointA1.x, lanePointA1.y,
          lanePointB1.x, lanePointB1.y,
          lanePointB2.x, lanePointB2.y,
          lanePointA2.x, lanePointA2.y
        ])
        graphics.endFill()
        
        prevOffset = currentOffset
      }
      
      // 绘制车道分隔线
      let offset = 0
      for (let lane = 0; lane < edge.nLane - 1; lane++) {
        offset += edge.laneWidths[lane]
        graphics.lineStyle(CONSTANTS.LANE_BORDER_WIDTH, CONSTANTS.LANE_INNER_COLOR)
        drawDashLine(graphics, pointA.moveAlong(pointAOffset, offset), pointB.moveAlong(pointBOffset, offset))
      }
    }
  }
}

// 根据车辆位置找到所属道路和车道
function findCarLane(carPoint) {
  let minDistance = Infinity
  let closestEdgeId = null
  let closestLaneIndex = 0
  let closestSegmentInfo = null
  
  // 遍历所有道路
  for (const edgeId in edges) {
    const edge = edges[edgeId]
    const points = edge.points
    const from = edge.from
    const to = edge.to
    
    // 遍历道路的每个线段
    for (let i = 1; i < points.length; i++) {
      let pointA, pointAOffset, pointB, pointBOffset
      let prevPointBOffset = null
      
      // 处理起点
      if (i === 1) {
        pointA = points[0].moveAlongDirectTo(points[1], from.virtual ? 0 : from.width)
        pointAOffset = points[0].directTo(points[1]).rotate(CONSTANTS.ROTATE)
      } else {
        pointA = points[i - 1]
        pointAOffset = prevPointBOffset
      }
      
      // 处理终点
      if (i === points.length - 1) {
        pointB = points[i].moveAlongDirectTo(points[i - 1], to.virtual ? 0 : to.width)
        pointBOffset = points[i - 1].directTo(points[i]).rotate(CONSTANTS.ROTATE)
      } else {
        pointB = points[i]
        pointBOffset = points[i - 1].directTo(points[i + 1]).rotate(CONSTANTS.ROTATE)
      }
      
      // 计算车辆到线段的距离
      const distance = pointToSegmentDistance(carPoint, pointA, pointB)
      
      if (distance < minDistance) {
        minDistance = distance
        closestEdgeId = edgeId
        closestSegmentInfo = { pointA, pointAOffset, pointB, pointBOffset }
      }
    }
  }
  
  if (!closestEdgeId || !closestSegmentInfo) {
    return null
  }
  
  // 计算车辆在找到的线段上的横向偏移量
  const edge = edges[closestEdgeId]
  const { pointA, pointAOffset, pointB, pointBOffset } = closestSegmentInfo
  
  // 计算车辆到道路中心线的投影
  const roadDir = pointA.directTo(pointB)
  const carToA = new Point(carPoint.x - pointA.x, carPoint.y - pointA.y)
  
  // 计算沿道路方向的距离
  const alongDist = carToA.x * roadDir.x + carToA.y * roadDir.y
  const t = alongDist / pointA.distanceTo(pointB)
  
  // 在线段上的投影点
  const projPoint = new Point(
    pointA.x + roadDir.x * alongDist,
    pointA.y + roadDir.y * alongDist
  )
  
  // 计算该投影点处的横向偏移方向（插值）
  const lateralOffset = new Point(
    pointAOffset.x * (1 - t) + pointBOffset.x * t,
    pointAOffset.y * (1 - t) + pointBOffset.y * t
  )
  
  // 归一化
  const lateralLen = Math.sqrt(lateralOffset.x ** 2 + lateralOffset.y ** 2)
  const lateralDir = new Point(lateralOffset.x / lateralLen, lateralOffset.y / lateralLen)
  
  // 计算车辆相对于道路中心线的横向距离
  const carToProjPoint = new Point(carPoint.x - projPoint.x, carPoint.y - projPoint.y)
  const lateralDist = carToProjPoint.x * lateralDir.x + carToProjPoint.y * lateralDir.y
  
  // 根据横向距离判断属于哪个车道
  let accumulatedWidth = 0
  let laneIndex = 0
  
  for (let lane = 0; lane < edge.nLane; lane++) {
    accumulatedWidth += edge.laneWidths[lane]
    if (lateralDist < accumulatedWidth) {
      laneIndex = lane
      break
    }
  }
  
  // 确保车道索引在有效范围内
  laneIndex = Math.max(0, Math.min(edge.nLane - 1, laneIndex))
  
  return { edgeId: closestEdgeId, laneIndex }
}

// 判断车辆是否在路口附近停留
function isCarStoppedNearIntersection(carId, carPoint, currentStep) {
  const HISTORY_FRAMES = 4  // 检查最近4帧
  const STOP_THRESHOLD = 1.0  // 位移阈值（米）
  const INTERSECTION_DISTANCE = 30  // 距离路口的阈值（米）
  
  // 更新车辆位置历史
  if (!carPositionHistory[carId]) {
    carPositionHistory[carId] = []
  }
  
  carPositionHistory[carId].push({
    x: carPoint.x,
    y: carPoint.y,
    step: currentStep
  })
  
  // 只保留最近的历史记录
  if (carPositionHistory[carId].length > HISTORY_FRAMES + 1) {
    carPositionHistory[carId].shift()
  }
  
  // 如果历史记录不足，认为不是停留状态
  if (carPositionHistory[carId].length < HISTORY_FRAMES) {
    return { isStopped: false, atStart: false, atEnd: false }
  }
  
  // 检查车辆是否停留（位移很小）
  const history = carPositionHistory[carId]
  const oldestPos = history[0]
  const newestPos = history[history.length - 1]
  const displacement = Math.sqrt(
    Math.pow(newestPos.x - oldestPos.x, 2) + 
    Math.pow(newestPos.y - oldestPos.y, 2)
  )
  
  // 如果位移大于阈值，说明在移动，不是停留状态
  if (displacement > STOP_THRESHOLD) {
    return { isStopped: false, atStart: false, atEnd: false }
  }
  
  // 检查车辆是否在路口附近
  // 找到车辆所在的道路
  const laneInfo = findCarLane(carPoint)
  if (!laneInfo) {
    return { isStopped: false, atStart: false, atEnd: false }
  }
  
  const edge = edges[laneInfo.edgeId]
  if (!edge) {
    return { isStopped: false, atStart: false, atEnd: false }
  }
  
  // 检查道路起点和终点
  const startPoint = edge.points[0]
  const endPoint = edge.points[edge.points.length - 1]
  
  // 计算车辆到道路起点（路口）的距离
  const distanceToStart = Math.sqrt(
    Math.pow(carPoint.x - startPoint.x, 2) + 
    Math.pow(carPoint.y - startPoint.y, 2)
  )
  
  // 计算车辆到道路终点（路口）的距离
  const distanceToEnd = Math.sqrt(
    Math.pow(carPoint.x - endPoint.x, 2) + 
    Math.pow(carPoint.y - endPoint.y, 2)
  )
  
  // 检查是否在起点路口附近（且起点不是虚拟节点）
  const atStart = !edge.from.virtual && distanceToStart <= INTERSECTION_DISTANCE
  
  // 检查是否在终点路口附近（且终点不是虚拟节点）
  const atEnd = !edge.to.virtual && distanceToEnd <= INTERSECTION_DISTANCE
  
  // 返回停留状态和位置信息
  return {
    isStopped: atStart || atEnd,
    atStart: atStart,
    atEnd: atEnd
  }
}

function setupCarPool() {
  const carG = new PIXI.Graphics()
  carG.lineStyle(0)
  carG.beginFill(0xFFFFFF, 0.8)
  carG.drawRect(0, 0, CONSTANTS.CAR_LENGTH, CONSTANTS.CAR_WIDTH)
  const carTexture = app.renderer.generateTexture(carG)

  const signalG = new PIXI.Graphics()
  signalG.beginFill(CONSTANTS.TURN_SIGNAL_COLOR, 0.7)
  signalG.drawRect(0, 0, CONSTANTS.CAR_LENGTH, CONSTANTS.CAR_WIDTH / 2)
  signalG.drawRect(0, 3 * CONSTANTS.CAR_WIDTH - CONSTANTS.CAR_WIDTH / 2, CONSTANTS.CAR_LENGTH, CONSTANTS.CAR_WIDTH / 2)
  signalG.endFill()
  const turnSignalTexture = app.renderer.generateTexture(signalG)

  turnSignalTextures = [
    new PIXI.Texture(turnSignalTexture, new PIXI.Rectangle(0, 0, CONSTANTS.CAR_LENGTH, CONSTANTS.CAR_WIDTH)),
    new PIXI.Texture(turnSignalTexture, new PIXI.Rectangle(0, CONSTANTS.CAR_WIDTH, CONSTANTS.CAR_LENGTH, CONSTANTS.CAR_WIDTH)),
    new PIXI.Texture(turnSignalTexture, new PIXI.Rectangle(0, CONSTANTS.CAR_WIDTH * 2, CONSTANTS.CAR_LENGTH, CONSTANTS.CAR_WIDTH))
  ]

  carContainer = new PIXI.ParticleContainer(CONSTANTS.NUM_CAR_POOL, { rotation: true, tint: true })
  turnSignalContainer = new PIXI.ParticleContainer(CONSTANTS.NUM_CAR_POOL, { rotation: true, tint: true })
  
  simulatorContainer.addChild(carContainer)
  simulatorContainer.addChild(turnSignalContainer)

  carPool = []
  for (let i = 0; i < CONSTANTS.NUM_CAR_POOL; i++) {
    const car = new PIXI.Sprite(carTexture)
    const signal = new PIXI.Sprite(turnSignalTextures[1])
    car.anchor.set(1, 0.5)
    signal.anchor.set(1, 0.5)
    carPool.push([car, signal])
  }
}

function drawStep(step) {
  const [carLogs, tlLogs] = logs[step].split(';')
  
  // 更新交通灯
  tlLogs.split(',').forEach(tlLog => {
    const parts = tlLog.split(' ')
    const tlEdge = parts[0]
    const tlStatus = parts.slice(1)
    
    if (trafficLightsG[tlEdge]) {
      tlStatus.forEach((status, j) => {
        trafficLightsG[tlEdge][j].tint = statusToColor(status)
        trafficLightsG[tlEdge][j].alpha = status === 'i' ? 0 : 1
      })
    }
  })

  // 重置所有道路每个车道的车辆计数
  for (const edgeId in edgeLaneCarCount) {
    edgeLaneCarCount[edgeId].fill(0)
  }

  // 更新车辆
  carContainer.removeChildren()
  turnSignalContainer.removeChildren()
  
  const carLogArray = carLogs.split(',')
  
  // 收集当前帧所有车辆ID
  const currentCarIds = new Set()
  for (let i = 0; i < carLogArray.length - 1; i++) {
    const carLog = carLogArray[i].split(' ')
    const carId = carLog[3]
    currentCarIds.add(carId)
  }
  
  // 清理不在当前场景中的车辆的历史记录
  for (const carId in carPositionHistory) {
    if (!currentCarIds.has(carId)) {
      delete carPositionHistory[carId]
    }
  }
  
  for (let i = 0; i < carLogArray.length - 1; i++) {
    const carLog = carLogArray[i].split(' ')
    const position = transCoord([parseFloat(carLog[0]), parseFloat(carLog[1])])
    const rotation = 2 * Math.PI - parseFloat(carLog[2])
    const carId = carLog[3]
    const laneChange = parseInt(carLog[4]) + 1
    const length = parseFloat(carLog[5])
    const width = parseFloat(carLog[6])

    // 检查车辆是否在路口附近停留
    const carPoint = new Point(position[0], position[1])
    const stoppedInfo = isCarStoppedNearIntersection(carId, carPoint, step)
    
    // 只统计停留在路口附近的车辆
    if (stoppedInfo.isStopped) {
      const laneInfo = findCarLane(carPoint)
      if (laneInfo) {
        const { edgeId, laneIndex } = laneInfo
        if (edgeLaneCarCount[edgeId]) {
          edgeLaneCarCount[edgeId][laneIndex]++
        }
      }
    }

    const [car, signal] = carPool[i]
    car.position.set(position[0], position[1])
    car.rotation = rotation
    car.name = carId
    car.tint = CONSTANTS.CAR_COLORS[stringHash(carId) % CONSTANTS.CAR_COLORS_NUM]
    car.width = length
    car.height = width
    carContainer.addChild(car)

    signal.position.set(position[0], position[1])
    signal.rotation = rotation
    signal.texture = turnSignalTextures[laneChange]
    signal.width = length
    signal.height = width
    turnSignalContainer.addChild(signal)
  }

  // 根据每个车道的车辆数量更新道路颜色
  updateEdgeColors()

  if (updateCallback) {
    updateCallback({
      carNum: carLogArray.length - 1,
      currentStep: step + 1,
      selectedEntity: ''
    })
  }
}

function animate() {
  if (!paused) {
    frameElapsed += 1
    if (frameElapsed >= 1 / (replaySpeed ** 2)) {
      currentStep = (currentStep + 1) % totalStep
      drawStep(currentStep)
      frameElapsed = 0
    }
  }
  animationId = requestAnimationFrame(animate)
}

function startSimulation(callback) {
  updateCallback = callback
  currentStep = 0
  drawStep(currentStep)
  animate()
}

function setPaused(value) {
  paused = value
}

function setReplaySpeed(value) {
  replaySpeed = value
}

function stepForward() {
  currentStep = (currentStep + 1) % totalStep
  drawStep(currentStep)
}

function stepBackward() {
  currentStep = (currentStep - 1 + totalStep) % totalStep
  drawStep(currentStep)
}

onMounted(() => {
  // 初始化会在initSimulator中进行
})

onUnmounted(() => {
  if (animationId) {
    cancelAnimationFrame(animationId)
  }
  if (app) {
    app.destroy(true)
  }
})
</script>

<style scoped>
.simulator-canvas {
  flex: 1;
  position: relative;
  background: #e8ebed;
  overflow: hidden;
}

.spinner {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 50px;
  height: 40px;
  text-align: center;
  font-size: 10px;
}

.spinner > div {
  background-color: #333;
  height: 100%;
  width: 6px;
  display: inline-block;
  margin: 0 2px;
  animation: sk-stretchdelay 1.2s infinite ease-in-out;
}

.spinner .rect2 {
  animation-delay: -1.1s;
}

.spinner .rect3 {
  animation-delay: -1.0s;
}

.spinner .rect4 {
  animation-delay: -0.9s;
}

.spinner .rect5 {
  animation-delay: -0.8s;
}

@keyframes sk-stretchdelay {
  0%, 40%, 100% { transform: scaleY(0.4); }
  20% { transform: scaleY(1.0); }
}
</style>
