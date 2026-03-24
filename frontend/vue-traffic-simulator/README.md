# Vue 交通流模拟器

这是一个使用 Vue 3 + PIXI.js 重构的交通流模拟器项目,用于可视化展示交通流模拟数据。

## 技术栈

- **Vue 3** - 使用 Composition API
- **Vite** - 快速的开发构建工具
- **PIXI.js 7** - 高性能 2D 渲染引擎
- **pixi-viewport** - 视口控制(拖拽、缩放等)

## 项目结构

```
vue-traffic-simulator/
├── src/
│   ├── components/
│   │   ├── ControlPanel.vue      # 控制面板组件
│   │   └── SimulatorCanvas.vue   # 模拟器画布组件
│   ├── utils/
│   │   ├── Point.js              # 点类工具
│   │   ├── constants.js          # 常量配置
│   │   └── helpers.js            # 辅助函数
│   ├── App.vue                   # 主应用组件
│   ├── main.js                   # 入口文件
│   └── style.css                 # 全局样式
├── index.html
├── package.json
└── vite.config.js
```

## 功能特点

- ✅ 路网可视化渲染
- ✅ 交通灯状态显示
- ✅ 车辆动画回放
- ✅ 回放速度控制
- ✅ 暂停/继续功能
- ✅ 单步前进/后退
- ✅ 视口拖拽和缩放
- ✅ 文件上传支持
- ✅ 调试模式

## 安装和运行

### 安装依赖

```bash
npm install
```

### 启动开发服务器

```bash
npm run dev
```

访问 `http://localhost:5173` 查看应用

### 构建生产版本

```bash
npm run build
```

### 预览生产构建

```bash
npm run preview
```

## 使用说明

1. **上传文件**
   - 点击"路网文件"上传 roadnet JSON 文件
   - 点击"回放文件"上传 replay TXT 文件
   - (可选) 上传图表文件

2. **开始模拟**
   - 点击"开始"按钮启动模拟
   - 可以勾选"调试模式"以显示更多信息

3. **控制回放**
   - 使用滑块或 +/- 按钮调整回放速度
   - 点击"暂停"按钮暂停/继续
   - 使用键盘快捷键:
     - `P` - 暂停/继续
     - `[` - 单步后退
     - `]` - 单步前进
     - `1` - 减速 (1.5倍)
     - `2` - 加速 (1.5倍)

4. **视口控制**
   - 🖱️ **鼠标拖拽**: 按住左键拖动画布
   - 🔍 **滚轮缩放**: 向上滚动放大,向下滚动缩小
   - 📱 **触摸缩放**: 双指捏合/展开(触摸屏)
   - 🎯 **缩放限制**: 0.1倍 ~ 5倍
   - ⚡ **惯性滚动**: 拖动松开后有平滑减速效果

## 数据格式

### 路网文件 (JSON)
```json
{
  "static": {
    "nodes": [...],
    "edges": [...]
  }
}
```

### 回放文件 (TXT)
每行格式: `车辆数据;交通灯数据`
- 车辆数据: `x y rotation id laneChange length width`
- 交通灯数据: `edgeId status1 status2 ...`

## 与原项目的区别

1. **架构改进**
   - 使用 Vue 3 组件化开发
   - 清晰的关注点分离
   - 更好的代码组织

2. **状态管理**
   - 使用 Vue 响应式系统
   - 统一的状态更新机制

3. **性能优化**
   - 按需渲染组件
   - 合理使用 watch 和 computed

4. **用户体验**
   - 更现代的 UI 设计
   - 更清晰的操作提示
   - 保持原有功能不变

## 开发说明

- 使用 ESM 模块系统
- 遵循 Vue 3 Composition API 规范
- PIXI.js 渲染逻辑封装在 SimulatorCanvas 组件中
- 所有常量统一在 `utils/constants.js` 中管理

## 许可证

本项目基于原 CityFlow 可视化项目重构。
